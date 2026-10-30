#include "core/Runtime.h"

#include "core/Dataset.h"
#include "core/Expression.h"
#include "core/NodeTypes.h"
#include "core/Parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace logievo {

namespace {

constexpr float kEps = 1e-5f;

int normAxis(int64_t axis, size_t rank) {
    if (axis < 0) axis += static_cast<int64_t>(rank);
    return static_cast<int>(std::clamp<int64_t>(axis, 0, std::max<int64_t>(0, static_cast<int64_t>(rank) - 1)));
}

// Splits a shape around an axis: how many blocks before it, its length, how many values after.
void around(const Shape& s, int axis, int64_t& outer, int64_t& len, int64_t& inner) {
    outer = inner = 1;
    for (int i = 0; i < axis; ++i) outer *= s[static_cast<size_t>(i)];
    len = s[static_cast<size_t>(axis)];
    for (size_t i = static_cast<size_t>(axis) + 1; i < s.size(); ++i) inner *= s[i];
}

Tensor reshaped(Tensor t, Shape shape) {
    t.shape = std::move(shape);
    return t;
}

void fill(Tensor& t, int init, std::mt19937& rng, int64_t fanIn, int64_t fanOut) {
    if (init == 2) std::fill(t.v.begin(), t.v.end(), 0.0f);
    else if (init == 3) std::fill(t.v.begin(), t.v.end(), 1.0f);
    else if (init == 1) {
        const float r = std::sqrt(6.0f / static_cast<float>(std::max<int64_t>(1, fanIn + fanOut)));
        std::uniform_real_distribution<float> d(-r, r);
        for (float& v : t.v) v = d(rng);
    } else {
        std::normal_distribution<float> d(0.0f, 0.1f);
        for (float& v : t.v) v = d(rng);
    }
}

} // namespace

// --- weights ---------------------------------------------------------------------------------

std::vector<Shape> weightShapes(const Graph& g, const Node& n, const NodeInfo& info) {
    if (info.state != NodeInfo::State::Ok) return {};
    auto num = [&](const char* id) { return static_cast<int64_t>(g.number(n, id)); };
    const Shape x = info.inputs.empty() ? Shape{} : info.inputs[0];
    if (n.type == "parameter") return {info.shape};
    if (n.type == "dense") {
        std::vector<Shape> s{{x.back(), num("units")}};
        if (g.number(n, "bias") != 0.0) s.push_back({num("units")});
        return s;
    }
    if (n.type == "conv2d") {
        std::vector<Shape> s{{num("filters"), x[1], num("kernel"), num("kernel")}};
        if (g.number(n, "bias") != 0.0) s.push_back({num("filters")});
        return s;
    }
    if (n.type == "rnn") return {{x.back(), num("units")}, {num("units"), num("units")}, {num("units")}};
    if (n.type == "embedding") return {{num("vocab"), num("dim")}};
    if (n.type == "layernorm") return {{x.back()}, {x.back()}};
    return {};
}

void prepareWeights(const Graph& g, const Analysis& a, Weights& w, uint32_t seed) {
    std::mt19937 rng(seed);
    Weights next;
    for (const Node& n : g.nodes) {
        const NodeInfo* info = a.info(n.id);
        if (!info) continue;
        const std::vector<Shape> shapes = weightShapes(g, n, *info);
        if (shapes.empty()) continue;
        auto old = w.find(n.id);
        bool fits = old != w.end() && old->second.size() == shapes.size();
        for (size_t i = 0; fits && i < shapes.size(); ++i) fits = old->second[i].shape == shapes[i];
        if (fits) {
            next[n.id] = std::move(old->second);
            continue;
        }
        std::vector<Tensor> fresh;
        for (size_t i = 0; i < shapes.size(); ++i) {
            Tensor t(shapes[i]);
            const Shape& s = shapes[i];
            if (n.type == "parameter") {
                const int64_t fanOut = s.back(), fanIn = s.size() > 1 ? s[s.size() - 2] : s.back();
                fill(t, static_cast<int>(g.number(n, "init")), rng, fanIn, fanOut);
            } else if (n.type == "dense") {
                fill(t, i == 0 ? 1 : 2, rng, s[0], i == 0 ? s[1] : 1);
            } else if (n.type == "conv2d") {
                fill(t, i == 0 ? 1 : 2, rng, i == 0 ? s[1] * s[2] * s[3] : 1, i == 0 ? s[0] * s[2] * s[3] : 1);
            } else if (n.type == "rnn") {
                fill(t, i < 2 ? 1 : 2, rng, s[0], i < 2 ? s[1] : 1); // Wx, Wh: Xavier; b: zeros
            } else if (n.type == "embedding") {
                fill(t, 0, rng, 1, 1);
            } else if (n.type == "layernorm") {
                fill(t, i == 0 ? 3 : 2, rng, 1, 1);
            }
            fresh.push_back(std::move(t));
        }
        next[n.id] = std::move(fresh);
    }
    w = std::move(next);
}

// --- forward ---------------------------------------------------------------------------------

Executor::Executor(const Graph& graph, const Analysis& analysis) : graph_(graph), analysis_(analysis) {}

const Tensor& Executor::input(const Node& n, int port) const {
    const Link* l = graph_.linkInto(n.id, port);
    return state_.at(l->fromNode).out[static_cast<size_t>(l->fromPort)];
}

const Tensor* Executor::output(int node, int port) const {
    auto it = state_.find(node);
    if (it == state_.end() || port >= static_cast<int>(it->second.out.size())) return nullptr;
    return &it->second.out[static_cast<size_t>(port)];
}

double Executor::forward(const Weights& weights, const Batches& data, bool training, std::mt19937& rng, Metrics* metrics) {
    state_.clear();
    double total = 0.0;
    for (const int id : analysis_.order) {
        const Node& n = *graph_.find(id);
        const NodeInfo& info = *analysis_.info(id);
        State& s = state_[id];
        const std::string& t = n.type;
        auto num = [&](const char* p) { return graph_.number(n, p); };
        auto w = [&](size_t i) -> const Tensor& { return weights.at(id)[i]; };
        Tensor y;
        if (t == "dataset") {
            const auto& batch = data.at(id);
            s.out = {batch[0], batch[1]};
            continue;
        }
        if (t == "parameter") y = w(0);
        else if (t == "constant") y = Tensor(info.shape, static_cast<float>(num("value")));
        else {
            const Tensor& x = input(n, 0);
            if (t == "add") y = binary(x, input(n, 1), [](float a, float b) { return a + b; });
            else if (t == "subtract") y = binary(x, input(n, 1), [](float a, float b) { return a - b; });
            else if (t == "multiply") y = binary(x, input(n, 1), [](float a, float b) { return a * b; });
            else if (t == "divide") y = binary(x, input(n, 1), [](float a, float b) { return a / b; });
            else if (t == "matmul") y = matmul(x, input(n, 1));
            else if (t == "transpose") y = transposeLast(x);
            else if (t == "sum" || t == "mean") {
                const bool mean = t == "mean";
                if (num("all") != 0.0) {
                    const double sum = std::accumulate(x.v.begin(), x.v.end(), 0.0);
                    y = Tensor({}, static_cast<float>(mean ? sum / std::max<int64_t>(1, x.size()) : sum));
                } else {
                    const int axis = normAxis(static_cast<int64_t>(num("axis")), x.shape.size());
                    int64_t outer, len, inner;
                    around(x.shape, axis, outer, len, inner);
                    Shape shape = x.shape;
                    if (num("keep") != 0.0) shape[static_cast<size_t>(axis)] = 1;
                    else shape.erase(shape.begin() + axis);
                    y = Tensor(shape);
                    for (int64_t o = 0; o < outer; ++o)
                        for (int64_t l = 0; l < len; ++l)
                            for (int64_t i = 0; i < inner; ++i) y[o * inner + i] += x[(o * len + l) * inner + i];
                    if (mean)
                        for (float& v : y.v) v /= static_cast<float>(std::max<int64_t>(1, len));
                }
            } else if (t == "power") {
                const float e = static_cast<float>(num("exponent"));
                y = unary(x, [e](float v) { return std::pow(v, e); });
            } else if (t == "exp") y = unary(x, [](float v) { return std::exp(std::min(v, 80.0f)); });
            else if (t == "log") y = unary(x, [](float v) { return std::log(std::max(v, 1e-12f)); });
            else if (t == "relu") y = unary(x, [](float v) { return v > 0 ? v : 0.0f; });
            else if (t == "leakyrelu") {
                const float k = static_cast<float>(num("slope"));
                y = unary(x, [k](float v) { return v > 0 ? v : v * k; });
            } else if (t == "gelu") {
                y = unary(x, [](float v) { return 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v))); });
            } else if (t == "sigmoid") y = unary(x, [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
            else if (t == "tanh") y = unary(x, [](float v) { return std::tanh(v); });
            else if (t == "softmax") {
                y = Tensor(x.shape);
                const int64_t d = x.shape.back(), rows = x.size() / std::max<int64_t>(1, d);
                for (int64_t r = 0; r < rows; ++r) {
                    const float* in = x.v.data() + r * d;
                    float* o = y.v.data() + r * d;
                    const float m = *std::max_element(in, in + d);
                    float sum = 0;
                    for (int64_t i = 0; i < d; ++i) sum += (o[i] = std::exp(in[i] - m));
                    for (int64_t i = 0; i < d; ++i) o[i] /= sum;
                }
            } else if (t == "dropout") {
                const float rate = static_cast<float>(num("rate"));
                if (training && rate > 0) {
                    std::uniform_real_distribution<float> u(0.0f, 1.0f);
                    s.cache = Tensor(x.shape);
                    for (float& m : s.cache.v) m = u(rng) >= rate ? 1.0f / (1.0f - rate) : 0.0f;
                    y = binary(x, s.cache, [](float a, float b) { return a * b; });
                } else {
                    y = x;
                }
            } else if (t == "dense") {
                const int64_t in = x.shape.back(), rows = x.size() / in;
                y = matmul(reshaped(x, {rows, in}), w(0));
                if (num("bias") != 0.0) {
                    const Tensor& b = w(1);
                    const int64_t units = b.size();
                    for (int64_t r = 0; r < rows; ++r)
                        for (int64_t u = 0; u < units; ++u) y[r * units + u] += b[u];
                }
                Shape shape = x.shape;
                shape.back() = w(0).shape[1];
                y.shape = shape;
            } else if (t == "conv2d") {
                const Tensor& k = w(0);
                const int64_t N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
                const int64_t F = k.shape[0], K = k.shape[2], st = static_cast<int64_t>(num("stride"));
                const int64_t pad = num("padding") == 0.0 ? K / 2 : 0;
                const int64_t Ho = (H + 2 * pad - K) / st + 1, Wo = (W + 2 * pad - K) / st + 1;
                y = Tensor({N, F, Ho, Wo});
                const bool bias = num("bias") != 0.0;
                const Tensor* bt = bias ? &w(1) : nullptr;
                // Each (example, filter) on its own: shared among the processor's threads.
                parallelFor(N * F, C * K * K * Ho * Wo, [&](int64_t begin, int64_t end) {
                for (int64_t bf = begin; bf < end; ++bf) {
                    const int64_t b = bf / F, f = bf % F;
                        for (int64_t oy = 0; oy < Ho; ++oy)
                            for (int64_t ox = 0; ox < Wo; ++ox) {
                                float sum = bt ? (*bt)[f] : 0.0f;
                                for (int64_t c = 0; c < C; ++c)
                                    for (int64_t ky = 0; ky < K; ++ky) {
                                        const int64_t iy = oy * st - pad + ky;
                                        if (iy < 0 || iy >= H) continue;
                                        for (int64_t kx = 0; kx < K; ++kx) {
                                            const int64_t ix = ox * st - pad + kx;
                                            if (ix < 0 || ix >= W) continue;
                                            sum += x[((b * C + c) * H + iy) * W + ix] * k[((f * C + c) * K + ky) * K + kx];
                                        }
                                    }
                                y[((b * F + f) * Ho + oy) * Wo + ox] = sum;
                            }
                }
                });
            } else if (t == "maxpool2d") {
                const int64_t N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
                const int64_t K = static_cast<int64_t>(num("kernel")), st = static_cast<int64_t>(num("stride"));
                const int64_t Ho = (H - K) / st + 1, Wo = (W - K) / st + 1;
                y = Tensor({N, C, Ho, Wo});
                s.index.assign(static_cast<size_t>(y.size()), 0);
                for (int64_t p = 0; p < N * C; ++p)
                    for (int64_t oy = 0; oy < Ho; ++oy)
                        for (int64_t ox = 0; ox < Wo; ++ox) {
                            int64_t best = (p * H + oy * st) * W + ox * st;
                            for (int64_t ky = 0; ky < K; ++ky)
                                for (int64_t kx = 0; kx < K; ++kx) {
                                    const int64_t i = (p * H + oy * st + ky) * W + ox * st + kx;
                                    if (x[i] > x[best]) best = i;
                                }
                            const int64_t o = (p * Ho + oy) * Wo + ox;
                            y[o] = x[best];
                            s.index[static_cast<size_t>(o)] = best;
                        }
            } else if (t == "globalpool") {
                const int64_t N = x.shape[0], C = x.shape[1], HW = x.shape[2] * x.shape[3];
                y = Tensor({N, C});
                for (int64_t p = 0; p < N * C; ++p) {
                    float sum = 0;
                    for (int64_t i = 0; i < HW; ++i) sum += x[p * HW + i];
                    y[p] = sum / static_cast<float>(HW);
                }
            } else if (t == "rnn") {
                // Step by step: h = tanh(x·Wx + h·Wh + b), from h = 0. The input part of every
                // step is one matrix product; every h is kept for backward.
                const Tensor &wx = w(0), &wh = w(1), &b = w(2);
                const int64_t F = x.shape.back(), T = x.shape[x.shape.size() - 2], H = wh.shape[0], R = x.size() / (T * F);
                const Tensor xw = matmul(reshaped(x, {R * T, F}), wx);
                s.cache = Tensor({R, T, H});
                Tensor h({R, H});
                for (int64_t st = 0; st < T; ++st) {
                    const Tensor hw = matmul(h, wh);
                    for (int64_t r = 0; r < R; ++r)
                        for (int64_t u = 0; u < H; ++u) {
                            const float v = std::tanh(hw[r * H + u] + xw[(r * T + st) * H + u] + b[u]);
                            h[r * H + u] = v;
                            s.cache[(r * T + st) * H + u] = v;
                        }
                }
                Shape shape = x.shape;
                shape.back() = H;
                if (num("all") != 0.0) y = reshaped(s.cache, shape);
                else {
                    shape.erase(shape.end() - 2);
                    y = reshaped(h, shape);
                }
            } else if (t == "flatten") {
                const bool batched = !info.inputs[0].empty() && info.inputs[0][0] == kBatch;
                y = reshaped(x, batched ? Shape{x.shape[0], x.size() / std::max<int64_t>(1, x.shape[0])} : Shape{x.size()});
            } else if (t == "reshape") {
                Shape shape = info.shape;
                if (!shape.empty() && shape[0] == kBatch) shape[0] = x.shape[0];
                y = reshaped(x, shape);
            } else if (t == "layernorm") {
                const int64_t d = x.shape.back(), rows = x.size() / d;
                y = Tensor(x.shape);
                s.cache = Tensor(x.shape); // x̂, and 1 / standard deviation per row
                s.rowScale.assign(static_cast<size_t>(rows), 0.0f);
                for (int64_t r = 0; r < rows; ++r) {
                    const float* in = x.v.data() + r * d;
                    double mean = 0, var = 0;
                    for (int64_t i = 0; i < d; ++i) mean += in[i];
                    mean /= static_cast<double>(d);
                    for (int64_t i = 0; i < d; ++i) var += (in[i] - mean) * (in[i] - mean);
                    var /= static_cast<double>(d);
                    const float is = static_cast<float>(1.0 / std::sqrt(var + kEps));
                    s.rowScale[static_cast<size_t>(r)] = is;
                    for (int64_t i = 0; i < d; ++i) {
                        const float xh = static_cast<float>(in[i] - mean) * is;
                        s.cache[r * d + i] = xh;
                        y[r * d + i] = xh * w(0)[i] + w(1)[i];
                    }
                }
            } else if (t == "embedding") {
                const Tensor& e = w(0);
                const int64_t vocab = e.shape[0], dim = e.shape[1];
                Shape shape = x.shape;
                shape.push_back(dim);
                y = Tensor(shape);
                s.index.resize(static_cast<size_t>(x.size()));
                for (int64_t i = 0; i < x.size(); ++i) {
                    const int64_t row = std::clamp<int64_t>(std::llround(x[i]), 0, vocab - 1);
                    s.index[static_cast<size_t>(i)] = row;
                    std::copy(e.v.begin() + row * dim, e.v.begin() + (row + 1) * dim, y.v.begin() + i * dim);
                }
            } else if (t == "concat") {
                const Tensor& b = input(n, 1);
                const int axis = normAxis(static_cast<int64_t>(num("axis")), x.shape.size());
                int64_t outer, la, inner, lb;
                around(x.shape, axis, outer, la, inner);
                lb = b.shape[static_cast<size_t>(axis)];
                Shape shape = x.shape;
                shape[static_cast<size_t>(axis)] = la + lb;
                y = Tensor(shape);
                for (int64_t o = 0; o < outer; ++o) {
                    std::copy_n(x.v.begin() + o * la * inner, la * inner, y.v.begin() + o * (la + lb) * inner);
                    std::copy_n(b.v.begin() + o * lb * inner, lb * inner, y.v.begin() + (o * (la + lb) + la) * inner);
                }
            } else if (t == "custom") {
                // The user's formula at every value, with its derivative for each input (kept for backward).
                const int count = static_cast<int>(inputPorts(n).size());
                Expression code;
                code.compile(graph_.text(n, "code"), count);
                std::vector<const Tensor*> ins;
                Shape shape = x.shape;
                for (int i = 0; i < count; ++i) {
                    ins.push_back(&input(n, i));
                    Shape joined;
                    broadcastShapes(shape, ins.back()->shape, joined);
                    shape = joined;
                }
                y = Tensor(shape);
                std::vector<std::vector<int64_t>> maps;
                for (const Tensor* in : ins) maps.push_back(broadcastIndex(shape, in->shape));
                s.derivs.assign(static_cast<size_t>(count), Tensor(shape));
                parallelFor(y.size(), 64, [&](int64_t begin, int64_t end) {
                    float values[Expression::kMaxInputs] = {}, grad[Expression::kMaxInputs] = {};
                    for (int64_t i = begin; i < end; ++i) {
                        for (int j = 0; j < count; ++j) values[j] = (*ins[static_cast<size_t>(j)])[maps[static_cast<size_t>(j)][static_cast<size_t>(i)]];
                        y[i] = code.eval(values, grad);
                        for (int j = 0; j < count; ++j) s.derivs[static_cast<size_t>(j)][i] = grad[j];
                    }
                });
            } else if (t == "loss") {
                const Tensor& target = input(n, 1);
                const int kind = static_cast<int>(num("kind"));
                const float weight = static_cast<float>(num("weight"));
                double loss = 0;
                if (kind == 0) { // cross entropy: softmax of the scores, -log of the right class
                    const int64_t N = x.shape[0], C = x.shape[1];
                    s.cache = Tensor(x.shape);
                    for (int64_t r = 0; r < N; ++r) {
                        const float* in = x.v.data() + r * C;
                        float* p = s.cache.v.data() + r * C;
                        const float m = *std::max_element(in, in + C);
                        float sum = 0;
                        for (int64_t c = 0; c < C; ++c) sum += (p[c] = std::exp(in[c] - m));
                        for (int64_t c = 0; c < C; ++c) p[c] /= sum;
                        const int64_t cls = std::clamp<int64_t>(std::llround(target[r]), 0, C - 1);
                        loss -= std::log(std::max(p[cls], 1e-12f));
                        if (metrics) {
                            metrics->correct += (std::max_element(p, p + C) - p) == cls;
                            metrics->counted += 1;
                        }
                    }
                    loss /= static_cast<double>(std::max<int64_t>(1, N));
                } else if (kind == 1) { // mean squared error
                    const Tensor d = binary(x, target, [](float a, float b) { return a - b; });
                    for (float v : d.v) loss += static_cast<double>(v) * v;
                    loss /= static_cast<double>(std::max<int64_t>(1, d.size()));
                } else { // binary cross entropy on scores (logits)
                    const Tensor l = binary(x, target, [](float z, float y) { return std::max(z, 0.0f) - z * y + std::log1p(std::exp(-std::abs(z))); });
                    loss = std::accumulate(l.v.begin(), l.v.end(), 0.0) / static_cast<double>(std::max<int64_t>(1, l.size()));
                    if (metrics) {
                        const Tensor right = binary(x, target, [](float z, float y) { return (z > 0) == (y > 0.5f) ? 1.0f : 0.0f; });
                        metrics->correct += std::accumulate(right.v.begin(), right.v.end(), 0.0);
                        metrics->counted += static_cast<double>(right.size());
                    }
                }
                y = Tensor({}, static_cast<float>(loss * weight));
                total += loss * weight;
            }
        }
        s.out.assign(1, std::move(y));
    }
    return total;
}

// --- backward --------------------------------------------------------------------------------

void Executor::addGrad(const Node& n, int port, Tensor g) {
    const Link* l = graph_.linkInto(n.id, port);
    if (!l) return;
    State& src = state_.at(l->fromNode);
    if (src.grad.size() < src.out.size()) src.grad.resize(src.out.size());
    const Tensor& out = src.out[static_cast<size_t>(l->fromPort)];
    Tensor& acc = src.grad[static_cast<size_t>(l->fromPort)];
    g = reduceTo(g, out.shape);
    if (acc.v.empty()) acc = std::move(g);
    else
        for (int64_t i = 0; i < acc.size(); ++i) acc[i] += g[i];
}

void Executor::backward(const Weights& weights, Weights& grads) {
    grads.clear();
    for (const auto& [id, ts] : weights) {
        auto& gs = grads[id];
        for (const Tensor& t : ts) gs.emplace_back(t.shape);
    }
    // Every loss starts with its own weight: d(total) / d(loss) = 1, and the loss holds weight × value.
    for (auto it = analysis_.order.rbegin(); it != analysis_.order.rend(); ++it) {
        const Node& n = *graph_.find(*it);
        State& s = state_.at(n.id);
        const std::string& t = n.type;
        auto num = [&](const char* p) { return graph_.number(n, p); };
        if (t == "loss") s.grad.assign(1, Tensor({}, static_cast<float>(num("weight"))));
        if (s.grad.empty() || s.grad[0].v.empty()) continue;
        const Tensor& g = s.grad[0];
        const Tensor& y = s.out[0];
        auto w = [&](size_t i) -> const Tensor& { return weights.at(n.id)[i]; };
        auto gw = [&](size_t i) -> Tensor& { return grads.at(n.id)[i]; };
        if (t == "dataset" || t == "constant") continue;
        if (t == "parameter") {
            const Tensor r = reduceTo(g, gw(0).shape);
            for (int64_t i = 0; i < r.size(); ++i) gw(0)[i] += r[i];
            continue;
        }
        const Tensor& x = input(n, 0);
        if (t == "add" || t == "subtract") {
            addGrad(n, 0, g);
            addGrad(n, 1, t == "add" ? g : unary(g, [](float v) { return -v; }));
        } else if (t == "multiply") {
            const Tensor& b = input(n, 1);
            addGrad(n, 0, binary(g, b, [](float a, float c) { return a * c; }));
            addGrad(n, 1, binary(g, x, [](float a, float c) { return a * c; }));
        } else if (t == "divide") {
            const Tensor& b = input(n, 1);
            addGrad(n, 0, binary(g, b, [](float a, float c) { return a / c; }));
            const Tensor gy = binary(g, y, [](float a, float c) { return a * c; });
            addGrad(n, 1, binary(gy, b, [](float a, float c) { return -a / c; }));
        } else if (t == "matmul") {
            const Tensor& b = input(n, 1);
            addGrad(n, 0, matmul(g, b, false, true));
            addGrad(n, 1, matmul(x, g, true, false));
        } else if (t == "transpose") {
            addGrad(n, 0, transposeLast(g));
        } else if (t == "sum" || t == "mean") {
            Tensor dx(x.shape);
            if (num("all") != 0.0) {
                const float v = t == "mean" ? g[0] / static_cast<float>(std::max<int64_t>(1, x.size())) : g[0];
                std::fill(dx.v.begin(), dx.v.end(), v);
            } else {
                const int axis = normAxis(static_cast<int64_t>(num("axis")), x.shape.size());
                int64_t outer, len, inner;
                around(x.shape, axis, outer, len, inner);
                const float scale = t == "mean" ? 1.0f / static_cast<float>(std::max<int64_t>(1, len)) : 1.0f;
                for (int64_t o = 0; o < outer; ++o)
                    for (int64_t l = 0; l < len; ++l)
                        for (int64_t i = 0; i < inner; ++i) dx[(o * len + l) * inner + i] = g[o * inner + i] * scale;
            }
            addGrad(n, 0, std::move(dx));
        } else if (t == "power") {
            const float e = static_cast<float>(num("exponent"));
            addGrad(n, 0, binary(g, x, [e](float a, float v) { return a * e * std::pow(v, e - 1.0f); }));
        } else if (t == "exp") {
            addGrad(n, 0, binary(g, y, [](float a, float v) { return a * v; }));
        } else if (t == "log") {
            addGrad(n, 0, binary(g, x, [](float a, float v) { return a / std::max(v, 1e-12f); }));
        } else if (t == "relu") {
            addGrad(n, 0, binary(g, x, [](float a, float v) { return v > 0 ? a : 0.0f; }));
        } else if (t == "leakyrelu") {
            const float k = static_cast<float>(num("slope"));
            addGrad(n, 0, binary(g, x, [k](float a, float v) { return v > 0 ? a : a * k; }));
        } else if (t == "gelu") {
            addGrad(n, 0, binary(g, x, [](float a, float v) {
                const float c = 0.7978845608f, u = c * (v + 0.044715f * v * v * v), th = std::tanh(u);
                return a * (0.5f * (1 + th) + 0.5f * v * (1 - th * th) * c * (1 + 3 * 0.044715f * v * v));
            }));
        } else if (t == "sigmoid") {
            addGrad(n, 0, binary(g, y, [](float a, float v) { return a * v * (1 - v); }));
        } else if (t == "tanh") {
            addGrad(n, 0, binary(g, y, [](float a, float v) { return a * (1 - v * v); }));
        } else if (t == "softmax") {
            Tensor dx(x.shape);
            const int64_t d = x.shape.back(), rows = x.size() / std::max<int64_t>(1, d);
            for (int64_t r = 0; r < rows; ++r) {
                float dot = 0;
                for (int64_t i = 0; i < d; ++i) dot += g[r * d + i] * y[r * d + i];
                for (int64_t i = 0; i < d; ++i) dx[r * d + i] = y[r * d + i] * (g[r * d + i] - dot);
            }
            addGrad(n, 0, std::move(dx));
        } else if (t == "dropout") {
            addGrad(n, 0, s.cache.v.empty() ? g : binary(g, s.cache, [](float a, float m) { return a * m; }));
        } else if (t == "dense") {
            const int64_t in = x.shape.back(), rows = x.size() / in, units = w(0).shape[1];
            const Tensor g2 = reshaped(g, {rows, units}), x2 = reshaped(x, {rows, in});
            const Tensor dw = matmul(x2, g2, true, false);
            for (int64_t i = 0; i < dw.size(); ++i) gw(0)[i] += dw[i];
            if (num("bias") != 0.0)
                for (int64_t r = 0; r < rows; ++r)
                    for (int64_t u = 0; u < units; ++u) gw(1)[u] += g2[r * units + u];
            addGrad(n, 0, reshaped(matmul(g2, w(0), false, true), x.shape));
        } else if (t == "rnn") {
            // Back through the steps, last to first: each step's h gets the gradient from the
            // output and from the step after it.
            const Tensor &wx = w(0), &wh = w(1);
            const int64_t F = x.shape.back(), T = x.shape[x.shape.size() - 2], H = wh.shape[0], R = x.size() / (T * F);
            const bool all = num("all") != 0.0;
            Tensor da({R * T, H}); // the gradient before tanh, at every step
            Tensor dh({R, H}), dat({R, H}), prev({R, H});
            for (int64_t st = T - 1; st >= 0; --st) {
                for (int64_t r = 0; r < R; ++r)
                    for (int64_t u = 0; u < H; ++u) {
                        const int64_t i = (r * T + st) * H + u;
                        const float out = all ? g[i] : st == T - 1 ? g[r * H + u] : 0.0f;
                        const float hv = s.cache[i];
                        dat[r * H + u] = da[i] = (dh[r * H + u] + out) * (1 - hv * hv);
                    }
                if (st > 0) {
                    for (int64_t r = 0; r < R; ++r)
                        for (int64_t u = 0; u < H; ++u) prev[r * H + u] = s.cache[(r * T + st - 1) * H + u];
                    const Tensor dwh = matmul(prev, dat, true, false);
                    for (int64_t i = 0; i < dwh.size(); ++i) gw(1)[i] += dwh[i];
                }
                dh = matmul(dat, wh, false, true);
            }
            const Tensor dwx = matmul(reshaped(x, {R * T, F}), da, true, false);
            for (int64_t i = 0; i < dwx.size(); ++i) gw(0)[i] += dwx[i];
            for (int64_t i = 0; i < R * T; ++i)
                for (int64_t u = 0; u < H; ++u) gw(2)[u] += da[i * H + u];
            addGrad(n, 0, reshaped(matmul(da, wx, false, true), x.shape));
        } else if (t == "conv2d") {
            const Tensor& k = w(0);
            Tensor& dk = gw(0);
            const int64_t N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
            const int64_t F = k.shape[0], K = k.shape[2], st = static_cast<int64_t>(num("stride"));
            const int64_t pad = num("padding") == 0.0 ? K / 2 : 0;
            const int64_t Ho = y.shape[2], Wo = y.shape[3];
            const bool bias = num("bias") != 0.0;
            Tensor dx(x.shape);
            Tensor* db = bias ? &gw(1) : nullptr;
            const int64_t work = N * Ho * Wo * C * K * K;
            // The kernel's gradient (and the bias's), filters shared among threads...
            parallelFor(F, work, [&](int64_t f0, int64_t f1) {
                for (int64_t f = f0; f < f1; ++f)
                    for (int64_t b = 0; b < N; ++b)
                        for (int64_t oy = 0; oy < Ho; ++oy)
                            for (int64_t ox = 0; ox < Wo; ++ox) {
                                const float go = g[((b * F + f) * Ho + oy) * Wo + ox];
                                if (go == 0.0f) continue;
                                if (db) (*db)[f] += go;
                                for (int64_t c = 0; c < C; ++c)
                                    for (int64_t ky = 0; ky < K; ++ky) {
                                        const int64_t iy = oy * st - pad + ky;
                                        if (iy < 0 || iy >= H) continue;
                                        for (int64_t kx = 0; kx < K; ++kx) {
                                            const int64_t ix = ox * st - pad + kx;
                                            if (ix < 0 || ix >= W) continue;
                                            dk[((f * C + c) * K + ky) * K + kx] += go * x[((b * C + c) * H + iy) * W + ix];
                                        }
                                    }
                            }
            });
            // ... and the input's, examples shared.
            parallelFor(N, F * Ho * Wo * C * K * K, [&](int64_t b0, int64_t b1) {
                for (int64_t b = b0; b < b1; ++b)
                    for (int64_t f = 0; f < F; ++f)
                        for (int64_t oy = 0; oy < Ho; ++oy)
                            for (int64_t ox = 0; ox < Wo; ++ox) {
                                const float go = g[((b * F + f) * Ho + oy) * Wo + ox];
                                if (go == 0.0f) continue;
                                for (int64_t c = 0; c < C; ++c)
                                    for (int64_t ky = 0; ky < K; ++ky) {
                                        const int64_t iy = oy * st - pad + ky;
                                        if (iy < 0 || iy >= H) continue;
                                        for (int64_t kx = 0; kx < K; ++kx) {
                                            const int64_t ix = ox * st - pad + kx;
                                            if (ix < 0 || ix >= W) continue;
                                            dx[((b * C + c) * H + iy) * W + ix] += go * k[((f * C + c) * K + ky) * K + kx];
                                        }
                                    }
                            }
            });
            addGrad(n, 0, std::move(dx));
        } else if (t == "maxpool2d") {
            Tensor dx(x.shape);
            for (int64_t o = 0; o < g.size(); ++o) dx[s.index[static_cast<size_t>(o)]] += g[o];
            addGrad(n, 0, std::move(dx));
        } else if (t == "globalpool") {
            Tensor dx(x.shape);
            const int64_t HW = x.shape[2] * x.shape[3];
            for (int64_t p = 0; p < g.size(); ++p)
                for (int64_t i = 0; i < HW; ++i) dx[p * HW + i] = g[p] / static_cast<float>(HW);
            addGrad(n, 0, std::move(dx));
        } else if (t == "flatten" || t == "reshape") {
            addGrad(n, 0, reshaped(g, x.shape));
        } else if (t == "layernorm") {
            const int64_t d = x.shape.back(), rows = x.size() / d;
            Tensor dx(x.shape);
            for (int64_t r = 0; r < rows; ++r) {
                const float is = s.rowScale[static_cast<size_t>(r)];
                double sumD = 0, sumDX = 0;
                for (int64_t i = 0; i < d; ++i) {
                    const float gi = g[r * d + i], xh = s.cache[r * d + i];
                    gw(0)[i] += gi * xh;
                    gw(1)[i] += gi;
                    const float dxh = gi * w(0)[i];
                    sumD += dxh;
                    sumDX += dxh * xh;
                }
                for (int64_t i = 0; i < d; ++i) {
                    const float dxh = g[r * d + i] * w(0)[i], xh = s.cache[r * d + i];
                    dx[r * d + i] = is / static_cast<float>(d) * (static_cast<float>(d) * dxh - static_cast<float>(sumD) - xh * static_cast<float>(sumDX));
                }
            }
            addGrad(n, 0, std::move(dx));
        } else if (t == "embedding") {
            const int64_t dim = w(0).shape[1];
            for (size_t i = 0; i < s.index.size(); ++i)
                for (int64_t j = 0; j < dim; ++j) gw(0)[s.index[i] * dim + j] += g[static_cast<int64_t>(i) * dim + j];
        } else if (t == "concat") {
            const Tensor& b = input(n, 1);
            const int axis = normAxis(static_cast<int64_t>(num("axis")), x.shape.size());
            int64_t outer, la, inner;
            around(x.shape, axis, outer, la, inner);
            const int64_t lb = b.shape[static_cast<size_t>(axis)];
            Tensor da(x.shape), db(b.shape);
            for (int64_t o = 0; o < outer; ++o) {
                std::copy_n(g.v.begin() + o * (la + lb) * inner, la * inner, da.v.begin() + o * la * inner);
                std::copy_n(g.v.begin() + (o * (la + lb) + la) * inner, lb * inner, db.v.begin() + o * lb * inner);
            }
            addGrad(n, 0, std::move(da));
            addGrad(n, 1, std::move(db));
        } else if (t == "custom") {
            for (size_t j = 0; j < s.derivs.size(); ++j)
                addGrad(n, static_cast<int>(j), binary(g, s.derivs[j], [](float a, float d) { return a * d; }));
        } else if (t == "loss") {
            const Tensor& target = input(n, 1);
            const int kind = static_cast<int>(num("kind"));
            const float scale = g[0];
            if (kind == 0) {
                const int64_t N = x.shape[0], C = x.shape[1];
                Tensor dx = s.cache;
                for (int64_t r = 0; r < N; ++r) {
                    const int64_t cls = std::clamp<int64_t>(std::llround(target[r]), 0, C - 1);
                    dx[r * C + cls] -= 1.0f;
                }
                for (float& v : dx.v) v *= scale / static_cast<float>(std::max<int64_t>(1, N));
                addGrad(n, 0, std::move(dx));
            } else {
                Tensor dx = kind == 1 ? binary(x, target, [](float a, float b) { return 2.0f * (a - b); })
                                      : binary(x, target, [](float z, float yv) { return 1.0f / (1.0f + std::exp(-z)) - yv; });
                for (float& v : dx.v) v *= scale / static_cast<float>(std::max<int64_t>(1, dx.size()));
                Tensor dt = kind == 1 ? unary(dx, [](float v) { return -v; })
                                      : binary(x, dx, [&](float z, float) { return -z * scale / static_cast<float>(std::max<int64_t>(1, dx.size())); });
                addGrad(n, 0, std::move(dx));
                addGrad(n, 1, std::move(dt));
            }
        }
    }
}

// --- training --------------------------------------------------------------------------------

Batches makeBatches(const Graph& g, const std::vector<int64_t>& rows) {
    Batches b;
    const int64_t count = static_cast<int64_t>(rows.size());
    for (const Node& n : g.nodes) {
        if (n.type != "dataset") continue;
        const auto d = loadDataset(g, n);
        if (!d->error.empty() || d->rows == 0) continue;
        Tensor x({count, d->features}), y(d->classes ? Shape{count} : Shape{count, 1});
        for (int64_t i = 0; i < count; ++i) {
            const int64_t r = rows[static_cast<size_t>(i)] % d->rows;
            if (!d->texts.empty()) {
                const std::vector<float> f = textFeatures(d->texts[static_cast<size_t>(r)]);
                std::copy(f.begin(), f.end(), x.v.begin() + i * d->features);
            } else {
                std::copy_n(d->x.begin() + r * d->features, d->features, x.v.begin() + i * d->features);
            }
            y[i] = d->y[static_cast<size_t>(r)];
        }
        b[n.id] = {std::move(x), std::move(y)};
    }
    return b;
}

Trainer::~Trainer() { stop(); }

bool Trainer::start(const Graph& graph, const Weights& weights) {
    stop();
    const Analysis a = analyze(graph);
    std::lock_guard lock(mutex_);
    report_ = {};
    report_.epochs = graph.training.epochs;
    if (!a.valid()) {
        report_.error = a.hasLoss ? "Fix the problems in the graph first" : "Add a Loss node: training makes it smaller";
        return false;
    }
    running_ = true;
    stop_ = false;
    report_.running = true;
    thread_ = std::thread([this, graph, weights] { run(graph, weights); });
    return true;
}

void Trainer::stop() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
    running_ = false;
}

TrainReport Trainer::report() const {
    std::lock_guard lock(mutex_);
    return report_;
}

bool Trainer::takeWeights(Weights& out) {
    std::lock_guard lock(mutex_);
    if (!fresh_) return false;
    out = latest_;
    fresh_ = false;
    return true;
}

void Trainer::run(Graph graph, Weights weights) {
    const Analysis a = analyze(graph);
    prepareWeights(graph, a, weights);
    const TrainSettings ts = graph.training;
    std::mt19937 rng(42);
    int64_t examples = 0;
    for (const Node& n : graph.nodes)
        if (n.type == "dataset") examples = std::max(examples, loadDataset(graph, n)->rows);
    const int64_t batch = std::max(1, ts.batchSize);
    std::vector<int64_t> order(static_cast<size_t>(std::max<int64_t>(1, examples)));
    std::iota(order.begin(), order.end(), 0);
    // Adam's running averages.
    Weights m, v;
    for (const auto& [id, list] : weights)
        for (const Tensor& t : list) {
            m[id].emplace_back(t.shape);
            v[id].emplace_back(t.shape);
        }
    Executor exec(graph, a);
    Weights grads;
    int64_t step = 0;
    const float lr = static_cast<float>(ts.learningRate), b1 = 0.9f, b2 = 0.999f;
    for (int epoch = 0; epoch < ts.epochs && !stop_; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double lossSum = 0;
        int batches = 0;
        Metrics metrics;
        for (int64_t start = 0; start < static_cast<int64_t>(order.size()) && !stop_; start += batch) {
            const std::vector<int64_t> rows(order.begin() + start, order.begin() + std::min<int64_t>(start + batch, static_cast<int64_t>(order.size())));
            const double loss = exec.forward(weights, makeBatches(graph, rows), true, rng, &metrics);
            if (!std::isfinite(loss)) {
                std::lock_guard lock(mutex_);
                report_.error = "The loss stopped being a number: try a smaller learning rate";
                stop_ = true;
                break;
            }
            exec.backward(weights, grads);
            ++step;
            const float c1 = 1.0f - std::pow(b1, static_cast<float>(step)), c2 = 1.0f - std::pow(b2, static_cast<float>(step));
            for (auto& [id, list] : weights)
                for (size_t k = 0; k < list.size(); ++k) {
                    Tensor& p = list[k];
                    const Tensor& gr = grads[id][k];
                    if (ts.optimizer == TrainSettings::Sgd) {
                        for (int64_t i = 0; i < p.size(); ++i) p[i] -= lr * gr[i];
                    } else {
                        Tensor &mm = m[id][k], &vv = v[id][k];
                        for (int64_t i = 0; i < p.size(); ++i) {
                            mm[i] = b1 * mm[i] + (1 - b1) * gr[i];
                            vv[i] = b2 * vv[i] + (1 - b2) * gr[i] * gr[i];
                            p[i] -= lr * (mm[i] / c1) / (std::sqrt(vv[i] / c2) + 1e-8f);
                        }
                    }
                }
            lossSum += loss;
            ++batches;
        }
        if (batches == 0) break;
        std::lock_guard lock(mutex_);
        report_.epoch = epoch + 1;
        report_.loss.push_back(static_cast<float>(lossSum / batches));
        if (metrics.counted > 0) report_.accuracy.push_back(static_cast<float>(metrics.correct / metrics.counted));
        latest_ = weights;
        fresh_ = true;
    }
    std::lock_guard lock(mutex_);
    report_.running = false;
    running_ = false;
}

} // namespace logievo
