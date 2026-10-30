#include "core/Analysis.h"

#include "core/Dataset.h"
#include "core/Expression.h"
#include "core/NodeTypes.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace logievo {

bool parseShape(const std::string& text, Shape& out) {
    out.clear();
    int64_t v = -1;
    for (char c : text + " ") {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            v = (v < 0 ? 0 : v) * 10 + (c - '0');
            if (v > (int64_t(1) << 40)) return false;
        } else if (c == 'x' || c == 'X' || c == ',' || c == ' ' || c == '*') {
            if (v == 0) return false;
            if (v > 0) out.push_back(v);
            v = -1;
        } else {
            return false;
        }
    }
    return !out.empty();
}

std::string shapeText(const Shape& shape) {
    if (shape.empty()) return "scalar";
    std::string s;
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? " × " : "") + (shape[i] == kBatch ? std::string("B") : std::to_string(shape[i]));
    return s;
}

std::string countText(int64_t count) {
    char buf[32];
    if (count >= 1'000'000'000) std::snprintf(buf, sizeof buf, "%.2fB", count / 1e9);
    else if (count >= 1'000'000) std::snprintf(buf, sizeof buf, "%.2fM", count / 1e6);
    else if (count >= 10'000) std::snprintf(buf, sizeof buf, "%.1fK", count / 1e3);
    else std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(count));
    return buf;
}

namespace {

bool known(int64_t d) { return d > 0; }

// An axis counted from the end when negative; -1 when there is no such axis.
int normAxis(int64_t axis, size_t rank) {
    if (axis < 0) axis += static_cast<int64_t>(rank);
    return axis >= 0 && axis < static_cast<int64_t>(rank) ? static_cast<int>(axis) : -1;
}

// Works out one node from what arrives at its inputs: its outputs and parameters, or the error.
std::string infer(const Graph& g, const Node& n, const std::vector<Shape>& in, std::vector<Shape>& out, int64_t& params) {
    auto num = [&](const char* id) { return static_cast<int64_t>(g.number(n, id)); };
    const std::string& t = n.type;
    const Shape x = in.empty() ? Shape{} : in[0];
    out = {x};
    params = 0;
    if (t == "dataset") {
        const auto d = loadDataset(g, n);
        if (!d->error.empty()) return d->error;
        out = {{kBatch, d->features}, d->classes ? Shape{kBatch} : Shape{kBatch, 1}};
        return {};
    }
    if (t == "parameter" || t == "constant") {
        Shape s;
        if (!parseShape(g.text(n, "shape"), s)) return "The shape should be sizes like 2x16";
        out = {s};
        if (t == "parameter") params = elementCount(s);
        return {};
    }
    if (t == "add" || t == "subtract" || t == "multiply" || t == "divide") {
        Shape s;
        if (!broadcastShapes(in[0], in[1], s)) return "The shapes do not fit: " + shapeText(in[0]) + " and " + shapeText(in[1]);
        out = {s};
        return {};
    }
    if (t == "matmul") {
        const Shape &a = in[0], &b = in[1];
        if (a.size() < 2 || b.size() < 2) return "Both need two dimensions or more: " + shapeText(a) + " and " + shapeText(b);
        if (a.back() != b[b.size() - 2]) return "The inner sizes differ: " + shapeText(a) + " times " + shapeText(b);
        Shape batch;
        if (!broadcastShapes(Shape(a.begin(), a.end() - 2), Shape(b.begin(), b.end() - 2), batch))
            return "The leading dimensions do not fit: " + shapeText(a) + " and " + shapeText(b);
        batch.push_back(a[a.size() - 2]);
        batch.push_back(b.back());
        out = {batch};
        return {};
    }
    if (t == "transpose") {
        if (x.size() < 2) return "Needs two dimensions or more, gets " + shapeText(x);
        std::swap(out[0][x.size() - 1], out[0][x.size() - 2]);
        return {};
    }
    if (t == "sum" || t == "mean") {
        if (g.number(n, "all") != 0.0) {
            out = {Shape{}};
            return {};
        }
        const int axis = normAxis(num("axis"), x.size());
        if (axis < 0) return "There is no axis " + std::to_string(num("axis")) + " in " + shapeText(x);
        if (g.number(n, "keep") != 0.0) out[0][static_cast<size_t>(axis)] = 1;
        else out[0].erase(out[0].begin() + axis);
        return {};
    }
    if (t == "power" || t == "exp" || t == "log" || t == "relu" || t == "leakyrelu" || t == "gelu" || t == "sigmoid" || t == "tanh" ||
        t == "dropout")
        return {};
    if (t == "softmax") return x.empty() ? "Needs at least one dimension" : std::string();
    if (t == "dense") {
        if (x.empty() || !known(x.back())) return "Needs features in its last dimension, gets " + shapeText(x);
        const int64_t units = num("units");
        out[0].back() = units;
        params = x.back() * units + (g.number(n, "bias") != 0.0 ? units : 0);
        return {};
    }
    if (t == "rnn") {
        if (x.size() < 2 || !known(x.back()) || !known(x[x.size() - 2]))
            return "Needs steps × features in its last two dimensions, gets " + shapeText(x);
        const int64_t units = num("units");
        out[0].back() = units;
        if (g.number(n, "all") == 0.0) out[0].erase(out[0].end() - 2);
        params = x.back() * units + units * units + units;
        return {};
    }
    if (t == "conv2d" || t == "maxpool2d") {
        if (x.size() != 4 || !known(x[1]) || !known(x[2]) || !known(x[3])) return "Needs batch × channels × height × width, gets " + shapeText(x);
        const int64_t k = num("kernel"), s = std::max<int64_t>(1, num("stride"));
        const int64_t pad = t == "conv2d" && g.number(n, "padding") == 0.0 ? k / 2 : 0;
        if (x[2] + 2 * pad < k || x[3] + 2 * pad < k) return "The kernel is larger than the image (" + shapeText(x) + ")";
        const int64_t h = (x[2] + 2 * pad - k) / s + 1, w = (x[3] + 2 * pad - k) / s + 1;
        const int64_t channels = t == "conv2d" ? num("filters") : x[1];
        out = {{x[0], channels, h, w}};
        if (t == "conv2d") params = x[1] * channels * k * k + (g.number(n, "bias") != 0.0 ? channels : 0);
        return {};
    }
    if (t == "globalpool") {
        if (x.size() != 4) return "Needs batch × channels × height × width, gets " + shapeText(x);
        out = {{x[0], x[1]}};
        return {};
    }
    if (t == "flatten") {
        const bool batched = !x.empty() && x[0] == kBatch;
        int64_t count = 1;
        for (size_t i = batched ? 1 : 0; i < x.size(); ++i) count *= x[i];
        out = {batched ? Shape{kBatch, count} : Shape{count}};
        return {};
    }
    if (t == "reshape") {
        Shape s;
        if (!parseShape(g.text(n, "shape"), s)) return "The shape should be sizes like 4x4";
        const bool batched = !x.empty() && x[0] == kBatch;
        const int64_t have = elementCount(batched ? Shape(x.begin() + 1, x.end()) : x);
        if (have != elementCount(s)) return "Has " + std::to_string(have) + " values per example, the shape holds " + std::to_string(elementCount(s));
        if (batched) s.insert(s.begin(), kBatch);
        out = {s};
        return {};
    }
    if (t == "layernorm") {
        if (x.empty() || !known(x.back())) return "Needs features in its last dimension, gets " + shapeText(x);
        params = 2 * x.back();
        return {};
    }
    if (t == "embedding") {
        out[0].push_back(num("dim"));
        params = num("vocab") * num("dim");
        return {};
    }
    if (t == "concat") {
        const Shape &a = in[0], &b = in[1];
        if (a.size() != b.size()) return "Different numbers of dimensions: " + shapeText(a) + " and " + shapeText(b);
        const int axis = normAxis(num("axis"), a.size());
        if (axis < 0) return "There is no axis " + std::to_string(num("axis")) + " in " + shapeText(a);
        for (size_t i = 0; i < a.size(); ++i)
            if (static_cast<int>(i) != axis && a[i] != b[i]) return "Only axis " + std::to_string(axis) + " may differ: " + shapeText(a) + " and " + shapeText(b);
        if (!known(a[static_cast<size_t>(axis)]) || !known(b[static_cast<size_t>(axis)])) return "Cannot join along the batch";
        out = {a};
        out[0][static_cast<size_t>(axis)] += b[static_cast<size_t>(axis)];
        return {};
    }
    if (t == "custom") {
        Expression code;
        std::string error;
        if (!code.compile(g.text(n, "code"), static_cast<int>(in.size()), &error)) return error;
        Shape s = in[0];
        for (size_t i = 1; i < in.size(); ++i) {
            Shape joined;
            if (!broadcastShapes(s, in[i], joined)) return "The inputs do not fit: " + shapeText(s) + " and " + shapeText(in[i]);
            s = joined;
        }
        out = {s};
        return {};
    }
    if (t == "loss") {
        const Shape &p = in[0], &y = in[1];
        out = {Shape{}};
        if (g.number(n, "kind") == 0.0) { // cross entropy
            if (p.size() != 2 || !known(p[1])) return "The prediction should be examples × class scores, gets " + shapeText(p);
            if (!(y.size() == 1 || (y.size() == 2 && y[1] == 1)) || y[0] != p[0])
                return "The target should be one class number per example (" + shapeText({p[0]}) + "), gets " + shapeText(y);
            // Targets straight from a dataset: a score for each of its classes.
            if (const Link* l = g.linkInto(n.id, 1); l && l->fromPort == 1)
                if (const Node* src = g.find(l->fromNode); src && src->type == "dataset")
                    if (const auto d = loadDataset(g, *src); d->classes && d->classCount > p[1])
                        return "The data has " + std::to_string(d->classCount) + " classes, the prediction scores " + std::to_string(p[1]) +
                               ": give the last layer " + std::to_string(d->classCount) + " units";
            return {};
        }
        Shape s;
        if (!broadcastShapes(p, y, s)) return "Prediction and target do not fit: " + shapeText(p) + " and " + shapeText(y);
        if (s != p) return "The target would stretch the prediction: " + shapeText(p) + " and " + shapeText(y);
        return {};
    }
    return "Unknown node type \"" + t + "\"";
}

} // namespace

Analysis analyze(const Graph& g) {
    Analysis a;
    // Kahn's order: a node once all the nodes feeding it are placed; what is left is in a loop.
    std::map<int, int> pending;
    for (const Node& n : g.nodes) pending[n.id] = 0;
    for (const Link& l : g.links)
        if (pending.count(l.fromNode) && pending.count(l.toNode)) ++pending[l.toNode];
    std::vector<int> ready;
    for (const Node& n : g.nodes)
        if (pending[n.id] == 0) ready.push_back(n.id);
    while (!ready.empty()) {
        const int id = ready.front();
        ready.erase(ready.begin());
        a.order.push_back(id);
        for (const Link& l : g.links)
            if (l.fromNode == id && pending.count(l.toNode) && --pending[l.toNode] == 0) ready.push_back(l.toNode);
    }

    for (const int id : a.order) {
        const Node& n = *g.find(id);
        NodeInfo& info = a.nodes[id];
        const NodeType* type = findNodeType(n.type);
        if (!type) {
            info.state = NodeInfo::State::Error;
            info.message = "Unknown node type \"" + n.type + "\"";
            ++a.errors;
            continue;
        }
        a.hasData |= n.type == "dataset";
        a.hasLoss |= n.type == "loss";
        const std::vector<std::string> ports = inputPorts(n);
        info.inputs.resize(ports.size());
        std::string missing;
        for (size_t p = 0; p < ports.size(); ++p) {
            const Link* l = g.linkInto(id, static_cast<int>(p));
            const NodeInfo* src = l ? a.info(l->fromNode) : nullptr;
            if (!l) missing = ports.size() > 1 ? "Connect input " + ports[p] : "Connect an input";
            else if (!src || src->state != NodeInfo::State::Ok) missing = "Waiting for " + g.find(l->fromNode)->name;
            else info.inputs[p] = src->outputs[static_cast<size_t>(l->fromPort)];
        }
        if (!missing.empty()) {
            info.state = NodeInfo::State::Waiting;
            info.message = missing;
            ++a.waiting;
            continue;
        }
        if (std::string error = infer(g, n, info.inputs, info.outputs, info.params); !error.empty()) {
            info.state = NodeInfo::State::Error;
            info.message = std::move(error);
            info.outputs.clear();
            ++a.errors;
            continue;
        }
        info.shape = info.outputs.empty() ? Shape{} : info.outputs[0];
        info.state = NodeInfo::State::Ok;
        a.totalParams += info.params;
    }
    for (const Node& n : g.nodes)
        if (!a.nodes.count(n.id)) {
            NodeInfo& info = a.nodes[n.id];
            info.state = NodeInfo::State::Error;
            info.message = "Part of a loop";
            ++a.errors;
        }
    return a;
}

} // namespace logievo
