// Core checks: shape inference, links, files, and the engine: every operation's gradient
// against finite differences, and a real training run that must learn.

#include "core/Analysis.h"
#include "core/Dataset.h"
#include "core/Export.h"
#include "core/Expression.h"
#include "core/Parallel.h"
#include "core/NodeTypes.h"
#include "core/Runtime.h"
#include "core/Serialization.h"
#include "core/Templates.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>

using namespace logievo;

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

namespace {

int add(Graph& g, const std::string& type, std::map<std::string, Json> params = {}) {
    Node& n = g.add(type, 0, 0);
    n.params = std::move(params);
    return n.id;
}

void link(Graph& g, int from, int to, int toPort = 0, int fromPort = 0) {
    if (!g.connect(from, fromPort, to, toPort)) std::printf("  could not link %d -> %d\n", from, to);
}

// The analytic gradient of every weight against (loss(w + h) - loss(w - h)) / 2h. A step can
// cross a kink (ReLU at 0, two values tying in a max pool), so a few values may differ there;
// a wrong derivative makes most of them differ.
bool gradientsMatch(const Graph& g, const char* what) {
    const Analysis a = analyze(g);
    if (!a.valid()) {
        std::printf("  %s: graph not valid\n", what);
        for (const auto& [id, info] : a.nodes)
            if (info.state != NodeInfo::State::Ok) std::printf("    %s: %s\n", g.find(id)->name.c_str(), info.message.c_str());
        return false;
    }
    Weights w;
    prepareWeights(g, a, w, 3);
    // Points far from the middle: the first spiral points sit near (0, 0), where ReLU and max
    // pool have kinks a finite difference steps across.
    const Batches data = makeBatches(g, {301, 352, 420, 487, 530, 593});
    Executor exec(g, a);
    std::mt19937 rng(1);
    exec.forward(w, data, false, rng);
    Weights grads;
    exec.backward(w, grads);
    double worst = 0;
    int checked = 0, off = 0;
    for (auto& [id, list] : w)
        for (size_t k = 0; k < list.size(); ++k)
            for (int64_t i = 0; i < list[k].size(); ++i) {
                const double analytic = grads[id][k][i];
                double error = 1.0;
                // A smaller step when the first one lands across a kink.
                for (float h : {2e-3f, 2e-4f}) {
                    const float keep = list[k][i];
                    list[k][i] = keep + h;
                    const double up = exec.forward(w, data, false, rng);
                    list[k][i] = keep - h;
                    const double down = exec.forward(w, data, false, rng);
                    list[k][i] = keep;
                    const double numeric = (up - down) / (2 * h);
                    error = std::min(error, std::abs(numeric - analytic) / std::max(0.05, std::abs(numeric) + std::abs(analytic)));
                    if (error <= 3e-2) break;
                }
                ++checked;
                if (error > 3e-2) ++off;
                else worst = std::max(worst, error);
            }
    std::printf("  %-36s %4d values, %d at a kink, worst error elsewhere %.2g\n", what, checked, off, worst);
    return off <= std::max(1, checked / 50);
}

} // namespace

int main() {
    // --- shapes
    {
        Shape s;
        CHECK(parseShape("3x224x224", s) && s == Shape({3, 224, 224}));
        CHECK(!parseShape("3x0", s));
        CHECK(shapeText({kBatch, 3}) == "B × 3");
        CHECK(countText(1234567) == "1.23M");
        Shape o;
        CHECK(broadcastShapes({kBatch, 3}, {3}, o) && o == Shape({kBatch, 3}));
        CHECK(!broadcastShapes({kBatch, 3}, {4}, o));
    }
    {
        // A two-layer network built from scratch: x · W1 + b1 → ReLU → · W2 → loss.
        Graph g;
        const int d = add(g, "dataset");
        const int w1 = add(g, "parameter", {{"shape", "2x8"}});
        const int b1 = add(g, "parameter", {{"shape", "8"}, {"init", 2}});
        const int mm = add(g, "matmul"), plus = add(g, "add"), relu = add(g, "relu");
        const int w2 = add(g, "parameter", {{"shape", "8x2"}});
        const int mm2 = add(g, "matmul"), loss = add(g, "loss");
        link(g, d, mm), link(g, w1, mm, 1), link(g, mm, plus), link(g, b1, plus, 1);
        link(g, plus, relu), link(g, relu, mm2), link(g, w2, mm2, 1);
        link(g, mm2, loss), link(g, d, loss, 1, 1);
        const Analysis a = analyze(g);
        CHECK(a.valid());
        CHECK(a.info(mm2)->shape == Shape({kBatch, 2}));
        CHECK(a.totalParams == 16 + 8 + 16);
        CHECK(!g.connect(loss, 0, d, 0)); // the loss has no output; a dataset no input
        CHECK(gradientsMatch(g, "matmul network"));
        Graph back;
        CHECK(fromJson(toJson(g), back) && back.links == g.links);
        // A wrong inner size is an error on that node; what follows waits.
        g.find(w1)->params["shape"] = "3x8";
        const Analysis bad = analyze(g);
        CHECK(bad.info(mm)->state == NodeInfo::State::Error);
        CHECK(bad.info(loss)->state == NodeInfo::State::Waiting);
    }
    // --- every operation's gradient: dataset → ops → dense to 3 classes → cross entropy
    {
        struct Case {
            const char* name;
            std::function<int(Graph&, int)> build; // from the dataset node, returns the node to score
        };
        const std::vector<Case> cases{
            {"subtract, multiply, divide", [](Graph& g, int x) {
                 const int p = add(g, "parameter", {{"shape", "2"}}), q = add(g, "parameter", {{"shape", "1x2"}, {"init", 3}});
                 const int s = add(g, "subtract"), m = add(g, "multiply"), dv = add(g, "divide"), c = add(g, "constant", {{"value", 2.0}});
                 const int ad = add(g, "add");
                 link(g, x, s), link(g, p, s, 1), link(g, s, m), link(g, q, m, 1);
                 link(g, q, ad), link(g, c, ad, 1), link(g, m, dv), link(g, ad, dv, 1);
                 return dv;
             }},
            {"gelu, tanh, leaky relu, sigmoid", [](Graph& g, int x) {
                 int prev = add(g, "dense", {{"units", 6}});
                 link(g, x, prev);
                 for (const char* t : {"gelu", "tanh", "leakyrelu", "sigmoid"}) {
                     const int n = add(g, t);
                     link(g, prev, n);
                     prev = n;
                 }
                 return prev;
             }},
            {"softmax, log, exp, power", [](Graph& g, int x) {
                 const int d = add(g, "dense", {{"units", 4}}), sm = add(g, "softmax"), lg = add(g, "log"), ex = add(g, "exp"), pw = add(g, "power");
                 link(g, x, d), link(g, d, sm), link(g, sm, lg), link(g, lg, ex), link(g, ex, pw);
                 return pw;
             }},
            {"layer norm, concat, transpose, mean", [](Graph& g, int x) {
                 const int d = add(g, "dense", {{"units", 5}}), ln = add(g, "layernorm"), cc = add(g, "concat");
                 link(g, x, d), link(g, d, ln), link(g, ln, cc), link(g, x, cc, 1);
                 const int r = add(g, "reshape", {{"shape", "7x1"}}), tr = add(g, "transpose"), mn = add(g, "mean", {{"axis", -2}});
                 link(g, cc, r), link(g, r, tr), link(g, tr, mn);
                 return mn;
             }},
            {"conv 2d, max pool, global pool", [](Graph& g, int x) {
                 const int d = add(g, "dense", {{"units", 32}}), r = add(g, "reshape", {{"shape", "2x4x4"}});
                 const int cv = add(g, "conv2d", {{"filters", 3}}), mp = add(g, "maxpool2d");
                 const int cv2 = add(g, "conv2d", {{"filters", 2}, {"padding", 1}, {"kernel", 2}});
                 const int gp = add(g, "globalpool");
                 link(g, x, d), link(g, d, r), link(g, r, cv), link(g, cv, mp), link(g, mp, cv2), link(g, cv2, gp);
                 return gp;
             }},
            {"recurrent (every step, then the last)", [](Graph& g, int x) {
                 const int d = add(g, "dense", {{"units", 12}}), r = add(g, "reshape", {{"shape", "4x3"}});
                 const int r1 = add(g, "rnn", {{"units", 5}, {"all", true}}), r2 = add(g, "rnn", {{"units", 4}});
                 link(g, x, d), link(g, d, r), link(g, r, r1), link(g, r1, r2);
                 return r2;
             }},
            {"embedding, flatten, sum", [](Graph& g, int x) {
                 const int sg = add(g, "sigmoid"), e = add(g, "embedding", {{"vocab", 3}, {"dim", 4}}), f = add(g, "flatten");
                 const int d = add(g, "dense", {{"units", 3}}), d2 = add(g, "dense", {{"units", 3}}), m = add(g, "add");
                 const int sm = add(g, "sum", {{"axis", -1}, {"keep", true}}), m2 = add(g, "multiply");
                 link(g, x, sg), link(g, sg, e), link(g, e, f), link(g, f, d);
                 link(g, x, d2), link(g, d, m), link(g, d2, m, 1);
                 link(g, d2, sm), link(g, m, m2), link(g, sm, m2, 1);
                 return m2;
             }},
        };
        for (const Case& c : cases) {
            Graph g;
            const int d = add(g, "dataset", {{"source", 1}}); // spirals: 3 classes
            const int out = c.build(g, d);
            const int head = add(g, "dense", {{"units", 3}}), loss = add(g, "loss");
            link(g, out, head), link(g, head, loss), link(g, d, loss, 1, 1);
            CHECK(gradientsMatch(g, c.name));
        }
        // Mean squared error (sine) and binary cross entropy (XOR).
        for (int kind : {1, 2}) {
            Graph g;
            const int d = add(g, "dataset", {{"source", kind == 1 ? 3 : 2}});
            const int h = add(g, "dense", {{"units", 1}}), loss = add(g, "loss", {{"kind", kind}});
            link(g, d, h), link(g, h, loss);
            if (kind == 1) link(g, d, loss, 1, 1);
            else {
                const int r = add(g, "reshape", {{"shape", "1"}});
                link(g, d, r, 0, 1), link(g, r, loss, 1);
            }
            CHECK(gradientsMatch(g, kind == 1 ? "mean squared error" : "binary cross entropy"));
        }
    }
    // --- templates: each one fits together, and the regression and the RNN learn
    {
        for (const ProjectTemplate& t : projectTemplates()) {
            const Graph g = makeTemplate(t.id, t.name);
            const Analysis a = analyze(g);
            const bool ok = t.id == "blank" || t.id == "tensors" ? a.errors == 0 : a.valid();
            if (!ok) std::printf("  template %s does not fit together\n", t.id.c_str());
            CHECK(ok);
            CHECK(g.name == t.name);
        }
        const Graph tensors = makeTemplate("tensors", "t");
        const Analysis ta = analyze(tensors);
        for (const Node& n : tensors.nodes)
            if (n.name == "total") CHECK(ta.info(n.id)->shape.empty()); // 3×1 + 1×4 → 3×4 → 3×2 → 2×3 → one number
        auto train = [](const Graph& g, Weights* out) {
            Trainer trainer;
            if (!trainer.start(g, {})) return TrainReport{};
            while (trainer.running()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (out) trainer.takeWeights(*out);
            return trainer.report();
        };
        {
            const Graph g = makeTemplate("linear", "line");
            Weights w;
            const TrainReport r = train(g, &w);
            CHECK(r.error.empty() && r.epoch == g.training.epochs);
            float slope = 0, bias = 0;
            for (const Node& n : g.nodes) {
                if (n.name == "w") slope = w.at(n.id)[0][0];
                if (n.name == "b") bias = w.at(n.id)[0][0];
            }
            std::printf("  linear regression: w %.3f, b %.3f\n", slope, bias);
            CHECK(std::abs(slope - 2.0f) < 0.1f && std::abs(bias + 1.0f) < 0.1f);
        }
        {
            Graph g = makeTemplate("rnn", "waves");
            g.training.epochs = 25;
            const TrainReport r = train(g, nullptr);
            CHECK(r.error.empty() && r.loss.size() == 25);
            if (!r.loss.empty()) std::printf("  recurrent: loss %.4f -> %.4f\n", r.loss.front(), r.loss.back());
            CHECK(!r.loss.empty() && r.loss.back() < r.loss.front() * 0.2f && r.loss.back() < 0.02f);
        }
        // Text commands: opens what is asked, refuses the rest, and survives being exported.
        {
            CHECK(normalizeText("  Abra a CALCULADORA, já!") == "abra a calculadora ja");
            CHECK(parseCommands(kDefaultCommands).size() == 7);
            const Graph g = makeTemplate("commands", "commands");
            Weights w;
            const TrainReport r = train(g, &w);
            CHECK(r.error.empty() && r.epoch == g.training.epochs);
            if (!r.accuracy.empty()) std::printf("  text commands: training accuracy %.1f%%\n", r.accuracy.back() * 100);
            CommandModel m;
            std::string why;
            CHECK(prepareCommandModel(g, w, m, &why));
            const std::vector<std::pair<std::string, std::string>> asks{
                {"abra a calculadora", "calc.exe"},
                {"abre o bloco de notas por favor", "notepad.exe"},
                {"Pode abrir o Paint pra mim?", "mspaint.exe"},
                {"open the browser", "https://www.google.com"},
                {"abrir meus documentos", "%USERPROFILE%\\Documents"},
                {"abra a calculdora", "calc.exe"}, // misheard
                {"não abra a calculadora", ""},
                {"feche o paint", ""},
                {"o que é o bloco de notas", ""},
                {"abra a porta", ""},
                {"bom dia", ""},
                {"desliga o computador", ""},
                {"don't open the terminal", ""},
            };
            int right = 0;
            for (const auto& [sentence, expected] : asks) {
                const Decision d = decide(m, sentence);
                const std::string got = d.command < 0 ? "" : m.commands[static_cast<size_t>(d.command)].target;
                if (got == expected) ++right;
                else std::printf("  text commands: \"%s\" -> \"%s\" (%.2f), expected \"%s\"\n", sentence.c_str(), got.c_str(), d.confidence, expected.c_str());
            }
            std::printf("  text commands: %d of %zu sentences right\n", right, asks.size());
            CHECK(right >= static_cast<int>(asks.size()) - 1);
            // Untrained: export says so. Exported: the model reads back from the end of the program.
            CHECK(!prepareCommandModel(g, {}, m, &why) && !why.empty());
            const auto dir = std::filesystem::temp_directory_path();
            const auto runner = dir / "logievo_fake_runner.bin", exe = dir / "logievo_exported.bin";
            std::ofstream(runner, std::ios::binary) << "a program";
            CHECK(exportExecutable(g, w, runner, exe, &why));
            Graph back;
            Weights backWeights;
            CHECK(readEmbeddedModel(exe, back, backWeights, &why) && back.nodes.size() == g.nodes.size() && backWeights.size() == w.size());
            CHECK(!readEmbeddedModel(runner, back, backWeights));
            // Exporting from an exported program carries one model, not two.
            CHECK(exportExecutable(g, w, exe, runner, &why));
            CHECK(std::filesystem::file_size(runner) == std::filesystem::file_size(exe));
            std::filesystem::remove(runner);
            std::filesystem::remove(exe);
        }
        // Text file: commands and examples from a file (as tools/textcommands/generate.py writes);
        // exported, the file's command list goes in the program.
        {
            const auto file = std::filesystem::temp_directory_path() / "logievo_text_examples.tsv";
            std::ofstream(file, std::ios::binary) << "# made up\n@command calculadora | calc = calc.exe\n@command paint = run: mspaint.exe\n"
                                                    "1\tabra a calculadora\n2\tabre o paint\n0\tnão abra a calculadora\r\n0\tbom dia\n";
            Graph g = makeTemplate("commands", "file");
            for (Node& n : g.nodes)
                if (n.type == "dataset") n.params = {{"source", kTextFileSource}, {"file", file.string()}};
            CHECK(fitOutputToClasses(g));
            const Analysis a = analyze(g);
            CHECK(a.valid());
            for (const Node& n : g.nodes)
                if (n.type == "dataset") {
                    const auto d = loadDataset(g, n);
                    CHECK(d->error.empty() && d->rows == 4 && d->classCount == 3 && d->texts[2] == "nao abra a calculadora");
                }
            Weights w;
            prepareWeights(g, a, w);
            CommandModel m;
            std::string why;
            CHECK(prepareCommandModel(g, w, m, &why) && m.commands.size() == 2 && m.commands[1].target == "run: mspaint.exe");
            for (const Node& n : m.graph.nodes)
                if (n.type == "dataset") CHECK(m.graph.number(n, "source") == kTextCommandsSource && m.graph.text(n, "file").empty());
            const auto bad = std::filesystem::temp_directory_path() / "logievo_text_bad.tsv";
            std::ofstream(bad, std::ios::binary) << "@command x = y\n5\tbad class\n";
            for (Node& n : g.nodes)
                if (n.type == "dataset") n.params["file"] = bad.string();
            CHECK(!prepareCommandModel(g, w, m, &why) && why.find("class 5") != std::string::npos);
            std::filesystem::remove(file);
            std::filesystem::remove(bad);
        }
    }
    // --- training learns: circles with a small network
    {
        Graph g;
        const int d = add(g, "dataset");
        const int h = add(g, "dense", {{"units", 16}}), act = add(g, "tanh"), o = add(g, "dense", {{"units", 2}}), loss = add(g, "loss");
        link(g, d, h), link(g, h, act), link(g, act, o), link(g, o, loss), link(g, d, loss, 1, 1);
        g.training.epochs = 60;
        g.training.learningRate = 0.02;
        Trainer trainer;
        CHECK(trainer.start(g, {}));
        while (trainer.running()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const TrainReport r = trainer.report();
        CHECK(r.error.empty() && r.epoch == 60);
        CHECK(!r.accuracy.empty() && r.accuracy.back() > 0.95f);
        CHECK(!r.loss.empty() && r.loss.back() < r.loss.front());
        if (!r.loss.empty() && !r.accuracy.empty())
            std::printf("  circles: loss %.3f -> %.3f, accuracy %.1f%%\n", r.loss.front(), r.loss.back(), r.accuracy.back() * 100);
        Weights w;
        CHECK(trainer.takeWeights(w) && w.size() == 2);
        // Weights survive the file.
        Weights back;
        weightsFromJson(weightsToJson(w), back);
        CHECK(back.size() == 2 && back.begin()->second[0].v == w.begin()->second[0].v);
        // Without a loss, it says so.
        g.remove(loss);
        CHECK(!trainer.start(g, {}));
        CHECK(!trainer.report().error.empty());
    }
    // --- Code nodes: the language, its derivatives, and a Code node training in a graph
    {
        Expression e;
        std::string error;
        CHECK(e.compile("t = sigmoid(b)  # a gate\na * t + (1 - t) * c ^ 2", 3, &error));
        const float in[3] = {0.7f, -0.3f, 1.5f};
        float grad[4];
        const float v = e.eval(in, grad);
        const float t = 1.0f / (1.0f + std::exp(0.3f));
        CHECK(std::abs(v - (0.7f * t + (1 - t) * 2.25f)) < 1e-5f);
        CHECK(std::abs(grad[0] - t) < 1e-5f);
        CHECK(std::abs(grad[1] - t * (1 - t) * (0.7f - 2.25f)) < 1e-5f);
        CHECK(std::abs(grad[2] - (1 - t) * 3.0f) < 1e-5f);
        CHECK(e.uses(0) && e.uses(1) && e.uses(2) && !e.uses(3));
        CHECK(!e.compile("a +", 1, &error) && error.find("Line 1") == 0);
        CHECK(!e.compile("c * 2", 2, &error) && error.find("more inputs") != std::string::npos);
        CHECK(!e.compile("q", 1, &error));
        CHECK(!e.compile("a\nb", 2, &error)); // only the last line gives the result
        CHECK(e.compile("max(a, 0) + if(a > 1, 2, clamp(a, -1, 1))", 1, &error));

        Graph g;
        const int d = add(g, "dataset", {{"source", 1}});
        const int h = add(g, "dense", {{"units", 4}}), p = add(g, "parameter", {{"shape", "4"}});
        const int c = add(g, "custom", {{"code", "u = softplus(a)\na * sigmoid(b) + u ^ 2 / 3"}});
        const int head = add(g, "dense", {{"units", 3}}), loss = add(g, "loss");
        link(g, d, h), link(g, h, c), link(g, p, c, 1), link(g, c, head), link(g, head, loss), link(g, d, loss, 1, 1);
        CHECK(analyze(g).info(c)->shape == Shape({kBatch, 4}));
        CHECK(gradientsMatch(g, "code node"));
        // Three inputs: a third port; with code that does not read, the node says why.
        g.find(c)->params["inputs"] = 3;
        CHECK(inputPorts(*g.find(c)).size() == 3);
        CHECK(analyze(g).info(c)->state == NodeInfo::State::Waiting);
        g.find(c)->params["code"] = "a +* b";
        link(g, p, c, 2);
        CHECK(analyze(g).info(c)->state == NodeInfo::State::Error);
    }
    // --- the processor's threads change the speed, never the result
    {
        std::mt19937 rng(5);
        std::normal_distribution<float> nd;
        Tensor a({3, 120, 90}), b({90, 70});
        for (float& v : a.v) v = nd(rng);
        for (float& v : b.v) v = nd(rng);
        setThreadCount(1);
        const Tensor one = matmul(a, b);
        setThreadCount(0);
        const Tensor all = matmul(a, b);
        CHECK(one.v == all.v && one.shape == Shape({3, 120, 70}));
        std::printf("  threads: %d of %d\n", threadCount(), hardwareThreads());
    }
    // --- a CSV file: a header row, the target first, classes counted
    {
        const auto file = std::filesystem::temp_directory_path() / "logievo-test.csv";
        {
            std::ofstream f(file);
            f << "label,a,b,c\n0,1,2,3\n2,4,5,6\r\n1,7,8,9\n";
        }
        Graph g;
        const std::string path = std::filesystem::path(file).generic_string();
        const int d = add(g, "dataset", {{"source", 4}, {"file", path}, {"column", 1}, {"scale", 0}});
        const auto data = loadDataset(g, *g.find(d));
        CHECK(data->error.empty() && data->rows == 3 && data->features == 3 && data->classCount == 3);
        CHECK(data->x[3] == 4.0f && data->y[1] == 2.0f);
        CHECK(analyze(g).info(d)->outputs[0] == Shape({kBatch, 3}));
        g.find(d)->params["file"] = path + ".missing";
        CHECK(analyze(g).info(d)->state == NodeInfo::State::Error);
        std::filesystem::remove(file);
    }
    if (failures == 0) std::printf("all core tests passed\n");
    return failures == 0 ? 0 : 1;
}
