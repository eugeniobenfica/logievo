#include "core/Templates.h"

#include "core/Dataset.h"

namespace logievo {

namespace {

constexpr float kStep = 250.0f; // between columns of cards

// Builds a template: nodes placed by column and row, links by node id.
struct Builder {
    Graph& g;
    int add(const std::string& type, float column, float row, std::map<std::string, Json> params = {}, const std::string& name = {}) {
        Node& n = g.add(type, column * kStep, row * 150.0f);
        n.params = std::move(params);
        if (!name.empty()) n.name = g.uniqueName(name);
        return n.id;
    }
    void link(int from, int to, int toPort = 0, int fromPort = 0) { g.connect(from, fromPort, to, toPort); }
};

// Dataset sources and loss kinds, as NodeTypes numbers them.
enum { Circles = 0, Spirals = 1, Line = 5, Sequences = 6, TextCommands = kTextCommandsSource };
enum { CrossEntropy = 0, MeanSquared = 1 };

} // namespace

const std::vector<ProjectTemplate>& projectTemplates() {
    static const std::vector<ProjectTemplate> list{
        {"blank", "Blank", "An empty canvas: build the model your way, from datasets, parameters and math up."},
        {"linear", "Linear Regression",
         "y = w·x + b from scratch: two Parameters, a Matrix Multiply and an Add learn the line through noisy points (w ends near 2, b near −1)."},
        {"scratch", "Network from Scratch",
         "A neural network with no layer nodes: weight and bias Parameters, Matrix Multiply, Add and Tanh tell two rings of points apart."},
        {"classifier", "Classifier (MLP)", "Dense layers with ReLU sort three interleaved spirals: the classic multilayer perceptron."},
        {"rnn", "Recurrent (RNN)",
         "A Recurrent layer reads 16 steps of a wave one at a time and predicts the value that comes next."},
        {"tensors", "Tensors",
         "A playground of shapes: broadcasting, matrix products, transposes and sums, each card showing the shape it gives. Nothing to train."},
        {"code", "Custom Code Node", "A gated unit written as code, tanh(a) · sigmoid(b), between Dense layers: your own math, trained like the rest."},
        {"commands", "Text Commands",
         "Reads a sentence (typed, or from Whisper) and opens a program or file, or does nothing. Trained on made-up requests, with the "
         "ones to refuse: negated, other verbs, things that are not programs. Export makes it a command-line executable."},
    };
    return list;
}

Graph makeTemplate(const std::string& id, const std::string& name) {
    Graph g;
    g.name = name;
    Builder b{g};
    TrainSettings& t = g.training;
    if (id == "linear") {
        const int d = b.add("dataset", 0, 0.5f, {{"source", Line}, {"samples", 200}, {"noise", 0.2}, {"scale", 0}}, "points");
        const int w = b.add("parameter", 0, 1.6f, {{"shape", "1x1"}, {"init", 0}}, "w");
        const int mm = b.add("matmul", 1, 0.5f);
        const int bias = b.add("parameter", 1, 1.6f, {{"shape", "1"}, {"init", 2}}, "b");
        const int ad = b.add("add", 2, 0.5f);
        const int loss = b.add("loss", 3, 0.5f, {{"kind", MeanSquared}});
        b.link(d, mm), b.link(w, mm, 1), b.link(mm, ad), b.link(bias, ad, 1), b.link(ad, loss), b.link(d, loss, 1, 1);
        t.optimizer = TrainSettings::Sgd, t.learningRate = 0.1, t.epochs = 100, t.batchSize = 32;
    } else if (id == "scratch") {
        const int d = b.add("dataset", 0, 0.5f, {{"source", Circles}}, "rings");
        const int w1 = b.add("parameter", 0, 1.6f, {{"shape", "2x16"}}, "w1");
        const int mm1 = b.add("matmul", 1, 0.5f);
        const int b1 = b.add("parameter", 1, 1.6f, {{"shape", "16"}, {"init", 2}}, "b1");
        const int ad1 = b.add("add", 2, 0.5f);
        const int act = b.add("tanh", 3, 0.5f);
        const int w2 = b.add("parameter", 3, 1.6f, {{"shape", "16x2"}}, "w2");
        const int mm2 = b.add("matmul", 4, 0.5f);
        const int b2 = b.add("parameter", 4, 1.6f, {{"shape", "2"}, {"init", 2}}, "b2");
        const int ad2 = b.add("add", 5, 0.5f);
        const int loss = b.add("loss", 6, 0.5f, {{"kind", CrossEntropy}});
        b.link(d, mm1), b.link(w1, mm1, 1), b.link(mm1, ad1), b.link(b1, ad1, 1), b.link(ad1, act);
        b.link(act, mm2), b.link(w2, mm2, 1), b.link(mm2, ad2), b.link(b2, ad2, 1), b.link(ad2, loss), b.link(d, loss, 1, 1);
        t.optimizer = TrainSettings::Adam, t.learningRate = 0.02, t.epochs = 80;
    } else if (id == "classifier") {
        const int d = b.add("dataset", 0, 0.5f, {{"source", Spirals}, {"samples", 900}});
        const int h1 = b.add("dense", 1, 0, {{"units", 32}});
        const int a1 = b.add("relu", 2, 0);
        const int h2 = b.add("dense", 3, 0, {{"units", 32}});
        const int a2 = b.add("relu", 4, 0);
        const int o = b.add("dense", 5, 0, {{"units", 3}}, "scores");
        const int loss = b.add("loss", 6, 0.5f, {{"kind", CrossEntropy}});
        b.link(d, h1), b.link(h1, a1), b.link(a1, h2), b.link(h2, a2), b.link(a2, o), b.link(o, loss), b.link(d, loss, 1, 1);
        t.optimizer = TrainSettings::Adam, t.learningRate = 0.01, t.epochs = 200;
    } else if (id == "rnn") {
        const int d = b.add("dataset", 0, 0.5f, {{"source", Sequences}, {"samples", 800}, {"scale", 0}, {"noise", 0.1}}, "waves");
        const int r = b.add("reshape", 1, 0, {{"shape", "16x1"}}, "steps");
        const int rnn = b.add("rnn", 2, 0, {{"units", 32}});
        const int o = b.add("dense", 3, 0, {{"units", 1}}, "next_value");
        const int loss = b.add("loss", 4, 0.5f, {{"kind", MeanSquared}});
        b.link(d, r), b.link(r, rnn), b.link(rnn, o), b.link(o, loss), b.link(d, loss, 1, 1);
        t.optimizer = TrainSettings::Adam, t.learningRate = 0.01, t.epochs = 60;
    } else if (id == "tensors") {
        // 3×1 + 1×4 broadcasts to 3×4; times 4×2 gives 3×2; transposed 2×3; summed, one number.
        const int a = b.add("constant", 0, 0, {{"shape", "3x1"}, {"value", 2.0}}, "a_3x1");
        const int c = b.add("parameter", 0, 1, {{"shape", "1x4"}, {"init", 3}}, "b_1x4");
        const int ad = b.add("add", 1, 0.5f, {}, "broadcast");
        const int m = b.add("parameter", 1, 1.6f, {{"shape", "4x2"}}, "m_4x2");
        const int mm = b.add("matmul", 2, 0.5f);
        const int tr = b.add("transpose", 3, 0.5f);
        const int rs = b.add("reshape", 4, 0, {{"shape", "6"}}, "flat");
        const int sm = b.add("sum", 4, 1, {{"all", true}}, "total");
        b.link(a, ad), b.link(c, ad, 1), b.link(ad, mm), b.link(m, mm, 1), b.link(mm, tr), b.link(tr, rs), b.link(tr, sm);
    } else if (id == "code") {
        const int d = b.add("dataset", 0, 0.5f, {{"source", Circles}});
        const int h = b.add("dense", 1, 0, {{"units", 16}}, "value");
        const int gate = b.add("dense", 1, 1, {{"units", 16}}, "gate");
        const int c = b.add("custom", 2, 0.5f, {{"code", "# a gated unit: the gate (0 to 1) lets the value through\ng = sigmoid(b)\ntanh(a) * g"}},
                            "gated_tanh");
        const int o = b.add("dense", 3, 0.5f, {{"units", 2}}, "scores");
        const int loss = b.add("loss", 4, 0.5f, {{"kind", CrossEntropy}});
        b.link(d, h), b.link(d, gate), b.link(h, c), b.link(gate, c, 1), b.link(c, o), b.link(o, loss), b.link(d, loss, 1, 1);
        t.optimizer = TrainSettings::Adam, t.learningRate = 0.02, t.epochs = 80;
    } else if (id == "commands") {
        // Words → a hidden layer → a score for "do nothing" and for each command.
        const int64_t classes = 1 + static_cast<int64_t>(parseCommands(kDefaultCommands).size());
        const int d = b.add("dataset", 0, 0.5f, {{"source", TextCommands}, {"commands", kDefaultCommands}, {"samples", 4000}, {"noise", 0.15}}, "requests");
        const int h = b.add("dense", 1, 0, {{"units", 64}}, "understand");
        const int a = b.add("relu", 2, 0);
        const int drop = b.add("dropout", 3, 0, {{"rate", 0.2}});
        const int o = b.add("dense", 4, 0, {{"units", classes}}, "command");
        const int loss = b.add("loss", 5, 0.5f, {{"kind", CrossEntropy}});
        b.link(d, h), b.link(h, a), b.link(a, drop), b.link(drop, o), b.link(o, loss), b.link(d, loss, 1, 1);
        t.optimizer = TrainSettings::Adam, t.learningRate = 0.005, t.epochs = 25, t.batchSize = 32;
    }
    return g;
}

} // namespace logievo
