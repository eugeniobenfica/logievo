#include "core/NodeTypes.h"

#include "core/Dataset.h"

#include <algorithm>

namespace logievo {

namespace {

ParamSpec intParam(std::string id, std::string label, int def, int min, int max, std::string tip = {}) {
    return {std::move(id), std::move(label), ParamKind::Int, static_cast<double>(def), static_cast<double>(min), static_cast<double>(max), {}, {}, std::move(tip)};
}
ParamSpec floatParam(std::string id, std::string label, double def, double min, double max, std::string tip = {}) {
    return {std::move(id), std::move(label), ParamKind::Float, def, min, max, {}, {}, std::move(tip)};
}
ParamSpec boolParam(std::string id, std::string label, bool def, std::string tip = {}) {
    return {std::move(id), std::move(label), ParamKind::Bool, def ? 1.0 : 0.0, 0, 1, {}, {}, std::move(tip)};
}
ParamSpec choiceParam(std::string id, std::string label, std::vector<std::string> choices, int def, std::string tip = {}) {
    return {std::move(id), std::move(label), ParamKind::Choice, static_cast<double>(def), 0, 0, std::move(choices), {}, std::move(tip)};
}
ParamSpec shapeParam(std::string id, std::string label, std::string def, std::string tip = {}) {
    return {std::move(id), std::move(label), ParamKind::Shape, 0, 0, 0, {}, std::move(def), std::move(tip)};
}
ParamSpec fileParam(std::string id, std::string label, std::string tip = {}) {
    return {std::move(id), std::move(label), ParamKind::File, 0, 0, 0, {}, {}, std::move(tip)};
}

std::vector<NodeType> build() {
    const std::vector<std::string> one{"in"}, two{"a", "b"}, out{"out"};
    std::vector<NodeType> t;
    // --- Data: where values come from
    t.push_back({"dataset", "Dataset", "Data",
                 "Examples to learn from: features go into the model, targets into a loss. Built-in sets to try ideas, or your own CSV "
                 "file (one example per row).",
                 {}, {"features", "targets"},
                 {choiceParam("source", "Source", {"Circles", "Spirals", "XOR", "Sine wave", "CSV file", "Line", "Sequences", "Text commands", "Text file"}, 0,
                              "Line: y = 2x − 1 with noise. Sequences: 16 steps of a wave, the target is the next value. Text commands: "
                              "made-up requests to open your commands, and ones to refuse (class 0, do nothing). Text file: the same at "
                              "scale, from tools/textcommands/generate.py"),
                  fileParam("file", "File", "CSV: one example per row, numbers separated by commas. Text file: @command lines, then class<TAB>sentence"),
                  {"commands", "Commands", ParamKind::Text, 0, 0, 0, {}, kDefaultCommands,
                   "One per line: names (separated by |) = what to open. Class 0 does nothing; class 1 is the first line, and so on"},
                  choiceParam("column", "Target column", {"Last", "First"}, 0, "Which column of the CSV file is the target"),
                  choiceParam("targets", "Targets are", {"Classes", "Numbers"}, 0, "Classes: whole numbers 0, 1, 2... (for cross entropy)"),
                  choiceParam("scale", "Scale features", {"None", "Standardize", "Divide by 255"}, 1),
                  intParam("samples", "Samples", 600, 8, 200000, "How many examples a built-in set makes"),
                  floatParam("noise", "Noise", 0.1, 0.0, 2.0, "How scattered a built-in set is")}});
    t.push_back({"parameter", "Parameter", "Data", "A tensor the model learns: weights, biases, embeddings, anything trainable you build with.", {}, {"value"},
                 {shapeParam("shape", "Shape", "2x16", "Sizes separated by x; it has no batch dimension"),
                  choiceParam("init", "Start at", {"Random", "Xavier", "Zeros", "Ones"}, 1)}});
    t.push_back({"constant", "Constant", "Data", "A fixed tensor, every value the same.", {}, {"value"},
                 {shapeParam("shape", "Shape", "1"), floatParam("value", "Value", 1.0, 0, 0)}});
    // --- Math: the building blocks
    t.push_back({"add", "Add", "Math", "a + b, value by value. Shapes broadcast: a size of 1 stretches to the other's.", two, out, {}});
    t.push_back({"subtract", "Subtract", "Math", "a − b, value by value (shapes broadcast).", two, out, {}});
    t.push_back({"multiply", "Multiply", "Math", "a × b, value by value (shapes broadcast).", two, out, {}});
    t.push_back({"divide", "Divide", "Math", "a ÷ b, value by value (shapes broadcast).", two, out, {}});
    t.push_back({"matmul", "Matrix Multiply", "Math", "Matrix product of the last two dimensions: n × k times k × m makes n × m.", two, out, {}});
    t.push_back({"sum", "Sum", "Math", "Adds the values along an axis, or all of them.", one, out,
                 {intParam("axis", "Axis", -1, -8, 8, "-1 is the last"), boolParam("all", "All axes", false), boolParam("keep", "Keep axis", false)}});
    t.push_back({"mean", "Mean", "Math", "Averages the values along an axis, or all of them.", one, out,
                 {intParam("axis", "Axis", -1, -8, 8, "-1 is the last"), boolParam("all", "All axes", false), boolParam("keep", "Keep axis", false)}});
    t.push_back({"power", "Power", "Math", "Each value raised to an exponent.", one, out, {floatParam("exponent", "Exponent", 2.0, 0, 0)}});
    t.push_back({"exp", "Exp", "Math", "e to the power of each value.", one, out, {}});
    t.push_back({"log", "Log", "Math", "Natural logarithm of each value.", one, out, {}});
    // --- Activations
    t.push_back({"relu", "ReLU", "Activations", "max(0, x).", one, out, {}});
    t.push_back({"leakyrelu", "Leaky ReLU", "Activations", "x, or slope × x below zero.", one, out, {floatParam("slope", "Slope", 0.01, 0.0, 1.0)}});
    t.push_back({"gelu", "GELU", "Activations", "Smooth ReLU, used in transformers.", one, out, {}});
    t.push_back({"sigmoid", "Sigmoid", "Activations", "Squashes values between 0 and 1.", one, out, {}});
    t.push_back({"tanh", "Tanh", "Activations", "Squashes values between -1 and 1.", one, out, {}});
    t.push_back({"softmax", "Softmax", "Activations", "Turns the last dimension into probabilities that add up to 1.", one, out, {}});
    // --- Layers: common blocks with their own learned weights
    t.push_back({"dense", "Dense", "Layers", "Fully connected layer: every input feature to every unit (x × W + b).", one, out,
                 {intParam("units", "Units", 16, 1, 1 << 20), boolParam("bias", "Bias", true)}});
    t.push_back({"conv2d", "Conv 2D", "Layers", "2D convolution over images shaped batch × channels × height × width.", one, out,
                 {intParam("filters", "Filters", 16, 1, 1 << 16), intParam("kernel", "Kernel", 3, 1, 64), intParam("stride", "Stride", 1, 1, 64),
                  choiceParam("padding", "Padding", {"Same", "Valid"}, 0, "Same keeps the size (at stride 1); Valid shrinks it"), boolParam("bias", "Bias", true)}});
    t.push_back({"rnn", "Recurrent", "Layers",
                 "A recurrent layer (RNN): reads steps × features one step at a time, h = tanh(x·Wx + h·Wh + b), and carries h "
                 "to the next step.",
                 one, out,
                 {intParam("units", "Units", 32, 1, 4096, "The size of h, what it remembers"),
                  boolParam("all", "Every step", false, "Gives h at every step (steps × units), not only the last")}});
    t.push_back({"embedding", "Embedding", "Layers", "Turns whole numbers (token ids) into learned vectors.", one, out,
                 {intParam("vocab", "Vocabulary", 1000, 1, 1 << 24), intParam("dim", "Dimension", 32, 1, 1 << 16)}});
    t.push_back({"layernorm", "Layer Norm", "Layers", "Normalizes the last dimension of each example, then scales and shifts it (learned).", one, out, {}});
    t.push_back({"dropout", "Dropout", "Layers", "Zeroes a share of the values while training, so the model does not lean on any one.", one, out,
                 {floatParam("rate", "Rate", 0.2, 0.0, 0.95)}});
    // --- Shape
    t.push_back({"reshape", "Reshape", "Shape", "Gives the values a new shape with as many of them (the batch stays).", one, out,
                 {shapeParam("shape", "Shape", "4x4", "Sizes without the batch, separated by x")}});
    t.push_back({"flatten", "Flatten", "Shape", "Lays each example's values out in one vector.", one, out, {}});
    t.push_back({"transpose", "Transpose", "Shape", "Swaps the last two dimensions.", one, out, {}});
    t.push_back({"concat", "Concatenate", "Shape", "Joins two tensors along an axis.", two, out, {intParam("axis", "Axis", -1, -8, 8, "-1 is the last")}});
    t.push_back({"maxpool2d", "Max Pool 2D", "Shape", "Keeps the largest value of each window of an image.", one, out,
                 {intParam("kernel", "Kernel", 2, 1, 64), intParam("stride", "Stride", 2, 1, 64)}});
    t.push_back({"globalpool", "Global Pool", "Shape", "Averages each channel over the whole image.", one, out, {}});
    // --- Custom: the user's own code
    t.push_back({"custom", "Code", "Custom",
                 "Your own formula, applied value by value to its inputs a, b, c, d (shapes broadcast). Its derivative is worked out "
                 "for you, so it trains like any other node. Lines can name values (t = sigmoid(b)); the last line is the result.",
                 {"a", "b"}, out,
                 {intParam("inputs", "Inputs", 2, 1, 4, "How many inputs it takes: a, b, c, d"),
                  {"code", "Code", ParamKind::Code, 0, 0, 0, {}, "a * b", "abs exp log sqrt sin cos tan tanh sigmoid relu softplus sign floor min max pow clamp if"}}});
    // --- Training
    t.push_back({"loss", "Loss", "Training",
                 "What training makes smaller: how far the prediction is from the target. Every Loss in the graph is added up.",
                 {"prediction", "target"}, {},
                 {choiceParam("kind", "Kind", {"Cross entropy", "Mean squared error", "Binary cross entropy"}, 0,
                              "Cross entropy: prediction scores per class, target class numbers"),
                  floatParam("weight", "Weight", 1.0, 0.0, 1000.0, "How much this loss counts")}});
    return t;
}

} // namespace

const std::vector<NodeType>& nodeTypes() {
    static const std::vector<NodeType> types = build();
    return types;
}

const std::vector<std::string>& nodeCategories() {
    static const std::vector<std::string> c{"Data", "Math", "Activations", "Layers", "Shape", "Custom", "Training"};
    return c;
}

const NodeType* findNodeType(const std::string& id) {
    for (const NodeType& t : nodeTypes())
        if (t.id == id) return &t;
    return nullptr;
}

const ParamSpec* findParam(const NodeType& type, const std::string& id) {
    for (const ParamSpec& p : type.params)
        if (p.id == id) return &p;
    return nullptr;
}

std::vector<std::string> inputPorts(const Node& node) {
    const NodeType* type = findNodeType(node.type);
    if (!type) return {};
    if (node.type != "custom") return type->inputs;
    int count = static_cast<int>(findParam(*type, "inputs")->defNumber);
    if (auto it = node.params.find("inputs"); it != node.params.end() && it->second.is_number()) count = it->second.get<int>();
    static const std::vector<std::string> names{"a", "b", "c", "d"};
    return {names.begin(), names.begin() + std::clamp(count, 1, 4)};
}

} // namespace logievo
