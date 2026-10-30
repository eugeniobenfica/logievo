#include "core/Export.h"

#include "core/Serialization.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>

namespace logievo {

namespace {

constexpr char kMagic[8] = {'L', 'G', 'V', 'M', 'O', 'D', 'E', 'L'};

bool fail(std::string* error, const std::string& why) {
    if (error) *error = why;
    return false;
}

} // namespace

bool prepareCommandModel(const Graph& graph, const Weights& weights, CommandModel& out, std::string* error) {
    out.graph = graph;
    out.weights = weights;
    const Node* data = nullptr;
    for (Node& n : out.graph.nodes) {
        if (n.type != "dataset") continue;
        const double source = out.graph.number(n, "source");
        if (data || (source != kTextCommandsSource && source != kTextFileSource))
            return fail(error, "Export works with models that read text: one Dataset, with the Text commands or Text file source");
        if (source == kTextFileSource) {
            // The file stays behind: what the model needs from it is its command list.
            const auto d = loadDataset(out.graph, n);
            if (!d->error.empty()) return fail(error, d->error);
            n.params["source"] = kTextCommandsSource;
            n.params["commands"] = d->commands;
            n.params.erase("file");
        }
        n.params["samples"] = 8; // only its shapes matter here: no need to make thousands of sentences
        data = &n;
    }
    if (!data) return fail(error, "Export works with models that read text: add a Dataset with the Text commands or Text file source");
    out.dataset = data->id;
    out.commands = parseCommands(out.graph.text(*data, "commands"));
    if (out.commands.empty()) return fail(error, "The dataset has no commands: write them as names = what to open");
    out.analysis = std::make_unique<Analysis>(analyze(out.graph));
    if (!out.analysis->valid()) return fail(error, "Fix the problems in the graph first");
    const Node* loss = nullptr;
    for (const Node& n : out.graph.nodes)
        if (n.type == "loss" && out.graph.number(n, "kind") == 0.0)
            if (const Link* t = out.graph.linkInto(n.id, 1); t && t->fromNode == out.dataset && t->fromPort == 1) loss = &n;
    if (!loss) return fail(error, "Connect the dataset's targets to a Cross entropy Loss: its prediction is what the model chooses");
    const Link* p = out.graph.linkInto(loss->id, 0);
    out.prediction = p->fromNode;
    out.port = p->fromPort;
    for (const int id : out.analysis->order) {
        const std::vector<Shape> shapes = weightShapes(out.graph, *out.graph.find(id), *out.analysis->info(id));
        if (shapes.empty()) continue;
        const auto it = out.weights.find(id);
        bool fits = it != out.weights.end() && it->second.size() == shapes.size();
        for (size_t k = 0; fits && k < shapes.size(); ++k) fits = it->second[k].shape == shapes[k];
        if (!fits) return fail(error, "Train the model first: the executable carries what it learned");
    }
    return true;
}

Decision decide(const CommandModel& model, const std::string& sentence, float threshold) {
    Tensor x({1, kTextFeatures});
    x.v = textFeatures(sentence);
    Batches data;
    data[model.dataset] = {std::move(x), Tensor(Shape{1})};
    Executor exec(model.graph, *model.analysis);
    std::mt19937 rng(1);
    exec.forward(model.weights, data, false, rng);
    const Tensor* scores = exec.output(model.prediction, model.port);
    Decision d;
    if (!scores || scores->v.empty()) return d;
    const std::vector<float>& s = scores->v;
    const float top = *std::max_element(s.begin(), s.end());
    double sum = 0;
    for (float v : s) sum += std::exp(v - top);
    const size_t best = static_cast<size_t>(std::max_element(s.begin(), s.end()) - s.begin());
    d.confidence = static_cast<float>(1.0 / sum);
    if (best > 0 && best <= model.commands.size() && d.confidence >= threshold) d.command = static_cast<int>(best) - 1;
    return d;
}

bool fitOutputToClasses(Graph& graph) {
    for (const Node& loss : graph.nodes) {
        if (loss.type != "loss" || graph.number(loss, "kind") != 0.0) continue;
        const Link* t = graph.linkInto(loss.id, 1);
        const Link* p = graph.linkInto(loss.id, 0);
        const Node* data = t ? graph.find(t->fromNode) : nullptr;
        Node* out = p ? graph.find(p->fromNode) : nullptr;
        if (!data || data->type != "dataset" || t->fromPort != 1 || !out || out->type != "dense") continue;
        const auto d = loadDataset(graph, *data);
        if (!d->error.empty() || !d->classes) return false;
        out->params["units"] = d->classCount;
        return true;
    }
    return false;
}

bool exportExecutable(const Graph& graph, const Weights& weights, const std::filesystem::path& runner, const std::filesystem::path& out,
                      std::string* error) {
    CommandModel check;
    if (!prepareCommandModel(graph, weights, check, error)) return false;
    std::ifstream in(runner, std::ios::binary);
    if (!in) return fail(error, "The runner (" + runner.filename().string() + ") is missing next to Logievo: build it with the app");
    std::string program((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // An exported model exporting again: the program without the model it carries.
    if (program.size() >= 16 && std::memcmp(program.data() + program.size() - 8, kMagic, 8) == 0) {
        uint64_t old = 0;
        for (int i = 0; i < 8; ++i) old |= static_cast<uint64_t>(static_cast<unsigned char>(program[program.size() - 16 + i])) << (8 * i);
        if (old <= program.size() - 16) program.resize(program.size() - 16 - old);
    }
    // The graph as the model reads it: a Text file dataset becomes its command list.
    Json j = toJson(check.graph);
    j["weights"] = weightsToJson(weights);
    const std::string model = j.dump();
    uint64_t size = model.size();
    char footer[16];
    for (int i = 0; i < 8; ++i) footer[i] = static_cast<char>((size >> (8 * i)) & 0xFF);
    std::memcpy(footer + 8, kMagic, 8);
    // Written beside, then moved over: a failed write never leaves half a program.
    std::filesystem::path tmp = out;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f || !f.write(program.data(), static_cast<std::streamsize>(program.size())) || !f.write(model.data(), static_cast<std::streamsize>(size)) ||
            !f.write(footer, sizeof footer))
            return fail(error, "Could not write " + out.filename().string());
    }
    std::error_code ec;
    std::filesystem::rename(tmp, out, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return fail(error, "Could not write " + out.filename().string() + " (is it running?)");
    }
    std::filesystem::permissions(out, std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec,
                                 std::filesystem::perm_options::add, ec);
    return true;
}

bool readEmbeddedModel(const std::filesystem::path& exe, Graph& graph, Weights& weights, std::string* error) {
    std::ifstream f(exe, std::ios::binary | std::ios::ate);
    if (!f) return fail(error, "Could not open " + exe.filename().string());
    const int64_t end = static_cast<int64_t>(f.tellg());
    char footer[16];
    if (end < 16 || !f.seekg(end - 16) || !f.read(footer, 16) || std::memcmp(footer + 8, kMagic, 8) != 0)
        return fail(error, "This program carries no model: export one from Logievo, or pass --model project.logievo");
    uint64_t size = 0;
    for (int i = 0; i < 8; ++i) size |= static_cast<uint64_t>(static_cast<unsigned char>(footer[i])) << (8 * i);
    if (size > static_cast<uint64_t>(end - 16)) return fail(error, "The model in this program is damaged");
    std::string text(static_cast<size_t>(size), '\0');
    f.seekg(end - 16 - static_cast<int64_t>(size));
    if (!f.read(text.data(), static_cast<std::streamsize>(size))) return fail(error, "The model in this program is damaged");
    const Json j = Json::parse(text, nullptr, false);
    if (j.is_discarded() || !fromJson(j, graph, error)) return fail(error, "The model in this program is damaged");
    weightsFromJson(j.value("weights", Json::object()), weights);
    return true;
}

} // namespace logievo
