#include "core/Serialization.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace logievo {

namespace {

constexpr char kBase64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string toBase64(const std::vector<float>& values) {
    std::vector<unsigned char> bytes(values.size() * 4);
    std::memcpy(bytes.data(), values.data(), bytes.size()); // little-endian float32 on every platform built for
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3) {
        const unsigned v = bytes[i] << 16 | (i + 1 < bytes.size() ? bytes[i + 1] << 8 : 0) | (i + 2 < bytes.size() ? bytes[i + 2] : 0);
        out += kBase64[(v >> 18) & 63];
        out += kBase64[(v >> 12) & 63];
        out += i + 1 < bytes.size() ? kBase64[(v >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? kBase64[v & 63] : '=';
    }
    return out;
}

std::vector<float> fromBase64(const std::string& text) {
    std::vector<unsigned char> bytes;
    unsigned v = 0;
    int bits = 0;
    for (char c : text) {
        const char* at = std::strchr(kBase64, c);
        if (!at || c == 0) continue;
        v = v << 6 | static_cast<unsigned>(at - kBase64);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            bytes.push_back(static_cast<unsigned char>((v >> bits) & 255));
        }
    }
    std::vector<float> out(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), out.size() * 4);
    return out;
}

} // namespace

Json weightsToJson(const Weights& weights) {
    Json out = Json::object();
    for (const auto& [id, list] : weights) {
        Json tensors = Json::array();
        for (const Tensor& t : list) tensors.push_back({{"shape", t.shape}, {"data", toBase64(t.v)}});
        out[std::to_string(id)] = tensors;
    }
    return out;
}

void weightsFromJson(const Json& j, Weights& out) {
    out.clear();
    if (!j.is_object()) return;
    for (const auto& [key, tensors] : j.items()) {
        std::vector<Tensor> list;
        for (const Json& jt : tensors) {
            Tensor t;
            t.shape = jt.value("shape", Shape{});
            t.v = fromBase64(jt.value("data", std::string()));
            if (static_cast<int64_t>(t.v.size()) != elementCount(t.shape)) return; // damaged: start fresh
            list.push_back(std::move(t));
        }
        out[std::stoi(key)] = std::move(list);
    }
}

Json toJson(const Graph& g) {
    Json nodes = Json::array();
    for (const Node& n : g.nodes) {
        Json params = Json::object();
        for (const auto& [k, v] : n.params) params[k] = v;
        nodes.push_back({{"id", n.id}, {"type", n.type}, {"name", n.name}, {"x", n.x}, {"y", n.y}, {"params", params}});
    }
    Json links = Json::array();
    for (const Link& l : g.links) links.push_back({{"from", {l.fromNode, l.fromPort}}, {"to", {l.toNode, l.toPort}}});
    const TrainSettings& t = g.training;
    const Json training{{"optimizer", t.optimizer == TrainSettings::Sgd ? "sgd" : "adam"}, {"learning_rate", t.learningRate}, {"epochs", t.epochs},
                        {"batch_size", t.batchSize}};
    return {{"format", "logievo"}, {"version", kFormatVersion}, {"name", g.name}, {"training", training}, {"nodes", nodes}, {"links", links}};
}

bool fromJson(const Json& j, Graph& out, std::string* error) {
    try {
        if (!j.is_object() || j.value("format", "") != "logievo") {
            if (error) *error = "Not an Logievo project";
            return false;
        }
        Graph g;
        g.name = j.value("name", std::string("Untitled"));
        if (const Json t = j.value("training", Json::object()); t.is_object()) {
            g.training.optimizer = t.value("optimizer", std::string("adam")) == "sgd" ? TrainSettings::Sgd : TrainSettings::Adam;
            g.training.learningRate = t.value("learning_rate", g.training.learningRate);
            g.training.epochs = t.value("epochs", g.training.epochs);
            g.training.batchSize = t.value("batch_size", g.training.batchSize);
        }
        for (const Json& jn : j.value("nodes", Json::array())) {
            Node n;
            n.id = jn.at("id").get<int>();
            n.type = jn.at("type").get<std::string>();
            n.name = jn.value("name", n.type);
            n.x = jn.value("x", 0.0f);
            n.y = jn.value("y", 0.0f);
            const Json params = jn.value("params", Json::object());
            for (const auto& [k, v] : params.items()) n.params[k] = v;
            if (!g.find(n.id)) g.nodes.push_back(std::move(n));
        }
        for (const Json& jl : j.value("links", Json::array())) {
            const Link l{jl.at("from").at(0).get<int>(), jl.at("from").at(1).get<int>(), jl.at("to").at(0).get<int>(), jl.at("to").at(1).get<int>()};
            g.connect(l.fromNode, l.fromPort, l.toNode, l.toPort); // checks ports and loops
        }
        out = std::move(g);
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool saveGraph(const std::filesystem::path& file, const Graph& graph, std::string* error, const Weights* weights) {
    Json j = toJson(graph);
    if (weights && !weights->empty()) j["weights"] = weightsToJson(*weights);
    const std::string text = j.dump(1);
    // Written beside, then moved over: a failed write never leaves half a project.
    std::filesystem::path tmp = file;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f || !(f << text)) {
            if (error) *error = "Could not write " + file.filename().string();
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, file, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        if (error) *error = "Could not write " + file.filename().string();
        return false;
    }
    return true;
}

bool loadGraph(const std::filesystem::path& file, Graph& out, std::string* error, Weights* weights) {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        if (error) *error = "Could not open the file";
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const Json j = Json::parse(ss.str(), nullptr, false);
    if (j.is_discarded()) {
        if (error) *error = "The file is not valid JSON";
        return false;
    }
    if (!fromJson(j, out, error)) return false;
    if (weights) weightsFromJson(j.value("weights", Json::object()), *weights);
    return true;
}

} // namespace logievo
