#pragma once

// The document: a model is a graph of nodes (data, parameters, math, layers, losses) joined by
// links from an output port to an input port. No Qt here: the editor, the file format and the
// engine (Runtime, Training) all work on this.

#include <nlohmann/json.hpp>

#include <map>
#include <string>
#include <vector>

namespace logievo {

using Json = nlohmann::json;

struct Node {
    int id = 0;
    std::string type;                   // a NodeType id ("dense", "conv2d"...)
    std::string name;                   // shown on the node
    float x = 0.0f, y = 0.0f;           // top-left, in canvas units
    std::map<std::string, Json> params; // by ParamSpec id; missing ones take the default
};

struct Link {
    int fromNode = 0, fromPort = 0; // an output
    int toNode = 0, toPort = 0;     // an input (takes one link at most)
    bool operator==(const Link&) const = default;
};

// How the model is trained (the Training section of the inspector).
struct TrainSettings {
    enum Optimizer { Sgd = 0, Adam = 1 };
    int optimizer = Adam;
    double learningRate = 0.01;
    int epochs = 200;
    int batchSize = 32;
    bool operator==(const TrainSettings&) const = default;
};

class Graph {
public:
    std::string name = "Untitled";
    TrainSettings training;
    std::vector<Node> nodes;
    std::vector<Link> links;

    Node* find(int id);
    const Node* find(int id) const;

    // A node of `type` at (x, y), with its default parameters and a name not taken yet.
    Node& add(const std::string& type, float x, float y);
    void remove(int id); // with its links
    // Joins an output to an input, replacing what the input had. False (and nothing changes)
    // when the ports do not exist, the nodes are the same or the link would close a loop.
    bool connect(int fromNode, int fromPort, int toNode, int toPort);
    void disconnect(int toNode, int toPort);
    const Link* linkInto(int toNode, int toPort) const;
    // Would `from` -> `to` make a cycle (`to` already reaches `from`)?
    bool reaches(int from, int to) const;

    std::string uniqueName(const std::string& base) const;
    int nextId() const;

    // A parameter's value, or its default when the node does not set it.
    Json param(const Node& node, const std::string& id) const;
    double number(const Node& node, const std::string& id) const;
    std::string text(const Node& node, const std::string& id) const;
};

} // namespace logievo
