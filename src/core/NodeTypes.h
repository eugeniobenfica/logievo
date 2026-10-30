#pragma once

// What nodes there are: their category, ports and parameters. The library panel, the
// inspector, shape inference and the engine all read this table.

#include "core/Graph.h"

#include <string>
#include <vector>

namespace logievo {

enum class ParamKind { Int, Float, Bool, Choice, Shape, File, Code, Text };

struct ParamSpec {
    std::string id;    // saved in files: never renamed
    std::string label; // shown in the inspector
    ParamKind kind = ParamKind::Int;
    double defNumber = 0.0;            // Int, Float, Bool (0 / 1), Choice (index)
    double min = 0.0, max = 0.0;       // min == max: unbounded
    std::vector<std::string> choices;  // Choice
    std::string defText;               // Shape: "3x224x224"; File: empty; Code, Text: the lines
    std::string tip;
};

struct NodeType {
    std::string id; // saved in files: never renamed
    std::string name;
    std::string category;
    std::string description;
    std::vector<std::string> inputs;  // port names
    std::vector<std::string> outputs;
    std::vector<ParamSpec> params;
};

// In library order.
const std::vector<NodeType>& nodeTypes();
const std::vector<std::string>& nodeCategories();
const NodeType* findNodeType(const std::string& id);
const ParamSpec* findParam(const NodeType& type, const std::string& id);
// A node's input ports: its type's, except a Code node, which has as many as its "inputs" says.
std::vector<std::string> inputPorts(const Node& node);

} // namespace logievo
