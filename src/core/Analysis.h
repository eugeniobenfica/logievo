#pragma once

// Shape inference: walks the graph in order and works out what each node outputs, how many
// trainable parameters it has, and what is wrong where something does not fit. Shapes are whole
// tensors: the batch (how many examples go through at once) is kBatch, shown as "B".

#include "core/Graph.h"
#include "core/Tensor.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace logievo {

struct NodeInfo {
    enum class State { Ok, Error, Waiting }; // Waiting: an input is missing or broken upstream
    State state = State::Waiting;
    std::vector<Shape> inputs;  // what arrives at each input port (empty when nothing does)
    std::vector<Shape> outputs; // what leaves each output port
    Shape shape;                // the first output (a Loss: the scalar it computes)
    int64_t params = 0;         // trainable values it holds
    std::string message;        // the error, or what it waits for
};

struct Analysis {
    std::map<int, NodeInfo> nodes;
    std::vector<int> order; // nodes in the order they compute (each after its inputs)
    int64_t totalParams = 0;
    int errors = 0, waiting = 0;
    bool hasData = false, hasLoss = false;
    // Everything fits and there is something to minimize: it can train.
    bool valid() const { return errors == 0 && waiting == 0 && hasLoss; }
    const NodeInfo* info(int id) const {
        auto it = nodes.find(id);
        return it == nodes.end() ? nullptr : &it->second;
    }
};

Analysis analyze(const Graph& graph);

// "3x224x224" (also "3, 224, 224", "3 224 224"); false when it is not a list of positive sizes.
bool parseShape(const std::string& text, Shape& out);
std::string shapeText(const Shape& shape); // "B × 3 × 224", "scalar"
std::string countText(int64_t count);      // "1.2M", "48.5K", "860"

} // namespace logievo
