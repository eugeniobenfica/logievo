#pragma once

// The engine, in plain C++: runs a graph forward on a batch of examples, works out the gradient
// of the loss for every trainable value (backpropagation, node by node in reverse order), and
// trains with SGD or Adam on a thread of its own.

#include "core/Analysis.h"
#include "core/Graph.h"
#include "core/Tensor.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace logievo {

// What a model has learned: the trainable tensors of each node (a Parameter's value, a Dense
// layer's weights and bias...), by node id.
using Weights = std::map<int, std::vector<Tensor>>;

// The trainable tensors a node holds, given what reaches it.
std::vector<Shape> weightShapes(const Graph& graph, const Node& node, const NodeInfo& info);
// Gives every node the weights it needs: keeps the ones whose shapes still fit, starts the others
// fresh (as each node's settings say) and drops the ones of nodes that are gone.
void prepareWeights(const Graph& graph, const Analysis& analysis, Weights& weights, uint32_t seed = 1);

// A batch for each Dataset node: its features and targets.
using Batches = std::map<int, std::array<Tensor, 2>>;

struct Metrics {
    double correct = 0, counted = 0; // classification losses: right answers among how many
};

class Executor {
public:
    Executor(const Graph& graph, const Analysis& analysis);

    // Runs every node; returns the total loss (each Loss times its weight).
    double forward(const Weights& weights, const Batches& data, bool training, std::mt19937& rng, Metrics* metrics = nullptr);
    // After forward: the gradient of the total loss for every weight (same layout as Weights).
    void backward(const Weights& weights, Weights& grads);
    // What a node's output port held in the last forward (nullptr when it did not run).
    const Tensor* output(int node, int port = 0) const;

private:
    struct State {
        std::vector<Tensor> out, grad; // per output port
        Tensor cache;                  // what backward needs (a dropout mask, softmax, x̂...)
        std::vector<int64_t> index;    // max pool's picks, embedding ids
        std::vector<float> rowScale;   // layer norm: 1 / standard deviation of each row
        std::vector<Tensor> derivs;    // a Code node: d output / d each input, value by value
    };
    const Tensor& input(const Node& n, int port) const;
    void addGrad(const Node& n, int port, Tensor g);

    const Graph& graph_;
    const Analysis& analysis_;
    std::map<int, State> state_;
};

struct TrainReport {
    int epoch = 0, epochs = 0;
    std::vector<float> loss;     // per epoch
    std::vector<float> accuracy; // per epoch, with a classification loss
    bool running = false;
    std::string error;
};

// Training on a thread: start() takes a copy of the graph and weights; report() and
// latestWeights() can be read at any time from the interface.
class Trainer {
public:
    ~Trainer();
    // False (with the reason in the report) when the graph cannot train.
    bool start(const Graph& graph, const Weights& weights);
    void stop(); // waits for the current batch
    bool running() const { return running_; }
    TrainReport report() const;
    // The weights after the last finished epoch; false when nothing new came since last asked.
    bool takeWeights(Weights& out);

private:
    void run(Graph graph, Weights weights);

    std::thread thread_;
    std::atomic<bool> running_{false}, stop_{false};
    mutable std::mutex mutex_;
    TrainReport report_;
    Weights latest_;
    bool fresh_ = false;
};

// Draws `count` examples from each dataset, rows picked by `rows` (shared by all datasets,
// each wrapping around its own size).
Batches makeBatches(const Graph& graph, const std::vector<int64_t>& rows);

} // namespace logievo
