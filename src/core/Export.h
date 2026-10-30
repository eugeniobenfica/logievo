#pragma once

// A trained text-commands model outside the editor: it reads a sentence and picks one of its
// commands (or none), and it is packed into a copy of the runner executable (logievo_run), the
// project's JSON with its weights appended after the program, so one file carries both.

#include "core/Analysis.h"
#include "core/Dataset.h"
#include "core/Graph.h"
#include "core/Runtime.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace logievo {

struct CommandModel {
    Graph graph; // with a Text file dataset turned into Text commands with the file's list
    Weights weights;
    std::unique_ptr<Analysis> analysis; // the Executor holds on to it: it stays put
    int dataset = 0;                    // the Text commands Dataset node
    int prediction = 0, port = 0;       // what its cross-entropy Loss scores
    std::vector<TextCommand> commands;
};

// False (with the reason) unless the graph trains on a Text commands or Text file dataset through
// a cross-entropy Loss, fits together, and has learned weights for every node that holds some.
bool prepareCommandModel(const Graph& graph, const Weights& weights, CommandModel& out, std::string* error = nullptr);

struct Decision {
    int command = -1;        // into CommandModel::commands; -1: do nothing
    float confidence = 0.0f; // the chosen class's probability
};

// Below `threshold`, nothing is safer than a guess.
Decision decide(const CommandModel& model, const std::string& sentence, float threshold = 0.6f);

// For a model trained on a dataset: its prediction node's units set to the dataset's class
// count when that is a Dense layer (more commands in the file, more scores). False otherwise.
bool fitOutputToClasses(Graph& graph);

// Writes `runner` with the model appended to `out`.
bool exportExecutable(const Graph& graph, const Weights& weights, const std::filesystem::path& runner, const std::filesystem::path& out,
                      std::string* error = nullptr);
// The model appended to an executable by exportExecutable; false when it has none.
bool readEmbeddedModel(const std::filesystem::path& exe, Graph& graph, Weights& weights, std::string* error = nullptr);

} // namespace logievo
