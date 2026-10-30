#pragma once

// The examples a Dataset node gives: built-in sets to try ideas on (circles, spirals, XOR, a
// sine wave, a line, sequences of 16 steps), a CSV file (one example per row), or text commands
// (made-up requests to open programs and files, and ones to refuse). Loaded once and kept while
// the node's settings (and the file) stay the same.

#include "core/Graph.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace logievo {

struct Dataset {
    int64_t rows = 0;
    int64_t features = 0;
    bool classes = true;     // targets are class numbers 0, 1, 2... (else numbers to predict)
    int64_t classCount = 0;  // with classes
    std::vector<float> x;    // rows × features (empty for text: see texts)
    std::vector<float> y;    // rows
    std::string error;       // why there is nothing (no file, a bad row...)
    // Text sources: one sentence per row, turned into features a batch at a time (hundreds of
    // thousands of rows would not fit as 1024 numbers each), and their command list.
    std::vector<std::string> texts;
    std::string commands;
};

std::shared_ptr<const Dataset> loadDataset(const Graph& graph, const Node& node);

// --- text commands -------------------------------------------------------------------------
// The "Text commands" source: each line of its list is "names = what to open" (names separated
// by |, # starts a comment; "run: program arguments" runs a command line instead of opening).
// Class 0 is "do nothing"; class k opens line k's target. A sentence becomes kTextFeatures
// numbers (hashed words, word pairs and letter triples), the same way in training and in the
// exported executable.
//
// The "Text file" source reads the same from a file, as made at scale by
// tools/textcommands/generate.py: "@command names = target" lines (class 1, 2... in order), then
// one example per line, "class<TAB>sentence". Blank lines and # comments are skipped.

inline constexpr int kTextCommandsSource = 7; // the Dataset "source" choices
inline constexpr int kTextFileSource = 8;
inline constexpr int64_t kTextFeatures = 4096; // few collisions: a new word seldom lands on "não"
extern const char* const kDefaultCommands;

struct TextCommand {
    std::vector<std::string> names;
    std::string target; // a program, a file, a folder or a link
};

std::vector<TextCommand> parseCommands(const std::string& text);
// Lower case, accents dropped, anything but letters and digits a space.
std::string normalizeText(const std::string& text);
std::vector<float> textFeatures(const std::string& text);

} // namespace logievo
