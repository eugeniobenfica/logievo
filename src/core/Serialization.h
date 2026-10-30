#pragma once

// The .logievo file: plain JSON (see toJson), so models can be read, diffed and written by
// other tools. Only compatible changes: new optional fields with defaults, never renames.

#include "core/Graph.h"
#include "core/Runtime.h"

#include <filesystem>
#include <string>

namespace logievo {

inline constexpr const char* kProjectExtension = ".logievo";
inline constexpr int kFormatVersion = 1;

Json toJson(const Graph& graph);
bool fromJson(const Json& json, Graph& out, std::string* error = nullptr);

// The learned weights go in the file too ("weights": shape and float32 values in base64).
Json weightsToJson(const Weights& weights);
void weightsFromJson(const Json& json, Weights& out);

bool saveGraph(const std::filesystem::path& file, const Graph& graph, std::string* error = nullptr, const Weights* weights = nullptr);
bool loadGraph(const std::filesystem::path& file, Graph& out, std::string* error = nullptr, Weights* weights = nullptr);

} // namespace logievo
