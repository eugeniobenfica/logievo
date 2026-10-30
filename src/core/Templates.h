#pragma once

// What a new project can start from (the New Project dialog): a blank canvas, or a small model
// built from the same nodes the library has, ready to train and to take apart.

#include "core/Graph.h"

#include <string>
#include <vector>

namespace logievo {

struct ProjectTemplate {
    std::string id;          // "blank", "linear", ...
    std::string name;        // "Linear Regression"
    std::string description; // a sentence for the dialog
};

const std::vector<ProjectTemplate>& projectTemplates();

// The template's graph, named `name`; an unknown id gives a blank one.
Graph makeTemplate(const std::string& id, const std::string& name);

} // namespace logievo
