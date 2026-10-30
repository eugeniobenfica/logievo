#pragma once

// Which of bestyle's icons stands for each node type and library category.

#include "bestyle/Icons.h"

#include <string>

namespace logievo::ui {

bestyle::Icon nodeIcon(const std::string& type);
bestyle::Icon categoryIcon(const std::string& category);

} // namespace logievo::ui
