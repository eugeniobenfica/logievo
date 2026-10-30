#pragma once

// Puts the settings to work: the look (theme, accent, rounding, font, text size) and
// performance (processor threads, the graphics card). Called at start and whenever Settings
// changes one of them.

namespace logievo {

void applyLook();
void applyPerformance();

} // namespace logievo
