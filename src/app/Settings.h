#pragma once

// What Logievo remembers between runs (settings.json in the app data folder): where projects
// live, the projects opened from elsewhere, the look, performance, and the user's own nodes.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace logievo {

struct RecentProject {
    std::filesystem::path path;
    std::string name;
    int64_t lastOpened = 0; // unix seconds
};

// A node the user wrote (Settings › Custom Nodes): it shows in the library under "Custom", and
// adds a Code node with this code and these inputs (the project keeps its own copy).
struct CustomNode {
    std::string name;
    std::string code;
    int inputs = 1;
    std::string description;
};

struct Settings {
    std::filesystem::path projectsFolder; // empty until the user picks one
    std::vector<RecentProject> recents;   // newest first
    // The look.
    std::string theme = "dark";           // dark, midnight, light
    std::string accent;                   // #RRGGBB; empty: white (black on the light theme)
    double rounding = 5.0;                // corners of fields and buttons, px
    double textScale = 1.0;               // every label
    std::string font = "Inter";
    bool showGrid = true;                 // the canvas's dots
    bool snapToGrid = true;               // cards move in steps
    // Performance.
    bool gpu = false;                     // matrix products on the graphics card (OpenGL compute)
    int threads = 0;                      // processor threads for the engine; 0: all of them
    std::vector<CustomNode> customNodes;
};

Settings& settings();
void loadSettings();
void saveSettings();

void addRecent(const std::filesystem::path& file, const std::string& name);
void removeRecent(const std::filesystem::path& file);
int64_t unixNow();

} // namespace logievo
