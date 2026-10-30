#include "app/Settings.h"

#include "app/Platform.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <sstream>

namespace logievo {

namespace {

constexpr size_t kMaxRecents = 50;

std::filesystem::path settingsFile() { return platform::appDataDir() / "settings.json"; }

std::string utf8(const std::filesystem::path& p) { return platform::qpath(p).toStdString(); }
std::filesystem::path fromUtf8(const std::string& s) { return platform::pathOf(QString::fromStdString(s)); }

} // namespace

Settings& settings() {
    static Settings s;
    return s;
}

int64_t unixNow() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

void loadSettings() {
    Settings& s = settings();
    std::ifstream f(settingsFile(), std::ios::binary);
    if (!f) return;
    std::stringstream ss;
    ss << f.rdbuf();
    const nlohmann::json j = nlohmann::json::parse(ss.str(), nullptr, false);
    if (!j.is_object()) return;
    s.projectsFolder = fromUtf8(j.value("projects_folder", std::string()));
    s.recents.clear();
    if (j.contains("recents") && j["recents"].is_array())
        for (const auto& r : j["recents"])
            if (r.is_object()) s.recents.push_back({fromUtf8(r.value("path", std::string())), r.value("name", std::string()), r.value("opened", int64_t(0))});
    s.theme = j.value("theme", s.theme);
    s.accent = j.value("accent", s.accent);
    s.rounding = j.value("rounding", s.rounding);
    s.textScale = j.value("text_scale", s.textScale);
    s.font = j.value("font", s.font);
    s.showGrid = j.value("show_grid", s.showGrid);
    s.snapToGrid = j.value("snap_to_grid", s.snapToGrid);
    s.gpu = j.value("gpu", s.gpu);
    s.threads = j.value("threads", s.threads);
    s.customNodes.clear();
    if (j.contains("custom_nodes") && j["custom_nodes"].is_array())
        for (const auto& c : j["custom_nodes"])
            if (c.is_object())
                s.customNodes.push_back({c.value("name", std::string("Custom")), c.value("code", std::string("a")), c.value("inputs", 1),
                                         c.value("description", std::string())});
}

void saveSettings() {
    const Settings& s = settings();
    nlohmann::json recents = nlohmann::json::array();
    for (const RecentProject& r : s.recents) recents.push_back({{"path", utf8(r.path)}, {"name", r.name}, {"opened", r.lastOpened}});
    nlohmann::json custom = nlohmann::json::array();
    for (const CustomNode& c : s.customNodes)
        custom.push_back({{"name", c.name}, {"code", c.code}, {"inputs", c.inputs}, {"description", c.description}});
    const nlohmann::json j{{"projects_folder", utf8(s.projectsFolder)}, {"recents", recents}, {"theme", s.theme}, {"accent", s.accent},
                           {"rounding", s.rounding}, {"text_scale", s.textScale}, {"font", s.font}, {"show_grid", s.showGrid},
                           {"snap_to_grid", s.snapToGrid}, {"gpu", s.gpu}, {"threads", s.threads}, {"custom_nodes", custom}};
    std::error_code ec;
    std::filesystem::create_directories(settingsFile().parent_path(), ec); // the app's own folder
    for (int attempt = 0; attempt < 2; ++attempt) { // OneDrive can hold a file for a moment
        std::ofstream f(settingsFile(), std::ios::binary);
        if (f << j.dump(2)) return;
    }
}

void addRecent(const std::filesystem::path& file, const std::string& name) {
    auto& r = settings().recents;
    removeRecent(file);
    r.insert(r.begin(), {file, name, unixNow()});
    if (r.size() > kMaxRecents) r.resize(kMaxRecents);
    saveSettings();
}

void removeRecent(const std::filesystem::path& file) {
    std::error_code ec;
    std::erase_if(settings().recents, [&](const RecentProject& r) { return r.path == file || std::filesystem::equivalent(r.path, file, ec); });
}

} // namespace logievo
