#pragma once

#include "app/Settings.h"
#include "core/Graph.h"

#include <QImage>
#include <QWidget>

#include <filesystem>
#include <functional>
#include <map>
#include <optional>

class QFileSystemWatcher;
class QLabel;
class QLineEdit;
class QScrollArea;

namespace logievo::ui {

// Start screen, like bemotion's: the sidebar (mark, New / Open Project, the projects folder),
// and the grid of projects (every project in the projects folder plus the ones opened from
// elsewhere), with the New Project dialog.
class HomeScreen : public QWidget {
    Q_OBJECT
public:
    explicit HomeScreen(QWidget* parent = nullptr);

    // Set by the application.
    std::function<void(const std::filesystem::path&)> onOpen;
    std::function<void(const std::filesystem::path&, Graph)> onCreate;

    void refresh(); // reloads the projects and their previews (when the screen shows)
    void setError(const QString& message);
    // Starts on the template `templateId` (projectTemplates(); default: Blank).
    void openNewProjectDialog(const std::string& templateId = {});
    bool chooseProjectsFolder(); // false if the user cancels

protected:
    void resizeEvent(QResizeEvent* e) override;
    bool eventFilter(QObject* watched, QEvent* e) override;

private:
    struct Preview {
        std::optional<Graph> graph;
        QImage thumbnail;
        int64_t stamp = 0; // file time when loaded
    };
    void buildUi();
    void rebuildGrid();
    void confirmDelete(const RecentProject& item);

    std::vector<RecentProject> entries_; // newest first
    std::map<std::filesystem::path, Preview> previews_;
    QWidget* content_ = nullptr;
    QFileSystemWatcher* watcher_ = nullptr;
    QLineEdit* search_ = nullptr;
    QLabel* error_ = nullptr;
    QWidget* errorRow_ = nullptr;
    QScrollArea* scroll_ = nullptr;
    int columns_ = 0;
};

} // namespace logievo::ui
