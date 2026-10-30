#pragma once

// The editing screen: a menu bar as bemotion's (the mark, which goes back to the projects; File,
// Edit, Add, View, Model, Help; on the right Settings, Train and Export), the node library on the
// left, the canvas in the middle and the inspector on the right.

#include "core/Graph.h"
#include "core/Runtime.h"

#include <QWidget>

#include <filesystem>
#include <functional>
#include <memory>

class QLabel;
class QMenuBar;
class QPushButton;
class QSplitter;

namespace logievo::ui {

class Document;
class Inspector;
class NodeCanvas;
class NodeLibrary;

class EditorWindow : public QWidget {
    Q_OBJECT
public:
    EditorWindow(Graph graph, std::filesystem::path file, Weights weights = {}, QWidget* parent = nullptr);
    ~EditorWindow() override;

    Document& document() { return *doc_; }
    NodeCanvas& canvas() { return *canvas_; }
    bool save();
    // Packs the trained model into a command-line executable (see core/Export.h).
    void exportModel();
    // Unsaved changes: asks to save them. False if the user cancels.
    bool confirmClose();

    std::function<void()> onBack;           // to the projects
    std::function<void()> onTitleChanged;   // the name or the unsaved mark changed
    std::function<void()> onNewProject;     // File › New Project
    std::function<void(const std::filesystem::path&)> onOpenProject;

private:
    void updateTrainButton();
    void updateHeader();
    void buildMenus();
    void toggleTraining();

    std::unique_ptr<Document> doc_;
    NodeLibrary* library_ = nullptr;
    NodeCanvas* canvas_ = nullptr;
    Inspector* inspector_ = nullptr;
    QMenuBar* menuBar_ = nullptr;
    QLabel* state_ = nullptr;
    QPushButton* train_ = nullptr;
};

} // namespace logievo::ui
