#pragma once

// The application: one window that shows the home screen or the editor.
//
//   logievo [project.logievo]
//
// Developer aid, to look at the interface without a person at the screen:
//   --capture FILE.png   saves the window once it has settled, then quits
//   --select ID          selects that node first (the inspector shows it)
//   --train              starts training once the project is open
//   --wait MS            how long --capture waits (1200 ms)
//   --size WxH           the window's size (instead of maximized)
//   --write-icon FILE    writes the 1024 px app icon (assets/branding) and quits
//   --settings PAGE / --help PAGE   opens that window (-1: its default page); --capture
//                        saves it too, as FILE-Settings.png
//   --new-project TEMPLATE   opens the New Project dialog on that template (captured as
//                        FILE-New Project.png)
//   --gpu-check FILE     compares matrix products on the graphics card and the processor, into FILE

#include "core/Graph.h"
#include "core/Runtime.h"

#include <QMainWindow>

#include <filesystem>
#include <functional>

class QStackedWidget;

namespace logievo {

namespace ui {
class EditorWindow;
class HomeScreen;
} // namespace ui

class MainWindow : public QMainWindow {
public:
    std::function<bool()> canClose;

protected:
    void closeEvent(QCloseEvent* e) override;
};

class Application {
public:
    int run(int argc, char** argv);

private:
    void openProject(const std::filesystem::path& file);
    void createProject(const std::filesystem::path& file, Graph graph);
    void showEditor(Graph graph, const std::filesystem::path& file, Weights weights = {});
    void showHome();
    void updateTitle();

    MainWindow* window_ = nullptr;
    QStackedWidget* stack_ = nullptr;
    ui::HomeScreen* home_ = nullptr;
    ui::EditorWindow* editor_ = nullptr;
};

} // namespace logievo
