#pragma once

// The app's secondary windows, like bemotion's: Settings (Theme, Interface, Performance, Custom
// Nodes) and Help (Shortcuts, Learn, About, Credits, License). One of each, opened from the home
// screen, the editor, Ctrl+, and F1.

#include <QObject>

namespace logievo::ui {

namespace SettingsPage {
enum { Theme, Interface, Performance, CustomNodes };
}
namespace HelpPage {
enum { Shortcuts, Learn, About, Credits, License };
}

void openSettings(int page = -1); // -1: the page it was on
void openHelp(int page = -1);
void closeWindows(); // before the application quits

// What Settings changes beyond the look (bestyle::Appearance says that one).
class AppSignals : public QObject {
    Q_OBJECT
public:
    static AppSignals& get();

signals:
    void customNodesChanged(); // the library shows them
    void interfaceChanged();   // grid, snapping
    void projectsFolderChanged();

private:
    AppSignals() = default;
};

} // namespace logievo::ui
