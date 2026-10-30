#pragma once

// What differs between systems or goes through Qt for files: UTF-8 paths, the file manager,
// the trash, file dialogs.

#include <QString>

#include <filesystem>
#include <optional>
#include <string>

class QWidget;

namespace logievo::platform {

QString qpath(const std::filesystem::path& p);   // for display and Qt APIs
std::filesystem::path pathOf(const QString& s);  // never through the local 8-bit code page
std::string env(const char* name);
std::filesystem::path appDataDir(); // LOGIEVO_APPDATA, else the system's place for app data
std::filesystem::path assetsDir();  // next to the executable
std::filesystem::path documentsDir();

void revealInFileManager(const std::filesystem::path& p);
bool moveToTrash(const std::filesystem::path& p, std::string* error = nullptr);
void openUrl(const QString& url);

std::optional<std::filesystem::path> openFileDialog(QWidget* parent, const QString& title, const QString& filter);
std::optional<std::filesystem::path> saveFileDialog(QWidget* parent, const QString& title, const std::filesystem::path& suggested,
                                                    const QString& filter);
std::optional<std::filesystem::path> chooseFolderDialog(QWidget* parent, const QString& title, const std::filesystem::path& start);

// Colors the window's system title bar like the app (Windows 11); elsewhere nothing.
void tintTitleBar(QWidget* window);

} // namespace logievo::platform
