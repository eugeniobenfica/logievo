#include "app/Platform.h"

#include "bestyle/Theme.h"

#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QProcess>
#include <QStandardPaths>
#include <QUrl>
#include <QWidget>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace logievo::platform {

namespace theme = bestyle::theme;

QString qpath(const std::filesystem::path& p) {
#ifdef _WIN32
    return QString::fromStdWString(p.wstring());
#else
    return QString::fromStdString(p.string());
#endif
}

std::filesystem::path pathOf(const QString& s) {
#ifdef _WIN32
    return std::filesystem::path(s.toStdWString());
#else
    return std::filesystem::path(s.toStdString());
#endif
}

std::string env(const char* name) { return qEnvironmentVariable(name).toStdString(); }

std::filesystem::path appDataDir() {
    if (const QString dir = qEnvironmentVariable("LOGIEVO_APPDATA"); !dir.isEmpty()) return pathOf(dir);
    return pathOf(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation));
}

std::filesystem::path assetsDir() { return pathOf(QCoreApplication::applicationDirPath()) / "assets"; }

std::filesystem::path documentsDir() { return pathOf(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)); }

void revealInFileManager(const std::filesystem::path& p) {
#ifdef _WIN32
    QProcess::startDetached("explorer.exe", {"/select,", QDir::toNativeSeparators(qpath(p))});
#else
    QDesktopServices::openUrl(QUrl::fromLocalFile(qpath(std::filesystem::is_directory(p) ? p : p.parent_path())));
#endif
}

bool moveToTrash(const std::filesystem::path& p, std::string* error) {
    if (QFile::moveToTrash(qpath(p))) return true;
    if (error) *error = "the system could not move it to the trash";
    return false;
}

void openUrl(const QString& url) { QDesktopServices::openUrl(QUrl(url)); }

std::optional<std::filesystem::path> openFileDialog(QWidget* parent, const QString& title, const QString& filter) {
    const QString f = QFileDialog::getOpenFileName(parent, title, {}, filter);
    if (f.isEmpty()) return std::nullopt;
    return pathOf(f);
}

std::optional<std::filesystem::path> saveFileDialog(QWidget* parent, const QString& title, const std::filesystem::path& suggested,
                                                    const QString& filter) {
    const QString f = QFileDialog::getSaveFileName(parent, title, qpath(suggested), filter);
    if (f.isEmpty()) return std::nullopt;
    return pathOf(f);
}

std::optional<std::filesystem::path> chooseFolderDialog(QWidget* parent, const QString& title, const std::filesystem::path& start) {
    const QString f = QFileDialog::getExistingDirectory(parent, title, qpath(start));
    if (f.isEmpty()) return std::nullopt;
    return pathOf(f);
}

void tintTitleBar(QWidget* window) {
#ifdef _WIN32
    using SetAttribute = HRESULT(WINAPI*)(HWND, DWORD, LPCVOID, DWORD);
    static const auto set = [] {
        const HMODULE dwm = LoadLibraryW(L"dwmapi.dll");
        return dwm ? reinterpret_cast<SetAttribute>(reinterpret_cast<void*>(GetProcAddress(dwm, "DwmSetWindowAttribute"))) : nullptr;
    }();
    if (!set) return;
    const HWND hwnd = reinterpret_cast<HWND>(window->winId());
    const bestyle::Rgba header = theme::kHeader;
    const BOOL dark = bestyle::red(header) + bestyle::green(header) + bestyle::blue(header) < 3 * 128;
    // COLORREF is 0x00BBGGRR, as Rgba without its alpha.
    const COLORREF caption = header & 0xFFFFFF, text = theme::kText & 0xFFFFFF, border = theme::kBorder & 0xFFFFFF;
    set(hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof dark);
    set(hwnd, 34 /* DWMWA_BORDER_COLOR */, &border, sizeof border);
    set(hwnd, 35 /* DWMWA_CAPTION_COLOR */, &caption, sizeof caption);
    set(hwnd, 36 /* DWMWA_TEXT_COLOR */, &text, sizeof text);
#else
    (void)window;
#endif
}

} // namespace logievo::platform
