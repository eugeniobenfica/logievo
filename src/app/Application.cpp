#include "app/Application.h"

#include "app/Gpu.h"
#include "app/Platform.h"
#include "app/Preferences.h"
#include "app/Settings.h"
#include "bestyle/BeStyle.h"
#include "core/Serialization.h"
#include "ui/Document.h"
#include "ui/EditorWindow.h"
#include "ui/HomeScreen.h"
#include "ui/Logo.h"
#include "ui/WindowChrome.h"
#include "ui/Windows.h"
#include "core/Tensor.h"

#include <QAction>
#include <QApplication>
#include <QMessageBox>
#include <QFile>
#include <QTextStream>

#include <chrono>
#include <random>
#include <tuple>
#include <QCloseEvent>
#include <QFontDatabase>
#include <QStackedWidget>
#include <QTimer>

namespace logievo {

namespace theme = bestyle::theme;

namespace {

// Developer aid: the graphics card against the processor on a few matrix products, written to
// `file` (the app has no console on Windows).
int checkGpu(const QString& file) {
    QFile out(file);
    if (!out.open(QIODevice::WriteOnly | QIODevice::Text)) return 1;
    QTextStream s(&out);
    const gpu::Info info = gpu::probe();
    s << "available " << info.available << "\ndevice " << info.device << "\nerror " << info.error << "\n";
    if (!info.available) return 2;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd;
    bool all = true;
    for (const auto& [n, k, m, ta, tb] : std::vector<std::tuple<int, int, int, bool, bool>>{
             {300, 257, 129, false, false}, {1000, 64, 1000, true, false}, {77, 513, 1031, false, true}, {512, 512, 512, true, true}}) {
        Tensor a(ta ? Shape{k, n} : Shape{n, k}), b(tb ? Shape{m, k} : Shape{k, m});
        for (float& v : a.v) v = nd(rng);
        for (float& v : b.v) v = nd(rng);
        gpu::enable(false);
        auto t0 = std::chrono::steady_clock::now();
        const Tensor cpu = matmul(a, b, ta, tb);
        const double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        gpu::enable(true);
        matmul(a, b, ta, tb); // warm up
        t0 = std::chrono::steady_clock::now();
        const Tensor gpuOut = matmul(a, b, ta, tb);
        const double gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        float worst = 0;
        for (int64_t i = 0; i < cpu.size(); ++i) worst = std::max(worst, std::abs(cpu[i] - gpuOut[i]));
        all &= worst < 1e-2f;
        s << n << "x" << k << "x" << m << " ta " << ta << " tb " << tb << ": worst " << worst << ", cpu " << cpuMs << " ms, gpu " << gpuMs << " ms\n";
    }
    gpu::shutdown();
    return all ? 0 : 3;
}

} // namespace

void MainWindow::closeEvent(QCloseEvent* e) {
    if (canClose && !canClose()) e->ignore();
    else e->accept();
}

int Application::run(int argc, char** argv) {
    QCoreApplication::setApplicationName("Logievo");
    QCoreApplication::setOrganizationName("Logievo");
    QCoreApplication::setApplicationVersion(LOGIEVO_VERSION);
    QApplication app(argc, argv);

    std::filesystem::path projectArg, capture;
    int selectId = 0, waitMs = 1200, settingsPage = -2, helpPage = -2;
    QString gpuCheck, newProject;
    bool train = false;
    QSize size;
    const QStringList args = QCoreApplication::arguments();
    for (int i = 1; i < args.size(); ++i) {
        const QString& a = args[i];
        const QString next = i + 1 < args.size() ? args[i + 1] : QString();
        if (a == "--capture") capture = platform::pathOf(next), ++i;
        else if (a == "--select") selectId = next.toInt(), ++i;
        else if (a == "--wait") waitMs = next.toInt(), ++i;
        else if (a == "--train") train = true;
        else if (a == "--new-project") newProject = next, ++i;
        else if (a == "--settings") settingsPage = next.toInt(), ++i;
        else if (a == "--help") helpPage = next.toInt(), ++i;
        else if (a == "--gpu-check") gpuCheck = next, ++i;
        else if (a == "--write-icon") return ui::appIconImage(1024).save(next) ? 0 : 1; // refreshes assets/branding
        else if (a == "--size") {
            const QStringList wh = next.split('x');
            if (wh.size() == 2) size = QSize(wh[0].toInt(), wh[1].toInt());
            ++i;
        } else projectArg = platform::pathOf(a);
    }

    // The interface font: Inter ships with Logievo, like bemotion.
    for (const char* style : {"Regular", "Medium", "SemiBold", "Bold"})
        QFontDatabase::addApplicationFont(platform::qpath(platform::assetsDir() / "fonts" / (std::string("Inter-") + style + ".ttf")));
    bestyle::setLogoPainter(ui::drawLogo);
    QApplication::setStyle(new bestyle::Style);
    QApplication::setWindowIcon(ui::appIcon());
    ui::installWindowChrome(); // Settings, Help and dialogs framed like the app
    loadSettings();
    applyLook();        // theme, accent, font, text size (Settings › Theme, Interface)
    applyPerformance(); // threads, graphics card (Settings › Performance)
    if (!gpuCheck.isEmpty()) return checkGpu(gpuCheck);

    window_ = new MainWindow;
    window_->setMinimumSize(960, 600);
    stack_ = new QStackedWidget;
    window_->setCentralWidget(stack_);
    home_ = new ui::HomeScreen;
    home_->onOpen = [this](const std::filesystem::path& p) { openProject(p); };
    home_->onCreate = [this](const std::filesystem::path& p, Graph g) { createProject(p, std::move(g)); };
    stack_->addWidget(home_);
    window_->canClose = [this] { return !editor_ || editor_->confirmClose(); };
    updateTitle();
    platform::tintTitleBar(window_);
    QObject::connect(&bestyle::Appearance::get(), &bestyle::Appearance::changed, window_, [this] { platform::tintTitleBar(window_); });
    // Settings and Help from anywhere in the window.
    auto shortcut = [this](const QKeySequence& keys, std::function<void()> run) {
        auto* a = new QAction(window_);
        a->setShortcut(keys);
        a->setShortcutContext(Qt::WindowShortcut);
        QObject::connect(a, &QAction::triggered, window_, [run] { run(); });
        window_->addAction(a);
    };
    shortcut(QKeySequence(Qt::CTRL | Qt::Key_Comma), [] { ui::openSettings(); });
    shortcut(QKeySequence(Qt::Key_F1), [] { ui::openHelp(); });
    if (size.isValid()) {
        window_->resize(size);
        window_->show();
    } else {
        window_->resize(1440, 900);
        window_->showMaximized();
    }
    home_->refresh();
    if (!projectArg.empty()) openProject(projectArg);
    if (editor_ && selectId) editor_->document().select({selectId});
    if (editor_ && train) editor_->document().startTraining();
    if (settingsPage > -2) ui::openSettings(settingsPage);
    if (!newProject.isEmpty()) QTimer::singleShot(0, window_, [this, newProject] { home_->openNewProjectDialog(newProject.toStdString()); });
    if (helpPage > -2) ui::openHelp(helpPage);

    if (!capture.empty())
        QTimer::singleShot(waitMs, window_, [this, capture] {
            window_->grab().save(platform::qpath(capture));
            // Other open windows next to it: shot-Settings.png...
            for (QWidget* w : QApplication::topLevelWidgets())
                if (w != window_ && w->isVisible() && !w->windowTitle().isEmpty()) {
                    std::filesystem::path other = capture;
                    other.replace_filename(platform::pathOf(platform::qpath(capture.stem()) + "-" + w->windowTitle() + platform::qpath(capture.extension())));
                    w->grab().save(platform::qpath(other));
                }
            QCoreApplication::exit(0);
        });

    const int result = QApplication::exec();
    ui::closeWindows();
    delete window_; // the editor first: it stops training
    gpu::shutdown();
    return result;
}

void Application::openProject(const std::filesystem::path& file) {
    Graph g;
    Weights weights;
    std::string error;
    if (!loadGraph(file, g, &error, &weights)) {
        if (editor_) { // opened from the editor's File menu
            QMessageBox::warning(window_, QObject::tr("Could Not Open"),
                                 QObject::tr("Could not open \"%1\": %2").arg(platform::qpath(file.filename()), QString::fromStdString(error)));
            return;
        }
        home_->setError(QObject::tr("Could not open \"%1\": %2").arg(platform::qpath(file.filename()), QString::fromStdString(error)));
        return;
    }
    addRecent(file, g.name);
    showEditor(std::move(g), file, std::move(weights));
}

void Application::createProject(const std::filesystem::path& file, Graph graph) {
    std::string error;
    if (!saveGraph(file, graph, &error)) {
        home_->setError(QObject::tr("Could not create the project: %1").arg(QString::fromStdString(error)));
        return;
    }
    addRecent(file, graph.name);
    showEditor(std::move(graph), file);
}

void Application::showEditor(Graph graph, const std::filesystem::path& file, Weights weights) {
    if (editor_ && !editor_->confirmClose()) return;
    if (editor_) {
        stack_->removeWidget(editor_);
        editor_->deleteLater();
    }
    editor_ = new ui::EditorWindow(std::move(graph), file, std::move(weights));
    editor_->onBack = [this] { showHome(); };
    editor_->onTitleChanged = [this] { updateTitle(); };
    editor_->onOpenProject = [this](const std::filesystem::path& p) { openProject(p); };
    editor_->onNewProject = [this] {
        showHome();
        if (!editor_) home_->openNewProjectDialog(); // unless the user kept the project open
    };
    stack_->addWidget(editor_);
    stack_->setCurrentWidget(editor_);
    editor_->setFocus();
    updateTitle();
}

void Application::showHome() {
    if (editor_ && !editor_->confirmClose()) return;
    if (editor_) {
        stack_->removeWidget(editor_);
        editor_->deleteLater();
        editor_ = nullptr;
    }
    stack_->setCurrentWidget(home_);
    home_->refresh();
    updateTitle();
}

void Application::updateTitle() {
    if (!editor_) {
        window_->setWindowTitle("Logievo");
        return;
    }
    const ui::Document& d = editor_->document();
    window_->setWindowTitle(QString::fromStdString(d.graph.name) + (d.dirty() ? " •" : "") + " — Logievo");
}

} // namespace logievo
