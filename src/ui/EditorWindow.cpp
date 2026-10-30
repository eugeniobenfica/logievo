#include "ui/EditorWindow.h"

#include "app/Platform.h"
#include "bestyle/BeStyle.h"
#include "ui/Document.h"
#include "ui/Icons.h"
#include "ui/Inspector.h"
#include "ui/Logo.h"
#include "ui/NodeCanvas.h"
#include "ui/NodeLibrary.h"
#include "ui/Windows.h"
#include "app/Settings.h"
#include "core/Export.h"
#include "core/NodeTypes.h"
#include "core/Serialization.h"

#include <QAction>
#include <QCoreApplication>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QPushButton>
#include <QSplitter>
#include <QVBoxLayout>


namespace logievo::ui {

namespace theme = bestyle::theme;
using bestyle::Icon;
using bestyle::Tone;

namespace {

// A panel's background, read from the theme when it paints.
class Panel : public QWidget {
public:
    explicit Panel(bestyle::Rgba (*color)(), QWidget* parent = nullptr) : QWidget(parent), color_(color) {}

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.fillRect(rect(), theme::q(color_()));
        p.setPen(theme::q(theme::kBorder));
        if (borderBottom) p.drawLine(0, height() - 1, width(), height() - 1);
    }

public:
    bool borderBottom = false;

private:
    bestyle::Rgba (*color_)();
};

bestyle::Rgba headerColor() { return theme::kHeader; }
bestyle::Rgba panelColor() { return theme::kBgPanel; }

// The Logievo mark, which is also the way back to the projects (as bemotion's): it lights up
// under the mouse.
class LogoButton : public QAbstractButton {
public:
    LogoButton() {
        setFixedSize(34, 34);
        setCursor(Qt::PointingHandCursor);
        setToolTip(tr("Back to the projects"));
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        bestyle::Draw d(p);
        if (underMouse()) d.rectFilled({1, 1}, {33, 33}, isDown() ? theme::kBgRaised : theme::kBgHover, 7.0f);
        drawLogo(p, {17, 17}, 11.0f, theme::kAccent);
    }
    void enterEvent(QEnterEvent*) override { update(); }
    void leaveEvent(QEvent*) override { update(); }
};

} // namespace

EditorWindow::EditorWindow(Graph graph, std::filesystem::path file, Weights weights, QWidget* parent)
    : QWidget(parent), doc_(std::make_unique<Document>(std::move(graph), std::move(file), std::move(weights))) {
    auto* col = new QVBoxLayout(this);
    col->setContentsMargins(0, 0, 0, 0);
    col->setSpacing(0);

    // --- the menu bar, as bemotion's: the mark (back to the projects), the menus, and on the
    // right Settings, Train and Export -----------------------------------------------------------
    auto* header = new Panel(headerColor);
    header->borderBottom = true;
    auto* headerRow = new QHBoxLayout(header);
    headerRow->setContentsMargins(0, 0, 0, 1);
    menuBar_ = new QMenuBar;
    headerRow->addWidget(menuBar_);
    col->addWidget(header);

    // --- library | canvas | inspector -----------------------------------------------------
    auto* split = new QSplitter(Qt::Horizontal);
    split->setHandleWidth(1);
    split->setChildrenCollapsible(false);
    auto* left = new Panel(panelColor);
    auto* leftCol = new QVBoxLayout(left);
    leftCol->setContentsMargins(0, 0, 0, 0);
    library_ = new NodeLibrary;
    leftCol->addWidget(library_);
    left->setMinimumWidth(200);
    canvas_ = new NodeCanvas(*doc_);
    auto* right = new Panel(panelColor);
    auto* rightCol = new QVBoxLayout(right);
    rightCol->setContentsMargins(0, 0, 0, 0);
    inspector_ = new Inspector(*doc_);
    rightCol->addWidget(inspector_);
    right->setMinimumWidth(240);
    split->addWidget(left);
    split->addWidget(canvas_);
    split->addWidget(right);
    split->setStretchFactor(0, 0);
    split->setStretchFactor(1, 1);
    split->setStretchFactor(2, 0);
    split->setSizes({272, 1000, 310});
    col->addWidget(split, 1);

    library_->onAdd = [this](const std::string& type, const std::string& name, const Json& params) {
        canvas_->addNode(type, {}, name, params);
        canvas_->setFocus();
    };
    inspector_->onDelete = [this] { canvas_->deleteSelection(); };
    inspector_->onDuplicate = [this] { canvas_->duplicateSelection(); };
    buildMenus();

    connect(doc_.get(), &Document::changed, this, &EditorWindow::updateHeader);
    connect(doc_.get(), &Document::saved, this, &EditorWindow::updateHeader);
    connect(doc_.get(), &Document::trainingChanged, this, &EditorWindow::updateTrainButton);
    connect(doc_.get(), &Document::changed, this, &EditorWindow::updateTrainButton);
    updateTrainButton();
    updateHeader();
    canvas_->setFocus();
}

EditorWindow::~EditorWindow() = default;

void EditorWindow::buildMenus() {
    QMenuBar* bar = menuBar_;
    // A menu entry with its shortcut, working while the editor (or anything in it) has the
    // focus; a text field keeps its own.
    auto action = [this](QMenu* menu, const QString& text, const QList<QKeySequence>& keys, std::function<void()> run) {
        auto* a = new QAction(text, this);
        a->setShortcuts(keys);
        a->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        a->setAutoRepeat(false);
        connect(a, &QAction::triggered, this, [run] { run(); });
        addAction(a);
        menu->addAction(a);
        return a;
    };
    // An entry whose key is handled elsewhere (the canvas, the window): the key is only shown.
    auto entry = [this](QMenu* menu, const QString& text, const QString& keys, std::function<void()> run) {
        return menu->addAction(keys.isEmpty() ? text : text + "\t" + keys, this, [run] { run(); });
    };

    // The mark: back to the projects. In a row with margins, so it sits centered on the titles.
    auto* back = new LogoButton;
    connect(back, &QAbstractButton::clicked, this, [this] {
        if (onBack) onBack();
    });
    auto* homeCorner = new QWidget;
    auto* homeRow = new QHBoxLayout(homeCorner);
    homeRow->setContentsMargins(6, 2, 2, 2);
    homeRow->addWidget(back, 0, Qt::AlignVCenter);
    bar->setCornerWidget(homeCorner, Qt::TopLeftCorner);

    QMenu* file = bar->addMenu(tr("File"));
    entry(file, tr("New Project..."), {}, [this] {
        if (onNewProject) onNewProject();
    });
    entry(file, tr("Open Project..."), {}, [this] {
        if (auto p = platform::openFileDialog(this, tr("Open Logievo Project"), tr("Logievo Project (*%1)").arg(kProjectExtension)); p && onOpenProject)
            onOpenProject(*p);
    });
    file->addSeparator();
    action(file, tr("Save"), {QKeySequence::Save}, [this] { save(); });
    entry(file, tr("Export..."), {}, [this] { exportModel(); });
    entry(file, tr("Show in Folder"), {}, [this] {
        if (!doc_->file.empty()) platform::revealInFileManager(doc_->file);
    });
    file->addSeparator();
    entry(file, tr("Close Project"), {}, [this] {
        if (onBack) onBack();
    });

    QMenu* edit = bar->addMenu(tr("Edit"));
    action(edit, tr("Undo"), {QKeySequence::Undo}, [this] { doc_->undo(); });
    action(edit, tr("Redo"), {QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Z), QKeySequence(Qt::CTRL | Qt::Key_Y)}, [this] { doc_->redo(); });
    edit->addSeparator();
    action(edit, tr("Duplicate"), {QKeySequence(Qt::CTRL | Qt::Key_D)}, [this] { canvas_->duplicateSelection(); });
    entry(edit, tr("Delete"), tr("Delete"), [this] { canvas_->deleteSelection(); });
    edit->addSeparator();
    action(edit, tr("Select All"), {QKeySequence::SelectAll}, [this] { canvas_->selectAll(); });
    entry(edit, tr("Select Nothing"), tr("Esc"), [this] { doc_->select({}); });
    edit->addSeparator();
    entry(edit, tr("Settings..."), "Ctrl+,", [] { openSettings(); });

    // Add: every node, by category, in the middle of the view (as a click in the library).
    QMenu* add = bar->addMenu(tr("Add"));
    connect(add, &QMenu::aboutToShow, this, [this, add] {
        add->clear();
        for (const std::string& category : nodeCategories()) {
            QMenu* sub = add->addMenu(bestyle::menuIcon(categoryIcon(category), theme::kTextDim), tr(category.c_str()));
            for (const NodeType& t : nodeTypes())
                if (t.category == category) {
                    const std::string id = t.id;
                    sub->addAction(bestyle::menuIcon(nodeIcon(id), theme::kTextDim), tr(t.name.c_str()), this,
                                   [this, id] { library_->onAdd(id, {}, Json::object()); });
                }
            if (category == "Custom")
                for (const CustomNode& c : settings().customNodes) {
                    const std::string name = c.name;
                    const Json params{{"code", c.code}, {"inputs", c.inputs}};
                    sub->addAction(bestyle::menuIcon(Icon::Edit, theme::kTextDim), QString::fromStdString(name), this,
                                   [this, name, params] { library_->onAdd("custom", name, params); });
                }
        }
    });

    QMenu* view = bar->addMenu(tr("View"));
    action(view, tr("Frame All"), {QKeySequence(Qt::Key_F)}, [this] { canvas_->frameAll(); });
    action(view, tr("Zoom In"), {QKeySequence(Qt::CTRL | Qt::Key_Equal), QKeySequence(Qt::CTRL | Qt::Key_Plus)}, [this] { canvas_->zoomBy(1.25); });
    action(view, tr("Zoom Out"), {QKeySequence(Qt::CTRL | Qt::Key_Minus)}, [this] { canvas_->zoomBy(1 / 1.25); });
    action(view, tr("Actual Size"), {QKeySequence(Qt::CTRL | Qt::Key_0)}, [this] { canvas_->zoomBy(1 / canvas_->zoom()); });
    view->addSeparator();
    QAction* grid = entry(view, tr("Dot Grid"), {}, [] {
        settings().showGrid = !settings().showGrid;
        saveSettings();
        emit AppSignals::get().interfaceChanged();
    });
    QAction* snap = entry(view, tr("Snap to Grid"), {}, [] {
        settings().snapToGrid = !settings().snapToGrid;
        saveSettings();
        emit AppSignals::get().interfaceChanged();
    });
    grid->setCheckable(true);
    snap->setCheckable(true);
    connect(view, &QMenu::aboutToShow, this, [grid, snap] {
        grid->setChecked(settings().showGrid);
        snap->setChecked(settings().snapToGrid);
    });

    QMenu* model = bar->addMenu(tr("Model"));
    QAction* trainAction = action(model, tr("Train"), {QKeySequence(Qt::CTRL | Qt::Key_Return)}, [this] { toggleTraining(); });
    QAction* startOver = entry(model, tr("Start Over"), {}, [this] { doc_->forgetWeights(); });
    model->addSeparator();
    entry(model, tr("Training Settings"), {}, [this] { doc_->select({}); });
    connect(model, &QMenu::aboutToShow, this, [this, trainAction, startOver] {
        trainAction->setText(doc_->training() ? tr("Stop Training") : tr("Train"));
        trainAction->setEnabled(doc_->training() || doc_->analysis().valid());
        startOver->setEnabled(!doc_->training() && !doc_->weights.empty());
    });

    // Help: the pages of the Help window, then the outside link.
    QMenu* help = bar->addMenu(tr("Help"));
    help->addAction(bestyle::menuIcon(Icon::Keyboard, theme::kTextDim), tr("Keyboard Shortcuts") + "\tF1", this, [] { openHelp(HelpPage::Shortcuts); });
    help->addAction(bestyle::menuIcon(Icon::Help, theme::kTextDim), tr("Learn Logievo"), this, [] { openHelp(HelpPage::Learn); });
    help->addSeparator();
    help->addAction(bestyle::menuIcon(Icon::Logo, theme::kTextDim), tr("About Logievo"), this, [] { openHelp(HelpPage::About); });
    help->addAction(bestyle::menuIcon(Icon::Star, theme::kTextDim), tr("Credits"), this, [] { openHelp(HelpPage::Credits); });
    help->addAction(bestyle::menuIcon(Icon::Lock, theme::kTextDim), tr("License"), this, [] { openHelp(HelpPage::License); });
    help->addSeparator();
    help->addAction(bestyle::menuIcon(Icon::Coffee, theme::kAccent), tr("Support Logievo (Buy Me a Coffee)..."), this,
                    [] { platform::openUrl("https://buymeacoffee.com/eugeniobenfica"); });

    // Right side: Settings, the unsaved mark, Train / Stop, Export (Save is Ctrl+S and File › Save).
    auto* corner = new QWidget;
    auto* row = new QHBoxLayout(corner);
    row->setContentsMargins(0, 2, 8, 2);
    row->setSpacing(8);
    auto* gear = new bestyle::IconButton(Icon::Gear, tr("Settings (Ctrl+,)"), 28);
    connect(gear, &QAbstractButton::clicked, this, [] { openSettings(); });
    row->addWidget(gear);
    state_ = new QLabel;
    bestyle::setTone(state_, Tone::TextFaint);
    row->addWidget(state_);
    // Train / Stop, with the epoch while it runs; the Training section (on the right, with
    // nothing selected) has the settings and the curve.
    train_ = new QPushButton;
    train_->setFixedHeight(28);
    train_->setMinimumWidth(104);
    connect(train_, &QPushButton::clicked, this, [this] { toggleTraining(); });
    row->addWidget(train_);
    auto* exportButton = bestyle::accentButton(tr("Export"));
    exportButton->setFixedHeight(28);
    exportButton->setMinimumWidth(78);
    exportButton->setToolTip(tr("Export the trained model as a command-line executable"));
    connect(exportButton, &QPushButton::clicked, this, [this] { exportModel(); });
    row->addWidget(exportButton);
    bar->setCornerWidget(corner, Qt::TopRightCorner);
}

void EditorWindow::exportModel() {
    if (doc_->training()) doc_->stopTraining();
    const QString title = tr("Could Not Export");
    std::string error;
    CommandModel model;
    if (!prepareCommandModel(doc_->graph, doc_->weights, model, &error)) {
        QMessageBox::warning(this, title, QString::fromStdString(error));
        return;
    }
#ifdef Q_OS_WIN
    const QString suffix = ".exe";
#else
    const QString suffix;
#endif
    const auto runner = platform::pathOf(QCoreApplication::applicationDirPath() + "/logievo_run" + suffix);
    const auto folder = doc_->file.empty() ? platform::documentsDir() : doc_->file.parent_path();
    const auto out = platform::saveFileDialog(this, tr("Export Model"), folder / platform::pathOf(QString::fromStdString(doc_->graph.name) + suffix),
                                              suffix.isEmpty() ? tr("All files (*)") : tr("Executable (*.exe)"));
    if (!out) return;
    if (!exportExecutable(doc_->graph, doc_->weights, runner, *out, &error)) {
        QMessageBox::warning(this, title, QString::fromStdString(error));
        return;
    }
    QMessageBox box(this);
    box.setWindowTitle(tr("Model Exported"));
    box.setText(tr("\"%1\" runs the model from the command line:").arg(platform::qpath(out->filename())));
    box.setInformativeText(tr("%1 \"abra a calculadora\"\n\nIt opens what the sentence asks for, or does nothing, and prints nothing. "
                              "With no sentence it reads one line from the input (pipe Whisper's text into it). "
                              "--dry-run only prints what it would open.")
                               .arg(platform::qpath(out->filename())));
    QPushButton* show = box.addButton(tr("Show in Folder"), QMessageBox::ActionRole);
    box.addButton(QMessageBox::Ok);
    box.exec();
    if (box.clickedButton() == show) platform::revealInFileManager(*out);
}

void EditorWindow::toggleTraining() {
    if (doc_->training()) {
        doc_->stopTraining();
        return;
    }
    if (!doc_->analysis().valid()) return;
    doc_->select({});
    doc_->startTraining();
}

void EditorWindow::updateHeader() {
    // The name is in the window's title; here only whether there is something to save.
    state_->setText(doc_->dirty() ? tr("Edited") : QString());
    state_->setToolTip(platform::qpath(doc_->file));
    state_->parentWidget()->adjustSize(); // the menu bar's corner keeps its size otherwise
    menuBar_->adjustSize();
    menuBar_->update();
    if (onTitleChanged) onTitleChanged();
}

bool EditorWindow::save() {
    std::string error;
    if (doc_->save(&error)) return true;
    QMessageBox::warning(this, tr("Could Not Save"), tr("The project could not be saved: %1").arg(QString::fromStdString(error)));
    return false;
}

bool EditorWindow::confirmClose() {
    if (!doc_->dirty()) return true;
    QMessageBox box(this);
    box.setWindowTitle(tr("Unsaved Changes"));
    box.setText(tr("Save the changes to \"%1\"?").arg(QString::fromStdString(doc_->graph.name)));
    box.setInformativeText(tr("They are lost if you don't."));
    QPushButton* saveButton = box.addButton(tr("Save"), QMessageBox::AcceptRole);
    saveButton->setProperty("accent", true);
    QPushButton* discard = box.addButton(tr("Don't Save"), QMessageBox::DestructiveRole);
    box.addButton(QMessageBox::Cancel);
    box.setDefaultButton(saveButton);
    box.exec();
    if (box.clickedButton() == saveButton) return save();
    return box.clickedButton() == discard;
}

void EditorWindow::updateTrainButton() {
    const TrainReport r = doc_->trainingReport();
    if (doc_->training()) {
        train_->setIcon(bestyle::makeIcon(Icon::Pause, theme::kText));
        train_->setText(tr("Stop  ·  %1 / %2").arg(r.epoch).arg(r.epochs));
        train_->setToolTip(tr("Stop training (what was learned so far stays)"));
    } else {
        train_->setIcon(bestyle::makeIcon(Icon::Play, theme::kTextDim));
        train_->setText(tr("Train"));
        train_->setToolTip(doc_->analysis().valid() ? tr("Train the model: the Loss nodes get smaller")
                                                    : tr("The model needs a Loss node, and every shape fitting, to train"));
    }
}

} // namespace logievo::ui
