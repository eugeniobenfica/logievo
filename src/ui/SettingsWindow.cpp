// Settings: Theme, Interface, Performance and Custom Nodes pages (ui/Windows.h).

#include "app/Gpu.h"
#include "app/Platform.h"
#include "app/Preferences.h"
#include "app/Settings.h"
#include "bestyle/BeStyle.h"
#include "core/Expression.h"
#include "core/Parallel.h"
#include "core/Tensor.h"
#include "ui/Icons.h"
#include "ui/PagedWindow.h"
#include "ui/Windows.h"

#include <QApplication>
#include <QColorDialog>
#include <QComboBox>
#include <QFontComboBox>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>

#include <chrono>
#include <cmath>
#include <random>

namespace logievo::ui {

namespace theme = bestyle::theme;
using bestyle::Icon;
using bestyle::Rgba;
using bestyle::Tone;

AppSignals& AppSignals::get() {
    static AppSignals s;
    return s;
}

namespace {

QPointer<PagedWindow> settingsWindow;

void changedLook() {
    saveSettings();
    applyLook();
}

bestyle::Switch* switchFor(bool on, std::function<void(bool)> changed) {
    auto* s = new bestyle::Switch;
    s->setChecked(on);
    QObject::connect(s, &QAbstractButton::toggled, s, [changed](bool v) { changed(v); });
    return s;
}

// A theme to pick: a small picture of the editor in its colors, its name under it.
class ThemeTile : public QAbstractButton {
public:
    ThemeTile(theme::Palette palette, Rgba accent, QString name, bool chosen)
        : palette_(theme::resolved(palette)), accent_(accent), name_(std::move(name)), chosen_(chosen) {
        setFixedSize(150, 112);
        setCursor(Qt::PointingHandCursor);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        bestyle::Draw d(p);
        const float w = static_cast<float>(width());
        const bool hot = underMouse();
        if (chosen_ || hot) d.rect({1, 1}, {w - 1, 111}, chosen_ ? theme::kAccent : theme::kBorder, 8.0f, chosen_ ? 2.0f : 1.0f);
        // The picture: the canvas, a side panel, two nodes joined, the accent button.
        const float x0 = 10, y0 = 10, x1 = w - 10, y1 = 78;
        d.rectFilled({x0, y0}, {x1, y1}, palette_.bgDeep, 5.0f);
        d.rectFilled({x0, y0}, {x0 + 34, y1}, palette_.bgPanel, 5.0f);
        d.rectFilled({x0 + 6, y0 + 8}, {x0 + 28, y0 + 14}, accent_, 2.0f);
        for (int i = 0; i < 3; ++i) d.rectFilled({x0 + 6, y0 + 22 + i * 10.0f}, {x0 + 28, y0 + 26 + i * 10.0f}, palette_.textFaint, 1.0f);
        d.rectFilled({x0 + 46, y0 + 14}, {x0 + 76, y0 + 34}, palette_.bgRaised, 3.0f);
        d.rectFilled({x0 + 88, y0 + 36}, {x0 + 118, y0 + 56}, palette_.bgRaised, 3.0f);
        d.rect({x0 + 88, y0 + 36}, {x0 + 118, y0 + 56}, accent_, 3.0f, 1.2f);
        d.bezier({x0 + 76, y0 + 24}, {x0 + 84, y0 + 24}, {x0 + 80, y0 + 46}, {x0 + 88, y0 + 46}, palette_.textDim, 1.2f);
        d.text({12, 86}, chosen_ ? theme::kText : theme::kTextDim, name_);
    }
    void enterEvent(QEnterEvent*) override { update(); }
    void leaveEvent(QEvent*) override { update(); }

private:
    theme::Palette palette_;
    Rgba accent_;
    QString name_;
    bool chosen_;
};

// A round swatch: an accent to pick.
class Swatch : public QAbstractButton {
public:
    Swatch(Rgba color, QString tip, bool chosen) : color_(color), chosen_(chosen) {
        setFixedSize(30, 30);
        setToolTip(tip);
        setCursor(Qt::PointingHandCursor);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        bestyle::Draw d(p);
        d.circleFilled({15, 15}, 10.0f, color_);
        d.circle({15, 15}, 10.0f, theme::kBorder, 1.0f);
        if (chosen_ || underMouse()) d.circle({15, 15}, 13.5f, chosen_ ? theme::kText : theme::kTextFaint, 1.5f);
    }
    void enterEvent(QEnterEvent*) override { update(); }
    void leaveEvent(QEvent*) override { update(); }

private:
    Rgba color_;
    bool chosen_;
};

// A slider with its value written beside it ("100%").
QWidget* sliderRow(int min, int max, int value, const QString& suffix, std::function<void(int)> changed) {
    auto* w = new QWidget;
    auto* r = new QHBoxLayout(w);
    r->setContentsMargins(0, 0, 0, 0);
    r->setSpacing(12);
    auto* s = new QSlider(Qt::Horizontal);
    s->setRange(min, max);
    s->setValue(value);
    s->setFixedWidth(190);
    auto* label = new QLabel(QString::number(value) + suffix);
    label->setMinimumWidth(44);
    label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    QObject::connect(s, &QSlider::valueChanged, label, [label, suffix](int v) { label->setText(QString::number(v) + suffix); });
    // Applied when let go: every panel restyles.
    QObject::connect(s, &QSlider::sliderReleased, s, [s, changed] { changed(s->value()); });
    QObject::connect(s, &QSlider::valueChanged, s, [s, changed](int v) {
        if (!s->isSliderDown()) changed(v);
    });
    r->addWidget(s);
    r->addWidget(label);
    return w;
}

// --- pages -----------------------------------------------------------------------------------

QWidget* themePage() {
    QVBoxLayout* col;
    QWidget* page = pageWidget(col);
    Settings& s = settings();
    {
        SettingsCard* card = addSection(col, QObject::tr("Theme"));
        auto* tiles = new QWidget;
        auto* row = new QHBoxLayout(tiles);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(10);
        const struct {
            const char* id;
            const char* name;
            theme::Palette palette;
        } themes[] = {{"dark", QT_TR_NOOP("Dark"), theme::darkPalette()},
                      {"midnight", QT_TR_NOOP("Midnight"), theme::midnightPalette()},
                      {"light", QT_TR_NOOP("Light"), theme::lightPalette()}};
        for (const auto& t : themes) {
            const std::string id = t.id;
            const Rgba accent = id == s.theme ? theme::kAccent : id == "light" ? bestyle::rgba(24, 24, 27) : bestyle::rgba(255, 255, 255);
            auto* tile = new ThemeTile(t.palette, accent, QObject::tr(t.name), s.theme == id);
            QObject::connect(tile, &QAbstractButton::clicked, tile, [id] {
                settings().theme = id;
                changedLook();
            });
            row->addWidget(tile);
        }
        row->addStretch(1);
        card->addContent(tiles);
    }
    {
        SettingsCard* card = addSection(col, QObject::tr("Accent"));
        auto* swatches = new QWidget;
        auto* row = new QHBoxLayout(swatches);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(4);
        const struct {
            const char* hex;
            const char* name;
        } colors[] = {{"", QT_TR_NOOP("Logievo (white; black on the light theme)")},
                      {"#E8B931", QT_TR_NOOP("Gold")},
                      {"#6EA8FF", QT_TR_NOOP("Blue")},
                      {"#4FD18B", QT_TR_NOOP("Green")},
                      {"#FF6B8A", QT_TR_NOOP("Pink")},
                      {"#B48CFF", QT_TR_NOOP("Violet")},
                      {"#FF9F43", QT_TR_NOOP("Orange")}};
        for (const auto& c : colors) {
            const std::string hex = c.hex;
            Rgba color = s.theme == "light" ? bestyle::rgba(24, 24, 27) : bestyle::rgba(255, 255, 255);
            if (!hex.empty()) theme::fromHex(hex, color);
            auto* sw = new Swatch(color | 0xFF000000u, QObject::tr(c.name), s.accent == hex);
            QObject::connect(sw, &QAbstractButton::clicked, sw, [hex] {
                settings().accent = hex;
                changedLook();
            });
            row->addWidget(sw);
        }
        row->addStretch(1);
        auto* pick = new QPushButton(QObject::tr("Other Color..."));
        QObject::connect(pick, &QPushButton::clicked, pick, [pick] {
            const QColor c = QColorDialog::getColor(theme::q(theme::kAccent), pick->window(), QObject::tr("Accent Color"));
            if (!c.isValid()) return;
            settings().accent = c.name(QColor::HexRgb).toUpper().toStdString();
            changedLook();
        });
        row->addWidget(pick);
        card->addRow(QObject::tr("Accent color"), QObject::tr("The main buttons, the selection and what is active"), nullptr);
        card->addContent(swatches);
    }
    {
        SettingsCard* card = addSection(col, QObject::tr("Shape"));
        card->addRow(QObject::tr("Rounding"), QObject::tr("Corners of fields, buttons and cards"),
                     sliderRow(0, 12, static_cast<int>(std::lround(s.rounding)), QObject::tr(" px"), [](int v) {
                         settings().rounding = v;
                         changedLook();
                     }));
    }
    return page;
}

QWidget* interfacePage() {
    QVBoxLayout* col;
    QWidget* page = pageWidget(col);
    Settings& s = settings();
    {
        SettingsCard* card = addSection(col, QObject::tr("Text"));
        card->addRow(QObject::tr("Text size"), QObject::tr("Every label in the interface"),
                     sliderRow(75, 150, static_cast<int>(std::lround(s.textScale * 100)), "%", [](int v) {
                         settings().textScale = v / 100.0;
                         changedLook();
                     }));
        auto* fonts = new QFontComboBox;
        fonts->setCurrentFont(QFont(QString::fromStdString(s.font)));
        fonts->setFixedWidth(240);
        QObject::connect(fonts, &QFontComboBox::currentFontChanged, fonts, [](const QFont& f) {
            settings().font = f.family().toStdString();
            changedLook();
        });
        card->addRow(QObject::tr("Interface font"), QObject::tr("Inter ships with Logievo; the others come from your system"), fonts);
    }
    {
        SettingsCard* card = addSection(col, QObject::tr("Canvas"));
        card->addRow(QObject::tr("Dot grid"), QObject::tr("Dots behind the nodes, to line them up by eye"), switchFor(s.showGrid, [](bool v) {
                         settings().showGrid = v;
                         saveSettings();
                         emit AppSignals::get().interfaceChanged();
                     }));
        card->addRow(QObject::tr("Snap to the grid"), QObject::tr("Dragged nodes move in steps, so they line up"), switchFor(s.snapToGrid, [](bool v) {
                         settings().snapToGrid = v;
                         saveSettings();
                         emit AppSignals::get().interfaceChanged();
                     }));
    }
    {
        SettingsCard* card = addSection(col, QObject::tr("Projects"));
        auto* folder = new QPushButton(bestyle::makeIcon(Icon::Folder, theme::kTextDim), QObject::tr("Change..."));
        QObject::connect(folder, &QPushButton::clicked, folder, [folder] {
            const auto current = settings().projectsFolder;
            if (auto f = platform::chooseFolderDialog(folder->window(), QObject::tr("Choose a Folder for Your Projects"), current.empty() ? platform::documentsDir() : current)) {
                settings().projectsFolder = *f;
                saveSettings();
                emit AppSignals::get().projectsFolderChanged();
                openSettings(SettingsPage::Interface); // shows the new path
            }
        });
        card->addRow(QObject::tr("Projects folder"),
                     s.projectsFolder.empty() ? QObject::tr("Not chosen yet") : platform::qpath(s.projectsFolder), folder);
    }
    return page;
}

QWidget* performancePage() {
    QVBoxLayout* col;
    QWidget* page = pageWidget(col);
    Settings& s = settings();
    {
        SettingsCard* card = addSection(col, QObject::tr("Graphics card"));
        const gpu::Info info = gpu::probe();
        const QString about = info.available
                                  ? QObject::tr("Big matrix products (Dense, Matrix Multiply) go to %1; the rest stays on the processor").arg(info.device)
                                  : info.error;
        auto* sw = switchFor(s.gpu && info.available, [](bool v) {
            settings().gpu = v;
            saveSettings();
            applyPerformance();
        });
        sw->setEnabled(info.available);
        card->addRow(QObject::tr("GPU acceleration"), about, sw);
    }
    {
        SettingsCard* card = addSection(col, QObject::tr("Processor"));
        auto* threads = new QComboBox;
        const int n = hardwareThreads();
        threads->addItem(QObject::tr("All (%1)").arg(n));
        for (int i = 1; i <= n; ++i) threads->addItem(i == 1 ? QObject::tr("1 thread") : QObject::tr("%1 threads").arg(i));
        threads->setCurrentIndex(std::clamp(s.threads, 0, n));
        threads->setFixedWidth(160);
        QObject::connect(threads, &QComboBox::currentIndexChanged, threads, [](int i) {
            settings().threads = std::max(0, i);
            saveSettings();
            applyPerformance();
        });
        card->addRow(QObject::tr("Threads"),
                     QObject::tr("How many of the processor's %1 threads training uses. Fewer leave the computer freer for other work").arg(n),
                     threads);
    }
    {
        SettingsCard* card = addSection(col, QObject::tr("Speed test"));
        auto* result = new QLabel(QObject::tr("A 768 × 768 matrix product, the heart of training, with the settings above"));
        result->setWordWrap(true);
        bestyle::setTone(result, Tone::TextFaint);
        auto* run = new QPushButton(bestyle::makeIcon(Icon::Speed, theme::kTextDim), QObject::tr("Run Test"));
        QObject::connect(run, &QPushButton::clicked, run, [result] {
            QApplication::setOverrideCursor(Qt::WaitCursor);
            std::mt19937 rng(3);
            std::normal_distribution<float> nd;
            Tensor a({768, 768}), b({768, 768});
            for (float& v : a.v) v = nd(rng);
            for (float& v : b.v) v = nd(rng);
            auto time = [&](Tensor& out) {
                const auto t0 = std::chrono::steady_clock::now();
                out = matmul(a, b);
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            };
            Tensor fast, check;
            const bool gpuOn = gpu::enabled();
            time(fast); // once to warm up (buffers, caches)
            const double ms = time(fast);
            QString text = QObject::tr("%1 ms (%2 billion multiply-adds a second) on %3")
                               .arg(ms, 0, 'f', 1)
                               .arg(768.0 * 768 * 768 / ms / 1e6, 0, 'f', 1)
                               .arg(gpuOn ? gpu::probe().device : QObject::tr("%1 processor threads").arg(threadCount()));
            if (gpuOn) {
                // The same product on the processor: the card must agree with it.
                gpu::enable(false);
                const double cpu = time(check);
                gpu::enable(true);
                float worst = 0;
                for (int64_t i = 0; i < fast.size(); ++i) worst = std::max(worst, std::abs(fast[i] - check[i]));
                text += QObject::tr("; the processor alone: %1 ms. Results agree to %2").arg(cpu, 0, 'f', 1).arg(worst, 0, 'g', 2);
            }
            QApplication::restoreOverrideCursor();
            result->setText(text);
            bestyle::setTone(result, Tone::Text);
        });
        card->addRow(QObject::tr("Matrix product"), QString(), run);
        card->addContent(result);
    }
    return page;
}

QWidget* customNodesPage() {
    QVBoxLayout* col;
    QWidget* page = pageWidget(col);
    col->addWidget(pageTitle(QObject::tr("Custom nodes")));
    col->addWidget(paragraph(QObject::tr("Nodes you write yourself: a formula applied value by value to the inputs a, b, c, d. Logievo works out "
                                         "its derivative, so it trains like any other node. Saved here, they show in the library under "
                                         "Custom; each project keeps its own copy of the code it uses.")));
    {
        SettingsCard* ref = addSection(col, QObject::tr("The language"));
        auto mono = [](const QString& text) {
            auto* l = new QLabel(text);
            QFont f = QFontDatabase::systemFont(QFontDatabase::FixedFont);
            l->setFont(f);
            l->setTextInteractionFlags(Qt::TextSelectableByMouse);
            bestyle::setTone(l, Tone::TextDim);
            return l;
        };
        ref->addRow(QObject::tr("Values"), QObject::tr("Inputs a b c d (x is a), numbers, pi, e"), nullptr);
        ref->addRow(QObject::tr("Operators"), QObject::tr("+ - * / ^, comparisons < > <= >= == != (1 or 0)"), nullptr);
        ref->addRow(QObject::tr("Functions"), QObject::tr("abs exp log sqrt sin cos tan tanh sigmoid relu softplus sign floor, "
                                                          "min(x, y) max(x, y) pow(x, y) clamp(x, lo, hi) if(cond, yes, no)"),
                    nullptr);
        ref->addContent(mono("# lines name values for the next ones\ngate = sigmoid(b)\na * gate + (1 - gate) * c   # the last line is the result"));
    }
    auto& list = settings().customNodes;
    for (size_t i = 0; i < list.size(); ++i) {
        SettingsCard* card = addSection(col, QString::fromStdString(list[i].name));
        auto* name = new QLineEdit(QString::fromStdString(list[i].name));
        name->setFixedWidth(240);
        auto* inputs = new QComboBox;
        inputs->addItems({"a", "a, b", "a, b, c", "a, b, c, d"});
        inputs->setCurrentIndex(std::clamp(list[i].inputs, 1, 4) - 1);
        inputs->setFixedWidth(240);
        auto* about = new QLineEdit(QString::fromStdString(list[i].description));
        about->setPlaceholderText(QObject::tr("What it does (shown on its tile)"));
        about->setFixedWidth(240);
        auto* code = new QPlainTextEdit(QString::fromStdString(list[i].code));
        code->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
        code->setMinimumHeight(110);
        code->setTabChangesFocus(true);
        auto* status = new QLabel;
        status->setWordWrap(true);
        auto* remove = new QPushButton(bestyle::makeIcon(Icon::Trash, theme::kDanger), QObject::tr("Delete"));
        // Every change is checked and saved a moment after typing stops.
        auto* later = new QTimer(code);
        later->setSingleShot(true);
        later->setInterval(400);
        auto check = [=] {
            Expression e;
            std::string error;
            const bool ok = e.compile(code->toPlainText().toStdString(), inputs->currentIndex() + 1, &error);
            status->setText(ok ? QObject::tr("Reads well.") : QString::fromStdString(error));
            bestyle::setTone(status, ok ? Tone::TextFaint : Tone::Danger);
        };
        check();
        QObject::connect(later, &QTimer::timeout, code, [=] {
            auto& nodes = settings().customNodes;
            if (i >= nodes.size()) return;
            nodes[i].name = name->text().trimmed().isEmpty() ? "Custom" : name->text().trimmed().toStdString();
            nodes[i].inputs = inputs->currentIndex() + 1;
            nodes[i].description = about->text().trimmed().toStdString();
            nodes[i].code = code->toPlainText().toStdString();
            saveSettings();
            check();
            emit AppSignals::get().customNodesChanged();
        });
        auto start = [later] { later->start(); };
        QObject::connect(name, &QLineEdit::textEdited, code, start);
        QObject::connect(about, &QLineEdit::textEdited, code, start);
        QObject::connect(inputs, &QComboBox::currentIndexChanged, code, start);
        QObject::connect(code, &QPlainTextEdit::textChanged, code, start);
        QObject::connect(remove, &QPushButton::clicked, remove, [i] {
            auto& nodes = settings().customNodes;
            if (i < nodes.size()) nodes.erase(nodes.begin() + static_cast<long>(i));
            saveSettings();
            emit AppSignals::get().customNodesChanged();
            QTimer::singleShot(0, [] { openSettings(SettingsPage::CustomNodes); });
        });
        card->addRow(QObject::tr("Name"), QString(), name);
        card->addRow(QObject::tr("Inputs"), QString(), inputs);
        card->addRow(QObject::tr("Description"), QString(), about);
        card->addRow(QObject::tr("Code"), QString(), code, true);
        auto* foot = new QHBoxLayout;
        foot->addWidget(status, 1);
        foot->addWidget(remove);
        card->addLayout(foot);
    }
    col->addSpacing(10);
    auto* add = bestyle::accentButton(QObject::tr("+  New Custom Node"));
    add->setMinimumHeight(34);
    QObject::connect(add, &QPushButton::clicked, add, [] {
        settings().customNodes.push_back({"Swish", "a * sigmoid(a)", 1, "x times sigmoid(x): a smooth, self-gated activation"});
        saveSettings();
        emit AppSignals::get().customNodesChanged();
        QTimer::singleShot(0, [] { openSettings(SettingsPage::CustomNodes); });
    });
    col->addWidget(add, 0, Qt::AlignLeft);
    return page;
}

} // namespace

void closeSettingsWindow() {
    if (settingsWindow) settingsWindow->close();
}

void openSettings(int page) {
    if (!settingsWindow) {
        // The footer: everything back as it came (Help is in the editor's Help menu, F1).
        auto* footer = new QWidget;
        auto* fcol = new QVBoxLayout(footer);
        fcol->setContentsMargins(0, 0, 0, 0);
        fcol->setSpacing(6);
        auto* reset = new QPushButton(QObject::tr("Reset All"));
        reset->setToolTip(QObject::tr("The look and performance as they came (projects and custom nodes stay)"));
        QObject::connect(reset, &QPushButton::clicked, reset, [] {
            Settings& s = settings();
            const Settings fresh;
            s.theme = fresh.theme, s.accent = fresh.accent, s.rounding = fresh.rounding, s.textScale = fresh.textScale, s.font = fresh.font;
            s.showGrid = fresh.showGrid, s.snapToGrid = fresh.snapToGrid, s.gpu = fresh.gpu, s.threads = fresh.threads;
            saveSettings();
            applyPerformance();
            applyLook();
            emit AppSignals::get().interfaceChanged();
        });
        fcol->addWidget(reset);
        settingsWindow = new PagedWindow(QObject::tr("Settings"),
                                         {{QObject::tr("Theme"), Icon::Palette, themePage},
                                          {QObject::tr("Interface"), Icon::Sliders, interfacePage},
                                          {QObject::tr("Performance"), Icon::Speed, performancePage},
                                          {QObject::tr("Custom Nodes"), Icon::Expression, customNodesPage}},
                                         footer);
        settingsWindow->setAttribute(Qt::WA_DeleteOnClose, true);
        settingsWindow->resize(860, 650);
        // A new look rebuilds the page (its picture of the themes, its colors).
        QObject::connect(&bestyle::Appearance::get(), &bestyle::Appearance::changed, settingsWindow.data(), [] {
            if (settingsWindow) QTimer::singleShot(0, settingsWindow.data(), [] { settingsWindow->rebuild(); });
        });
        settingsWindow->showPage(page >= 0 ? page : SettingsPage::Theme);
    } else if (page >= 0) {
        settingsWindow->showPage(page);
    } else {
        settingsWindow->rebuild();
    }
    settingsWindow->present();
}

} // namespace logievo::ui
