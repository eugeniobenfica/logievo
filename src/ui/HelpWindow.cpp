// Help: Shortcuts, Learn, About, Credits and License pages (ui/Windows.h).

#include "app/Platform.h"
#include "bestyle/BeStyle.h"
#include "ui/Logo.h"
#include "ui/PagedWindow.h"
#include "ui/Windows.h"

#include <QFontMetrics>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

namespace logievo::ui {

namespace theme = bestyle::theme;
using bestyle::Icon;
using bestyle::Tone;

namespace {

QPointer<PagedWindow> helpWindow;
QString shortcutFilter;

constexpr const char* kSupportUrl = "https://buymeacoffee.com/eugeniobenfica";

constexpr const char* kLicense = R"(MIT License

Copyright (c) 2026 The Logievo contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
)";

// "Ctrl+Shift+Z / Ctrl+Y" drawn as key caps, as bemotion's: each key a raised cap with its side
// showing, "+" between the keys of a combination and "/" between alternatives.
class KeyCaps : public QWidget {
public:
    explicit KeyCaps(QString keys) : keys_(std::move(keys)) {
        setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        setMinimumHeight(fontMetrics().height() + 12);
        setMinimumWidth(static_cast<int>(layoutWidth()));
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        bestyle::Draw d(p);
        float x = 0;
        const float h = static_cast<float>(fontMetrics().height() + 8);
        for (const Part& part : parts()) {
            if (part.separator) {
                const float w = d.textWidth(part.text);
                d.text({x + 3, (h - d.textHeight()) * 0.5f}, theme::kTextFaint, part.text);
                x += w + 6;
                continue;
            }
            const float w = std::max(h, d.textWidth(part.text) + 14.0f);
            const float r = std::min(6.0f, h * 0.25f);
            d.rectFilled({x, 2}, {x + w, h + 2}, theme::kBorder, r); // the key's side
            d.rectFilled({x, 0}, {x + w, h}, theme::kBgRaised, r);
            d.rect({x, 0}, {x + w, h}, theme::kBorder, r);
            d.text({x + (w - d.textWidth(part.text)) * 0.5f, (h - d.textHeight()) * 0.5f}, theme::kText, part.text);
            x += w + 3;
        }
    }

private:
    struct Part {
        QString text;
        bool separator;
    };
    std::vector<Part> parts() const {
        std::vector<Part> out;
        const QStringList combos = keys_.split(" / ");
        for (int c = 0; c < combos.size(); ++c) {
            if (c) out.push_back({" / ", true});
            // "+" joins keys, except a key that is itself "+" (Ctrl++).
            const QString& combo = combos[c];
            int k = 0;
            bool first = true;
            while (k < combo.size()) {
                int plus = combo.indexOf('+', k + 1);
                if (plus < 0) plus = static_cast<int>(combo.size());
                if (!first) out.push_back({"+", true});
                first = false;
                out.push_back({combo.mid(k, plus - k), false});
                k = plus + 1;
            }
        }
        return out;
    }
    float layoutWidth() const {
        const QFontMetricsF fm(font());
        const float h = static_cast<float>(fm.height() + 8);
        float x = 0;
        for (const Part& part : parts())
            x += part.separator ? static_cast<float>(fm.horizontalAdvance(part.text)) + 6
                                : std::max(h, static_cast<float>(fm.horizontalAdvance(part.text)) + 14.0f) + 3;
        return x + 4;
    }
    QString keys_;
};

struct Shortcut {
    const char* group;
    const char* what;
    const char* keys; // "Ctrl+S"; alternatives split by " / "
};

const std::vector<Shortcut>& shortcuts() {
    static const std::vector<Shortcut> list{
        {QT_TR_NOOP("Files"), QT_TR_NOOP("Save"), "Ctrl+S"},
        {QT_TR_NOOP("Files"), QT_TR_NOOP("Settings"), "Ctrl+,"},
        {QT_TR_NOOP("Files"), QT_TR_NOOP("Help"), "F1"},
        {QT_TR_NOOP("Editing"), QT_TR_NOOP("Undo"), "Ctrl+Z"},
        {QT_TR_NOOP("Editing"), QT_TR_NOOP("Redo"), "Ctrl+Shift+Z / Ctrl+Y"},
        {QT_TR_NOOP("Editing"), QT_TR_NOOP("Duplicate"), "Ctrl+D"},
        {QT_TR_NOOP("Editing"), QT_TR_NOOP("Delete"), "Delete / Backspace"},
        {QT_TR_NOOP("Editing"), QT_TR_NOOP("Select all"), "Ctrl+A"},
        {QT_TR_NOOP("Editing"), QT_TR_NOOP("Select nothing"), "Esc"},
        {QT_TR_NOOP("Model"), QT_TR_NOOP("Train / Stop"), "Ctrl+Enter"},
        {QT_TR_NOOP("Canvas"), QT_TR_NOOP("Frame every node"), "F"},
        {QT_TR_NOOP("Canvas"), QT_TR_NOOP("Zoom in"), "Ctrl+= / Ctrl++"},
        {QT_TR_NOOP("Canvas"), QT_TR_NOOP("Zoom out"), "Ctrl+-"},
        {QT_TR_NOOP("Canvas"), QT_TR_NOOP("Actual size"), "Ctrl+0"},
        {QT_TR_NOOP("Mouse"), QT_TR_NOOP("Zoom"), QT_TR_NOOP("Wheel")},
        {QT_TR_NOOP("Mouse"), QT_TR_NOOP("Pan"), QT_TR_NOOP("Space+Drag / Middle Drag")},
        {QT_TR_NOOP("Mouse"), QT_TR_NOOP("Add to the selection"), QT_TR_NOOP("Ctrl+Click")},
        {QT_TR_NOOP("Mouse"), QT_TR_NOOP("Select with a box"), QT_TR_NOOP("Drag on empty space")},
        {QT_TR_NOOP("Mouse"), QT_TR_NOOP("Add a node here"), QT_TR_NOOP("Right Click")},
    };
    return list;
}

QWidget* shortcutsPage() {
    QVBoxLayout* col;
    QWidget* page = pageWidget(col);
    auto* head = new QHBoxLayout;
    head->addWidget(pageTitle(QObject::tr("Keyboard shortcuts")));
    head->addStretch(1);
    col->addLayout(head);
    auto* search = new QLineEdit(shortcutFilter);
    search->setPlaceholderText(QObject::tr("Search shortcuts"));
    search->setClearButtonEnabled(true);
    col->addWidget(search);
    // The list under the field, rebuilt as it is typed in (the field itself stays).
    auto* list = new QWidget;
    auto* lcol = new QVBoxLayout(list);
    lcol->setContentsMargins(0, 0, 0, 0);
    lcol->setSpacing(6);
    col->addWidget(list);
    auto fill = [lcol](const QString& filter) {
        while (QLayoutItem* item = lcol->takeAt(0)) {
            if (QWidget* w = item->widget()) w->deleteLater();
            delete item;
        }
        QString group;
        SettingsCard* card = nullptr;
        for (const Shortcut& s : shortcuts()) {
            const QString keys = QObject::tr(s.keys);
            const QString what = QObject::tr(s.what);
            if (!filter.isEmpty() && !what.contains(filter, Qt::CaseInsensitive) && !keys.contains(filter, Qt::CaseInsensitive)) continue;
            if (group != s.group) {
                group = s.group;
                lcol->addWidget(bestyle::sectionLabel(QObject::tr(s.group)));
                card = new SettingsCard;
                lcol->addWidget(card);
            }
            card->addRow(what, QString(), new KeyCaps(keys));
        }
        if (!card) lcol->addWidget(hintLabel(QObject::tr("No shortcut matches \"%1\".").arg(filter)));
    };
    fill(shortcutFilter);
    QObject::connect(search, &QLineEdit::textChanged, search, [fill](const QString& text) {
        shortcutFilter = text;
        fill(text);
    });
    return page;
}

QWidget* learnPage() {
    QVBoxLayout* col;
    QWidget* page = pageWidget(col);
    col->addWidget(pageTitle(QObject::tr("Learn")));
    {
        SettingsCard* c = addSection(col, QObject::tr("Your first model, from zero"));
        c->addRow(QObject::tr("1  ·  A Dataset"), QObject::tr("Drag it in from the library. Circles, Spirals, XOR or a sine wave to try ideas, or your CSV file"), nullptr);
        c->addRow(QObject::tr("2  ·  Something that learns"),
                  QObject::tr("A Parameter (2 × 16) and a Matrix Multiply from the features, then an Add with a bias Parameter (16): a layer "
                              "made by hand. Or a Dense node, the same in one"),
                  nullptr);
        c->addRow(QObject::tr("3  ·  A bend"), QObject::tr("An activation (Tanh, ReLU...) between layers, so the model can draw curves, not just lines"), nullptr);
        c->addRow(QObject::tr("4  ·  A Loss"), QObject::tr("The last layer's scores into its prediction, the dataset's targets into its target. Cross entropy for classes"), nullptr);
        c->addRow(QObject::tr("5  ·  Train"), QObject::tr("Press Train. The curve on the right shows the loss going down and the right answers going up"), nullptr);
    }
    {
        SettingsCard* c = addSection(col, QObject::tr("Shapes"));
        c->addRow(QObject::tr("B is the batch"), QObject::tr("How many examples go through at once. B × 2 is a batch of examples with 2 values each"), nullptr);
        c->addRow(QObject::tr("Broadcasting"), QObject::tr("Add, Multiply and Code nodes stretch a size of 1 (or a missing one) to fit the other side: B × 16 + 16 adds the bias to every example"), nullptr);
        c->addRow(QObject::tr("Red means it does not fit"), QObject::tr("The node says why at its foot; with nothing selected, Problems on the right lists every one"), nullptr);
    }
    {
        SettingsCard* c = addSection(col, QObject::tr("Your own nodes"));
        c->addRow(QObject::tr("Code nodes"), QObject::tr("Under Custom in the library: write a formula of a, b, c, d (sigmoid(a) * b...). Its derivative is worked out for you"), nullptr);
        auto* open = new QPushButton(QObject::tr("Custom Nodes..."));
        QObject::connect(open, &QPushButton::clicked, open, [] { openSettings(SettingsPage::CustomNodes); });
        c->addRow(QObject::tr("A library of your own"), QObject::tr("Save formulas you use in Settings › Custom Nodes: they show as tiles"), open);
    }
    {
        SettingsCard* c = addSection(col, QObject::tr("When training goes wrong"));
        c->addRow(QObject::tr("The loss jumps around or becomes NaN"), QObject::tr("Lower the learning rate (0.001 is a safe start with Adam)"), nullptr);
        c->addRow(QObject::tr("The loss stops going down"), QObject::tr("More units, another layer, or more epochs; Standardize the features"), nullptr);
        c->addRow(QObject::tr("Too slow"), QObject::tr("Settings › Performance: every processor thread, and the graphics card for big layers"), nullptr);
    }
    return page;
}

// The big mark of the About page.
class BigLogo : public QWidget {
public:
    BigLogo() { setFixedSize(64, 64); }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        drawLogo(p, {32, 32}, 30.0f, theme::kAccent);
    }
};

QWidget* aboutPage() {
    QVBoxLayout* col;
    QWidget* page = pageWidget(col);
    {
        auto* card = new SettingsCard;
        auto* row = new QHBoxLayout;
        row->setSpacing(18);
        row->addWidget(new BigLogo, 0, Qt::AlignVCenter);
        auto* names = new QVBoxLayout;
        names->setSpacing(2);
        auto* name = new QLabel("Logievo");
        QFont f = name->font();
        f.setPointSizeF(f.pointSizeF() * 1.6);
        f.setWeight(QFont::Bold);
        name->setFont(f);
        names->addWidget(name);
        auto* tagline = new QLabel(QObject::tr("AI models, from scratch"));
        bestyle::setTone(tagline, Tone::TextDim);
        names->addWidget(tagline);
        auto* version = new QLabel(QObject::tr("Version %1  ·  free and open source (MIT)").arg(LOGIEVO_VERSION));
        bestyle::setTone(version, Tone::TextFaint);
        names->addWidget(version);
        row->addLayout(names, 1);
        card->addLayout(row);
        col->addWidget(card);
    }
    {
        SettingsCard* c = addSection(col, QObject::tr("The idea"));
        c->addRow(QObject::tr("From zero, your way"), QObject::tr("Parameters and math up, or ready layers: the model is what you draw"), nullptr);
        c->addRow(QObject::tr("See as you build"), QObject::tr("Every node shows the shape it gives and what it holds; what does not fit says so"), nullptr);
        c->addRow(QObject::tr("Train right here"), QObject::tr("Press Train and watch it learn, on the processor's threads or the graphics card"), nullptr);
        c->addRow(QObject::tr("Plain C++ inside"), QObject::tr("Tensors, gradients and training written for Logievo: no machine-learning library underneath"), nullptr);
    }
    col->addSpacing(10);
    auto* buttons = new QHBoxLayout;
    auto* support = bestyle::accentButton(QObject::tr("Support Logievo"));
    support->setIcon(bestyle::makeIcon(Icon::Coffee, theme::kOnAccent));
    QObject::connect(support, &QPushButton::clicked, support, [] { platform::openUrl(kSupportUrl); });
    buttons->addWidget(support);
    auto* credits = new QPushButton(bestyle::makeIcon(Icon::Star, theme::kTextDim), QObject::tr("Credits"));
    QObject::connect(credits, &QPushButton::clicked, credits, [] { openHelp(HelpPage::Credits); });
    buttons->addWidget(credits);
    buttons->addStretch(1);
    col->addLayout(buttons);
    col->addSpacing(8);
    col->addWidget(hintLabel(QObject::tr("Written in C++20 with Qt and bestyle; the engine is plain C++, with OpenGL compute for the graphics card.")));
    return page;
}

QWidget* creditsPage() {
    QVBoxLayout* col;
    QWidget* page = pageWidget(col);
    col->addWidget(pageTitle(QObject::tr("Credits")));
    col->addWidget(paragraph(QObject::tr("Logievo stands on the work of others. Thank you.")));
    {
        SettingsCard* c = addSection(col, QObject::tr("Built with"));
        c->addRow("Qt 6", QObject::tr("The interface, windows and OpenGL  ·  The Qt Company and contributors  ·  LGPL v3"), nullptr);
        c->addRow("bestyle", QObject::tr("The look, shared with bemotion: theme, controls and icons"), nullptr);
        c->addRow("nlohmann/json", QObject::tr("Project files and settings  ·  Niels Lohmann  ·  MIT"), nullptr);
        c->addRow("Inter", QObject::tr("The interface font  ·  Rasmus Andersson  ·  SIL Open Font License 1.1"), nullptr);
    }
    {
        SettingsCard* c = addSection(col, QObject::tr("Ideas from"));
        c->addRow(QObject::tr("Adam"), QObject::tr("The optimizer  ·  Kingma and Ba, 2014"), nullptr);
        c->addRow(QObject::tr("Xavier initialization"), QObject::tr("Glorot and Bengio, 2010"), nullptr);
        c->addRow(QObject::tr("GELU"), QObject::tr("Hendrycks and Gimpel, 2016"), nullptr);
        c->addRow(QObject::tr("Layer normalization"), QObject::tr("Ba, Kiros and Hinton, 2016"), nullptr);
    }
    return page;
}

QWidget* licensePage() {
    QVBoxLayout* col;
    QWidget* page = pageWidget(col);
    col->addWidget(pageTitle(QObject::tr("License")));
    {
        SettingsCard* c = addSection(col, QObject::tr("In short"));
        c->addRow(QObject::tr("Free for anything"), QObject::tr("Personal and commercial work alike, with no price, account or watermark"), nullptr);
        c->addRow(QObject::tr("Yours to change and share"), QObject::tr("As long as the notice below travels with it"), nullptr);
        c->addRow(QObject::tr("Your models are yours"), QObject::tr("Projects, data and what they learn belong entirely to you"), nullptr);
    }
    SettingsCard* c = addSection(col, QObject::tr("MIT License"));
    auto* text = new QPlainTextEdit(QString::fromUtf8(kLicense));
    text->setReadOnly(true);
    text->setMinimumHeight(300);
    c->addContent(text);
    return page;
}

} // namespace

void openHelp(int page) {
    if (!helpWindow) {
        helpWindow = new PagedWindow(QObject::tr("Help"),
                                     {{QObject::tr("Shortcuts"), Icon::Keyboard, shortcutsPage},
                                      {QObject::tr("Learn"), Icon::Help, learnPage},
                                      {QObject::tr("About"), Icon::Logo, aboutPage},
                                      {QObject::tr("Credits"), Icon::Star, creditsPage},
                                      {QObject::tr("License"), Icon::Lock, licensePage}});
        helpWindow->setAttribute(Qt::WA_DeleteOnClose, true);
        helpWindow->resize(860, 650);
        QObject::connect(&bestyle::Appearance::get(), &bestyle::Appearance::changed, helpWindow.data(), [] {
            if (helpWindow) QTimer::singleShot(0, helpWindow.data(), [] { helpWindow->rebuild(); });
        });
        helpWindow->showPage(page >= 0 ? page : HelpPage::About);
    } else if (page >= 0) {
        helpWindow->showPage(page);
    }
    helpWindow->present();
}

void closeWindows() {
    if (helpWindow) helpWindow->close();
    extern void closeSettingsWindow();
    closeSettingsWindow();
}

} // namespace logievo::ui
