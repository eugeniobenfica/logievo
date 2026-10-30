#include "ui/Inspector.h"

#include "bestyle/BeStyle.h"
#include "core/Dataset.h"
#include "core/NodeTypes.h"
#include "ui/Document.h"
#include "ui/Icons.h"
#include "ui/Windows.h"
#include "app/Settings.h"
#include "app/Platform.h"

#include <QComboBox>
#include <QDir>
#include <QFontDatabase>
#include <QPainterPath>
#include <QPlainTextEdit>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPushButton>
#include <QScrollArea>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

namespace logievo::ui {

namespace theme = bestyle::theme;
namespace metrics = bestyle::metrics;
using bestyle::Icon;
using bestyle::Tone;

namespace {

QLabel* toned(const QString& text, Tone tone, bool wrap = false) {
    auto* l = new QLabel(text);
    bestyle::setTone(l, tone);
    l->setWordWrap(wrap);
    return l;
}

QLabel* heading(const QString& text) {
    auto* l = toned(text, Tone::Text);
    QFont f = l->font();
    f.setPointSizeF(f.pointSizeF() * 1.25);
    f.setWeight(QFont::DemiBold);
    l->setFont(f);
    return l;
}

// The node type's icon in a raised square, beside its name in the panel's header.
class IconBadge : public QWidget {
public:
    explicit IconBadge(Icon icon) : icon_(icon) { setFixedSize(36, 36); }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        bestyle::Draw d(p);
        d.rectFilled({0.5f, 0.5f}, {35.5f, 35.5f}, theme::kBgRaised, 8.0f);
        d.rect({0.5f, 0.5f}, {35.5f, 35.5f}, theme::kBorder, 8.0f);
        bestyle::drawIcon(d, icon_, {18, 18}, 18.0f, theme::kAccent);
    }

private:
    Icon icon_;
};

// The loss per epoch as a line (scaled to its own range), and the share of right answers on a
// 0 to 100% scale when the loss is a classification.
class LossChart : public QWidget {
public:
    LossChart() { setFixedHeight(130); }
    void setReport(TrainReport r) {
        report_ = std::move(r);
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QRectF frame = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
        p.setPen(QPen(theme::q(theme::kBorder), 1));
        p.setBrush(theme::q(theme::kBgDeep));
        p.drawRoundedRect(frame, 6, 6);
        const QRectF plot = frame.adjusted(10, 22, -10, -10);
        p.setPen(QPen(theme::q(theme::withAlpha(theme::kBorder, 140)), 1, Qt::DotLine));
        for (int i = 1; i < 4; ++i) p.drawLine(QPointF(plot.left(), plot.top() + plot.height() * i / 4), QPointF(plot.right(), plot.top() + plot.height() * i / 4));
        const auto& loss = report_.loss;
        const auto& acc = report_.accuracy;
        const int span = std::max(2, report_.epochs);
        auto line = [&](const std::vector<float>& values, float lo, float hi, QColor color, Qt::PenStyle style) {
            if (values.empty()) return;
            QPainterPath path;
            for (size_t i = 0; i < values.size(); ++i) {
                const qreal x = plot.left() + plot.width() * static_cast<qreal>(i) / (span - 1);
                const qreal y = plot.bottom() - plot.height() * std::clamp((values[i] - lo) / std::max(hi - lo, 1e-9f), 0.0f, 1.0f);
                if (i == 0) path.moveTo(x, y);
                else path.lineTo(x, y);
            }
            p.setPen(QPen(color, 1.6, style));
            p.setBrush(Qt::NoBrush);
            p.drawPath(path);
        };
        if (!loss.empty()) {
            const float hi = *std::max_element(loss.begin(), loss.end()), lo = std::min(0.0f, *std::min_element(loss.begin(), loss.end()));
            line(acc, 0.0f, 1.0f, theme::q(theme::kTextFaint), Qt::DashLine);
            line(loss, lo, hi, theme::q(theme::kAccent), Qt::SolidLine);
        }
        QFont small = font();
        small.setPointSizeF(small.pointSizeF() * 0.86);
        p.setFont(small);
        p.setPen(theme::q(theme::kText));
        p.drawText(QRectF(frame.left() + 10, frame.top() + 4, frame.width(), 16), Qt::AlignVCenter | Qt::AlignLeft, tr("Loss"));
        if (!acc.empty()) {
            p.setPen(theme::q(theme::kTextFaint));
            p.drawText(QRectF(frame.left(), frame.top() + 4, frame.width() - 10, 16), Qt::AlignVCenter | Qt::AlignRight, tr("- - right answers"));
        }
    }

private:
    TrainReport report_;
};

QFrame* rule() {
    auto* line = new QFrame;
    line->setFrameShape(QFrame::HLine);
    bestyle::setTone(line, Tone::Border);
    return line;
}

} // namespace

Inspector::Inspector(Document& doc, QWidget* parent) : QWidget(parent), doc_(doc) {
    auto* col = new QVBoxLayout(this);
    col->setContentsMargins(0, 0, 0, 0);
    scroll_ = new QScrollArea;
    scroll_->setWidgetResizable(true);
    scroll_->setFrameShape(QFrame::NoFrame);
    scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll_->viewport()->setAutoFillBackground(false);
    col->addWidget(scroll_);
    connect(&doc_, &Document::selectionChanged, this, &Inspector::scheduleRebuild);
    connect(&doc_, &Document::replaced, this, &Inspector::scheduleRebuild);
    connect(&doc_, &Document::changed, this, [this] {
        if (refresh_) refresh_();
    });
    connect(&doc_, &Document::trainingChanged, this, [this] {
        if (refreshTraining_) refreshTraining_();
    });
    rebuild();
}

void Inspector::scheduleRebuild() {
    // Later, never inside a signal of a field that is about to go.
    if (pending_) return;
    pending_ = true;
    QTimer::singleShot(0, this, [this] {
        pending_ = false;
        rebuild();
    });
}

void Inspector::rebuild() {
    refresh_ = nullptr;
    refreshTraining_ = nullptr;
    const auto& sel = doc_.selection();
    QWidget* content = sel.size() == 1 && doc_.graph.find(sel[0]) ? buildNode(sel[0]) : sel.size() > 1 ? buildMany() : buildModel();
    content->setAutoFillBackground(false);
    scroll_->setWidget(content);
    if (refresh_) refresh_();
}

QWidget* Inspector::buildNode(int id) {
    const Node& node = *doc_.graph.find(id);
    const NodeType* type = findNodeType(node.type);
    auto* w = new QWidget;
    auto* col = new QVBoxLayout(w);
    col->setContentsMargins(metrics::kPanelMargin + 6, metrics::kPanelMargin + 8, metrics::kPanelMargin + 6, metrics::kPanelMargin);
    col->setSpacing(metrics::kGap + 2);

    auto* head = new QHBoxLayout;
    head->setSpacing(10);
    head->addWidget(new IconBadge(nodeIcon(node.type)));
    auto* names = new QVBoxLayout;
    names->setSpacing(0);
    names->addWidget(heading(type ? tr(type->name.c_str()) : QString::fromStdString(node.type)));
    names->addWidget(toned(type ? tr(type->category.c_str()) : tr("Unknown"), Tone::TextFaint));
    head->addLayout(names, 1);
    col->addLayout(head);
    if (type) col->addWidget(toned(tr(type->description.c_str()), Tone::TextDim, true));
    col->addSpacing(metrics::kGap);
    col->addWidget(rule());

    col->addWidget(bestyle::sectionLabel(tr("Node")));
    auto* name = new QLineEdit(QString::fromStdString(node.name));
    connect(name, &QLineEdit::editingFinished, this, [this, id, name] {
        Node* n = doc_.graph.find(id);
        const std::string text = name->text().trimmed().toStdString();
        if (!n || text.empty() || text == n->name) return;
        n->name = text;
        doc_.edited();
    });
    col->addWidget(new bestyle::LabeledRow(tr("Name"), name, 120));

    if (type && !type->params.empty()) {
        col->addSpacing(metrics::kGap);
        col->addWidget(bestyle::sectionLabel(tr("Settings")));
        for (const ParamSpec& spec : type->params) {
            const std::string pid = spec.id;
            if (node.type == "dataset") {
                const double source = doc_.graph.number(node, "source");
                const bool csv = source == 4.0, text = source == kTextCommandsSource, textFile = source == kTextFileSource;
                const bool csvOnly = pid == "column" || pid == "targets", builtInOnly = pid == "samples" || pid == "noise";
                if ((csvOnly && !csv) || (pid == "file" && !csv && !textFile) || (builtInOnly && (csv || textFile)) ||
                    (pid == "commands" && !text) || (pid == "scale" && (text || textFile)))
                    continue;
            }
            QWidget* control = nullptr;
            const bool ports = node.type == "custom" && pid == "inputs";
            auto set = [this, id, pid, ports](Json value, bool record) {
                if (Node* n = doc_.graph.find(id)) {
                    n->params[pid] = std::move(value);
                    if (ports) {
                        const int count = static_cast<int>(inputPorts(*n).size());
                        std::erase_if(doc_.graph.links, [&](const Link& l) { return l.toNode == id && l.toPort >= count; });
                    }
                    doc_.edited(record);
                }
            };
            switch (spec.kind) {
            case ParamKind::Int:
            case ParamKind::Float: {
                auto* v = new bestyle::DragValue;
                const bool integer = spec.kind == ParamKind::Int;
                v->setRange(static_cast<float>(spec.min), static_cast<float>(spec.max));
                v->setFormat(integer ? "%.0f" : "%.3f");
                v->setSpeed(integer ? (spec.max - spec.min > 1000 ? 4.0f : 0.25f) : 0.005f);
                v->setValue(static_cast<float>(doc_.graph.number(node, pid)));
                connect(v, &bestyle::DragValue::valueChanged, this, [set, integer](float value) {
                    set(integer ? Json(static_cast<int64_t>(std::lround(value))) : Json(std::round(value * 1000.0) / 1000.0), false);
                });
                connect(v, &bestyle::DragValue::editFinished, this, [this, ports] {
                    doc_.edited();
                    if (ports) scheduleRebuild();
                });
                control = v;
                break;
            }
            case ParamKind::Bool: {
                auto* s = new bestyle::Switch;
                s->setChecked(doc_.graph.number(node, pid) != 0.0);
                connect(s, &QAbstractButton::toggled, this, [set](bool on) { set(on, true); });
                auto* holder = new QWidget;
                auto* r = new QHBoxLayout(holder);
                r->setContentsMargins(0, 0, 0, 0);
                r->addWidget(s);
                r->addStretch(1);
                control = holder;
                break;
            }
            case ParamKind::Choice: {
                auto* c = new QComboBox;
                for (const std::string& choice : spec.choices) c->addItem(tr(choice.c_str()));
                c->setCurrentIndex(static_cast<int>(doc_.graph.number(node, pid)));
                const bool reshapes = node.type == "dataset" && pid == "source";
                connect(c, &QComboBox::currentIndexChanged, this, [this, set, reshapes](int i) {
                    set(i, true);
                    if (reshapes) scheduleRebuild();
                });
                control = c;
                break;
            }
            case ParamKind::Code: {
                // Monospaced, saved a moment after typing stops; the foot of the panel says if it reads.
                auto* holder = new QWidget;
                auto* c = new QVBoxLayout(holder);
                c->setContentsMargins(0, 0, 0, 0);
                c->setSpacing(metrics::kGap);
                auto* edit = new QPlainTextEdit(QString::fromStdString(doc_.graph.text(node, pid)));
                edit->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
                edit->setMinimumHeight(120);
                edit->setTabChangesFocus(true);
                auto* later = new QTimer(edit);
                later->setSingleShot(true);
                later->setInterval(350);
                connect(edit, &QPlainTextEdit::textChanged, later, qOverload<>(&QTimer::start));
                connect(later, &QTimer::timeout, this, [this, set, edit, id, pid] {
                    const Node* n = doc_.graph.find(id);
                    const std::string text = edit->toPlainText().toStdString();
                    if (n && text != doc_.graph.text(*n, pid)) set(text, true);
                });
                c->addWidget(edit);
                auto* keep = new QPushButton(bestyle::makeIcon(Icon::Plus, theme::kTextDim), tr("Save to Library"));
                keep->setToolTip(tr("Keep this code as a node of your own: it shows under Custom in every project"));
                connect(keep, &QPushButton::clicked, this, [this, id] {
                    const Node* n = doc_.graph.find(id);
                    if (!n) return;
                    settings().customNodes.push_back({n->name, doc_.graph.text(*n, "code"), static_cast<int>(inputPorts(*n).size()), {}});
                    saveSettings();
                    emit AppSignals::get().customNodesChanged();
                    openSettings(SettingsPage::CustomNodes);
                });
                c->addWidget(keep, 0, Qt::AlignLeft);
                col->addWidget(toned(tr(spec.label.c_str()), Tone::Text));
                if (!spec.tip.empty()) edit->setToolTip(tr(spec.tip.c_str()));
                col->addWidget(holder);
                continue;
            }
            case ParamKind::Text: {
                // Lines of text (a dataset's commands), saved a moment after typing stops.
                auto* edit = new QPlainTextEdit(QString::fromStdString(doc_.graph.text(node, pid)));
                edit->setMinimumHeight(140);
                edit->setTabChangesFocus(true);
                edit->setLineWrapMode(QPlainTextEdit::NoWrap);
                auto* later = new QTimer(edit);
                later->setSingleShot(true);
                later->setInterval(500);
                connect(edit, &QPlainTextEdit::textChanged, later, qOverload<>(&QTimer::start));
                connect(later, &QTimer::timeout, this, [this, set, edit, id, pid] {
                    const Node* n = doc_.graph.find(id);
                    const std::string text = edit->toPlainText().toStdString();
                    if (n && text != doc_.graph.text(*n, pid)) set(text, true);
                });
                col->addWidget(toned(tr(spec.label.c_str()), Tone::Text));
                if (!spec.tip.empty()) edit->setToolTip(tr(spec.tip.c_str()));
                col->addWidget(edit);
                continue;
            }
            case ParamKind::File: {
                auto* holder = new QWidget;
                auto* r = new QHBoxLayout(holder);
                r->setContentsMargins(0, 0, 0, 0);
                r->setSpacing(metrics::kGap);
                auto* e = new QLineEdit(QString::fromStdString(doc_.graph.text(node, pid)));
                e->setPlaceholderText(tr("No file"));
                auto* browse = new QPushButton(tr("Browse..."));
                r->addWidget(e, 1);
                r->addWidget(browse);
                connect(e, &QLineEdit::editingFinished, this, [this, set, e, id, pid] {
                    const Node* n = doc_.graph.find(id);
                    const std::string text = e->text().trimmed().toStdString();
                    if (n && text != doc_.graph.text(*n, pid)) set(text, true);
                });
                const bool textFile = doc_.graph.number(node, "source") == kTextFileSource;
                connect(browse, &QPushButton::clicked, this, [this, set, e, textFile] {
                    const QString title = textFile ? tr("Choose a Text File") : tr("Choose a CSV File");
                    const QString filter = textFile ? tr("Text examples (*.tsv *.txt);;All files (*)") : tr("CSV files (*.csv *.txt);;All files (*)");
                    if (auto f = platform::openFileDialog(this, title, filter)) {
                        e->setText(QDir::fromNativeSeparators(platform::qpath(*f)));
                        set(e->text().toStdString(), true);
                    }
                });
                control = holder;
                break;
            }
            case ParamKind::Shape: {
                auto* e = new QLineEdit(QString::fromStdString(doc_.graph.text(node, pid)));
                e->setPlaceholderText("3x224x224");
                connect(e, &QLineEdit::editingFinished, this, [this, set, e, id, pid] {
                    const Node* n = doc_.graph.find(id);
                    const std::string text = e->text().trimmed().toStdString();
                    if (n && text != doc_.graph.text(*n, pid)) set(text, true);
                });
                control = e;
                break;
            }
            }
            if (!spec.tip.empty()) control->setToolTip(tr(spec.tip.c_str()));
            col->addWidget(new bestyle::LabeledRow(tr(spec.label.c_str()), control, 120));
        }
    }

    // What comes in and out, worked out from the graph.
    col->addSpacing(metrics::kGap);
    col->addWidget(bestyle::sectionLabel(tr("Result")));
    std::vector<QLabel*> inputs;
    if (type)
        for (const std::string& port : inputPorts(node)) {
            auto* l = toned({}, Tone::TextDim);
            inputs.push_back(l);
            col->addWidget(new bestyle::LabeledRow(inputPorts(node).size() > 1 ? tr("Input %1").arg(QString::fromStdString(port)) : tr("Input"), l, 120));
        }
    std::vector<QLabel*> outputs;
    if (type)
        for (const std::string& port : type->outputs) {
            auto* l = toned({}, Tone::Text);
            outputs.push_back(l);
            QString label = tr("Output");
            if (type->outputs.size() > 1) {
                label = QString::fromStdString(port);
                label[0] = label[0].toUpper();
            }
            col->addWidget(new bestyle::LabeledRow(label, l, 120));
        }
    auto* params = toned({}, Tone::Text);
    col->addWidget(new bestyle::LabeledRow(node.type == "dataset" ? tr("Rows") : tr("Parameters"), params, 120));
    auto* status = toned({}, Tone::TextFaint, true);
    col->addWidget(status);
    refresh_ = [this, id, inputs, outputs, params, status] {
        const NodeInfo* info = doc_.analysis().info(id);
        const Node* node = doc_.graph.find(id);
        if (!info || !node) return;
        for (size_t i = 0; i < inputs.size(); ++i)
            inputs[i]->setText(i < info->inputs.size() && info->state != NodeInfo::State::Waiting ? QString::fromStdString(shapeText(info->inputs[i])) : QString("—"));
        const bool ok = info->state == NodeInfo::State::Ok;
        for (size_t i = 0; i < outputs.size(); ++i)
            outputs[i]->setText(ok && i < info->outputs.size() ? QString::fromStdString(shapeText(info->outputs[i])) : QString("—"));
        if (node->type == "dataset" && ok) {
            const auto data = loadDataset(doc_.graph, *node);
            QString rows = QString("%L1").arg(data->rows);
            if (data->classes) rows += tr("  ·  %1 classes").arg(data->classCount);
            params->setText(rows);
            status->hide();
            return;
        }
        params->setText(ok ? QString::fromStdString(countText(info->params)) + (info->params >= 10'000 ? QString("  (%L1)").arg(info->params) : QString())
                           : QString("—"));
        status->setVisible(!ok);
        status->setText(QString::fromStdString(info->message));
        bestyle::setTone(status, info->state == NodeInfo::State::Error ? Tone::Danger : Tone::TextFaint);
    };

    col->addSpacing(metrics::kGroupGap);
    auto* buttons = new QHBoxLayout;
    auto* dup = new QPushButton(bestyle::makeIcon(Icon::Duplicate, theme::kTextDim), tr("Duplicate"));
    connect(dup, &QPushButton::clicked, this, [this] {
        if (onDuplicate) onDuplicate();
    });
    auto* del = new QPushButton(bestyle::makeIcon(Icon::Trash, theme::kDanger), tr("Delete"));
    connect(del, &QPushButton::clicked, this, [this] {
        if (onDelete) onDelete();
    });
    buttons->addWidget(dup);
    buttons->addWidget(del);
    col->addLayout(buttons);
    col->addStretch(1);
    return w;
}

QWidget* Inspector::buildMany() {
    auto* w = new QWidget;
    auto* col = new QVBoxLayout(w);
    col->setContentsMargins(metrics::kPanelMargin + 6, metrics::kPanelMargin + 8, metrics::kPanelMargin + 6, metrics::kPanelMargin);
    col->setSpacing(metrics::kGap + 2);
    col->addWidget(heading(tr("%1 nodes selected").arg(doc_.selection().size())));
    auto* params = toned({}, Tone::TextDim);
    col->addWidget(params);
    refresh_ = [this, params] {
        int64_t total = 0;
        for (int id : doc_.selection())
            if (const NodeInfo* info = doc_.analysis().info(id)) total += info->params;
        params->setText(tr("%1 parameters together").arg(QString::fromStdString(countText(total))));
    };
    col->addSpacing(metrics::kGap);
    auto* buttons = new QHBoxLayout;
    auto* dup = new QPushButton(bestyle::makeIcon(Icon::Duplicate, theme::kTextDim), tr("Duplicate"));
    connect(dup, &QPushButton::clicked, this, [this] {
        if (onDuplicate) onDuplicate();
    });
    auto* del = new QPushButton(bestyle::makeIcon(Icon::Trash, theme::kDanger), tr("Delete"));
    connect(del, &QPushButton::clicked, this, [this] {
        if (onDelete) onDelete();
    });
    buttons->addWidget(dup);
    buttons->addWidget(del);
    col->addLayout(buttons);
    col->addStretch(1);
    return w;
}

QWidget* Inspector::buildModel() {
    auto* w = new QWidget;
    auto* col = new QVBoxLayout(w);
    col->setContentsMargins(metrics::kPanelMargin + 6, metrics::kPanelMargin + 8, metrics::kPanelMargin + 6, metrics::kPanelMargin);
    col->setSpacing(metrics::kGap + 2);
    auto* head = new QHBoxLayout;
    head->setSpacing(10);
    head->addWidget(new IconBadge(Icon::Cube));
    auto* names = new QVBoxLayout;
    names->setSpacing(0);
    names->addWidget(heading(tr("Model")));
    names->addWidget(toned(tr("Select a node to edit its settings"), Tone::TextFaint));
    head->addLayout(names, 1);
    col->addLayout(head);
    col->addSpacing(metrics::kGap);
    col->addWidget(rule());

    col->addWidget(bestyle::sectionLabel(tr("Project")));
    auto* name = new QLineEdit(QString::fromStdString(doc_.graph.name));
    connect(name, &QLineEdit::editingFinished, this, [this, name] {
        const std::string text = name->text().trimmed().toStdString();
        if (text.empty() || text == doc_.graph.name) return;
        doc_.graph.name = text;
        doc_.edited();
    });
    col->addWidget(new bestyle::LabeledRow(tr("Name"), name, 120));

    col->addSpacing(metrics::kGap);
    col->addWidget(bestyle::sectionLabel(tr("Summary")));
    auto* nodes = toned({}, Tone::Text);
    auto* links = toned({}, Tone::Text);
    auto* params = toned({}, Tone::Text);
    auto* state = toned({}, Tone::Text);
    col->addWidget(new bestyle::LabeledRow(tr("Nodes"), nodes, 120));
    col->addWidget(new bestyle::LabeledRow(tr("Links"), links, 120));
    col->addWidget(new bestyle::LabeledRow(tr("Parameters"), params, 120));
    col->addWidget(new bestyle::LabeledRow(tr("Status"), state, 120));

    col->addSpacing(metrics::kGap);
    col->addWidget(bestyle::sectionLabel(tr("Training")));
    TrainSettings& ts = doc_.graph.training;
    auto* optimizer = new QComboBox;
    optimizer->addItems({tr("SGD"), tr("Adam")});
    optimizer->setCurrentIndex(ts.optimizer);
    optimizer->setToolTip(tr("SGD: plain steps down the gradient. Adam: steps that adapt to each value (usually faster)"));
    connect(optimizer, &QComboBox::currentIndexChanged, this, [this](int i) {
        doc_.graph.training.optimizer = i;
        doc_.edited();
    });
    col->addWidget(new bestyle::LabeledRow(tr("Optimizer"), optimizer, 120));
    auto number = [this, col](const QString& label, double value, float min, float max, const char* format, float speed, const QString& tip,
                              std::function<void(double)> apply) {
        auto* v = new bestyle::DragValue;
        v->setRange(min, max);
        v->setFormat(format);
        v->setSpeed(speed);
        v->setValue(static_cast<float>(value));
        v->setToolTip(tip);
        connect(v, &bestyle::DragValue::valueChanged, this, [apply](float x) { apply(x); });
        connect(v, &bestyle::DragValue::editFinished, this, [this] { doc_.edited(); });
        col->addWidget(new bestyle::LabeledRow(label, v, 120));
    };
    number(tr("Learning rate"), ts.learningRate, 0.00001f, 10.0f, "%.4f", 0.0005f, tr("How big each step is: too big and the loss jumps around"),
           [this](double x) { doc_.graph.training.learningRate = x; });
    number(tr("Epochs"), ts.epochs, 1.0f, 1000000.0f, "%.0f", 1.0f, tr("How many times training goes through every example"),
           [this](double x) { doc_.graph.training.epochs = static_cast<int>(std::lround(x)); });
    number(tr("Batch size"), ts.batchSize, 1.0f, 65536.0f, "%.0f", 0.5f, tr("How many examples each step looks at"),
           [this](double x) { doc_.graph.training.batchSize = static_cast<int>(std::lround(x)); });
    auto* buttons = new QHBoxLayout;
    auto* train = bestyle::accentButton(tr("Train"));
    train->setMinimumHeight(30);
    connect(train, &QPushButton::clicked, this, [this] {
        if (doc_.training()) doc_.stopTraining();
        else doc_.startTraining();
    });
    auto* forget = new QPushButton(bestyle::makeIcon(Icon::Reset, theme::kTextDim), tr("Start Over"));
    forget->setMinimumHeight(30);
    forget->setToolTip(tr("Forget what the model learned: the next training starts from fresh weights"));
    connect(forget, &QPushButton::clicked, this, [this] { doc_.forgetWeights(); });
    buttons->addWidget(train, 1);
    buttons->addWidget(forget);
    col->addLayout(buttons);
    auto* progress = toned({}, Tone::TextDim, true);
    col->addWidget(progress);
    auto* chart = new LossChart;
    col->addWidget(chart);
    refreshTraining_ = [this, train, forget, progress, chart] {
        const TrainReport r = doc_.trainingReport();
        const bool running = doc_.training();
        train->setText(running ? tr("Stop") : r.loss.empty() || doc_.weights.empty() ? tr("Train") : tr("Train More"));
        train->setEnabled(running || doc_.analysis().valid());
        forget->setEnabled(!running && !doc_.weights.empty());
        QString text;
        if (!r.error.empty()) text = QString::fromStdString(r.error);
        else if (!r.loss.empty()) {
            text = tr("Epoch %1 of %2  ·  loss %3").arg(r.epoch).arg(r.epochs).arg(QString::number(r.loss.back(), 'g', 4));
            if (!r.accuracy.empty()) text += tr("  ·  %1% right").arg(QString::number(r.accuracy.back() * 100, 'f', 1));
        } else if (running) text = tr("Starting...");
        else if (!doc_.weights.empty()) text = tr("Has learned weights (saved with the project). Train more, or start over.");
        else text = doc_.analysis().valid() ? tr("Ready: training makes the Loss smaller.") : tr("Needs a Loss node and every shape fitting.");
        progress->setText(text);
        bestyle::setTone(progress, r.error.empty() ? Tone::TextDim : Tone::Danger);
        chart->setReport(r);
        chart->setVisible(!r.loss.empty());
    };

    col->addSpacing(metrics::kGap);
    col->addWidget(bestyle::sectionLabel(tr("Problems")));
    auto* problems = new QWidget;
    auto* list = new QVBoxLayout(problems);
    list->setContentsMargins(0, 0, 0, 0);
    list->setSpacing(2);
    col->addWidget(problems);
    col->addStretch(1);

    refresh_ = [this, nodes, links, params, state, list] {
        if (refreshTraining_) refreshTraining_();
        const Analysis& a = doc_.analysis();
        const Graph& g = doc_.graph;
        nodes->setText(QString::number(g.nodes.size()));
        links->setText(QString::number(g.links.size()));
        params->setText(QString::fromStdString(countText(a.totalParams)) + (a.totalParams >= 10'000 ? QString("  (%L1)").arg(a.totalParams) : QString()));
        state->setText(a.valid() ? tr("Ready to train") : a.errors ? tr("Has problems") : a.hasLoss ? tr("Incomplete") : tr("Needs a Loss"));
        bestyle::setTone(state, a.valid() ? Tone::Text : a.errors ? Tone::Danger : Tone::TextDim);
        while (QLayoutItem* item = list->takeAt(0)) {
            if (QWidget* x = item->widget()) x->deleteLater();
            delete item;
        }
        // Each problem a row: click it to select the node.
        int count = 0;
        auto add = [&](const QString& text, Tone tone, int nodeId) {
            auto* b = new QPushButton(text);
            b->setFlat(true);
            b->setStyleSheet("QPushButton { text-align: left; padding: 4px 6px; border: none; }");
            b->setCursor(nodeId ? Qt::PointingHandCursor : Qt::ArrowCursor);
            bestyle::setTone(b, tone);
            if (nodeId) connect(b, &QPushButton::clicked, this, [this, nodeId] { doc_.select({nodeId}); });
            list->addWidget(b);
            ++count;
        };
        if (!a.hasData) add(tr("Add a Dataset: the examples to learn from"), Tone::TextDim, 0);
        if (!a.hasLoss) add(tr("Add a Loss: what training makes smaller"), Tone::TextDim, 0);
        for (const Node& n : g.nodes)
            if (const NodeInfo* info = a.info(n.id); info && info->state == NodeInfo::State::Error && count < 30)
                add(QString::fromStdString(n.name + ": " + info->message), Tone::Danger, n.id);
        for (const Node& n : g.nodes)
            if (const NodeInfo* info = a.info(n.id); info && info->state == NodeInfo::State::Waiting && info->message.rfind("Connect", 0) == 0 && count < 30)
                add(QString::fromStdString(n.name + ": " + info->message), Tone::TextDim, n.id);
        if (count == 0) {
            auto* none = toned(tr("None. Every shape fits."), Tone::TextFaint);
            list->addWidget(none);
        }
    };
    return w;
}

} // namespace logievo::ui
