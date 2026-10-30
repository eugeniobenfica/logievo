#include "ui/NodeCanvas.h"

#include "bestyle/BeStyle.h"
#include "core/Dataset.h"
#include "core/NodeTypes.h"
#include "ui/Document.h"
#include "ui/GraphPreview.h"
#include "ui/Icons.h"
#include "ui/Logo.h"
#include "ui/Windows.h"
#include "app/Settings.h"

#include <QDragEnterEvent>
#include <QFileInfo>
#include <QRegularExpression>
#include <QFontMetrics>
#include <QKeyEvent>
#include <QMenu>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace logievo::ui {

namespace theme = bestyle::theme;
using bestyle::Rgba;

namespace {

constexpr qreal kMinZoom = 0.2, kMaxZoom = 2.5;
constexpr float kGrid = 24.0f;
constexpr float kPortRadius = 5.0f;
constexpr float kPortHit = 10.0f; // view pixels around a port that grab it

QString fmt(double v) { return QString::number(v, 'g', 6); }

QString choice(const Node& n, const Graph& g, const char* param) {
    const NodeType* type = findNodeType(n.type);
    const ParamSpec* spec = type ? findParam(*type, param) : nullptr;
    if (!spec || spec->choices.empty()) return {};
    const int i = std::clamp(static_cast<int>(g.number(n, param)), 0, static_cast<int>(spec->choices.size()) - 1);
    return QObject::tr(spec->choices[static_cast<size_t>(i)].c_str());
}

// The node's main settings in a few words, on its first row.
QString summary(const Graph& g, const Node& n) {
    auto num = [&](const char* id) { return static_cast<long long>(g.number(n, id)); };
    auto shape = [&] { return QString::fromStdString(g.text(n, "shape")).replace("x", " × "); };
    const std::string& t = n.type;
    if (t == "dataset") {
        if (num("source") == 4 || num("source") == kTextFileSource) {
            const QString file = QString::fromStdString(g.text(n, "file"));
            return file.isEmpty() ? choice(n, g, "source") : QFileInfo(file).fileName();
        }
        if (num("source") == kTextCommandsSource) return QObject::tr("%1 commands").arg(parseCommands(g.text(n, "commands")).size());
        return choice(n, g, "source");
    }
    if (t == "parameter") return shape() + "  ·  " + choice(n, g, "init");
    if (t == "constant") return shape() + "  =  " + fmt(g.number(n, "value"));
    if (t == "reshape") return shape();
    if (t == "sum" || t == "mean") return g.number(n, "all") != 0.0 ? QObject::tr("all") : QObject::tr("axis %1").arg(num("axis"));
    if (t == "power") return QObject::tr("^ %1").arg(fmt(g.number(n, "exponent")));
    if (t == "leakyrelu") return QObject::tr("slope %1").arg(fmt(g.number(n, "slope")));
    if (t == "dense") return QObject::tr("%1 units").arg(num("units"));
    if (t == "rnn") return QObject::tr("%1 units").arg(num("units")) + (g.number(n, "all") != 0.0 ? QObject::tr(" · every step") : QString());
    if (t == "conv2d") {
        QString s = QObject::tr("%1 filters · %2×%2").arg(num("filters")).arg(num("kernel"));
        if (num("stride") > 1) s += QObject::tr(" / %1").arg(num("stride"));
        return s;
    }
    if (t == "maxpool2d") return QString("%1×%1 / %2").arg(num("kernel")).arg(num("stride"));
    if (t == "embedding") return QObject::tr("%1 ids → %2").arg(QString::fromStdString(countText(num("vocab")))).arg(num("dim"));
    if (t == "dropout") return QObject::tr("rate %1").arg(fmt(g.number(n, "rate")));
    if (t == "concat") return QObject::tr("axis %1").arg(num("axis"));
    if (t == "custom") { // its result: the last line that is not a comment
        QString last;
        for (QString line : QString::fromStdString(g.text(n, "code")).split(QRegularExpression("[\n;]"))) {
            line = line.section('#', 0, 0).section("//", 0, 0).trimmed();
            if (!line.isEmpty()) last = line;
        }
        return last;
    }
    if (t == "loss") return choice(n, g, "kind") + (g.number(n, "weight") != 1.0 ? QObject::tr(" × %1").arg(fmt(g.number(n, "weight"))) : QString());
    return {};
}

} // namespace

NodeCanvas::NodeCanvas(Document& doc, QWidget* parent) : QWidget(parent), doc_(doc) {
    setMouseTracking(true);
    setAcceptDrops(true);
    setFocusPolicy(Qt::StrongFocus);
    setMinimumSize(200, 200);
    connect(&doc_, &Document::changed, this, qOverload<>(&QWidget::update));
    connect(&doc_, &Document::selectionChanged, this, qOverload<>(&QWidget::update));
    connect(&doc_, &Document::trainingChanged, this, qOverload<>(&QWidget::update));
    connect(&AppSignals::get(), &AppSignals::interfaceChanged, this, qOverload<>(&QWidget::update));
    connect(&bestyle::Appearance::get(), &bestyle::Appearance::changed, this, qOverload<>(&QWidget::update));
}

// --- editing ---------------------------------------------------------------------------------

int NodeCanvas::addNode(const std::string& type, std::optional<QPointF> scene, const std::string& name, const Json& params) {
    QPointF at;
    if (scene) at = *scene;
    else {
        // The middle of the view, moved along while a node is already there.
        at = toScene(QPointF(width() * 0.5, height() * 0.4)) - QPointF(kNodeWidth * 0.5, 40);
        auto taken = [&](QPointF p) {
            return std::any_of(doc_.graph.nodes.begin(), doc_.graph.nodes.end(),
                               [&](const Node& n) { return std::abs(n.x - p.x()) < 12 && std::abs(n.y - p.y()) < 12; });
        };
        while (taken(at)) at += QPointF(28, 28);
    }
    Node& n = doc_.graph.add(type, static_cast<float>(std::round(at.x() / 4) * 4), static_cast<float>(std::round(at.y() / 4) * 4));
    if (params.is_object())
        for (const auto& [k, v] : params.items()) n.params[k] = v;
    if (!name.empty()) n.name = doc_.graph.uniqueName(name);
    const int id = n.id;
    doc_.edited();
    doc_.select({id});
    return id;
}

void NodeCanvas::deleteSelection() {
    if (doc_.selection().empty()) return;
    for (int id : doc_.selection()) doc_.graph.remove(id);
    doc_.select({});
    doc_.edited();
}

void NodeCanvas::duplicateSelection() {
    if (doc_.selection().empty()) return;
    // The copies keep the links among themselves, and the links coming in from outside.
    std::map<int, int> copyOf;
    for (int id : doc_.selection()) {
        const Node original = *doc_.graph.find(id);
        Node& n = doc_.graph.add(original.type, original.x + 32, original.y + 32);
        n.params = original.params;
        copyOf[id] = n.id;
    }
    const std::vector<Link> links = doc_.graph.links;
    for (const Link& l : links) {
        if (!copyOf.count(l.toNode)) continue;
        const int from = copyOf.count(l.fromNode) ? copyOf[l.fromNode] : l.fromNode;
        doc_.graph.connect(from, l.fromPort, copyOf[l.toNode], l.toPort);
    }
    std::vector<int> ids;
    for (const auto& [from, to] : copyOf) ids.push_back(to);
    doc_.edited();
    doc_.select(ids);
}

void NodeCanvas::selectAll() {
    std::vector<int> ids;
    for (const Node& n : doc_.graph.nodes) ids.push_back(n.id);
    doc_.select(ids);
}

// --- view ------------------------------------------------------------------------------------

void NodeCanvas::setZoom(qreal zoom, QPointF anchorView) {
    zoom = std::clamp(zoom, kMinZoom, kMaxZoom);
    const QPointF anchor = toScene(anchorView);
    zoom_ = zoom;
    pan_ = anchorView - anchor * zoom_;
    update();
}

void NodeCanvas::zoomBy(qreal factor) { setZoom(zoom_ * factor, QPointF(width() * 0.5, height() * 0.5)); }

void NodeCanvas::frameAll() {
    const QRectF b = graphBounds(doc_.graph);
    if (b.isNull() || width() < 50) {
        zoom_ = 1.0;
        pan_ = QPointF(width() * 0.5, height() * 0.5);
    } else {
        const qreal margin = 60;
        zoom_ = std::clamp(std::min((width() - margin * 2) / b.width(), (height() - margin * 2) / b.height()), kMinZoom, 1.0);
        pan_ = QPointF(width() * 0.5, height() * 0.5) - b.center() * zoom_;
    }
    update();
}

void NodeCanvas::resizeEvent(QResizeEvent* e) {
    QWidget::resizeEvent(e);
    if (!framed_ && width() > 100 && height() > 100) {
        framed_ = true;
        frameAll();
    }
}

// --- hit testing -----------------------------------------------------------------------------

NodeCanvas::Hit NodeCanvas::hitTest(QPointF view) const {
    const QPointF s = toScene(view);
    const qreal reach = kPortHit / zoom_;
    // Ports first (they stick out of the cards), then the cards, the top one first.
    for (auto it = doc_.graph.nodes.rbegin(); it != doc_.graph.nodes.rend(); ++it) {
        const NodeType* t = findNodeType(it->type);
        if (!t) continue;
        for (int i = 0; i < static_cast<int>(inputPorts(*it).size()); ++i)
            if (QLineF(s, inputPort(*it, i)).length() <= reach) return {Hit::Kind::Input, it->id, i};
        for (int i = 0; i < static_cast<int>(t->outputs.size()); ++i)
            if (QLineF(s, outputPort(*it, i)).length() <= reach) return {Hit::Kind::Output, it->id, i};
    }
    for (auto it = doc_.graph.nodes.rbegin(); it != doc_.graph.nodes.rend(); ++it)
        if (nodeRect(*it).contains(s)) return {Hit::Kind::Node, it->id, 0};
    return {};
}

// --- painting --------------------------------------------------------------------------------

void NodeCanvas::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.fillRect(rect(), theme::q(theme::kBgDeep));

    // A dot grid, fading out when zoomed far out.
    const qreal step = kGrid * zoom_;
    if (step >= 9 && settings().showGrid) {
        const QColor dot = theme::q(theme::withAlpha(theme::kTextFaint, step >= 16 ? 90 : 50));
        const qreal x0 = std::fmod(pan_.x(), step), y0 = std::fmod(pan_.y(), step);
        const qreal r = std::max(0.8, 1.1 * zoom_);
        p.setPen(Qt::NoPen);
        p.setBrush(dot);
        for (qreal y = y0; y < height(); y += step)
            for (qreal x = x0; x < width(); x += step) p.drawEllipse(QPointF(x, y), r, r);
    }

    const Graph& g = doc_.graph;
    if (g.nodes.empty()) {
        drawLogo(p, {width() * 0.5f, height() * 0.5f - 34}, 26.0f, theme::withAlpha(theme::kTextFaint, 120));
        p.setPen(theme::q(theme::kTextDim));
        p.drawText(QRectF(0, height() * 0.5 + 6, width(), 24), Qt::AlignCenter, tr("Start from nothing: a Dataset, then Parameters and math, or layers"));
        p.setPen(theme::q(theme::kTextFaint));
        p.drawText(QRectF(0, height() * 0.5 + 28, width(), 24), Qt::AlignCenter,
                   tr("click or drag nodes in from the left, or right-click the canvas · a Loss tells training what to learn"));
    }

    p.save();
    p.translate(pan_);
    p.scale(zoom_, zoom_);
    // Links under the nodes; brighter when they touch the selection.
    const Analysis& a = doc_.analysis();
    for (const Link& l : g.links) {
        const Node* from = g.find(l.fromNode);
        const Node* to = g.find(l.toNode);
        if (!from || !to) continue;
        if (linkDetached_ && drag_ == Drag::Link && l.toNode == linkNode_) continue;
        const bool lit = doc_.selected(l.fromNode) || doc_.selected(l.toNode);
        const NodeInfo* info = a.info(l.fromNode);
        const bool ok = info && info->state == NodeInfo::State::Ok;
        QPen pen(theme::q(lit ? theme::kText : ok ? theme::kTextDim : theme::kTextFaint), lit ? 2.2 : 1.6);
        pen.setCosmetic(false);
        p.setPen(pen);
        p.setBrush(Qt::NoBrush);
        p.drawPath(wirePath(outputPort(*from, l.fromPort), inputPort(*to, l.toPort)));
    }
    if (drag_ == Drag::Link) {
        if (const Node* n = g.find(linkNode_)) {
            const QPointF port = linkFromOutput_ ? outputPort(*n, linkPort_) : inputPort(*n, linkPort_);
            const QPointF cursor = toScene(cursorView_);
            QPen pen(theme::q(theme::kAccent), 2.0, Qt::DashLine);
            p.setPen(pen);
            p.drawPath(linkFromOutput_ ? wirePath(port, cursor) : wirePath(cursor, port));
        }
    }
    for (const Node& n : g.nodes) paintNode(p, n);
    // A node dragged in from the library: the card itself where it would land.
    if (ghost_) {
        p.setOpacity(0.8);
        paintNode(p, *ghost_, true);
        p.setOpacity(1.0);
    }
    p.restore();

    if (drag_ == Drag::Box) {
        const QRectF box = QRectF(pressView_, cursorView_).normalized();
        p.setPen(QPen(theme::q(theme::accent(150)), 1));
        p.setBrush(theme::q(theme::accent(18)));
        p.drawRect(box);
    }

    // The model at a glance, bottom left.
    if (!g.nodes.empty()) {
        QString text = tr("%1 nodes  ·  %2 parameters").arg(g.nodes.size()).arg(QString::fromStdString(countText(a.totalParams)));
        Rgba color = theme::kTextDim;
        if (a.errors) {
            text += "  ·  " + (a.errors == 1 ? tr("1 problem") : tr("%1 problems").arg(a.errors));
            color = theme::kDanger;
        } else if (!a.valid()) {
            text += "  ·  " + (a.hasLoss ? tr("incomplete") : tr("add a Loss to train"));
        } else if (const TrainReport r = doc_.trainingReport(); doc_.training()) {
            text += "  ·  " + tr("training, epoch %1 of %2").arg(r.epoch).arg(r.epochs);
            color = theme::kText;
        } else if (!r.loss.empty()) {
            text += "  ·  " + tr("trained: loss %1").arg(QString::number(r.loss.back(), 'g', 3));
            if (!r.accuracy.empty()) text += tr(", %1% right").arg(QString::number(r.accuracy.back() * 100, 'f', 1));
        } else {
            text += "  ·  " + tr("ready to train");
        }
        const QFontMetrics fm(font());
        const QRectF r(12, height() - 12 - 28, fm.horizontalAdvance(text) + 24, 28);
        p.setPen(QPen(theme::q(theme::kBorder), 1));
        p.setBrush(theme::q(theme::withAlpha(theme::kBgPanel, 235)));
        p.drawRoundedRect(r, 6, 6);
        p.setPen(theme::q(color));
        p.drawText(r, Qt::AlignCenter, text);
        // ... and the zoom, bottom right.
        const QString z = QString("%1%").arg(std::lround(zoom_ * 100));
        const QRectF zr(width() - 12 - fm.horizontalAdvance(z) - 20, height() - 12 - 28, fm.horizontalAdvance(z) + 20, 28);
        p.setPen(QPen(theme::q(theme::kBorder), 1));
        p.setBrush(theme::q(theme::withAlpha(theme::kBgPanel, 235)));
        p.drawRoundedRect(zr, 6, 6);
        p.setPen(theme::q(theme::kTextFaint));
        p.drawText(zr, Qt::AlignCenter, z);
    }
}

void NodeCanvas::paintNode(QPainter& p, const Node& n, bool ghost) {
    const Graph& g = doc_.graph;
    const int id = ghost ? 0 : n.id;
    const NodeType* type = findNodeType(n.type);
    const NodeInfo* info = ghost ? nullptr : doc_.analysis().info(id);
    const QRectF r = nodeRect(n);
    const bool selected = ghost || doc_.selected(id);
    const bool hovered = hover_.node == id && drag_ == Drag::None;
    const bool error = info && info->state == NodeInfo::State::Error;
    const qreal radius = 8;

    // Card: a soft shadow, the body, the header band.
    p.setPen(Qt::NoPen);
    p.setBrush(theme::q(theme::shade(70)));
    p.drawRoundedRect(r.translated(0, 3), radius, radius);
    p.setBrush(theme::q(theme::kBgPanel));
    p.drawRoundedRect(r, radius, radius);
    QPainterPath header;
    header.addRoundedRect(QRectF(r.left(), r.top(), r.width(), kNodeHeader + radius), radius, radius);
    p.save();
    p.setClipRect(QRectF(r.left(), r.top(), r.width(), kNodeHeader));
    p.fillPath(header, theme::q(selected ? theme::mix(theme::kBgRaised, theme::kAccent, 0.08f) : theme::kBgRaised));
    p.restore();
    p.setPen(QPen(theme::q(theme::kBorder), 1));
    p.drawLine(QPointF(r.left(), r.top() + kNodeHeader), QPointF(r.right(), r.top() + kNodeHeader));

    // Header: icon, name; the type on the right when the name says something else.
    bestyle::Draw d(p);
    bestyle::drawIcon(d, nodeIcon(n.type), {static_cast<float>(r.left() + 17), static_cast<float>(r.top() + kNodeHeader * 0.5)}, 15.0f,
                      selected ? theme::kAccent : theme::kTextDim);
    QFont bold = font();
    bold.setWeight(QFont::DemiBold);
    p.setFont(bold);
    const QFontMetricsF fmb(bold);
    QString typeName = type ? QString::fromStdString(type->name) : QString::fromStdString(n.type);
    const QString name = QString::fromStdString(n.name);
    const bool showType = type && name.compare(QString(typeName).remove(' '), Qt::CaseInsensitive) != 0 &&
                          !name.startsWith(QString(typeName).toLower().replace(' ', '_'));
    QFont small = font();
    small.setPointSizeF(small.pointSizeF() * 0.86);
    const QFontMetricsF fms(small);
    const qreal typeW = showType ? fms.horizontalAdvance(typeName) + 10 : 0;
    p.setPen(theme::q(theme::kText));
    p.drawText(QRectF(r.left() + 32, r.top(), r.width() - 42 - typeW, kNodeHeader), Qt::AlignVCenter | Qt::AlignLeft,
               fmb.elidedText(name, Qt::ElideRight, r.width() - 42 - typeW));
    if (showType) {
        p.setFont(small);
        p.setPen(theme::q(theme::kTextFaint));
        p.drawText(QRectF(r.right() - typeW - 4, r.top(), typeW, kNodeHeader), Qt::AlignVCenter | Qt::AlignLeft, typeName);
    }

    // Rows: port names when there are several, and the node's settings.
    p.setFont(small);
    const std::vector<std::string> ports = inputPorts(n);
    const int inputs = static_cast<int>(ports.size());
    const int outputs = type ? static_cast<int>(type->outputs.size()) : 0;
    qreal left = 14, right = 14; // room the port names take
    p.setPen(theme::q(theme::kTextFaint));
    if (inputs > 1)
        for (int i = 0; i < inputs; ++i) {
            const QString label = QString::fromStdString(ports[static_cast<size_t>(i)]);
            p.drawText(QRectF(r.left() + 12, inputPort(n, i).y() - kNodeRow * 0.5, 80, kNodeRow), Qt::AlignVCenter | Qt::AlignLeft, label);
            if (i == 0) left = std::max(left, 22 + fms.horizontalAdvance(label));
        }
    if (outputs > 1)
        for (int i = 0; i < outputs; ++i) {
            const QString label = QString::fromStdString(type->outputs[static_cast<size_t>(i)]);
            p.drawText(QRectF(r.right() - 92, outputPort(n, i).y() - kNodeRow * 0.5, 80, kNodeRow), Qt::AlignVCenter | Qt::AlignRight, label);
            if (i == 0) right = std::max(right, 22 + fms.horizontalAdvance(label));
        }
    if (const QString s = summary(g, n); !s.isEmpty()) {
        p.setPen(theme::q(theme::kTextDim));
        p.drawText(QRectF(r.left() + left, r.top() + kNodeHeader, r.width() - left - right, kNodeRow), Qt::AlignVCenter | Qt::AlignLeft,
                   fms.elidedText(s, Qt::ElideRight, r.width() - left - right));
    }

    // Footer: what comes out and how many parameters, or what is wrong.
    const qreal fy = r.bottom() - kNodeFooter;
    p.setPen(QPen(theme::q(theme::withAlpha(theme::kBorder, 160)), 1));
    p.drawLine(QPointF(r.left() + 10, fy), QPointF(r.right() - 10, fy));
    const QRectF footer(r.left() + 14, fy, r.width() - 28, kNodeFooter);
    if (info && info->state == NodeInfo::State::Ok) {
        // On the right: the trainable values it holds (a dataset: its rows).
        QString params = info->params ? QString::fromStdString(countText(info->params)) : QString();
        if (n.type == "dataset") params = tr("%1 rows").arg(QString::fromStdString(countText(loadDataset(g, n)->rows)));
        const qreal pw = params.isEmpty() ? 0 : fms.horizontalAdvance(params) + 8;
        QString out;
        for (const Shape& s : info->outputs) out += (out.isEmpty() ? "" : "  ·  ") + QString::fromStdString(shapeText(s));
        if (n.type == "loss") {
            const TrainReport rep = doc_.trainingReport();
            out = rep.loss.empty() ? tr("to make smaller") : tr("total %1").arg(QString::number(rep.loss.back(), 'g', 3));
        }
        p.setPen(theme::q(theme::kText));
        p.drawText(footer.adjusted(0, 0, -pw, 0), Qt::AlignVCenter | Qt::AlignLeft,
                   fms.elidedText(QString::fromUtf8("→ ") + out, Qt::ElideRight, footer.width() - pw));
        if (!params.isEmpty()) {
            p.setPen(theme::q(theme::kTextFaint));
            p.drawText(footer, Qt::AlignVCenter | Qt::AlignRight, params);
        }
    } else if (ghost) {
        p.setPen(theme::q(theme::kTextFaint));
        p.drawText(footer, Qt::AlignVCenter | Qt::AlignLeft, tr("drop to add"));
    } else if (info) {
        p.setPen(theme::q(error ? theme::kDanger : theme::kTextFaint));
        p.drawText(footer, Qt::AlignVCenter | Qt::AlignLeft, fms.elidedText(QString::fromStdString(info->message), Qt::ElideRight, footer.width()));
    }

    // Outline: the accent when selected, red with a problem.
    const Rgba outline = selected ? theme::kAccent : error ? theme::kDanger : hovered ? theme::kTextFaint : theme::kBorder;
    p.setPen(QPen(theme::q(outline), selected ? 1.8 : 1.0));
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(r, radius, radius);

    // Ports: filled when connected; the one a link would land on rings in the accent.
    auto port = [&](QPointF at, bool connected, bool target) {
        p.setPen(QPen(theme::q(target ? theme::kAccent : connected ? theme::kText : theme::kTextDim), target ? 2.0 : 1.4));
        p.setBrush(theme::q(connected ? theme::kText : theme::kBgDeep));
        p.drawEllipse(at, target ? kPortRadius + 1.5 : kPortRadius, target ? kPortRadius + 1.5 : kPortRadius);
    };
    for (int i = 0; i < inputs; ++i) {
        const bool target = ((drag_ == Drag::Link && linkFromOutput_) || drag_ == Drag::None) && hover_.kind == Hit::Kind::Input &&
                            hover_.node == id && hover_.port == i;
        port(inputPort(n, i), g.linkInto(id, i) != nullptr && !(linkDetached_ && drag_ == Drag::Link && linkNode_ == id && linkPort_ == i), target);
    }
    for (int i = 0; i < outputs; ++i) {
        const bool connected = std::any_of(g.links.begin(), g.links.end(), [&](const Link& l) { return l.fromNode == id && l.fromPort == i; });
        const bool target = ((drag_ == Drag::Link && !linkFromOutput_) || drag_ == Drag::None) && hover_.kind == Hit::Kind::Output &&
                            hover_.node == id && hover_.port == i;
        port(outputPort(n, i), connected, target);
    }
}

// --- mouse -----------------------------------------------------------------------------------

void NodeCanvas::mousePressEvent(QMouseEvent* e) {
    setFocus();
    const QPointF v = e->position();
    pressView_ = lastView_ = cursorView_ = v;
    moved_ = false;
    if (e->button() == Qt::MiddleButton || (e->button() == Qt::LeftButton && space_)) {
        drag_ = Drag::Pan;
        setCursor(Qt::ClosedHandCursor);
        return;
    }
    if (e->button() == Qt::RightButton) {
        showContextMenu(v, e->globalPosition().toPoint());
        return;
    }
    if (e->button() != Qt::LeftButton) return;
    const Hit hit = hitTest(v);
    const bool toggle = e->modifiers() & (Qt::ControlModifier | Qt::ShiftModifier);
    switch (hit.kind) {
    case Hit::Kind::Output:
        drag_ = Drag::Link;
        linkNode_ = hit.node;
        linkPort_ = hit.port;
        linkFromOutput_ = true;
        linkDetached_ = false;
        break;
    case Hit::Kind::Input:
        if (const Link* l = doc_.graph.linkInto(hit.node, hit.port)) {
            // Takes the link off this input: it follows the cursor from its output.
            drag_ = Drag::Link;
            linkFromOutput_ = true;
            linkDetached_ = true;
            linkNode_ = l->fromNode;
            linkPort_ = l->fromPort;
            doc_.graph.disconnect(hit.node, hit.port);
            doc_.edited(false);
        } else {
            drag_ = Drag::Link;
            linkNode_ = hit.node;
            linkPort_ = hit.port;
            linkFromOutput_ = false;
            linkDetached_ = false;
        }
        break;
    case Hit::Kind::Node: {
        std::vector<int> sel = doc_.selection();
        if (toggle) {
            if (doc_.selected(hit.node)) std::erase(sel, hit.node);
            else sel.push_back(hit.node);
        } else if (!doc_.selected(hit.node)) {
            sel = {hit.node};
        }
        doc_.select(sel);
        // The card comes to the top.
        auto it = std::find_if(doc_.graph.nodes.begin(), doc_.graph.nodes.end(), [&](const Node& n) { return n.id == hit.node; });
        std::rotate(it, it + 1, doc_.graph.nodes.end());
        drag_ = Drag::Nodes;
        startPos_.clear();
        for (int id : doc_.selection())
            if (const Node* n = doc_.graph.find(id)) startPos_[id] = QPointF(n->x, n->y);
        break;
    }
    case Hit::Kind::None:
        if (!toggle) doc_.select({});
        drag_ = Drag::Box;
        break;
    }
    update();
}

void NodeCanvas::mouseMoveEvent(QMouseEvent* e) {
    const QPointF v = e->position();
    cursorView_ = v;
    if ((v - pressView_).manhattanLength() > 3) moved_ = true;
    switch (drag_) {
    case Drag::Pan:
        pan_ += v - lastView_;
        break;
    case Drag::Nodes: {
        const QPointF delta = (v - pressView_) / zoom_;
        for (const auto& [id, start] : startPos_)
            if (Node* n = doc_.graph.find(id)) {
                // On the grid's step (Settings › Interface), so cards line up.
                const float snap = settings().snapToGrid ? kGrid / 2 : 1.0f;
                n->x = static_cast<float>(std::round((start.x() + delta.x()) / snap) * snap);
                n->y = static_cast<float>(std::round((start.y() + delta.y()) / snap) * snap);
            }
        if (moved_) doc_.edited(false);
        break;
    }
    case Drag::Link:
    case Drag::Box:
    case Drag::None:
        break;
    }
    lastView_ = v;
    const Hit h = drag_ == Drag::Pan ? Hit{} : hitTest(v);
    if (h.kind != hover_.kind || h.node != hover_.node || h.port != hover_.port) hover_ = h;
    if (drag_ == Drag::None && !space_)
        setCursor(h.kind == Hit::Kind::Input || h.kind == Hit::Kind::Output ? Qt::CrossCursor : Qt::ArrowCursor);
    update();
}

void NodeCanvas::mouseReleaseEvent(QMouseEvent* e) {
    const QPointF v = e->position();
    const Drag drag = drag_;
    drag_ = Drag::None;
    switch (drag) {
    case Drag::Pan:
        setCursor(space_ ? Qt::OpenHandCursor : Qt::ArrowCursor);
        break;
    case Drag::Nodes:
        if (moved_) doc_.edited();
        break;
    case Drag::Link:
        drag_ = Drag::Link; // drawn while a menu asks what to add
        finishLink(v, e->globalPosition().toPoint());
        drag_ = Drag::None;
        linkDetached_ = false;
        break;
    case Drag::Box: {
        const QRectF box = QRectF(toScene(pressView_), toScene(v)).normalized();
        std::vector<int> sel = (e->modifiers() & (Qt::ControlModifier | Qt::ShiftModifier)) ? doc_.selection() : std::vector<int>{};
        for (const Node& n : doc_.graph.nodes)
            if (moved_ && box.intersects(nodeRect(n)) && std::find(sel.begin(), sel.end(), n.id) == sel.end()) sel.push_back(n.id);
        doc_.select(sel);
        break;
    }
    case Drag::None:
        break;
    }
    update();
}

void NodeCanvas::finishLink(QPointF view, QPoint global) {
    Graph& g = doc_.graph;
    const Hit hit = hitTest(view);
    bool linked = false;
    if (linkFromOutput_) {
        int to = 0, port = -1;
        if (hit.kind == Hit::Kind::Input) to = hit.node, port = hit.port;
        else if (hit.kind == Hit::Kind::Node) {
            // Dropped on a card: its first free input (or its first).
            if (const auto ports = inputPorts(*g.find(hit.node)); !ports.empty()) {
                to = hit.node;
                port = 0;
                for (int i = 0; i < static_cast<int>(ports.size()); ++i)
                    if (!g.linkInto(to, i)) {
                        port = i;
                        break;
                    }
            }
        }
        if (port >= 0) linked = g.connect(linkNode_, linkPort_, to, port);
        else if (hit.kind == Hit::Kind::None && !linkDetached_ && moved_) {
            // Dropped on empty space: pick a node to add there, fed by this output.
            QMenu menu;
            const int from = linkNode_, fromPort = linkPort_;
            const QPointF at = toScene(view) - QPointF(0, kNodeHeader + kNodeRow * 0.5);
            addNodeMenu(menu, [&](const std::string& type) {
                const int id = addNode(type, at);
                linked = g.connect(from, fromPort, id, 0);
            });
            menu.exec(global);
        }
    } else if (hit.kind == Hit::Kind::Output || hit.kind == Hit::Kind::Node) {
        const NodeType* t = findNodeType(g.find(hit.node)->type);
        if (t && !t->outputs.empty()) linked = g.connect(hit.node, hit.kind == Hit::Kind::Output ? hit.port : 0, linkNode_, linkPort_);
    }
    if (linked || linkDetached_) doc_.edited();
}

void NodeCanvas::wheelEvent(QWheelEvent* e) {
    const qreal steps = e->angleDelta().y() / 120.0;
    if (steps != 0) setZoom(zoom_ * std::pow(1.15, steps), e->position());
    else pan_ += e->pixelDelta(); // a touchpad
    update();
}

void NodeCanvas::leaveEvent(QEvent*) {
    hover_ = {};
    update();
}

// --- keys ------------------------------------------------------------------------------------

void NodeCanvas::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Space && !e->isAutoRepeat()) {
        space_ = true;
        setCursor(Qt::OpenHandCursor);
        return;
    }
    if (e->key() == Qt::Key_Delete || e->key() == Qt::Key_Backspace) {
        deleteSelection();
        return;
    }
    if (e->key() == Qt::Key_Escape) {
        doc_.select({});
        return;
    }
    QWidget::keyPressEvent(e);
}

void NodeCanvas::keyReleaseEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Space && !e->isAutoRepeat()) {
        space_ = false;
        if (drag_ != Drag::Pan) setCursor(Qt::ArrowCursor);
        return;
    }
    QWidget::keyReleaseEvent(e);
}

// --- menus and drops -------------------------------------------------------------------------

void NodeCanvas::addNodeMenu(QMenu& menu, const std::function<void(const std::string&)>& chosen) {
    for (const std::string& category : nodeCategories()) {
        QMenu* sub = menu.addMenu(bestyle::menuIcon(categoryIcon(category), theme::kTextDim), tr(category.c_str()));
        for (const NodeType& t : nodeTypes())
            if (t.category == category) {
                const std::string id = t.id;
                sub->addAction(bestyle::menuIcon(nodeIcon(t.id), theme::kTextDim), tr(t.name.c_str()), [chosen, id] { chosen(id); });
            }
    }
}

void NodeCanvas::showContextMenu(QPointF view, QPoint global) {
    const Hit hit = hitTest(view);
    QMenu menu;
    if (hit.kind != Hit::Kind::None) {
        if (!doc_.selected(hit.node)) doc_.select({hit.node});
        menu.addAction(bestyle::menuIcon(bestyle::Icon::Duplicate, theme::kTextDim), tr("Duplicate"), [this] { duplicateSelection(); });
        menu.addAction(tr("Disconnect"), [this] {
            for (int id : doc_.selection())
                std::erase_if(doc_.graph.links, [id](const Link& l) { return l.fromNode == id || l.toNode == id; });
            doc_.edited();
        });
        menu.addSeparator();
        menu.addAction(bestyle::menuIcon(bestyle::Icon::Trash, theme::kDanger), tr("Delete"), [this] { deleteSelection(); });
    } else {
        const QPointF at = toScene(view);
        QMenu* add = menu.addMenu(bestyle::menuIcon(bestyle::Icon::Plus, theme::kTextDim), tr("Add Node"));
        addNodeMenu(*add, [this, at](const std::string& type) { addNode(type, at); });
        menu.addSeparator();
        menu.addAction(bestyle::menuIcon(bestyle::Icon::FitView, theme::kTextDim), tr("Frame All"), [this] { frameAll(); });
        menu.addAction(tr("Select All"), [this] { selectAll(); });
    }
    menu.exec(global);
}

namespace {

// The node a library drag carries: its type, and a custom node's name and parameters.
std::optional<Node> draggedNode(const QMimeData* mime) {
    const std::string type = mime->data(kNodeMime).toStdString();
    const NodeType* t = findNodeType(type);
    if (!t) return std::nullopt;
    Node n;
    n.type = type;
    n.name = t->name;
    const Json extra = Json::parse(mime->data(kNodeParamsMime).toStdString(), nullptr, false);
    if (extra.is_object()) {
        if (const std::string name = extra.value("name", std::string()); !name.empty()) n.name = name;
        if (const Json params = extra.value("params", Json::object()); params.is_object())
            for (const auto& [k, v] : params.items()) n.params[k] = v;
    }
    return n;
}

} // namespace

void NodeCanvas::dragEnterEvent(QDragEnterEvent* e) {
    ghost_ = draggedNode(e->mimeData());
    if (!ghost_) return;
    ghost_->name = doc_.graph.uniqueName(ghost_->name);
    e->acceptProposedAction();
    dragMoveEvent(e);
}

void NodeCanvas::dragMoveEvent(QDragMoveEvent* e) {
    if (!ghost_) return;
    // The card's header under the cursor.
    const QPointF at = toScene(e->position()) - QPointF(kNodeWidth * 0.5, kNodeHeader * 0.5);
    ghost_->x = static_cast<float>(at.x());
    ghost_->y = static_cast<float>(at.y());
    e->acceptProposedAction();
    update();
}

void NodeCanvas::dragLeaveEvent(QDragLeaveEvent*) {
    ghost_.reset();
    update();
}

void NodeCanvas::dropEvent(QDropEvent* e) {
    const std::optional<Node> n = draggedNode(e->mimeData());
    ghost_.reset();
    if (!n) return;
    Json params = Json::object();
    for (const auto& [k, v] : n->params) params[k] = v;
    addNode(n->type, toScene(e->position()) - QPointF(kNodeWidth * 0.5, kNodeHeader * 0.5), n->name, params);
    e->acceptProposedAction();
    setFocus();
}

} // namespace logievo::ui
