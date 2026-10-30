#include "ui/GraphPreview.h"

#include "bestyle/Theme.h"
#include "core/NodeTypes.h"
#include "ui/Logo.h"

#include <QPainter>
#include <QPainterPath>

#include <algorithm>

namespace logievo::ui {

namespace theme = bestyle::theme;

namespace {
int rows(const Node& n) {
    const NodeType* t = findNodeType(n.type);
    return t ? std::max<int>({1, static_cast<int>(inputPorts(n).size()), static_cast<int>(t->outputs.size())}) : 1;
}
} // namespace

QRectF nodeRect(const Node& n) { return {n.x, n.y, kNodeWidth, kNodeHeader + rows(n) * kNodeRow + kNodeFooter}; }

QPointF inputPort(const Node& n, int port) { return {n.x, n.y + kNodeHeader + kNodeRow * (port + 0.5f)}; }

QPointF outputPort(const Node& n, int port) { return {n.x + kNodeWidth, n.y + kNodeHeader + kNodeRow * (port + 0.5f)}; }

QRectF graphBounds(const Graph& g) {
    QRectF r;
    for (const Node& n : g.nodes) r = r.isNull() ? nodeRect(n) : r.united(nodeRect(n));
    return r;
}

QPainterPath wirePath(QPointF s, QPointF e) {
    const qreal dx = e.x() >= s.x() ? std::max(50.0, (e.x() - s.x()) * 0.5) : std::clamp(std::abs(e.y() - s.y()) * 0.5, 60.0, 140.0);
    QPainterPath path(s);
    path.cubicTo(s + QPointF(dx, 0), e - QPointF(dx, 0), e);
    return path;
}

QImage renderPreview(const Graph& g, QSize size) {
    QImage img(size, QImage::Format_ARGB32_Premultiplied);
    img.fill(theme::q(theme::kBgDeep));
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing);
    const QRectF b = graphBounds(g);
    if (b.isNull()) {
        drawLogo(p, {size.width() * 0.5f, size.height() * 0.5f}, size.height() * 0.11f, theme::kTextFaint);
        return img;
    }
    // Fitted with a margin, never larger than life.
    const qreal pad = size.height() * 0.12;
    const qreal zoom = std::min({(size.width() - pad * 2) / b.width(), (size.height() - pad * 2) / b.height(), 0.6 * size.height() / 225.0});
    p.translate(size.width() * 0.5, size.height() * 0.5);
    p.scale(zoom, zoom);
    p.translate(-b.center());
    QPen wire(theme::q(theme::kTextFaint), 2.0 / zoom * (size.height() / 225.0));
    p.setPen(wire);
    p.setBrush(Qt::NoBrush);
    for (const Link& l : g.links) {
        const Node* a = g.find(l.fromNode);
        const Node* c = g.find(l.toNode);
        if (!a || !c) continue;
        p.drawPath(wirePath(outputPort(*a, l.fromPort), inputPort(*c, l.toPort)));
    }
    p.setPen(QPen(theme::q(theme::kBorder), 2.0));
    for (const Node& n : g.nodes) {
        const QRectF r = nodeRect(n);
        p.setBrush(theme::q(theme::kBgRaised));
        p.drawRoundedRect(r, 10, 10);
        p.fillRect(QRectF(r.left() + 14, r.top() + 12, r.width() * 0.55, 9), theme::q(theme::kTextDim));
    }
    return img;
}

} // namespace logievo::ui
