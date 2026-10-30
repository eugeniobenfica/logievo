#include "ui/Logo.h"

#include <QPainterPath>
#include <QPolygonF>

namespace logievo::ui {

using bestyle::Rgba;

namespace {

// The mark on a 100-unit grid (y down); it spans x 14..90, y 6..94, centered on (52, 50).
constexpr QPointF kStem[] = {{14, 18}, {40, 6}, {40, 66}, {14, 94}};          // upright, to the fold
constexpr QPointF kFoot[] = {{14, 94}, {40, 66}, {90, 66}, {78, 94}};         // past the fold
constexpr float kCenterX = 52.0f, kCenterY = 50.0f, kHalf = 44.0f;

QPolygonF place(const QPointF* pts, int n, bestyle::Pt center, float radius) {
    QPolygonF poly;
    const float s = radius / kHalf;
    for (int i = 0; i < n; ++i) poly << QPointF(center.x + (pts[i].x() - kCenterX) * s, center.y + (pts[i].y() - kCenterY) * s);
    return poly;
}

} // namespace

void drawLogo(QPainter& p, bestyle::Pt center, float radius, Rgba color) {
    p.save();
    p.setRenderHint(QPainter::Antialiasing);
    p.setPen(Qt::NoPen);
    QColor c = bestyle::qcolor(color);
    // Both faces in one path, so the shared fold edge has no seam.
    QPainterPath whole;
    whole.addPolygon(place(kStem, 4, center, radius));
    whole.addPolygon(place(kFoot, 4, center, radius));
    whole.setFillRule(Qt::WindingFill);
    QColor foot = c;
    foot.setAlphaF(c.alphaF() * 0.62f);
    p.setBrush(foot);
    p.drawPath(whole.simplified());
    p.setBrush(c);
    p.drawPolygon(place(kStem, 4, center, radius));
    p.restore();
}

QImage appIconImage(int size) {
    QImage img(size, size, QImage::Format_ARGB32_Premultiplied);
    img.fill(Qt::transparent);
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing);
    const qreal r = size * 0.22;
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(18, 18, 21));
    p.drawRoundedRect(QRectF(0, 0, size, size), r, r);
    drawLogo(p, {size * 0.5f, size * 0.5f}, size * 0.34f, bestyle::rgba(255, 255, 255));
    return img;
}

QIcon appIcon() {
    QIcon icon;
    for (int size : {16, 24, 32, 48, 64, 128, 256}) icon.addPixmap(QPixmap::fromImage(appIconImage(size)));
    return icon;
}

} // namespace logievo::ui
