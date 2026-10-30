#pragma once

// Where a node's box and ports are on the canvas (shared by the editor's canvas and the home
// screen's thumbnails), and a still of a whole graph for the project cards.

#include "core/Graph.h"

#include <QImage>
#include <QPainterPath>
#include <QRectF>

namespace logievo::ui {

constexpr float kNodeWidth = 196.0f;
constexpr float kNodeHeader = 32.0f;
constexpr float kNodeRow = 22.0f;
constexpr float kNodeFooter = 28.0f;

QRectF nodeRect(const Node& node);
QPointF inputPort(const Node& node, int port);
QPointF outputPort(const Node& node, int port);
QRectF graphBounds(const Graph& graph); // empty for an empty graph
// A link's curve from an output to an input; one going back left makes a short S, not a loop
// across the whole graph.
QPainterPath wirePath(QPointF from, QPointF to);

// The graph fitted in `size` pixels (device pixels: pass size * devicePixelRatio).
QImage renderPreview(const Graph& graph, QSize size);

} // namespace logievo::ui
