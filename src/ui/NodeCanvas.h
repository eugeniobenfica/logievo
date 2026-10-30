#pragma once

// The graph editor's canvas: nodes as cards with their ports, links as curves. Drag a node to
// move it, from an output to an input to connect (or from a connected input to take the link
// off), on empty space to select with a box; wheel zooms, the middle button or Space + drag
// pans. Nodes come from the library (click or drag) or the right-click menu.

#include "core/Graph.h"

#include <QPointF>
#include <QWidget>

#include <map>
#include <functional>
#include <optional>
#include <string>

class QMenu;

namespace logievo::ui {

class Document;

inline constexpr const char* kNodeMime = "application/x-logievo-node"; // a NodeType id
inline constexpr const char* kNodeParamsMime = "application/x-logievo-node-params"; // {"name", "params"} (JSON)

class NodeCanvas : public QWidget {
    Q_OBJECT
public:
    explicit NodeCanvas(Document& doc, QWidget* parent = nullptr);

    // A node of `type` at `scene` (its top-left), or near the middle of the view; selected.
    // A custom node comes with its name and parameters (its code).
    int addNode(const std::string& type, std::optional<QPointF> scene = {}, const std::string& name = {}, const Json& params = {});
    void frameAll();
    void deleteSelection();
    void duplicateSelection();
    void selectAll();
    void zoomBy(qreal factor); // around the middle of the view
    float zoom() const { return static_cast<float>(zoom_); }
    QPointF toScene(QPointF view) const { return (view - pan_) / zoom_; }
    QPointF toView(QPointF scene) const { return scene * zoom_ + pan_; }

protected:
    void paintEvent(QPaintEvent*) override;
    void resizeEvent(QResizeEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void keyPressEvent(QKeyEvent*) override;
    void keyReleaseEvent(QKeyEvent*) override;
    void dragEnterEvent(QDragEnterEvent*) override;
    void dragMoveEvent(QDragMoveEvent*) override;
    void dragLeaveEvent(QDragLeaveEvent*) override;
    void dropEvent(QDropEvent*) override;
    void leaveEvent(QEvent*) override;

private:
    struct Hit {
        enum class Kind { None, Node, Input, Output } kind = Kind::None;
        int node = 0, port = 0;
    };
    Hit hitTest(QPointF view) const;
    void setZoom(qreal zoom, QPointF anchorView);
    // A menu of every node type; `chosen` gets the one picked.
    void addNodeMenu(QMenu& menu, const std::function<void(const std::string&)>& chosen);
    void showContextMenu(QPointF view, QPoint global);
    void finishLink(QPointF view, QPoint global);

    // `ghost`: a node being dragged in, not in the graph yet.
    void paintNode(QPainter& p, const Node& n, bool ghost = false);

    Document& doc_;
    QPointF pan_{0, 0};
    qreal zoom_ = 1.0;
    bool framed_ = false;

    enum class Drag { None, Nodes, Link, Pan, Box } drag_ = Drag::None;
    QPointF pressView_, lastView_;
    bool moved_ = false;
    std::map<int, QPointF> startPos_; // dragged nodes where they were
    // A link being drawn: from an output (or back from an input) to the cursor.
    int linkNode_ = 0, linkPort_ = 0;
    bool linkFromOutput_ = true, linkDetached_ = false;
    QPointF cursorView_;
    bool space_ = false;
    Hit hover_;
    std::optional<Node> ghost_; // a node dragged in from the library, where it would land
};

} // namespace logievo::ui
