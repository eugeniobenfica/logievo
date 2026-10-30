#pragma once

// The left panel: every node type in its category (Data, Layers, Activations...), as tiles.
// Click one to add it in the middle of the canvas, or drag it where it goes; the search field
// narrows the list. Custom shows the Code node and the user's own (Settings › Custom Nodes).

#include "core/Graph.h"

#include <QWidget>

#include <functional>
#include <set>
#include <string>

class QLineEdit;
class QVBoxLayout;

namespace logievo::ui {

class NodeLibrary : public QWidget {
    Q_OBJECT
public:
    explicit NodeLibrary(QWidget* parent = nullptr);
    // A node to add: its type, and for a custom node its name and starting parameters.
    std::function<void(const std::string& type, const std::string& name, const Json& params)> onAdd;

private:
    void rebuild();
    QLineEdit* search_ = nullptr;
    QWidget* list_ = nullptr;
    QVBoxLayout* listLayout_ = nullptr;
    std::set<std::string> closed_; // categories folded away
};

} // namespace logievo::ui
