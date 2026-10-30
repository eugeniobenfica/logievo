#include "ui/NodeLibrary.h"

#include "bestyle/BeStyle.h"
#include "core/NodeTypes.h"
#include "ui/Icons.h"
#include "ui/NodeCanvas.h"
#include "ui/Windows.h"
#include "app/Settings.h"

#include <QLabel>
#include <QLineEdit>
#include <QMimeData>
#include <QScrollArea>
#include <QVBoxLayout>

namespace logievo::ui {

namespace metrics = bestyle::metrics;

NodeLibrary::NodeLibrary(QWidget* parent) : QWidget(parent) {
    auto* col = new QVBoxLayout(this);
    col->setContentsMargins(metrics::kPanelMargin, metrics::kPanelMargin + 4, metrics::kPanelMargin, 0);
    col->setSpacing(metrics::kGroupGap);
    auto* title = bestyle::sectionLabel(tr("Nodes"));
    col->addWidget(title);
    search_ = new QLineEdit;
    search_->setPlaceholderText(tr("Search nodes"));
    search_->setClearButtonEnabled(true);
    connect(search_, &QLineEdit::textChanged, this, [this] { rebuild(); });
    connect(&AppSignals::get(), &AppSignals::customNodesChanged, this, [this] { rebuild(); });
    col->addWidget(search_);
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->viewport()->setAutoFillBackground(false);
    list_ = new QWidget;
    list_->setAutoFillBackground(false);
    listLayout_ = new QVBoxLayout(list_);
    listLayout_->setContentsMargins(0, 0, 0, metrics::kPanelMargin);
    listLayout_->setSpacing(metrics::kGap);
    scroll->setWidget(list_);
    col->addWidget(scroll, 1);
    rebuild();
}

void NodeLibrary::rebuild() {
    while (QLayoutItem* item = listLayout_->takeAt(0)) {
        if (QWidget* w = item->widget()) w->deleteLater();
        delete item;
    }
    const QString search = search_->text().trimmed();
    int shown = 0;
    // A tile: clicked it adds the node, dragged it carries its type (and a custom node's settings).
    auto tile = [this](bestyle::Icon icon, const QString& label, const QString& tip, const std::string& type, const std::string& name, const Json& params) {
        auto* t = new bestyle::Tile(icon, label);
        t->setTileSize(QSize(76, 64));
        t->setToolTip(tip);
        t->setDragData([type, name, params] {
            auto* mime = new QMimeData;
            mime->setData(kNodeMime, QByteArray::fromStdString(type));
            mime->setData(kNodeParamsMime, QByteArray::fromStdString(Json{{"name", name}, {"params", params}}.dump()));
            return mime;
        });
        t->setDragPixmap([] { return QPixmap(); }); // the canvas draws the card itself where it would land
        connect(t, &QAbstractButton::clicked, this, [this, type, name, params] {
            if (onAdd) onAdd(type, name, params);
        });
        return t;
    };
    for (const std::string& category : nodeCategories()) {
        std::vector<const NodeType*> types;
        for (const NodeType& t : nodeTypes())
            if (t.category == category && (search.isEmpty() || QString::fromStdString(t.name).contains(search, Qt::CaseInsensitive) ||
                                           tr(t.name.c_str()).contains(search, Qt::CaseInsensitive)))
                types.push_back(&t);
        // The user's own nodes join the Code node.
        std::vector<const CustomNode*> mine;
        if (category == "Custom")
            for (const CustomNode& c : settings().customNodes)
                if (search.isEmpty() || QString::fromStdString(c.name).contains(search, Qt::CaseInsensitive)) mine.push_back(&c);
        if (types.empty() && mine.empty()) continue;
        shown += static_cast<int>(types.size() + mine.size());
        // A search opens every category it finds something in.
        const bool closed = search.isEmpty() && closed_.count(category);
        auto* header = new bestyle::CategoryHeader(categoryIcon(category), tr(category.c_str()), static_cast<int>(types.size() + mine.size()), closed);
        header->toggled = [this, category] {
            if (!closed_.erase(category)) closed_.insert(category);
            QMetaObject::invokeMethod(this, [this] { rebuild(); }, Qt::QueuedConnection);
        };
        listLayout_->addWidget(header);
        if (closed) continue;
        auto* grid = new bestyle::ResponsiveGrid(76);
        for (const NodeType* t : types) grid->addWidget(tile(nodeIcon(t->id), tr(t->name.c_str()), tr(t->description.c_str()), t->id, {}, Json::object()));
        for (const CustomNode* c : mine) {
            const QString tip = (c->description.empty() ? QString() : QString::fromStdString(c->description) + "\n\n") + QString::fromStdString(c->code);
            grid->addWidget(tile(bestyle::Icon::Edit, QString::fromStdString(c->name), tip, "custom", c->name,
                                 Json{{"code", c->code}, {"inputs", c->inputs}}));
        }
        listLayout_->addWidget(new bestyle::CategoryBody(grid));
    }
    if (shown == 0) {
        auto* none = new QLabel(tr("No node matches \"%1\".").arg(search));
        none->setWordWrap(true);
        bestyle::setTone(none, bestyle::Tone::TextDim);
        listLayout_->addWidget(none);
    }
    listLayout_->addStretch(1);
}

} // namespace logievo::ui
