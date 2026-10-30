#pragma once

// The right panel: the selected node's name and settings, with what it outputs and how many
// parameters it has; with nothing selected, the model as a whole: its summary, training (settings,
// Train / Stop, the loss curve) and problems.

#include <QWidget>

#include <functional>

class QScrollArea;

namespace logievo::ui {

class Document;

class Inspector : public QWidget {
    Q_OBJECT
public:
    explicit Inspector(Document& doc, QWidget* parent = nullptr);
    std::function<void()> onDelete, onDuplicate; // the selection (the canvas does it)

private:
    void scheduleRebuild();
    void rebuild();
    QWidget* buildNode(int id);
    QWidget* buildModel();
    QWidget* buildMany();

    Document& doc_;
    QScrollArea* scroll_ = nullptr;
    bool pending_ = false;
    std::function<void()> refresh_; // updates what the analysis decides (shapes, counts) in place
    std::function<void()> refreshTraining_; // the Training section, as training goes
};

} // namespace logievo::ui
