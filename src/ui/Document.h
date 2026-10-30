#pragma once

// The open project: its graph, file, selection and undo history, the analysis of the graph as
// it is now, and what the model has learned (its weights, and the training that runs). Every
// panel edits the graph through it and listens to it.

#include "core/Analysis.h"
#include "core/Graph.h"
#include "core/Runtime.h"

#include <QObject>

#include <filesystem>
#include <string>
#include <vector>

class QTimer;

namespace logievo::ui {

class Document : public QObject {
    Q_OBJECT
public:
    Document(Graph graph, std::filesystem::path file, Weights weights = {});
    ~Document() override;

    Graph graph;
    std::filesystem::path file;
    Weights weights; // learned so far (empty: never trained)

    const Analysis& analysis() const { return analysis_; }

    // After changing `graph`: analyzes it again and says so. `record` makes it one step of the
    // undo history (a drag records once, when it ends).
    void edited(bool record = true);
    void undo();
    void redo();
    bool canUndo() const { return !undo_.empty(); }
    bool canRedo() const { return !redo_.empty(); }
    bool dirty() const { return current_ != saved_ || weightsDirty_; }
    bool save(std::string* error = nullptr);

    const std::vector<int>& selection() const { return selection_; }
    bool selected(int id) const;
    void select(std::vector<int> ids);

    // Training on its own thread: the graph as it is now, from the weights learned so far.
    bool startTraining();
    void stopTraining();
    bool training() const { return trainer_.running(); }
    TrainReport trainingReport() const { return report_; }
    void forgetWeights(); // the next training starts fresh

signals:
    void changed();          // the graph (or its analysis) changed
    void replaced();         // ... entirely (undo, redo): panels rebuild
    void selectionChanged();
    void saved();
    void trainingChanged();  // started, an epoch went by, stopped

private:
    std::string snapshot() const;
    void restore(const std::string& state);
    void poll();

    Analysis analysis_;
    std::vector<int> selection_;
    std::string current_, saved_;
    std::vector<std::string> undo_, redo_;
    bool weightsDirty_ = false;
    Trainer trainer_;
    TrainReport report_;
    QTimer* poll_ = nullptr;
};

} // namespace logievo::ui
