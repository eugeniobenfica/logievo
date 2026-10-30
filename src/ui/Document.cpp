#include "ui/Document.h"

#include "core/Serialization.h"

#include <QTimer>

#include <algorithm>

namespace logievo::ui {

namespace {
constexpr size_t kUndoSteps = 200;
}

Document::Document(Graph g, std::filesystem::path f, Weights w) : graph(std::move(g)), file(std::move(f)), weights(std::move(w)) {
    analysis_ = analyze(graph);
    current_ = saved_ = snapshot();
    poll_ = new QTimer(this);
    poll_->setInterval(100);
    connect(poll_, &QTimer::timeout, this, &Document::poll);
}

Document::~Document() { trainer_.stop(); }

std::string Document::snapshot() const { return toJson(graph).dump(); }

void Document::restore(const std::string& state) {
    Graph g;
    if (!fromJson(Json::parse(state, nullptr, false), g)) return;
    graph = std::move(g);
    current_ = state;
    analysis_ = analyze(graph);
    std::erase_if(selection_, [this](int id) { return !graph.find(id); });
    emit replaced();
    emit changed();
    emit selectionChanged();
}

void Document::edited(bool record) {
    analysis_ = analyze(graph);
    if (record) {
        if (std::string now = snapshot(); now != current_) {
            undo_.push_back(std::move(current_));
            if (undo_.size() > kUndoSteps) undo_.erase(undo_.begin());
            current_ = std::move(now);
            redo_.clear();
        }
    }
    emit changed();
}

void Document::undo() {
    if (undo_.empty()) return;
    redo_.push_back(current_);
    const std::string state = std::move(undo_.back());
    undo_.pop_back();
    restore(state);
}

void Document::redo() {
    if (redo_.empty()) return;
    undo_.push_back(current_);
    const std::string state = std::move(redo_.back());
    redo_.pop_back();
    restore(state);
}

bool Document::save(std::string* error) {
    // Only the weights that still fit the graph as it is.
    Weights keep;
    for (const auto& [id, list] : weights)
        if (const Node* n = graph.find(id); n && analysis_.info(id)) {
            const std::vector<Shape> shapes = weightShapes(graph, *n, *analysis_.info(id));
            bool fits = shapes.size() == list.size();
            for (size_t i = 0; fits && i < list.size(); ++i) fits = shapes[i] == list[i].shape;
            if (fits) keep[id] = list;
        }
    if (!saveGraph(file, graph, error, &keep)) return false;
    saved_ = current_ = snapshot();
    weightsDirty_ = false;
    emit saved();
    return true;
}

bool Document::selected(int id) const { return std::find(selection_.begin(), selection_.end(), id) != selection_.end(); }

void Document::select(std::vector<int> ids) {
    if (ids == selection_) return;
    selection_ = std::move(ids);
    emit selectionChanged();
}

bool Document::startTraining() {
    const bool started = trainer_.start(graph, weights);
    report_ = trainer_.report();
    if (started) poll_->start();
    emit trainingChanged();
    return started;
}

void Document::stopTraining() {
    trainer_.stop();
    poll();
}

void Document::forgetWeights() {
    if (training()) return;
    weightsDirty_ = !weights.empty() || weightsDirty_;
    weights.clear();
    report_ = {};
    emit trainingChanged();
    emit changed();
}

void Document::poll() {
    // Read "done" first: the trainer hands over its last weights before it says it is done.
    const bool done = !trainer_.running();
    report_ = trainer_.report();
    if (trainer_.takeWeights(weights)) weightsDirty_ = true;
    if (done) {
        poll_->stop();
        emit changed(); // unsaved weights
    }
    emit trainingChanged();
}

} // namespace logievo::ui
