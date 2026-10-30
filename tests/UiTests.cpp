// The editor driven like a person would: links drawn and taken off with the mouse, nodes
// dragged and box-selected, delete, undo and redo, then a training run from the document.
// Runs offscreen (QT_QPA_PLATFORM=offscreen).

#include "bestyle/BeStyle.h"
#include "core/Graph.h"
#include "ui/Document.h"
#include "ui/EditorWindow.h"
#include "ui/GraphPreview.h"
#include "ui/NodeCanvas.h"

#include <QApplication>
#include <QDragEnterEvent>
#include <QMimeData>
#include <QTest>

#include <cstdio>
#include <cstdlib>

using namespace logievo;
using namespace logievo::ui;

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

namespace {

void drag(QWidget* w, QPointF from, QPointF to) {
    QTest::mousePress(w, Qt::LeftButton, {}, from.toPoint());
    for (int i = 1; i <= 8; ++i) QTest::mouseMove(w, (from + (to - from) * (i / 8.0)).toPoint());
    QTest::mouseRelease(w, Qt::LeftButton, {}, to.toPoint());
}

} // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    bestyle::install(bestyle::theme::darkPalette());

    Graph g;
    g.add("dataset", 0, 0);
    g.add("dense", 300, 0).params["units"] = 2;
    g.add("loss", 600, 0);
    const int data = g.nodes[0].id, dense = g.nodes[1].id, loss = g.nodes[2].id;
    EditorWindow editor(std::move(g), std::filesystem::temp_directory_path() / "logievo-ui-test.logievo");
    editor.resize(1400, 800);
    editor.show();
    QTest::qWait(50);
    Document& doc = editor.document();
    NodeCanvas& canvas = editor.canvas();
    auto view = [&](QPointF scene) { return canvas.toView(scene); };
    auto node = [&](int id) { return *doc.graph.find(id); };

    // Output port -> input port draws a link.
    drag(&canvas, view(outputPort(node(data), 0)), view(inputPort(node(dense), 0)));
    CHECK(doc.graph.links.size() == 1);
    // ... dropping on a card (not its port) takes its first free input: the prediction.
    drag(&canvas, view(outputPort(node(dense), 0)), view(nodeRect(node(loss)).center()));
    CHECK(doc.graph.links.size() == 2 && doc.graph.linkInto(loss, 0));
    // ... and the dataset's second output, the targets, into the loss's second input.
    drag(&canvas, view(outputPort(node(data), 1)), view(inputPort(node(loss), 1)));
    CHECK(doc.graph.links.size() == 3);
    CHECK(doc.analysis().valid());

    // Dragging a card moves it; undo puts it back, redo moves it again.
    const QPointF before(node(dense).x, node(dense).y);
    const QPointF grab = view(nodeRect(node(dense)).topLeft() + QPointF(60, 12));
    drag(&canvas, grab, grab + QPointF(80, 120));
    CHECK(node(dense).y > before.y() + 50);
    CHECK(doc.selection() == std::vector<int>{dense});
    QTest::keyClick(&canvas, Qt::Key_Z, Qt::ControlModifier);
    CHECK(node(dense).x == before.x() && node(dense).y == before.y());
    QTest::keyClick(&canvas, Qt::Key_Z, Qt::ControlModifier | Qt::ShiftModifier);
    CHECK(node(dense).y > before.y() + 50);

    // Delete takes the node and its links; undo brings both back.
    QTest::mouseClick(&canvas, Qt::LeftButton, {}, view(nodeRect(node(dense)).topLeft() + QPointF(60, 12)).toPoint());
    QTest::keyClick(&canvas, Qt::Key_Delete);
    CHECK(doc.graph.nodes.size() == 2 && doc.graph.links.size() == 1);
    QTest::keyClick(&canvas, Qt::Key_Z, Qt::ControlModifier);
    CHECK(doc.graph.nodes.size() == 3 && doc.graph.links.size() == 3);

    // Pulling a link off an input and letting go in empty space removes it.
    const QPointF port = view(inputPort(node(loss), 0));
    drag(&canvas, port, port + QPointF(0, 200));
    CHECK(doc.graph.links.size() == 2 && !doc.graph.linkInto(loss, 0));
    CHECK(!doc.analysis().valid());

    // A box around everything selects everything; a click on empty space clears it.
    const QRectF all = graphBounds(doc.graph).adjusted(-40, -40, 40, 40);
    drag(&canvas, view(all.topLeft()), view(all.bottomRight()));
    CHECK(doc.selection().size() == 3);
    QTest::mouseClick(&canvas, Qt::LeftButton, {}, view(all.bottomRight() + QPointF(30, 30)).toPoint());
    CHECK(doc.selection().empty());

    // Duplicate keeps the incoming link.
    doc.select({dense});
    QTest::keyClick(&canvas, Qt::Key_D, Qt::ControlModifier);
    CHECK(doc.graph.nodes.size() == 4 && doc.graph.links.size() == 3);

    // Linked back up, it trains; the learned weights make the project unsaved.
    QTest::keyClick(&canvas, Qt::Key_Z, Qt::ControlModifier);
    drag(&canvas, view(outputPort(node(dense), 0)), view(inputPort(node(loss), 0)));
    CHECK(doc.analysis().valid());
    doc.graph.training.epochs = 15;
    CHECK(doc.startTraining());
    for (int i = 0; i < 300 && doc.training(); ++i) QTest::qWait(20);
    QTest::qWait(250);
    const TrainReport r = doc.trainingReport();
    CHECK(!doc.training() && r.epoch == 15 && r.error.empty());
    CHECK(!r.loss.empty() && r.loss.back() <= r.loss.front());
    CHECK(doc.weights.count(dense) == 1);
    CHECK(doc.dirty());

    // A node dragged in from the library: its card follows the cursor over the canvas, and the
    // drop adds it there (the header under the cursor).
    {
        QMimeData mime;
        mime.setData(kNodeMime, "relu");
        const QPoint at(420, 380);
        const QImage before = canvas.grab().toImage();
        QDragEnterEvent enter(QPointF(at - QPoint(30, 30)), Qt::CopyAction, &mime, Qt::LeftButton, {});
        QApplication::sendEvent(&canvas, &enter);
        CHECK(enter.isAccepted());
        QDragMoveEvent move(QPointF(at), Qt::CopyAction, &mime, Qt::LeftButton, {});
        QApplication::sendEvent(&canvas, &move);
        const QImage during = canvas.grab().toImage();
        if (const std::string shot = std::getenv("LOGIEVO_UI_SHOT") ? std::getenv("LOGIEVO_UI_SHOT") : ""; !shot.empty())
            during.save(QString::fromStdString(shot));
        int changed = 0;
        for (int y = at.y() - 10; y < at.y() + 40; ++y)
            for (int x = at.x() - 90; x < at.x() + 90; ++x) changed += before.pixel(x, y) != during.pixel(x, y);
        CHECK(changed > 2000);
        const size_t count = doc.graph.nodes.size();
        QDropEvent drop(QPointF(at), Qt::CopyAction, &mime, Qt::LeftButton, {});
        QApplication::sendEvent(&canvas, &drop);
        CHECK(doc.graph.nodes.size() == count + 1 && doc.graph.nodes.back().type == "relu");
        CHECK(nodeRect(doc.graph.nodes.back()).contains(canvas.toScene(QPointF(at))));
    }
    if (failures == 0) std::printf("all ui tests passed\n");
    return failures == 0 ? 0 : 1;
}
