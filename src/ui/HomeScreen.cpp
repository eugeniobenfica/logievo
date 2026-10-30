#include "ui/HomeScreen.h"

#include "app/Platform.h"
#include "bestyle/BeStyle.h"
#include "core/Analysis.h"
#include "core/Serialization.h"
#include "core/Templates.h"
#include "ui/GraphPreview.h"
#include "ui/Logo.h"
#include "ui/Windows.h"

#include <QDialog>
#include <QDialogButtonBox>
#include <QFileSystemWatcher>
#include <QFontMetrics>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>

namespace logievo::ui {

namespace theme = bestyle::theme;
using bestyle::Icon;
using bestyle::Pt;
using bestyle::Rgba;
using bestyle::Tone;

namespace {

constexpr int kCardMin = 240, kGap = 20;
constexpr const char* kSupportUrl = "https://buymeacoffee.com/eugeniobenfica";

QString relativeTime(int64_t unixSeconds) {
    const int64_t d = std::max<int64_t>(0, unixNow() - unixSeconds);
    if (d < 60) return QObject::tr("just now");
    if (d < 3600) return QObject::tr("%1 min ago").arg(d / 60);
    if (d < 86400) return QObject::tr("%1 h ago").arg(d / 3600);
    if (d < 86400 * 30) return d < 86400 * 2 ? QObject::tr("1 day ago") : QObject::tr("%1 days ago").arg(d / 86400);
    char buf[32];
    const std::time_t t = static_cast<std::time_t>(unixSeconds);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&t));
    return buf;
}

int64_t modified(const std::filesystem::path& f) {
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(f, ec);
    if (ec) return 0;
    using namespace std::chrono;
    // File clock -> system clock through "now" (std::clock_cast is missing from libc++).
    const auto sys = system_clock::now() + duration_cast<system_clock::duration>(t - std::filesystem::file_time_type::clock::now());
    return duration_cast<seconds>(sys.time_since_epoch()).count();
}

QString sanitizeFileName(QString name) {
    for (QChar& ch : name)
        if (QString("<>:\"/\\|?*").contains(ch) || ch.unicode() < 32) ch = '_';
    name = name.trimmed();
    while (name.endsWith('.')) name.chop(1);
    return name.isEmpty() ? QObject::tr("Untitled") : name;
}

std::filesystem::path uniqueProjectPath(const std::filesystem::path& folder, const QString& name) {
    const QString base = sanitizeFileName(name);
    std::error_code ec;
    auto candidate = folder / platform::pathOf(base + kProjectExtension);
    for (int i = 2; std::filesystem::exists(candidate, ec); ++i) candidate = folder / platform::pathOf(base + " " + QString::number(i) + kProjectExtension);
    return candidate;
}

// A label in one of the theme's tones, larger and bold when asked.
QLabel* styled(const QString& text, Tone tone, float scale = 1.0f, bool bold = false) {
    auto* l = new QLabel(text);
    bestyle::setTone(l, tone);
    if (scale != 1.0f || bold) {
        QFont f = l->font();
        f.setPointSizeF(f.pointSizeF() * scale);
        if (bold) f.setWeight(QFont::Bold);
        l->setFont(f);
    }
    return l;
}

// A template in the New Project dialog: a picture of its graph and its name; one is picked.
class TemplateCard : public QAbstractButton {
public:
    TemplateCard(const QString& name, const Graph& graph) {
        setText(name);
        setCheckable(true);
        setAutoExclusive(true);
        setCursor(Qt::PointingHandCursor);
        setFixedSize(156, 122);
        const qreal dpr = devicePixelRatioF();
        if (!graph.nodes.empty()) {
            thumbnail_ = renderPreview(graph, QSize(static_cast<int>(156 * dpr), static_cast<int>(88 * dpr)));
            thumbnail_.setDevicePixelRatio(dpr);
        }
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        bestyle::Draw d(p);
        const float w = static_cast<float>(width()), h = static_cast<float>(height()), thumbH = 88.0f;
        const bool hot = underMouse();
        d.rectFilled({0, 0}, {w, h}, hot && !isChecked() ? theme::kBgHover : theme::kBgPanel, 8.0f);
        QPainterPath clip;
        clip.addRoundedRect(QRectF(0, 0, w, thumbH + 8), 8, 8);
        p.save();
        p.setClipPath(clip);
        p.setClipRect(QRectF(0, 0, w, thumbH), Qt::IntersectClip);
        p.fillRect(QRectF(0, 0, w, thumbH), theme::q(theme::kBgDeep));
        if (!thumbnail_.isNull()) p.drawImage(QRectF(0, 0, w, thumbH), thumbnail_);
        else drawLogo(p, {w * 0.5f, thumbH * 0.5f}, 16.0f, theme::kTextFaint);
        p.restore();
        d.rect({0.5f, 0.5f}, {w - 0.5f, h - 0.5f}, isChecked() ? theme::kAccent : hot ? theme::kTextFaint : theme::kBorder, 8.0f,
               isChecked() ? 2.0f : 1.0f);
        QFont f = font();
        if (isChecked()) f.setWeight(QFont::DemiBold);
        p.setFont(f);
        p.setPen(theme::q(isChecked() ? theme::kText : theme::kTextDim));
        p.drawText(QRectF(10, thumbH, w - 20, h - thumbH), Qt::AlignCenter, QFontMetrics(f).elidedText(text(), Qt::ElideRight, static_cast<int>(w) - 20));
    }
    void enterEvent(QEnterEvent*) override { update(); }
    void leaveEvent(QEvent*) override { update(); }

private:
    QImage thumbnail_;
};

// A project: thumbnail, name and details; click to open, "..." (or right-click) for more.
class ProjectCard : public QWidget {
public:
    ProjectCard(QString name, QString meta, const QImage* thumbnail, std::function<void()> open, std::function<void(QPoint)> menu)
        : name_(std::move(name)), meta_(std::move(meta)), thumbnail_(thumbnail), open_(std::move(open)), menu_(std::move(menu)) {
        setMouseTracking(true);
        setCursor(Qt::PointingHandCursor);
        setToolTip(name_);
    }
    void setCardWidth(int w) { setFixedSize(w, w * 9 / 16 + 62); }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        bestyle::Draw d(p);
        const bool hot = underMouse();
        const float w = static_cast<float>(width()), h = static_cast<float>(height());
        const float thumbH = w * 9.0f / 16.0f;
        d.rectFilled({0, 0}, {w, h}, hot ? theme::kBgHover : theme::kBgPanel, 8.0f);
        // Thumbnail with rounded top corners.
        QPainterPath clip;
        clip.addRoundedRect(QRectF(0, 0, w, thumbH + 8), 8, 8);
        p.save();
        p.setClipPath(clip);
        p.setClipRect(QRectF(0, 0, w, thumbH), Qt::IntersectClip);
        if (thumbnail_ && !thumbnail_->isNull()) p.drawImage(QRectF(0, 0, w, thumbH), *thumbnail_);
        else {
            p.fillRect(QRectF(0, 0, w, thumbH), theme::q(theme::kBgDeep));
            drawLogo(p, {w * 0.5f, thumbH * 0.5f}, 18.0f, theme::kTextFaint);
        }
        p.restore();
        d.rect({0.5f, 0.5f}, {w - 0.5f, h - 0.5f}, hot ? theme::kAccent : theme::kBorder, 8.0f, hot ? 1.5f : 1.0f);
        if (hot) {
            const QRectF r = dotsRect();
            d.rectFilled(Pt(r.topLeft()), Pt(r.bottomRight()), r.contains(mapFromGlobal(QCursor::pos())) ? theme::kBgHover : theme::shade(160), 6.0f);
            for (int i = -1; i <= 1; ++i) d.circleFilled({static_cast<float>(r.center().x()) + i * 6.0f, static_cast<float>(r.center().y())}, 2.0f, theme::kText);
        }
        p.setClipRect(QRectF(12, 0, w - 24, h));
        QFont bold = font();
        bold.setWeight(QFont::DemiBold);
        p.setFont(bold);
        const int room = static_cast<int>(w) - 28;
        d.text({14, thumbH + 12}, theme::kText, QFontMetrics(bold).elidedText(name_, Qt::ElideRight, room));
        p.setFont(font());
        d.text({14, thumbH + 34}, theme::kTextFaint, fontMetrics().elidedText(meta_, Qt::ElideRight, room));
    }
    void mouseMoveEvent(QMouseEvent*) override { update(); }
    void enterEvent(QEnterEvent*) override { update(); }
    void leaveEvent(QEvent*) override { update(); }
    void mouseReleaseEvent(QMouseEvent* e) override {
        if (!rect().contains(e->position().toPoint())) return;
        if (e->button() == Qt::RightButton || (e->button() == Qt::LeftButton && dotsRect().contains(e->position())))
            menu_(e->globalPosition().toPoint());
        else if (e->button() == Qt::LeftButton) open_();
    }

private:
    QRectF dotsRect() const { return {width() - 28.0 - 8, 8, 28, 28}; }
    QString name_, meta_;
    const QImage* thumbnail_;
    std::function<void()> open_;
    std::function<void(QPoint)> menu_;
};

// The Logievo mark as a widget, in the accent (white).
class LogoMark : public QWidget {
public:
    explicit LogoMark(int size) { setFixedSize(size, size); }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        drawLogo(p, {width() * 0.5f, height() * 0.5f}, width() * 0.48f, theme::kAccent);
    }
};

// A framed box with a title, a line under it and a control on the right.
QFrame* bannerCard(const QString& title, const QString& detail, QWidget* control) {
    auto* card = new QFrame;
    card->setObjectName("banner");
    card->setStyleSheet(QString("QFrame#banner { background: %1; border: 1px solid %2; border-radius: 8px; }")
                            .arg(theme::q(theme::kBgPanel).name(), theme::q(theme::kBorder).name()));
    auto* row = new QHBoxLayout(card);
    row->setContentsMargins(18, 14, 18, 14);
    auto* text = new QVBoxLayout;
    text->setSpacing(2);
    text->addWidget(styled(title, Tone::Text, 1.0f, true));
    text->addWidget(styled(detail, Tone::TextDim));
    row->addLayout(text, 1);
    row->addWidget(control);
    return card;
}

} // namespace

HomeScreen::HomeScreen(QWidget* parent) : QWidget(parent) {
    setAutoFillBackground(true);
    auto* outer = new QHBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    buildUi();
    connect(&bestyle::Appearance::get(), &bestyle::Appearance::changed, this, [this] {
        const QString search = search_->text();
        buildUi();
        search_->setText(search);
        rebuildGrid();
    });
    // Projects saved, copied or removed in the projects folder show up by themselves.
    watcher_ = new QFileSystemWatcher(this);
    if (!settings().projectsFolder.empty()) watcher_->addPath(platform::qpath(settings().projectsFolder));
    auto* later = new QTimer(this);
    later->setSingleShot(true);
    later->setInterval(400);
    connect(watcher_, &QFileSystemWatcher::directoryChanged, later, qOverload<>(&QTimer::start));
    connect(later, &QTimer::timeout, this, [this] {
        if (isVisible()) refresh();
    });
    connect(&AppSignals::get(), &AppSignals::projectsFolderChanged, this, [this] {
        if (!watcher_->directories().isEmpty()) watcher_->removePaths(watcher_->directories());
        if (!settings().projectsFolder.empty()) watcher_->addPath(platform::qpath(settings().projectsFolder));
        const QString search = search_->text();
        buildUi();
        search_->setText(search);
        refresh();
    });
}

void HomeScreen::buildUi() {
    if (content_) {
        layout()->removeWidget(content_);
        content_->deleteLater();
    }
    content_ = new QWidget;
    layout()->addWidget(content_);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, theme::q(theme::kBgDeep));
    setPalette(pal);
    columns_ = 0;
    auto* row = new QHBoxLayout(content_);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(0);

    // --- sidebar -----------------------------------------------------------------------
    auto* sidebar = new QFrame;
    sidebar->setFixedWidth(270);
    sidebar->setAutoFillBackground(true);
    QPalette sp = sidebar->palette();
    sp.setColor(QPalette::Window, theme::q(theme::kBgPanel));
    sidebar->setPalette(sp);
    auto* side = new QVBoxLayout(sidebar);
    side->setContentsMargins(22, 26, 22, 22);
    side->setSpacing(8);
    auto* brand = new QHBoxLayout;
    brand->setSpacing(12);
    brand->addWidget(new LogoMark(40));
    auto* names = new QVBoxLayout;
    names->setSpacing(0);
    names->addWidget(styled("Logievo", Tone::Text, 1.6f, true));
    names->addWidget(styled(QString("v%1").arg(LOGIEVO_VERSION), Tone::TextFaint));
    brand->addLayout(names, 1);
    side->addLayout(brand);
    side->addSpacing(18);
    auto* create = bestyle::accentButton(tr("+  New Project"));
    create->setMinimumHeight(38);
    connect(create, &QPushButton::clicked, this, [this] { openNewProjectDialog(); });
    side->addWidget(create);
    auto* open = new QPushButton(tr("Open Project..."));
    open->setMinimumHeight(34);
    connect(open, &QPushButton::clicked, this, [this] {
        if (auto file = platform::openFileDialog(this, tr("Open Logievo Project"), tr("Logievo Project (*%1)").arg(kProjectExtension)))
            if (onOpen) onOpen(*file);
    });
    side->addWidget(open);
    side->addStretch(1);
    auto* line = new QFrame;
    line->setFrameShape(QFrame::HLine);
    bestyle::setTone(line, Tone::Border);
    side->addWidget(line);
    // The projects folder (a button of its own: it needs its words), then the small icons.
    auto* folder = new QPushButton(bestyle::makeIcon(Icon::Folder, theme::kTextDim, 24), tr("Projects Folder"));
    folder->setIconSize(QSize(24, 24));
    folder->setToolTip(settings().projectsFolder.empty() ? tr("Choose where your projects live") : platform::qpath(settings().projectsFolder));
    connect(folder, &QPushButton::clicked, this, [this, folder] {
        if (settings().projectsFolder.empty()) {
            chooseProjectsFolder();
            return;
        }
        QMenu menu;
        menu.addAction(bestyle::menuIcon(Icon::Folder, theme::kTextDim), tr("Show in File Manager"),
                       [] { platform::revealInFileManager(settings().projectsFolder); });
        menu.addAction(tr("Change Folder..."), [this] { chooseProjectsFolder(); });
        menu.exec(folder->mapToGlobal(QPoint(0, folder->height())));
    });
    side->addWidget(folder);
    side->addSpacing(2);
    auto* footer = new QHBoxLayout;
    footer->setSpacing(6);
    footer->addStretch(1);
    auto* coffee = new bestyle::IconButton(Icon::Coffee, tr("Support Logievo: buy a coffee"), 30);
    connect(coffee, &QAbstractButton::clicked, this, [] { platform::openUrl(kSupportUrl); });
    footer->addWidget(coffee);
    auto* gear = new bestyle::IconButton(Icon::Gear, tr("Settings (Ctrl+,)"), 30);
    connect(gear, &QAbstractButton::clicked, this, [] { openSettings(); });
    footer->addWidget(gear);
    auto* help = new bestyle::IconButton(Icon::Help, tr("Help, shortcuts, credits and license (F1)"), 30);
    connect(help, &QAbstractButton::clicked, this, [] { openHelp(HelpPage::About); });
    footer->addWidget(help);
    side->addLayout(footer);
    row->addWidget(sidebar);

    // --- projects ----------------------------------------------------------------------
    auto* main = new QVBoxLayout;
    main->setContentsMargins(36, 30, 36, 0);
    main->setSpacing(6);
    auto* head = new QHBoxLayout;
    auto* title = styled(tr("Projects"), Tone::Text, 1.8f, true);
    if (!settings().projectsFolder.empty())
        title->setToolTip(tr("Every project in %1, and the ones you opened from elsewhere").arg(platform::qpath(settings().projectsFolder)));
    head->addWidget(title);
    head->addStretch(1);
    search_ = new QLineEdit;
    search_->setPlaceholderText(tr("Search projects"));
    search_->setClearButtonEnabled(true);
    search_->setFixedWidth(260);
    connect(search_, &QLineEdit::textChanged, this, [this] { rebuildGrid(); });
    head->addWidget(search_);
    main->addLayout(head);

    errorRow_ = new QWidget;
    auto* er = new QHBoxLayout(errorRow_);
    er->setContentsMargins(0, 4, 0, 4);
    error_ = styled("", Tone::Danger);
    error_->setWordWrap(true);
    er->addWidget(error_, 1);
    auto* dismiss = new QPushButton(tr("Dismiss"));
    connect(dismiss, &QPushButton::clicked, errorRow_, &QWidget::hide);
    er->addWidget(dismiss);
    errorRow_->hide();
    main->addWidget(errorRow_);
    main->addSpacing(8);
    scroll_ = new QScrollArea;
    scroll_->setWidgetResizable(true);
    scroll_->setFrameShape(QFrame::NoFrame);
    scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll_->viewport()->setAutoFillBackground(false);
    scroll_->viewport()->installEventFilter(this); // its width decides the columns
    main->addWidget(scroll_, 1);
    row->addLayout(main, 1);
}

void HomeScreen::setError(const QString& message) {
    error_->setText(message);
    errorRow_->setVisible(!message.isEmpty());
}

void HomeScreen::refresh() {
    // Every project in the projects folder (and its subfolders), plus the ones opened from elsewhere.
    namespace fs = std::filesystem;
    entries_.clear();
    for (const RecentProject& r : settings().recents) {
        std::error_code ec;
        if (!fs::exists(r.path, ec)) continue;
        RecentProject e = r;
        e.lastOpened = std::max(r.lastOpened, modified(r.path));
        entries_.push_back(std::move(e));
    }
    const fs::path folder = settings().projectsFolder;
    std::error_code ec;
    int scanned = 0;
    if (!folder.empty())
        for (auto it = fs::recursive_directory_iterator(folder, fs::directory_options::skip_permission_denied, ec);
             !ec && it != fs::recursive_directory_iterator() && scanned < 500; it.increment(ec)) {
            if (it.depth() >= 3) it.disable_recursion_pending();
            std::error_code fe;
            if (!it->is_regular_file(fe) || it->path().extension() != kProjectExtension) continue;
            ++scanned;
            const fs::path path = it->path();
            if (std::any_of(entries_.begin(), entries_.end(), [&](const RecentProject& e) { return fs::equivalent(e.path, path, fe); })) continue;
            entries_.push_back({path, platform::qpath(path.stem()).toStdString(), modified(path)});
        }
    std::stable_sort(entries_.begin(), entries_.end(), [](const RecentProject& a, const RecentProject& b) { return a.lastOpened > b.lastOpened; });

    // Previews: only projects changed since the last time are loaded again.
    std::map<fs::path, Preview> previews;
    const qreal dpr = devicePixelRatioF();
    for (RecentProject& e : entries_) {
        const int64_t stamp = modified(e.path);
        if (auto old = previews_.find(e.path); old != previews_.end() && old->second.stamp == stamp) {
            previews[e.path] = std::move(old->second);
        } else {
            Preview pv;
            pv.stamp = stamp;
            Graph g;
            if (loadGraph(e.path, g)) {
                pv.thumbnail = renderPreview(g, QSize(static_cast<int>(400 * dpr), static_cast<int>(225 * dpr)));
                pv.graph = std::move(g);
            }
            previews[e.path] = std::move(pv);
        }
        if (const Preview& pv = previews[e.path]; pv.graph && !pv.graph->name.empty()) e.name = pv.graph->name;
    }
    previews_ = std::move(previews);
    rebuildGrid();
}

void HomeScreen::resizeEvent(QResizeEvent* e) {
    QWidget::resizeEvent(e);
    const int cols = std::max(1, (scroll_->viewport()->width() + kGap) / (kCardMin + kGap));
    if (cols != columns_) rebuildGrid();
}

bool HomeScreen::eventFilter(QObject* watched, QEvent* e) {
    // A rebuilt screen gets its final width after the grid was first laid out.
    if (scroll_ && watched == scroll_->viewport() && e->type() == QEvent::Resize) {
        const int cols = std::max(1, (scroll_->viewport()->width() + kGap) / (kCardMin + kGap));
        if (cols != columns_) QMetaObject::invokeMethod(this, [this] { rebuildGrid(); }, Qt::QueuedConnection);
    }
    return QWidget::eventFilter(watched, e);
}

void HomeScreen::rebuildGrid() {
    const int avail = std::max(kCardMin, scroll_->viewport()->width() - 2);
    columns_ = std::max(1, (avail + kGap) / (kCardMin + kGap));
    const int cardW = (avail - kGap * (columns_ - 1)) / columns_;

    auto* grid = new QWidget;
    auto* col = new QVBoxLayout(grid);
    col->setContentsMargins(0, 0, 0, 24);
    // No projects folder yet: a banner on top says what it is for, with the button that picks it.
    if (settings().projectsFolder.empty()) {
        auto* choose = bestyle::accentButton(tr("Choose Folder..."));
        connect(choose, &QPushButton::clicked, this, [this] { chooseProjectsFolder(); });
        col->addWidget(bannerCard(tr("Choose a folder for your projects"), tr("New projects are saved there, and every project in it shows here"), choose));
        col->addSpacing(14);
    }
    if (entries_.empty()) {
        if (!settings().projectsFolder.empty()) {
            col->addStretch(1);
            auto* t1 = styled(tr("No projects yet"), Tone::Text, 1.5f, true);
            t1->setAlignment(Qt::AlignCenter);
            col->addWidget(t1);
            auto* t2 = styled(tr("Design your first model in a few clicks."), Tone::TextDim);
            t2->setAlignment(Qt::AlignCenter);
            col->addWidget(t2);
            col->addSpacing(12);
            auto* b = bestyle::accentButton(tr("Create a project"));
            b->setFixedSize(200, 40);
            connect(b, &QPushButton::clicked, this, [this] { openNewProjectDialog(); });
            col->addWidget(b, 0, Qt::AlignHCenter);
            col->addStretch(2);
        } else {
            col->addStretch(1);
        }
        scroll_->setWidget(grid);
        return;
    }
    auto* cards = new QGridLayout;
    cards->setSpacing(kGap);
    cards->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    col->addLayout(cards);
    const QString search = search_->text();
    int shown = 0;
    for (const RecentProject& item : entries_) {
        const QString name = QString::fromStdString(item.name);
        if (!name.contains(search, Qt::CaseInsensitive)) continue;
        const auto it = previews_.find(item.path);
        const Preview* pv = it != previews_.end() ? &it->second : nullptr;
        QString meta;
        if (pv && pv->graph) {
            const Analysis a = analyze(*pv->graph);
            const int nodes = static_cast<int>(pv->graph->nodes.size());
            meta = relativeTime(item.lastOpened) + "  ·  " + (nodes == 1 ? tr("1 node") : tr("%1 nodes").arg(nodes)) + "  ·  " +
                   tr("%1 parameters").arg(QString::fromStdString(countText(a.totalParams)));
        } else {
            meta = tr("Unreadable file") + "  ·  " + relativeTime(item.lastOpened);
        }
        const RecentProject copy = item;
        auto* card = new ProjectCard(
            name, meta, pv ? &pv->thumbnail : nullptr,
            [this, copy] {
                if (onOpen) onOpen(copy.path);
            },
            [this, copy](QPoint at) {
                QMenu menu;
                menu.addAction(bestyle::menuIcon(Icon::Play, theme::kTextDim), tr("Open"), [this, copy] {
                    if (onOpen) onOpen(copy.path);
                });
                menu.addAction(bestyle::menuIcon(Icon::Folder, theme::kTextDim), tr("Show in Folder"), [copy] { platform::revealInFileManager(copy.path); });
                menu.addSeparator();
                menu.addAction(bestyle::menuIcon(Icon::Trash, theme::kDanger), tr("Delete Project..."), [this, copy] { confirmDelete(copy); });
                menu.exec(at);
            });
        card->setCardWidth(cardW);
        cards->addWidget(card, shown / columns_, shown % columns_);
        ++shown;
    }
    if (shown == 0) col->addWidget(styled(tr("No projects match \"%1\".").arg(search), Tone::TextDim));
    col->addStretch(1);
    scroll_->setWidget(grid);
}

bool HomeScreen::chooseProjectsFolder() {
    const auto current = settings().projectsFolder;
    const auto folder = platform::chooseFolderDialog(this, tr("Choose a Folder for Your Projects"), current.empty() ? platform::documentsDir() : current);
    if (!folder) return false;
    settings().projectsFolder = *folder;
    saveSettings();
    if (!watcher_->directories().isEmpty()) watcher_->removePaths(watcher_->directories());
    watcher_->addPath(platform::qpath(*folder));
    const QString search = search_->text();
    buildUi(); // the sidebar's tooltips name the folder
    search_->setText(search);
    refresh();
    return true;
}

void HomeScreen::confirmDelete(const RecentProject& item) {
#ifdef _WIN32
    const QString bin = tr("Recycle Bin");
#else
    const QString bin = tr("Trash");
#endif
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Delete Project"));
    dialog.setMinimumWidth(460);
    auto* col = new QVBoxLayout(&dialog);
    col->setContentsMargins(24, 22, 24, 18);
    col->setSpacing(12);
    auto* head = new QHBoxLayout;
    head->setSpacing(12);
    auto* icon = new QLabel;
    icon->setPixmap(bestyle::makeIcon(Icon::Trash, theme::kDanger, 26).pixmap(26, 26));
    head->addWidget(icon, 0, Qt::AlignTop);
    auto* title = styled(tr("Delete \"%1\"?").arg(QString::fromStdString(item.name)), Tone::Text, 1.25f, true);
    title->setWordWrap(true);
    head->addWidget(title, 1);
    col->addLayout(head);
    auto* where = styled(platform::qpath(item.path), Tone::TextFaint, 0.92f);
    where->setWordWrap(true);
    where->setTextInteractionFlags(Qt::TextSelectableByMouse);
    col->addWidget(where);
    auto* what = styled(tr("It goes to the %1, where you can still restore it.").arg(bin), Tone::TextDim);
    what->setWordWrap(true);
    col->addWidget(what);
    col->addSpacing(4);
    auto* buttons = new QHBoxLayout;
    buttons->addStretch();
    auto* keep = new QPushButton(tr("Cancel"));
    keep->setMinimumWidth(96);
    keep->setDefault(true);
    auto* del = new QPushButton(tr("Delete Project"));
    del->setProperty("danger", true);
    del->setAutoDefault(false);
    del->setMinimumWidth(120);
    connect(keep, &QPushButton::clicked, &dialog, &QDialog::reject);
    connect(del, &QPushButton::clicked, &dialog, &QDialog::accept);
    buttons->addWidget(keep);
    buttons->addWidget(del);
    col->addLayout(buttons);
    keep->setFocus();
    if (dialog.exec() != QDialog::Accepted) return;
    std::string err;
    std::error_code ec;
    if (!std::filesystem::exists(item.path, ec) || platform::moveToTrash(item.path, &err)) {
        removeRecent(item.path);
        saveSettings();
        previews_.erase(item.path);
        std::erase_if(entries_, [&](const RecentProject& e) { return e.path == item.path; });
    } else {
        setError(tr("Could not delete \"%1\": %2").arg(QString::fromStdString(item.name), QString::fromStdString(err)));
    }
    QMetaObject::invokeMethod(this, [this] { rebuildGrid(); }, Qt::QueuedConnection);
}

void HomeScreen::openNewProjectDialog(const std::string& templateId) {
    // New projects go to the projects folder: asked for first if there is none yet.
    if (settings().projectsFolder.empty() && !chooseProjectsFolder()) return;
    QDialog dialog(this);
    dialog.setWindowTitle(tr("New Project"));
    auto* form = new QFormLayout;
    form->setLabelAlignment(Qt::AlignLeft);
    form->setHorizontalSpacing(16);
    form->setVerticalSpacing(8);
    auto* name = new QLineEdit(tr("Untitled"));
    name->selectAll();
    form->addRow(tr("Name"), name);

    // Start from: a blank canvas or a template. The name follows the template until it is typed.
    auto* grid = new QGridLayout;
    grid->setSpacing(10);
    auto* about = styled(QString(), Tone::TextFaint, 0.92f);
    about->setWordWrap(true);
    about->setMinimumHeight(QFontMetrics(about->font()).height() * 2 + 4);
    std::string chosen = "blank";
    bool named = false;
    connect(name, &QLineEdit::textEdited, &dialog, [&named] { named = true; });
    const auto& list = projectTemplates();
    for (size_t i = 0; i < list.size(); ++i) {
        const ProjectTemplate& t = list[i];
        auto* card = new TemplateCard(tr(t.name.c_str()), makeTemplate(t.id, t.name));
        card->setToolTip(tr(t.description.c_str()));
        connect(card, &QAbstractButton::toggled, &dialog, [&, card, t](bool on) {
            if (!on) return;
            chosen = t.id;
            about->setText(tr(t.description.c_str()));
            if (!named) {
                name->setText(t.id == "blank" ? tr("Untitled") : card->text());
                name->selectAll();
            }
        });
        if (t.id == templateId || (i == 0 && templateId.empty())) card->setChecked(true);
        grid->addWidget(card, static_cast<int>(i / 4), static_cast<int>(i % 4));
    }
    auto* start = new QVBoxLayout;
    start->setSpacing(8);
    start->addLayout(grid);
    start->addWidget(about);
    form->addRow(tr("Start from"), start);
    std::filesystem::path folderPath = settings().projectsFolder;
    auto* folderRow = new QHBoxLayout;
    auto* folder = new QLineEdit(platform::qpath(folderPath));
    folder->setReadOnly(true);
    folderRow->addWidget(folder, 1);
    auto* change = new QPushButton(tr("Change..."));
    folderRow->addWidget(change);
    form->addRow(tr("Location"), folderRow);
    connect(change, &QPushButton::clicked, &dialog, [&] {
        if (auto f = platform::chooseFolderDialog(&dialog, tr("Choose where to save the project"), folderPath)) {
            folderPath = *f;
            folder->setText(platform::qpath(folderPath));
        }
    });

    auto* buttons = new QDialogButtonBox;
    QPushButton* createButton = buttons->addButton(tr("Create"), QDialogButtonBox::AcceptRole);
    createButton->setProperty("accent", true);
    createButton->setDefault(true);
    buttons->addButton(QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    auto* col = new QVBoxLayout(&dialog);
    col->setContentsMargins(24, 22, 24, 18);
    col->addLayout(form);
    col->addSpacing(10);
    col->addWidget(buttons);
    name->setFocus();
    if (dialog.exec() != QDialog::Accepted) return;

    Graph g = makeTemplate(chosen, name->text().trimmed().isEmpty() ? tr("Untitled").toStdString() : name->text().trimmed().toStdString());
    if (onCreate) onCreate(uniqueProjectPath(folderPath, QString::fromStdString(g.name)), std::move(g));
}

} // namespace logievo::ui
