#include "app/Preferences.h"

#include "app/Gpu.h"
#include "app/Settings.h"
#include "bestyle/BeStyle.h"
#include "core/Parallel.h"

#include <QApplication>
#include <QFont>

#include <algorithm>

namespace logievo {

namespace theme = bestyle::theme;

void applyLook() {
    const Settings& s = settings();
    theme::Palette p = s.theme == "light" ? theme::lightPalette() : s.theme == "midnight" ? theme::midnightPalette() : theme::darkPalette();
    p.name = s.theme;
    // Logievo's accent is white; on the light theme, its opposite.
    bestyle::Rgba accent = s.theme == "light" ? bestyle::rgba(24, 24, 27) : bestyle::rgba(255, 255, 255);
    if (bestyle::Rgba custom = 0; !s.accent.empty() && theme::fromHex(s.accent, custom)) accent = custom | 0xFF000000u;
    p.accent = accent;
    p.onAccent = 0; // worked out from the accent: dark text on a light one
    p.focus = theme::withAlpha(accent, 170);
    p.selection = theme::withAlpha(accent, s.theme == "light" ? 40 : 46);
    p.playhead = p.keyframe = accent;
    p.rounding = static_cast<float>(std::clamp(s.rounding, 0.0, 12.0));
    QFont font(QString::fromStdString(s.font.empty() ? "Inter" : s.font));
    font.setPointSizeF(9.5 * std::clamp(s.textScale, 0.75, 1.6));
    font.setHintingPreference(QFont::PreferNoHinting);
    QApplication::setFont(font);
    theme::apply(p); // tells every panel (bestyle::Appearance)
}

void applyPerformance() {
    const Settings& s = settings();
    setThreadCount(s.threads);
    gpu::enable(s.gpu);
}

} // namespace logievo
