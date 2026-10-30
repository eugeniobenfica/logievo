#pragma once

// The Logievo mark: a folded "L", the Logievo letter, in one color. The upright stroke is
// solid and the foot, past the fold, a shade lighter, which gives it its depth.

#include "bestyle/Draw.h"

#include <QIcon>
#include <QImage>
#include <QPainter>

namespace logievo::ui {

// The mark in a square of side 2 * radius around `center` (the bestyle::Icon::Logo painter).
void drawLogo(QPainter& p, bestyle::Pt center, float radius, bestyle::Rgba color);
// The app icon: the white mark on a dark rounded square.
QImage appIconImage(int size);
QIcon appIcon();

} // namespace logievo::ui
