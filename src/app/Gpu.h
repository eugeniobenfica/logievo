#pragma once

// GPU acceleration (Settings › Performance): matrix products on the graphics card, through an
// OpenGL 4.3 compute shader on a thread of its own. Installed as the engine's matmul kernel
// (core/Parallel.h); big products go there, everything else stays on the processor.

#include <QString>

namespace logievo::gpu {

struct Info {
    bool available = false;
    QString device; // "NVIDIA GeForce ..." (the OpenGL renderer)
    QString error;  // why it is not available
};

// Starts the graphics thread the first time (from the interface thread) and says what it found.
Info probe();
// Sends big matrix products to the graphics card, or stops; false when there is no card to use.
bool enable(bool on);
bool enabled();
// Stops the graphics thread (before the application quits).
void shutdown();

} // namespace logievo::gpu
