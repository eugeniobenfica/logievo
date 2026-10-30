#pragma once

// How the engine uses the machine: the processor's threads (a pool shared by every heavy
// operation) and, when one is installed, a graphics-card kernel for matrix products. Set from
// Settings › Performance; plain C++ (the GPU kernel itself lives in the app).

#include <cstdint>
#include <functional>

namespace logievo {

// 0: every thread the processor has.
void setThreadCount(int count);
int threadCount();    // what is used now
int hardwareThreads(); // what the processor has

// Runs fn(begin, end) over [0, count) split across the threads; `cost` (work per item, about one
// multiply-add each) keeps small jobs on the calling thread, where splitting would cost more.
void parallelFor(int64_t count, int64_t cost, const std::function<void(int64_t begin, int64_t end)>& fn);

// C = A × B for row-major n × k and k × m matrices (`ta` / `tb`: A / B are stored transposed).
// Returns false when it could not (the engine then does it on the processor).
using MatmulKernel = std::function<bool(const float* a, const float* b, float* c, int64_t n, int64_t k, int64_t m, bool ta, bool tb)>;
void setMatmulKernel(MatmulKernel kernel); // empty: none
// Products of at least this many multiply-adds go to the kernel (smaller ones are faster here).
inline constexpr int64_t kKernelMinWork = int64_t(1) << 20;
bool runMatmulKernel(const float* a, const float* b, float* c, int64_t n, int64_t k, int64_t m, bool ta, bool tb);

} // namespace logievo
