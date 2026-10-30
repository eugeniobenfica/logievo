#include "core/Parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace logievo {

namespace {

// Workers started once, as many as the processor has (minus the caller); how many a job uses is
// decided per call, so changing the setting never stops anything running.
class Pool {
public:
    Pool() {
        const int n = std::max(0, hardwareThreads() - 1);
        for (int i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::lock_guard lock(mutex_);
            quit_ = true;
        }
        cv_.notify_all();
        for (std::thread& t : workers_) t.join();
    }
    void push(std::function<void()> job) {
        {
            std::lock_guard lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        cv_.notify_one();
    }
    int size() const { return static_cast<int>(workers_.size()); }

private:
    void loop() {
        inWorker() = true;
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return quit_ || !jobs_.empty(); });
                if (quit_ && jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }

public:
    static bool& inWorker() {
        thread_local bool in = false;
        return in;
    }

private:
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> jobs_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool quit_ = false;
};

Pool& pool() {
    static Pool p;
    return p;
}

std::atomic<int> wanted{0};
std::mutex kernelMutex;
MatmulKernel kernel;

} // namespace

int hardwareThreads() { return std::max(1, static_cast<int>(std::thread::hardware_concurrency())); }

void setThreadCount(int count) { wanted = std::max(0, count); }

int threadCount() {
    const int w = wanted;
    return w <= 0 ? hardwareThreads() : std::min(w, hardwareThreads());
}

void parallelFor(int64_t count, int64_t cost, const std::function<void(int64_t, int64_t)>& fn) {
    if (count <= 0) return;
    constexpr int64_t kMinWorkPerThread = 32 * 1024;
    int threads = std::min<int64_t>(threadCount(), std::max<int64_t>(1, count * std::max<int64_t>(1, cost) / kMinWorkPerThread));
    threads = std::min<int64_t>(threads, count);
    // A job inside a job (or a pool too small) runs here.
    if (threads <= 1 || Pool::inWorker() || pool().size() == 0) {
        fn(0, count);
        return;
    }
    threads = std::min(threads, pool().size() + 1);
    const int64_t chunk = (count + threads - 1) / threads;
    std::mutex m;
    std::condition_variable done;
    int pending = 0;
    for (int t = 1; t < threads; ++t) {
        const int64_t begin = t * chunk, end = std::min(count, begin + chunk);
        if (begin >= end) break;
        {
            std::lock_guard lock(m);
            ++pending;
        }
        pool().push([&, begin, end] {
            fn(begin, end);
            std::lock_guard lock(m);
            if (--pending == 0) done.notify_one();
        });
    }
    fn(0, std::min(count, chunk)); // the caller takes the first share
    std::unique_lock lock(m);
    done.wait(lock, [&] { return pending == 0; });
}

void setMatmulKernel(MatmulKernel k) {
    std::lock_guard lock(kernelMutex);
    kernel = std::move(k);
}

bool runMatmulKernel(const float* a, const float* b, float* c, int64_t n, int64_t k, int64_t m, bool ta, bool tb) {
    MatmulKernel use;
    {
        std::lock_guard lock(kernelMutex);
        use = kernel;
    }
    return use && n * k * m >= kKernelMinWork && use(a, b, c, n, k, m, ta, tb);
}

} // namespace logievo
