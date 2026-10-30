#include "app/Gpu.h"

#include "core/Parallel.h"

#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QSurfaceFormat>
#include <QThread>

#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <mutex>

namespace logievo::gpu {

namespace {

// One 16 × 16 tile of C per work group, A and B read through shared memory a tile at a time.
const char* kShader = R"(#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer C { float c[]; };
uniform ivec3 dims;   // n, k, m
uniform ivec2 trans;  // A, B stored transposed
shared float ta[16][16];
shared float tb[16][16];
void main() {
    int row = int(gl_GlobalInvocationID.y), col = int(gl_GlobalInvocationID.x);
    int lr = int(gl_LocalInvocationID.y), lc = int(gl_LocalInvocationID.x);
    int n = dims.x, k = dims.y, m = dims.z;
    float sum = 0.0;
    for (int t = 0; t < k; t += 16) {
        int ac = t + lc, br = t + lr;
        ta[lr][lc] = (row < n && ac < k) ? (trans.x == 1 ? a[ac * n + row] : a[row * k + ac]) : 0.0;
        tb[lr][lc] = (br < k && col < m) ? (trans.y == 1 ? b[col * k + br] : b[br * m + col]) : 0.0;
        barrier();
        for (int i = 0; i < 16; ++i) sum += ta[lr][i] * tb[i][lc];
        barrier();
    }
    if (row < n && col < m) c[row * m + col] = sum;
}
)";

struct Job {
    const float *a, *b;
    float* c;
    int64_t n, k, m;
    bool ta, tb;
    std::promise<bool> done;
};

class Worker : public QThread {
public:
    explicit Worker(QOffscreenSurface* surface) : surface_(surface) {}

    Info info() { return ready_.get_future().get(); }

    bool multiply(const float* a, const float* b, float* c, int64_t n, int64_t k, int64_t m, bool ta, bool tb) {
        // Too big for one buffer (or one dispatch): the processor does it.
        if (n * k > (int64_t(1) << 28) || k * m > (int64_t(1) << 28) || n * m > (int64_t(1) << 28) || n > 65535 * 16 || m > 65535 * 16) return false;
        auto job = std::make_shared<Job>(Job{a, b, c, n, k, m, ta, tb, {}});
        std::future<bool> result = job->done.get_future();
        {
            std::lock_guard lock(mutex_);
            if (quit_) return false;
            jobs_.push_back(job);
        }
        cv_.notify_one();
        return result.get();
    }

    void quit() {
        {
            std::lock_guard lock(mutex_);
            quit_ = true;
        }
        cv_.notify_all();
        wait();
    }

protected:
    void run() override {
        QOpenGLContext context;
        QSurfaceFormat format = surface_->format();
        context.setFormat(format);
        Info info;
        if (!context.create() || !context.makeCurrent(surface_)) {
            info.error = QObject::tr("No OpenGL graphics card could be opened");
            ready_.set_value(info);
            return;
        }
        auto* f = context.extraFunctions();
        info.device = QString::fromLatin1(reinterpret_cast<const char*>(f->glGetString(GL_RENDERER)));
        const QSurfaceFormat got = context.format();
        if (got.majorVersion() * 10 + got.minorVersion() < 43) {
            info.error = QObject::tr("%1 has OpenGL %2.%3; matrix products on it need 4.3").arg(info.device).arg(got.majorVersion()).arg(got.minorVersion());
            ready_.set_value(info);
            return;
        }
        // The compute program.
        const GLuint shader = f->glCreateShader(GL_COMPUTE_SHADER);
        f->glShaderSource(shader, 1, &kShader, nullptr);
        f->glCompileShader(shader);
        GLint ok = 0;
        f->glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        const GLuint program = f->glCreateProgram();
        if (ok) {
            f->glAttachShader(program, shader);
            f->glLinkProgram(program);
            f->glGetProgramiv(program, GL_LINK_STATUS, &ok);
        }
        if (!ok) {
            char log[512] = {};
            f->glGetShaderInfoLog(shader, sizeof log, nullptr, log);
            info.error = QObject::tr("The graphics driver did not take the compute program: %1").arg(QString::fromLatin1(log).trimmed());
            ready_.set_value(info);
            return;
        }
        const GLint dims = f->glGetUniformLocation(program, "dims"), trans = f->glGetUniformLocation(program, "trans");
        GLuint buffers[3];
        f->glGenBuffers(3, buffers);
        int64_t capacity[3] = {0, 0, 0};
        info.available = true;
        ready_.set_value(info);

        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return quit_ || !jobs_.empty(); });
                if (quit_) break;
                job = jobs_.front();
                jobs_.pop_front();
            }
            const int64_t sizes[3] = {job->n * job->k, job->k * job->m, job->n * job->m};
            const float* sources[2] = {job->a, job->b};
            for (int i = 0; i < 3; ++i) {
                f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[i]);
                const GLsizeiptr bytes = static_cast<GLsizeiptr>(sizes[i] * 4);
                if (sizes[i] > capacity[i]) {
                    f->glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, i < 2 ? sources[i] : nullptr, GL_DYNAMIC_DRAW);
                    capacity[i] = sizes[i];
                } else if (i < 2) {
                    f->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, sources[i]);
                }
                f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, static_cast<GLuint>(i), buffers[i]);
            }
            f->glUseProgram(program);
            f->glUniform3i(dims, static_cast<GLint>(job->n), static_cast<GLint>(job->k), static_cast<GLint>(job->m));
            f->glUniform2i(trans, job->ta ? 1 : 0, job->tb ? 1 : 0);
            f->glDispatchCompute(static_cast<GLuint>((job->m + 15) / 16), static_cast<GLuint>((job->n + 15) / 16), 1);
            f->glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
            f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            const void* mapped = f->glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(sizes[2] * 4), GL_MAP_READ_BIT);
            const bool done = mapped && f->glGetError() == GL_NO_ERROR;
            if (mapped) {
                std::memcpy(job->c, mapped, static_cast<size_t>(sizes[2] * 4));
                f->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            }
            job->done.set_value(done);
        }
        // Whatever still waits goes back to the processor.
        std::lock_guard lock(mutex_);
        for (auto& job : jobs_) job->done.set_value(false);
        jobs_.clear();
        f->glDeleteBuffers(3, buffers);
        f->glDeleteProgram(program);
        f->glDeleteShader(shader);
        context.doneCurrent();
    }

private:
    QOffscreenSurface* surface_;
    std::promise<Info> ready_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> jobs_;
    bool quit_ = false;
};

struct State {
    std::unique_ptr<QOffscreenSurface> surface;
    std::unique_ptr<Worker> worker;
    Info info;
    bool probed = false, on = false;
};

State& state() {
    static State s;
    return s;
}

} // namespace

Info probe() {
    State& s = state();
    if (s.probed) return s.info;
    s.probed = true;
    // The surface is made here, on the interface thread (as Qt asks); the context on the worker.
    QSurfaceFormat format;
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setRenderableType(QSurfaceFormat::OpenGL);
    s.surface = std::make_unique<QOffscreenSurface>();
    s.surface->setFormat(format);
    s.surface->create();
    if (!s.surface->isValid()) {
        s.info.error = QObject::tr("This system offers no OpenGL surface");
        return s.info;
    }
    s.worker = std::make_unique<Worker>(s.surface.get());
    s.worker->start();
    s.info = s.worker->info();
    if (!s.info.available) {
        s.worker->wait();
        s.worker.reset();
    }
    return s.info;
}

bool enable(bool on) {
    State& s = state();
    if (on && !probe().available) on = false;
    s.on = on;
    if (on) {
        Worker* w = s.worker.get();
        setMatmulKernel([w](const float* a, const float* b, float* c, int64_t n, int64_t k, int64_t m, bool ta, bool tb) {
            return w->multiply(a, b, c, n, k, m, ta, tb);
        });
    } else {
        setMatmulKernel({});
    }
    return on;
}

bool enabled() { return state().on; }

void shutdown() {
    State& s = state();
    setMatmulKernel({});
    s.on = false;
    if (s.worker) {
        s.worker->quit();
        s.worker.reset();
    }
    s.surface.reset();
}

} // namespace logievo::gpu
