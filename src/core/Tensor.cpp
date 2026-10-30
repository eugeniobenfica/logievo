#include "core/Tensor.h"

#include "core/Parallel.h"

#include <algorithm>

namespace logievo {

int64_t elementCount(const Shape& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

bool broadcastShapes(const Shape& a, const Shape& b, Shape& out) {
    const size_t rank = std::max(a.size(), b.size());
    out.assign(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t x = i < a.size() ? a[a.size() - 1 - i] : 1;
        const int64_t y = i < b.size() ? b[b.size() - 1 - i] : 1;
        int64_t& o = out[rank - 1 - i];
        if (x == y) o = x;
        else if (x == 1) o = y;
        else if (y == 1) o = x;
        else return false;
    }
    return true;
}

std::vector<int64_t> broadcastIndex(const Shape& out, const Shape& in) {
    const size_t rank = out.size();
    // The input's strides lined up with the output's dimensions; 0 where it broadcasts.
    std::vector<int64_t> stride(rank, 0);
    int64_t s = 1;
    for (size_t i = 0; i < in.size(); ++i) {
        const size_t d = in.size() - 1 - i;
        const size_t o = rank - 1 - i;
        stride[o] = in[d] == 1 ? 0 : s;
        s *= in[d];
    }
    const int64_t n = elementCount(out);
    std::vector<int64_t> map(static_cast<size_t>(n));
    std::vector<int64_t> counter(rank, 0);
    int64_t offset = 0;
    for (int64_t i = 0; i < n; ++i) {
        map[static_cast<size_t>(i)] = offset;
        // Odometer: step the last dimension, carry into the ones before.
        for (size_t k = rank; k-- > 0;) {
            if (++counter[k] < out[k]) {
                offset += stride[k];
                break;
            }
            offset -= stride[k] * (out[k] - 1);
            counter[k] = 0;
        }
    }
    return map;
}

Tensor reduceTo(const Tensor& grad, const Shape& shape) {
    if (grad.shape == shape) return grad;
    Tensor out(shape);
    const auto map = broadcastIndex(grad.shape, shape);
    for (int64_t i = 0; i < grad.size(); ++i) out[map[static_cast<size_t>(i)]] += grad[i];
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b, bool ta, bool tb) {
    const size_t ra = a.shape.size(), rb = b.shape.size();
    const int64_t n = ta ? a.shape[ra - 1] : a.shape[ra - 2];
    const int64_t k = ta ? a.shape[ra - 2] : a.shape[ra - 1];
    const int64_t m = tb ? b.shape[rb - 2] : b.shape[rb - 1];
    const Shape batchA(a.shape.begin(), a.shape.end() - 2), batchB(b.shape.begin(), b.shape.end() - 2);
    Shape batch;
    broadcastShapes(batchA, batchB, batch);
    Shape shape = batch;
    shape.push_back(n);
    shape.push_back(m);
    Tensor out(shape);
    const auto ia = broadcastIndex(batch, batchA), ib = broadcastIndex(batch, batchB);
    const int64_t sizeA = a.shape[ra - 2] * a.shape[ra - 1], sizeB = b.shape[rb - 2] * b.shape[rb - 1];
    const int64_t mats = static_cast<int64_t>(ia.size());
    // Big products go to the graphics card when one is set up (Settings › Performance).
    std::vector<bool> done(static_cast<size_t>(mats), false);
    for (int64_t bi = 0; bi < mats; ++bi)
        done[static_cast<size_t>(bi)] = runMatmulKernel(a.v.data() + ia[static_cast<size_t>(bi)] * sizeA, b.v.data() + ib[static_cast<size_t>(bi)] * sizeB,
                                                        out.v.data() + bi * n * m, n, k, m, ta, tb);
    // The rest here, rows shared among the processor's threads.
    parallelFor(mats * n, k * m, [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; ++r) {
            const int64_t bi = r / n, i = r % n;
            if (done[static_cast<size_t>(bi)]) continue;
            const float* A = a.v.data() + ia[static_cast<size_t>(bi)] * sizeA;
            const float* B = b.v.data() + ib[static_cast<size_t>(bi)] * sizeB;
            float* row = out.v.data() + bi * n * m + i * m;
            for (int64_t p = 0; p < k; ++p) {
                const float x = ta ? A[p * n + i] : A[i * k + p];
                if (x == 0.0f) continue;
                if (tb)
                    for (int64_t j = 0; j < m; ++j) row[j] += x * B[j * k + p];
                else {
                    const float* brow = B + p * m;
                    for (int64_t j = 0; j < m; ++j) row[j] += x * brow[j];
                }
            }
        }
    });
    return out;
}

Tensor transposeLast(const Tensor& a) {
    Shape shape = a.shape;
    const size_t r = shape.size();
    std::swap(shape[r - 1], shape[r - 2]);
    Tensor out(shape);
    const int64_t rows = a.shape[r - 2], cols = a.shape[r - 1], mats = a.size() / std::max<int64_t>(1, rows * cols);
    for (int64_t b = 0; b < mats; ++b)
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < cols; ++j) out[b * rows * cols + j * rows + i] = a[b * rows * cols + i * cols + j];
    return out;
}

} // namespace logievo
