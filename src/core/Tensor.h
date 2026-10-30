#pragma once

// Tensors for the engine, in plain C++: a shape and its values (float, row-major). Shapes
// broadcast like NumPy's: aligned from the right, a size of 1 stretches to the other's.

#include <cstdint>
#include <string>
#include <vector>

namespace logievo {

using Shape = std::vector<int64_t>;
// In shapes the analysis works out, the batch dimension (how many examples go through at
// once) is not known yet: it is written as kBatch and shown as "B".
inline constexpr int64_t kBatch = -1;

int64_t elementCount(const Shape& shape); // 1 for a scalar (empty shape)

struct Tensor {
    Shape shape;
    std::vector<float> v;

    Tensor() = default;
    explicit Tensor(Shape s, float fill = 0.0f) : shape(std::move(s)), v(static_cast<size_t>(elementCount(shape)), fill) {}
    int64_t size() const { return static_cast<int64_t>(v.size()); }
    int rank() const { return static_cast<int>(shape.size()); }
    float& operator[](int64_t i) { return v[static_cast<size_t>(i)]; }
    float operator[](int64_t i) const { return v[static_cast<size_t>(i)]; }
};

// The broadcast shape of a and b; false when they do not fit. Sizes may be kBatch: it fits
// itself and 1.
bool broadcastShapes(const Shape& a, const Shape& b, Shape& out);
// For each value of a tensor shaped `out`, the index of the value of a tensor shaped `in` that
// broadcasts to it.
std::vector<int64_t> broadcastIndex(const Shape& out, const Shape& in);
// A gradient for a broadcast result, summed back to the shape of the input that broadcast.
Tensor reduceTo(const Tensor& grad, const Shape& shape);

// Value by value with broadcasting.
template <class F>
Tensor binary(const Tensor& a, const Tensor& b, F f) {
    if (a.shape == b.shape) {
        Tensor out(a.shape);
        for (int64_t i = 0; i < out.size(); ++i) out[i] = f(a[i], b[i]);
        return out;
    }
    Shape shape;
    broadcastShapes(a.shape, b.shape, shape);
    Tensor out(shape);
    const auto ia = broadcastIndex(shape, a.shape), ib = broadcastIndex(shape, b.shape);
    for (int64_t i = 0; i < out.size(); ++i) out[i] = f(a[ia[static_cast<size_t>(i)]], b[ib[static_cast<size_t>(i)]]);
    return out;
}

template <class F>
Tensor unary(const Tensor& a, F f) {
    Tensor out(a.shape);
    for (int64_t i = 0; i < out.size(); ++i) out[i] = f(a[i]);
    return out;
}

// Matrix product of the last two dimensions, the ones before broadcast (a batch of matrices).
// `ta` / `tb`: use a / b transposed (their last two dimensions swapped).
Tensor matmul(const Tensor& a, const Tensor& b, bool ta = false, bool tb = false);
// The last two dimensions swapped.
Tensor transposeLast(const Tensor& a);

} // namespace logievo
