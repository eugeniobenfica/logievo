#pragma once

// The language of Code nodes: a small formula language applied value by value, with its own
// derivatives (forward-mode, dual numbers), so a node the user writes trains like any other.
//
//     t = sigmoid(b)          // lines (or ';') name values for the next ones
//     a * t + (1 - t) * c     // the last line is what the node gives
//
// Inputs are a, b, c, d (x is a). Numbers, + - * / ^, comparisons (1 or 0), pi, e and
// abs exp log sqrt sin cos tan tanh sigmoid relu softplus sign floor min max pow clamp if.
// "#" or "//" start a comment.

#include <string>
#include <vector>

namespace logievo {

class Expression {
public:
    static constexpr int kMaxInputs = 4;

    // False with the reason (and the line) when the code does not read.
    bool compile(const std::string& code, int inputs, std::string* error = nullptr);
    bool uses(int input) const { return input >= 0 && input < kMaxInputs && uses_[input]; }
    // The value at `in` (one value per input); `grad` (when given) gets d value / d input.
    float eval(const float* in, float* grad = nullptr) const;

private:
    struct Op {
        int code = 0; // see Expression.cpp
        int arg = 0;
        float value = 0.0f;
    };
    std::vector<Op> program_;
    int slots_ = 0;
    bool uses_[kMaxInputs] = {};
};

} // namespace logievo
