#include "core/Expression.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>

namespace logievo {

namespace {

// Stack machine operations.
enum Code {
    Num,     // push value
    Input,   // push input `arg`
    Load,    // push local `arg`
    Store,   // pop into local `arg`
    Neg, Add, Sub, Mul, Div, Pow,
    Lt, Gt, Le, Ge, Eq, Ne,
    Fn1,     // function `arg` of one value
    Fn2,     // ... of two
    Clamp, If,
};

enum Fn { Abs, Exp, Log, Sqrt, Sin, Cos, Tan, Tanh, Sigmoid, Relu, Softplus, Sign, Floor, Min, Max, PowFn };

const std::map<std::string, std::pair<int, int>>& functions() { // name -> (Fn or special, arguments)
    static const std::map<std::string, std::pair<int, int>> f{
        {"abs", {Abs, 1}},       {"exp", {Exp, 1}},   {"log", {Log, 1}},     {"sqrt", {Sqrt, 1}},     {"sin", {Sin, 1}},
        {"cos", {Cos, 1}},       {"tan", {Tan, 1}},   {"tanh", {Tanh, 1}},   {"sigmoid", {Sigmoid, 1}}, {"relu", {Relu, 1}},
        {"softplus", {Softplus, 1}}, {"sign", {Sign, 1}}, {"floor", {Floor, 1}}, {"min", {Min, 2}},     {"max", {Max, 2}},
        {"pow", {PowFn, 2}},     {"clamp", {-1, 3}},  {"if", {-2, 3}},
    };
    return f;
}

struct Token {
    enum Kind { End, Number, Name, Symbol, Newline } kind = End;
    std::string text;
    float number = 0;
    int line = 1;
};

std::vector<Token> tokenize(const std::string& code, std::string& error) {
    std::vector<Token> out;
    int line = 1;
    for (size_t i = 0; i < code.size();) {
        const char c = code[i];
        if (c == '\n' || c == ';') {
            out.push_back({Token::Newline, ";", 0, line});
            if (c == '\n') ++line;
            ++i;
        } else if (c == '#' || (c == '/' && i + 1 < code.size() && code[i + 1] == '/')) {
            while (i < code.size() && code[i] != '\n') ++i;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
        } else if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && i + 1 < code.size() && std::isdigit(static_cast<unsigned char>(code[i + 1])))) {
            char* end = nullptr;
            const float v = std::strtof(code.c_str() + i, &end);
            out.push_back({Token::Number, code.substr(i, static_cast<size_t>(end - (code.c_str() + i))), v, line});
            i = static_cast<size_t>(end - code.c_str());
        } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t j = i;
            while (j < code.size() && (std::isalnum(static_cast<unsigned char>(code[j])) || code[j] == '_')) ++j;
            out.push_back({Token::Name, code.substr(i, j - i), 0, line});
            i = j;
        } else {
            static const char* two[] = {"<=", ">=", "==", "!=", "**"};
            std::string sym(1, c);
            for (const char* t : two)
                if (code.compare(i, 2, t) == 0) sym = t;
            if (std::string("+-*/^()<>,=!").find(c) == std::string::npos) {
                error = "Line " + std::to_string(line) + ": \"" + std::string(1, c) + "\" is not part of the language";
                return {};
            }
            out.push_back({Token::Symbol, sym == "**" ? "^" : sym, 0, line});
            i += sym.size();
        }
    }
    out.push_back({Token::End, "", 0, line});
    return out;
}

// Recursive descent: statement lines, then expressions by precedence.
class Parser {
public:
    Parser(std::vector<Token> tokens, int inputs, std::vector<int>& ops, std::vector<int>& args, std::vector<float>& values)
        : t_(std::move(tokens)), inputs_(inputs), ops_(ops), args_(args), values_(values) {}

    bool run(std::string& error, int& slots, bool* uses) {
        bool any = false;
        for (;;) {
            while (peek().kind == Token::Newline) ++p_;
            if (peek().kind == Token::End) break;
            const bool last = isLastStatement();
            // name = expression
            if (peek().kind == Token::Name && p_ + 1 < t_.size() && t_[p_ + 1].kind == Token::Symbol && t_[p_ + 1].text == "=") {
                const std::string name = peek().text;
                if (inputIndex(name) >= 0 || functions().count(name) || name == "pi" || name == "e") return fail(error, "\"" + name + "\" is taken: pick another name");
                p_ += 2;
                if (!expression()) return fail(error, error_);
                int& slot = locals_[name];
                if (!slot) slot = static_cast<int>(locals_.size());
                emit(Store, slot - 1);
                if (last) emit(Load, slot - 1);
                any = last;
            } else {
                if (!expression()) return fail(error, error_);
                if (!last) return fail(error, "Only the last line gives the result: name this value (t = ...) or remove it");
                any = true;
            }
            if (peek().kind != Token::Newline && peek().kind != Token::End) return fail(error, "Unexpected \"" + peek().text + "\"");
        }
        if (!any) return fail(error, "Write what the node gives, for example: a * 2");
        slots = static_cast<int>(locals_.size());
        for (int i = 0; i < Expression::kMaxInputs; ++i) uses[i] = uses_[i];
        return true;
    }

private:
    const Token& peek() const { return t_[std::min(p_, t_.size() - 1)]; }
    bool isLastStatement() const {
        for (size_t i = p_; i < t_.size(); ++i)
            if (t_[i].kind == Token::Newline) {
                for (size_t j = i; j < t_.size(); ++j)
                    if (t_[j].kind != Token::Newline && t_[j].kind != Token::End) return false;
                return true;
            }
        return true;
    }
    bool accept(const char* sym) {
        if (peek().kind == Token::Symbol && peek().text == sym) {
            ++p_;
            return true;
        }
        return false;
    }
    void emit(int op, int arg = 0, float value = 0) {
        ops_.push_back(op);
        args_.push_back(arg);
        values_.push_back(value);
    }
    bool fail(std::string& error, const std::string& message) {
        error = "Line " + std::to_string(peek().line) + ": " + message;
        return false;
    }
    bool err(const std::string& m) {
        error_ = m;
        return false;
    }
    int inputIndex(const std::string& name) const {
        if (name == "x") return 0;
        if (name.size() == 1 && name[0] >= 'a' && name[0] < 'a' + inputs_) return name[0] - 'a';
        return -1;
    }

    bool expression() { return comparison(); }
    bool comparison() {
        if (!additive()) return false;
        static const std::pair<const char*, int> ops[] = {{"<=", Le}, {">=", Ge}, {"==", Eq}, {"!=", Ne}, {"<", Lt}, {">", Gt}};
        for (;;) {
            bool matched = false;
            for (const auto& [sym, op] : ops)
                if (accept(sym)) {
                    if (!additive()) return false;
                    emit(op);
                    matched = true;
                    break;
                }
            if (!matched) return true;
        }
    }
    bool additive() {
        if (!term()) return false;
        for (;;) {
            if (accept("+")) {
                if (!term()) return false;
                emit(Add);
            } else if (accept("-")) {
                if (!term()) return false;
                emit(Sub);
            } else return true;
        }
    }
    bool term() {
        if (!unary()) return false;
        for (;;) {
            if (accept("*")) {
                if (!unary()) return false;
                emit(Mul);
            } else if (accept("/")) {
                if (!unary()) return false;
                emit(Div);
            } else return true;
        }
    }
    bool unary() {
        if (accept("-")) {
            if (!unary()) return false;
            emit(Neg);
            return true;
        }
        if (accept("+")) return unary();
        return power();
    }
    bool power() {
        if (!primary()) return false;
        if (accept("^")) { // right to left: 2^3^2 = 2^9
            if (!unary()) return false;
            emit(Pow);
        }
        return true;
    }
    bool primary() {
        const Token tok = peek();
        if (tok.kind == Token::Number) {
            ++p_;
            emit(Num, 0, tok.number);
            return true;
        }
        if (accept("(")) {
            if (!expression()) return false;
            return accept(")") || err("A \")\" is missing");
        }
        if (tok.kind != Token::Name) return err(tok.kind == Token::End || tok.kind == Token::Newline ? "Something is missing at the end" : "Unexpected \"" + tok.text + "\"");
        ++p_;
        const std::string& name = tok.text;
        if (auto f = functions().find(name); f != functions().end()) {
            if (!accept("(")) return err(name + " needs its value in parentheses: " + name + "(...)");
            for (int i = 0; i < f->second.second; ++i) {
                if (i && !accept(",")) return err(name + " takes " + std::to_string(f->second.second) + " values");
                if (!expression()) return false;
            }
            if (!accept(")")) return err(name + " takes " + std::to_string(f->second.second) + " values");
            if (f->second.first == -1) emit(Clamp);
            else if (f->second.first == -2) emit(If);
            else emit(f->second.second == 1 ? Fn1 : Fn2, f->second.first);
            return true;
        }
        if (name == "pi") return emit(Num, 0, 3.14159265f), true;
        if (name == "e") return emit(Num, 0, 2.71828183f), true;
        if (const int in = inputIndex(name); in >= 0) {
            uses_[in] = true;
            emit(Input, in);
            return true;
        }
        if (auto l = locals_.find(name); l != locals_.end()) {
            emit(Load, l->second - 1);
            return true;
        }
        if (name.size() == 1 && name[0] >= 'a' && name[0] <= 'd')
            return err("\"" + name + "\" is an input this node does not have: give it more inputs");
        return err("\"" + name + "\" is not known");
    }

    std::vector<Token> t_;
    size_t p_ = 0;
    int inputs_;
    std::vector<int>& ops_;
    std::vector<int>& args_;
    std::vector<float>& values_;
    std::map<std::string, int> locals_; // name -> slot + 1
    bool uses_[Expression::kMaxInputs] = {};
    std::string error_;
};

// A value and its derivatives with respect to each input.
struct Dual {
    float v = 0, d[Expression::kMaxInputs] = {};
};

Dual scaled(const Dual& a, float value, float slope) {
    Dual r;
    r.v = value;
    for (int i = 0; i < Expression::kMaxInputs; ++i) r.d[i] = a.d[i] * slope;
    return r;
}

Dual constant(float v) {
    Dual r;
    r.v = v;
    return r;
}

} // namespace

bool Expression::compile(const std::string& code, int inputs, std::string* error) {
    program_.clear();
    std::string e;
    std::vector<Token> tokens = tokenize(code, e);
    if (!e.empty()) {
        if (error) *error = e;
        return false;
    }
    std::vector<int> ops, args;
    std::vector<float> values;
    Parser parser(std::move(tokens), std::clamp(inputs, 1, kMaxInputs), ops, args, values);
    if (!parser.run(e, slots_, uses_)) {
        if (error) *error = e;
        return false;
    }
    for (size_t i = 0; i < ops.size(); ++i) program_.push_back({ops[i], args[i], values[i]});
    return true;
}

float Expression::eval(const float* in, float* grad) const {
    Dual stack[64];
    Dual locals[64];
    int top = 0;
    for (const Op& op : program_) {
        if (top >= 62) break; // deeper than any sensible formula
        switch (op.code) {
        case Num:
            stack[top++] = constant(op.value);
            break;
        case Input: {
            Dual d = constant(in[op.arg]);
            d.d[op.arg] = 1;
            stack[top++] = d;
            break;
        }
        case Load:
            stack[top++] = locals[std::min(op.arg, 63)];
            break;
        case Store:
            locals[std::min(op.arg, 63)] = stack[--top];
            break;
        case Neg:
            stack[top - 1] = scaled(stack[top - 1], -stack[top - 1].v, -1);
            break;
        case Fn1: {
            Dual& a = stack[top - 1];
            const float x = a.v;
            switch (op.arg) {
            case Abs: a = scaled(a, std::abs(x), x >= 0 ? 1.0f : -1.0f); break;
            case Exp: {
                const float y = std::exp(std::min(x, 80.0f));
                a = scaled(a, y, y);
                break;
            }
            case Log: a = scaled(a, std::log(std::max(x, 1e-12f)), 1.0f / std::max(x, 1e-12f)); break;
            case Sqrt: {
                const float y = std::sqrt(std::max(x, 0.0f));
                a = scaled(a, y, 0.5f / std::max(y, 1e-6f));
                break;
            }
            case Sin: a = scaled(a, std::sin(x), std::cos(x)); break;
            case Cos: a = scaled(a, std::cos(x), -std::sin(x)); break;
            case Tan: {
                const float y = std::tan(x);
                a = scaled(a, y, 1 + y * y);
                break;
            }
            case Tanh: {
                const float y = std::tanh(x);
                a = scaled(a, y, 1 - y * y);
                break;
            }
            case Sigmoid: {
                const float y = 1.0f / (1.0f + std::exp(-x));
                a = scaled(a, y, y * (1 - y));
                break;
            }
            case Relu: a = scaled(a, x > 0 ? x : 0.0f, x > 0 ? 1.0f : 0.0f); break;
            case Softplus: a = scaled(a, std::max(x, 0.0f) + std::log1p(std::exp(-std::abs(x))), 1.0f / (1.0f + std::exp(-x))); break;
            case Sign: a = constant(x > 0 ? 1.0f : x < 0 ? -1.0f : 0.0f); break;
            case Floor: a = constant(std::floor(x)); break;
            }
            break;
        }
        case Fn2:
        case Add: case Sub: case Mul: case Div: case Pow:
        case Lt: case Gt: case Le: case Ge: case Eq: case Ne: {
            const Dual b = stack[--top];
            Dual& a = stack[top - 1];
            Dual r;
            const int what = op.code == Fn2 ? (op.arg == Min ? -1 : op.arg == Max ? -2 : Pow) : op.code;
            switch (what) {
            case Add:
                r.v = a.v + b.v;
                for (int i = 0; i < kMaxInputs; ++i) r.d[i] = a.d[i] + b.d[i];
                break;
            case Sub:
                r.v = a.v - b.v;
                for (int i = 0; i < kMaxInputs; ++i) r.d[i] = a.d[i] - b.d[i];
                break;
            case Mul:
                r.v = a.v * b.v;
                for (int i = 0; i < kMaxInputs; ++i) r.d[i] = a.d[i] * b.v + b.d[i] * a.v;
                break;
            case Div:
                r.v = a.v / b.v;
                for (int i = 0; i < kMaxInputs; ++i) r.d[i] = (a.d[i] * b.v - b.d[i] * a.v) / (b.v * b.v);
                break;
            case Pow: {
                r.v = std::pow(a.v, b.v);
                const float da = b.v * std::pow(a.v, b.v - 1), db = a.v > 0 ? r.v * std::log(a.v) : 0.0f;
                for (int i = 0; i < kMaxInputs; ++i) r.d[i] = a.d[i] * da + b.d[i] * db;
                break;
            }
            case -1: r = a.v <= b.v ? a : b; break;
            case -2: r = a.v >= b.v ? a : b; break;
            case Lt: r = constant(a.v < b.v); break;
            case Gt: r = constant(a.v > b.v); break;
            case Le: r = constant(a.v <= b.v); break;
            case Ge: r = constant(a.v >= b.v); break;
            case Eq: r = constant(a.v == b.v); break;
            case Ne: r = constant(a.v != b.v); break;
            }
            a = r;
            break;
        }
        case Clamp: {
            const Dual hi = stack[--top], lo = stack[--top];
            Dual& x = stack[top - 1];
            if (x.v < lo.v) x = lo;
            else if (x.v > hi.v) x = hi;
            break;
        }
        case If: {
            const Dual no = stack[--top], yes = stack[--top];
            Dual& c = stack[top - 1];
            c = c.v > 0 ? yes : no;
            break;
        }
        }
    }
    const Dual result = top > 0 ? stack[top - 1] : Dual{};
    if (grad)
        for (int i = 0; i < kMaxInputs; ++i) grad[i] = result.d[i];
    return result.v;
}

} // namespace logievo
