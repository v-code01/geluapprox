// The three standard GELU forms, as straightforward C++ f32 loops compiled at -O3 -march=native. GELU is
// the FFN activation in most transformers, and it is computed three ways in practice:
//
//   exact    : x/2 * (1 + erf(x/sqrt2))                          - the definition (uses erf).
//   tanh     : 0.5x * (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))   - "gelu_new", GPT-2 and many others.
//   sigmoid  : x * sigmoid(1.702 x)                              - the cheapest approximation.
//
// The f64 exact form is the reference. The comparison is between the three forms (fidelity and cost), not
// scalar-vs-SIMD: each loop body is dominated by one scalar libm call per element (erff/tanhf/expf), and a
// loop containing a scalar transcendental call cannot be SIMD-vectorized across iterations, so the
// throughput reflects the relative cost of those library functions - which is what real code pays.
// Header-only, no I/O.
#ifndef GELU_HPP
#define GELU_HPP

#include <cmath>
#include <cstddef>
#include <vector>

namespace gelu {
inline const double INV_SQRT2 = 0.7071067811865476;      // 1/sqrt(2)
inline const float INV_SQRT2_F = 0.70710678f;
inline const float TANH_C = 0.7978845608f;               // sqrt(2/pi)
inline const float TANH_A = 0.044715f;
inline const float SIGMOID_K = 1.702f;
}  // namespace gelu

// f64 exact-erf reference (scalar).
inline double gelu_exact_f64(double x) {
    return x * 0.5 * (1.0 + std::erf(x * gelu::INV_SQRT2));
}

// f32 exact-erf form (uses the f32 error function).
inline void gelu_exact_f32(const std::vector<float>& x, std::vector<float>& out) {
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = x[i] * 0.5f * (1.0f + std::erff(x[i] * gelu::INV_SQRT2_F));
}

// f32 tanh approximation ("gelu_new").
inline void gelu_tanh_f32(const std::vector<float>& x, std::vector<float>& out) {
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + std::tanh(gelu::TANH_C * (v + gelu::TANH_A * v * v * v)));
    }
}

// f32 sigmoid approximation.
inline void gelu_sigmoid_f32(const std::vector<float>& x, std::vector<float>& out) {
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = x[i] / (1.0f + std::exp(-gelu::SIGMOID_K * x[i]));
}

// Maximum absolute error of an f32 output against the f64 reference.
inline double max_abs_error(const std::vector<float>& a, const std::vector<double>& ref) {
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::fmax(e, std::fabs(static_cast<double>(a[i]) - ref[i]));
    return e;
}

// Root-mean-square error of an f32 output against the f64 reference.
inline double rms_error(const std::vector<float>& a, const std::vector<double>& ref) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = static_cast<double>(a[i]) - ref[i];
        s += d * d;
    }
    return a.empty() ? 0.0 : std::sqrt(s / static_cast<double>(a.size()));
}

#endif  // GELU_HPP
