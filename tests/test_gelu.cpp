// Correctness tests for the GELU kernels, written before the header exists (TDD). The f64 exact-erf form
// is the oracle. Each f32 vector kernel must match its own scalar definition, and the pre-registered
// fidelity ordering must hold: the tanh approximation is close to exact (< 1e-3), the sigmoid
// approximation is far (> 1e-2).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gelu.hpp"

namespace {
int g_checks = 0;
void check(bool cond, const char* what) {
    ++g_checks;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

// Scalar definitions, used to check the vector kernels apply the right formula elementwise.
double s_tanh(double x) {
    const double c = std::sqrt(2.0 / 3.141592653589793);
    return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
}
double s_sigmoid(double x) { return x / (1.0 + std::exp(-1.702 * x)); }

void test_exact_reference_values() {
    check(std::fabs(gelu_exact_f64(0.0)) < 1e-12, "gelu(0) = 0");
    check(std::fabs(gelu_exact_f64(8.0) - 8.0) < 1e-6, "gelu(8) ~= 8");
    check(std::fabs(gelu_exact_f64(-8.0)) < 1e-6, "gelu(-8) ~= 0");
    check(gelu_exact_f64(1.0) > 0.84 && gelu_exact_f64(1.0) < 0.85, "gelu(1) ~= 0.8413");
}

void test_kernels_match_scalar_definition() {
    std::vector<float> x = {-3.0f, -1.0f, -0.2f, 0.0f, 0.5f, 2.0f, 4.0f}, out;
    gelu_tanh_f32(x, out);
    for (size_t i = 0; i < x.size(); ++i)
        check(std::fabs(out[i] - s_tanh(x[i])) < 1e-5, "tanh kernel matches scalar def");
    gelu_sigmoid_f32(x, out);
    for (size_t i = 0; i < x.size(); ++i)
        check(std::fabs(out[i] - s_sigmoid(x[i])) < 1e-5, "sigmoid kernel matches scalar def");
    gelu_exact_f32(x, out);
    for (size_t i = 0; i < x.size(); ++i)
        check(std::fabs(out[i] - gelu_exact_f64(x[i])) < 1e-5, "exact f32 kernel matches f64 def");
}

void test_fidelity_ordering() {
    // Over a dense grid, tanh-approx is faithful and sigmoid-approx is not.
    std::vector<float> x, ref_in;
    for (int i = -6000; i <= 6000; ++i) x.push_back(static_cast<float>(i) * 1e-3f);
    std::vector<double> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i) ref[i] = gelu_exact_f64(x[i]);
    std::vector<float> out;
    gelu_tanh_f32(x, out);
    double e_tanh = max_abs_error(out, ref);
    gelu_sigmoid_f32(x, out);
    double e_sig = max_abs_error(out, ref);
    check(e_tanh < 1e-3, "tanh-approx max error < 1e-3");
    check(e_sig > 1e-2, "sigmoid-approx max error > 1e-2");
    check(e_sig > 10.0 * e_tanh, "sigmoid-approx an order of magnitude worse than tanh-approx");
}

}  // namespace

int main() {
    test_exact_reference_values();
    test_kernels_match_scalar_definition();
    test_fidelity_ordering();
    std::printf("OK: %d checks passed\n", g_checks);
    return 0;
}
