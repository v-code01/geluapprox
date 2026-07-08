// Benchmark the three GELU forms: accuracy against the f64 exact reference over a dense grid split by
// x-region (so the error can be localized), and throughput over a large buffer. Deterministic input, so
// the run is self-contained. Emits JSON lines: accuracy rows {kind:"acc", variant, region, max_err,
// rms_err} and throughput rows {kind:"thr", variant, n, ns}.
//
// Usage: bench REPS
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gelu.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using Kernel = void (*)(const std::vector<float>&, std::vector<float>&);

struct Region {
    const char* name;
    float lo;
    float hi;
};

void acc_rows(std::string& out, const char* variant, Kernel k,
              const std::vector<Region>& regions) {
    for (const Region& r : regions) {
        std::vector<float> x;
        for (int i = 0; i <= 16000; ++i) {
            float v = -8.0f + static_cast<float>(i) * 1e-3f;
            if (v >= r.lo && v < r.hi) x.push_back(v);
        }
        if (x.empty()) continue;
        std::vector<double> ref(x.size());
        for (size_t i = 0; i < x.size(); ++i) ref[i] = gelu_exact_f64(x[i]);
        std::vector<float> y;
        k(x, y);
        char row[192];
        std::snprintf(row, sizeof(row),
                      "{\"kind\": \"acc\", \"variant\": \"%s\", \"region\": \"%s\", "
                      "\"max_err\": %.6e, \"rms_err\": %.6e}\n",
                      variant, r.name, max_abs_error(y, ref), rms_error(y, ref));
        out += row;
    }
}

double min_ns(Kernel k, const std::vector<float>& x, int reps) {
    double best = 1e300;
    volatile float sink = 0.0f;
    std::vector<float> out;
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        k(x, out);
        auto t1 = Clock::now();
        sink += out.empty() ? 0.0f : out[0];
        best = std::fmin(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    (void)sink;
    return best;
}
}  // namespace

int main(int argc, char** argv) {
    const int reps = (argc > 1) ? std::atoi(argv[1]) : 300;
    const std::vector<Region> regions = {
        {"all", -8.0f, 8.001f}, {"neg_tail", -8.0f, -3.0f}, {"neg_mid", -3.0f, 0.0f},
        {"pos_mid", 0.0f, 3.0f}, {"pos_tail", 3.0f, 8.001f}};
    const std::vector<std::pair<const char*, Kernel>> variants = {
        {"exact_f32", gelu_exact_f32}, {"tanh", gelu_tanh_f32}, {"sigmoid", gelu_sigmoid_f32}};

    std::string out;
    for (const auto& v : variants) acc_rows(out, v.first, v.second, regions);

    // Throughput over a fixed buffer of representative inputs.
    const size_t n = 65536;
    std::vector<float> x(n);
    uint64_t s = 0x1234567;
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x[i] = static_cast<float>((static_cast<double>(s >> 11) / 9007199254740992.0) * 16.0 - 8.0);
    }
    for (const auto& v : variants) {
        char row[128];
        std::snprintf(row, sizeof(row),
                      "{\"kind\": \"thr\", \"variant\": \"%s\", \"n\": %zu, \"ns\": %.1f}\n",
                      v.first, n, min_ns(v.second, x, reps));
        out += row;
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
