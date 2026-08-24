#ifndef BENCHMARK_OPTIONS_HPP
#define BENCHMARK_OPTIONS_HPP

#include <cstdint>
#include <filesystem>

constexpr float kMlmcAlpha = 1.0f;
constexpr float kMlmcBeta = 1.0f;
constexpr float kMlmcGamma = 1.0f;
constexpr double kBiasFraction = 0.25;
// The one-step Euler grid is a poor coarsest level for this path-dependent
// Greek: millions of tiny samples make allocator/RNG overhead dominate.
// Level 0 therefore uses 2^3=8 time steps; this is still standard MLMC,
// with every later level doubling the resolution.
constexpr int kBaseTimeLevel = 3;

inline int euler_steps_for_level(int level) {
    return 1 << (level + kBaseTimeLevel);
}

struct Options {
    double eps = 0.1;
    int N0 = 100;
    int Lmin = 2;
    int Lmax = 20;
    int repeats = 5;
    int dimension = 5;
    std::uint32_t seed = 20260820U;
    std::filesystem::path output_dir = "results_eps_010";
};

Options parse_options(int argc, char** argv);

#endif
