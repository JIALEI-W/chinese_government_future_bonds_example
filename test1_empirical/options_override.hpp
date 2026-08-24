#ifndef BENCHMARK_OPTIONS_HPP
#define BENCHMARK_OPTIONS_HPP

#include <cstdint>
#include <filesystem>

constexpr float kMlmcAlpha = 1.0f;
constexpr float kMlmcBeta = 1.0f;
constexpr float kMlmcGamma = 1.0f;
constexpr double kBiasFraction = 0.25;

// Keep the level convention of /Users/km/real_data_mlmc exactly:
// level l has 2^l Euler steps, including one step on level zero.
inline int euler_steps_for_level(int level) {
    return 1 << level;
}

struct Options {
    double eps = 0.01;
    int N0 = 200;
    int Lmin = 2;
    int Lmax = 20;
    int repeats = 5;
    int dimension = 4;
    std::uint32_t seed = 20260820U;
    std::filesystem::path output_dir = "../outputs/test1_empirical_eps001_original";
};

Options parse_options(int argc, char** argv);

#endif
