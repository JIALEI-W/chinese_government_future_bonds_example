#include "benchmark_model.hpp"
#include "path_sampling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

volatile double g_checksum = 0.0;

struct Timing {
    double milliseconds = 0.0;
    double checksum = 0.0;
};

Timing time_mode(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    int steps,
    std::uint64_t samples,
    std::uint32_t seed
) {
    std::mt19937 rng(seed);
    double checksum = 0.0;
    const auto start = Clock::now();
    for (std::uint64_t sample = 0; sample < samples; ++sample) {
        const Vec gradient = evaluate_gradient(
            model, mode, make_fine_normals(steps, model.n, rng));
        for (double value : gradient) checksum += value;
    }
    const double elapsed = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    g_checksum += checksum;
    return {elapsed, checksum};
}

std::uint64_t sample_count(int dimension, int steps) {
    const std::uint64_t directions =
        static_cast<std::uint64_t>(dimension) * (dimension - 1) / 2;
    const double proxy_per_path = static_cast<double>(directions)
        * dimension * dimension * steps;
    const double target_proxy_work = 6.0e7;
    const auto planned = static_cast<std::uint64_t>(
        std::ceil(target_proxy_work / proxy_per_path));
    return std::max<std::uint64_t>(64,
        std::min<std::uint64_t>(planned, 10000));
}

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path output = "experiment_data/dimension_scaling.csv";
        if (argc == 3 && std::string(argv[1]) == "--output") {
            output = argv[2];
        } else if (argc != 1) {
            throw std::invalid_argument(
                "usage: dimension_benchmark [--output PATH]");
        }

        fs::create_directories(output.parent_path());
        std::ofstream csv(output);
        csv << "dimension,directions,steps,samples,repetition,method,"
               "runtime_ms,time_per_path_us,speedup\n";
        csv << std::setprecision(17);

        const std::vector<int> dimensions = {2, 3, 5, 8, 12, 16, 20};
        constexpr int steps = 64;
        constexpr int repeats = 5;
        for (int dimension : dimensions) {
            const BenchmarkModel model = make_benchmark_model(
                dimension, 20260820U + static_cast<std::uint32_t>(dimension));
            const std::uint64_t samples = sample_count(dimension, steps);

            // Untimed warm-up removes one-off code and allocator effects.
            (void)time_mode(model, DifferentiationMode::Forward,
                steps, 2, 0x12340000U + dimension);
            (void)time_mode(model, DifferentiationMode::Adjoint,
                steps, 2, 0x12340000U + dimension);

            for (int repetition = 0; repetition < repeats; ++repetition) {
                const std::uint32_t seed = 0x5a170000U
                    + static_cast<std::uint32_t>(100 * dimension + repetition);
                const Timing forward = time_mode(model,
                    DifferentiationMode::Forward, steps, samples, seed);
                const Timing adjoint = time_mode(model,
                    DifferentiationMode::Adjoint, steps, samples, seed);
                const double tolerance = 1e-8 * std::max(
                    1.0, std::abs(forward.checksum));
                if (std::abs(forward.checksum - adjoint.checksum) > tolerance) {
                    throw std::runtime_error(
                        "forward and adjoint checksums disagree");
                }
                const double speedup = forward.milliseconds / adjoint.milliseconds;
                const std::uint64_t directions =
                    static_cast<std::uint64_t>(dimension) * (dimension - 1) / 2;
                const auto write_row = [&](const char* method, double runtime) {
                    csv << dimension << ',' << directions << ',' << steps << ','
                        << samples << ',' << repetition + 1 << ',' << method << ','
                        << runtime << ',' << 1000.0 * runtime / samples << ','
                        << speedup << '\n';
                };
                write_row("Forward", forward.milliseconds);
                write_row("Adjoint", adjoint.milliseconds);
            }
            std::cout << "dimension " << dimension << " complete ("
                      << samples << " paths/repetition)\n";
        }
        std::cout << "wrote " << output << ", checksum=" << g_checksum << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
