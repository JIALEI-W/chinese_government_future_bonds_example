#include "benchmark_model.hpp"
#include "mlmc_correlation_greeks.hpp"
#include "options.hpp"
#include "standard_monte_carlo.hpp"
#include "statistics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    try {
        fs::path output_dir = "experiment_data";
        if (argc == 3 && std::string(argv[1]) == "--output-dir") {
            output_dir = argv[2];
        } else if (argc != 1) {
            throw std::invalid_argument(
                "usage: complexity_sweep [--output-dir PATH]");
        }
        fs::create_directories(output_dir);
        std::ofstream sweep(output_dir / "epsilon_scaling.csv");
        std::ofstream levels(output_dir / "mlmc_level_diagnostics.csv");
        sweep << "epsilon,method,final_level,finest_steps,total_samples,"
                 "actual_euler_steps,max_standard_error,bias_bound,"
                 "estimated_rmse,planning_runtime_ms\n";
        levels << "epsilon,level,euler_steps,h,samples,max_abs_mean,"
                  "max_variance,mean_abs_mean,mean_variance\n";
        sweep << std::setprecision(17);
        levels << std::setprecision(17);

        const BenchmarkModel model = make_benchmark_model(5, 20260820U);
        const Vec continuous = analytic_continuous_greeks(model, 65536);
        const std::vector<double> epsilons = {0.20, 0.15, 0.10, 0.075, 0.05};

        for (std::size_t e = 0; e < epsilons.size(); ++e) {
            const double eps = epsilons[e];
            const int controlled_Lmin = minimum_bias_controlled_level(
                model, eps, kBiasFraction, 2, 20);
            const auto mlmc_start = Clock::now();
            const MlmcVectorStatistics plan = plan_mlmc_correlation_greeks(
                model,
                DifferentiationMode::Adjoint,
                controlled_Lmin,
                20,
                100,
                static_cast<float>(eps),
                kMlmcAlpha,
                kMlmcBeta,
                kMlmcGamma,
                0x31000000U + static_cast<std::uint32_t>(e));
            const double mlmc_plan_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - mlmc_start).count();

            const auto standard_start = Clock::now();
            const std::uint64_t standard_samples = plan_standard_samples(
                model,
                plan.final_level,
                100,
                eps,
                kBiasFraction,
                0x42000000U + static_cast<std::uint32_t>(e));
            const double standard_plan_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - standard_start).count();

            const int finest_steps = euler_steps_for_level(plan.final_level);
            const Vec euler = analytic_euler_greeks(model, finest_steps);
            const double analytic_bias = max_abs_difference(euler, continuous);
            const double mlmc_bias = reconstructed_mlmc_bias(plan, kMlmcAlpha);
            const double standard_se = std::sqrt(1.0 - kBiasFraction) * eps;
            const double mlmc_se = *std::max_element(
                plan.standard_error.begin(), plan.standard_error.end());

            std::uint64_t mlmc_samples = 0;
            std::uint64_t mlmc_work = 0;
            for (int level = 0; level <= plan.final_level; ++level) {
                const std::uint64_t count = static_cast<std::uint64_t>(
                    plan.samples_per_level[level]);
                const std::uint64_t fine = static_cast<std::uint64_t>(
                    euler_steps_for_level(level));
                const std::uint64_t per_sample = level == 0
                    ? fine : fine + fine / 2ULL;
                mlmc_samples += count;
                mlmc_work += count * per_sample;

                double max_abs_mean = 0.0;
                double max_variance = 0.0;
                double mean_abs_mean = 0.0;
                double mean_variance = 0.0;
                for (std::size_t output = 0;
                     output < plan.level_mean.size(); ++output) {
                    const double abs_mean = std::abs(
                        plan.level_mean[output][level]);
                    const double variance =
                        plan.level_variance[output][level];
                    max_abs_mean = std::max(max_abs_mean, abs_mean);
                    max_variance = std::max(max_variance, variance);
                    mean_abs_mean += abs_mean;
                    mean_variance += variance;
                }
                mean_abs_mean /= plan.level_mean.size();
                mean_variance /= plan.level_mean.size();
                levels << eps << ',' << level << ',' << fine << ','
                       << 1.0 / static_cast<double>(fine) << ',' << count << ','
                       << max_abs_mean << ',' << max_variance << ','
                       << mean_abs_mean << ',' << mean_variance << '\n';
            }

            const std::uint64_t standard_work = standard_samples
                * static_cast<std::uint64_t>(finest_steps);
            sweep << eps << ",Standard MC," << plan.final_level << ','
                  << finest_steps << ',' << standard_samples << ','
                  << standard_work << ',' << standard_se << ','
                  << analytic_bias << ',' << std::hypot(standard_se, analytic_bias)
                  << ',' << standard_plan_ms << '\n';
            sweep << eps << ",MLMC," << plan.final_level << ','
                  << finest_steps << ',' << mlmc_samples << ',' << mlmc_work
                  << ',' << mlmc_se << ',' << mlmc_bias << ','
                  << std::hypot(mlmc_se, mlmc_bias) << ',' << mlmc_plan_ms << '\n';

            std::cout << "eps=" << eps << ", level=" << plan.final_level
                      << ", standard work=" << standard_work
                      << ", MLMC work=" << mlmc_work << '\n';
        }
        std::cout << "wrote complexity data to " << output_dir << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
