#include "benchmark_model.hpp"
#include "benchmark_report.hpp"
#include "mlmc_correlation_greeks.hpp"
#include "options.hpp"
#include "standard_monte_carlo.hpp"
#include "statistics.hpp"
#include "validation.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const BenchmarkModel model =
            make_benchmark_model(options.dimension, options.seed);
        const ValidationChecks checks = run_validation_checks(model);
        const int controlled_Lmin = minimum_bias_controlled_level(
            model, options.eps, kBiasFraction, options.Lmin, options.Lmax);

        const auto mlmc_plan_start = std::chrono::steady_clock::now();
        const MlmcVectorStatistics mlmc_plan =
            plan_mlmc_correlation_greeks(
                model,
                DifferentiationMode::Adjoint,
                controlled_Lmin,
                options.Lmax,
                options.N0,
                static_cast<float>(options.eps),
                kMlmcAlpha,
                kMlmcBeta,
                kMlmcGamma,
                options.seed ^ 0x11111111U);
        const auto mlmc_plan_finish = std::chrono::steady_clock::now();
        const double mlmc_plan_runtime_ms =
            std::chrono::duration<double, std::milli>(
                mlmc_plan_finish - mlmc_plan_start).count();

        const auto standard_plan_start = std::chrono::steady_clock::now();
        const std::uint64_t standard_samples = plan_standard_samples(
            model,
            mlmc_plan.final_level,
            options.N0,
            options.eps,
            kBiasFraction,
            options.seed ^ 0x22222222U);
        const auto standard_plan_finish = std::chrono::steady_clock::now();
        const double standard_plan_runtime_ms =
            std::chrono::duration<double, std::milli>(
                standard_plan_finish - standard_plan_start).count();

        const int final_steps = euler_steps_for_level(mlmc_plan.final_level);
        const Vec analytic_euler = analytic_euler_greeks(model, final_steps);
        const Vec analytic_continuous =
            analytic_continuous_greeks(model, 65536);
        const double analytic_bias =
            max_abs_difference(analytic_euler, analytic_continuous);
        const double reconstructed_bias =
            reconstructed_mlmc_bias(mlmc_plan, kMlmcAlpha);

        constexpr std::uint32_t kStandardSeed = 0x6b8b4567U;
        constexpr std::uint32_t kMlmcSeed = 0x327b23c6U;

        std::vector<TimedMethod> methods;
        methods.push_back({
            "standard",
            "forward",
            "standard MC",
            [&]() {
                return standard_monte_carlo(
                    model,
                    DifferentiationMode::Forward,
                    mlmc_plan.final_level,
                    standard_samples,
                    kStandardSeed);
            },
            {}, {}, false});
        methods.push_back({
            "mlmc",
            "forward",
            "MLMC",
            [&]() {
                return fixed_mlmc_correlation_greeks(
                    model,
                    DifferentiationMode::Forward,
                    mlmc_plan.samples_per_level,
                    kMlmcSeed);
            },
            {}, {}, false});
        methods.push_back({
            "adjoint",
            "standard reverse Cholesky",
            "standard MC",
            [&]() {
                return standard_monte_carlo(
                    model,
                    DifferentiationMode::Adjoint,
                    mlmc_plan.final_level,
                    standard_samples,
                    kStandardSeed);
            },
            {}, {}, false});
        methods.push_back({
            "mlmc_adjoint",
            "standard reverse Cholesky",
            "MLMC",
            [&]() {
                return fixed_mlmc_correlation_greeks(
                    model,
                    DifferentiationMode::Adjoint,
                    mlmc_plan.samples_per_level,
                    kMlmcSeed);
            },
            {}, {}, false});

        for (int repetition = 0;
             repetition < options.repeats;
             ++repetition) {
            for (std::size_t offset = 0;
                 offset < methods.size();
                 ++offset) {
                const std::size_t index =
                    (static_cast<std::size_t>(repetition) + offset)
                    % methods.size();
                TimedMethod& method = methods[index];
                const auto start = std::chrono::steady_clock::now();
                Estimate result = method.run();
                const auto finish = std::chrono::steady_clock::now();
                method.runtimes_ms.push_back(
                    std::chrono::duration<double, std::milli>(
                        finish - start).count());
                if (!method.has_result) {
                    method.result = std::move(result);
                    method.has_result = true;
                }
            }
        }

        if (max_abs_difference(
                methods[0].result.mean,
                methods[2].result.mean) > 1.0e-9
            || max_abs_difference(
                methods[1].result.mean,
                methods[3].result.mean) > 1.0e-9) {
            throw std::runtime_error(
                "forward and adjoint estimators disagree on identical paths");
        }

        for (const TimedMethod& method : methods) {
            const double bias_bound = method.sampling == "MLMC"
                ? reconstructed_bias : analytic_bias;
            const double estimated_rmse = std::hypot(
                method.result.max_standard_error, bias_bound);
            if (estimated_rmse > options.eps) {
                throw std::runtime_error(
                    method.name + " failed the common eps RMSE condition");
            }
        }
        const double standard_runtime = median(methods[0].runtimes_ms);
        const double mlmc_speedup =
            standard_runtime / median(methods[1].runtimes_ms);
        const double adjoint_speedup =
            standard_runtime / median(methods[2].runtimes_ms);
        if (mlmc_speedup < 1.5 || adjoint_speedup < 1.5) {
            throw std::runtime_error(
                "required significant speedup (>=1.5x) was not observed");
        }

        write_benchmark_report(
            options,
            model,
            mlmc_plan,
            mlmc_plan_runtime_ms,
            standard_plan_runtime_ms,
            standard_samples,
            reconstructed_bias,
            analytic_bias,
            checks,
            analytic_euler,
            methods);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
