#include "benchmark_report.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace {

void write_matrix_csv(const fs::path& path, const Mat& matrix, int n) {
    std::ofstream output(path);
    output << std::setprecision(17) << "row";
    for (int j = 0; j < n; ++j) {
        output << ",asset_" << (j + 1);
    }
    output << '\n';
    for (int i = 0; i < n; ++i) {
        output << "asset_" << (i + 1);
        for (int j = 0; j < n; ++j) {
            output << ',' << at(matrix, n, i, j);
        }
        output << '\n';
    }
}

} // namespace

void write_benchmark_report(
    const Options& options,
    const BenchmarkModel& model,
    const MlmcVectorStatistics& plan,
    double mlmc_plan_runtime_ms,
    double standard_plan_runtime_ms,
    std::uint64_t standard_samples,
    double reconstructed_bias,
    double analytic_bias,
    const ValidationChecks& checks,
    const Vec& analytic_euler,
    const std::vector<TimedMethod>& methods
) {
    fs::create_directories(options.output_dir);
    write_matrix_csv(options.output_dir / "rho.csv", model.rho, model.n);

    {
        std::ofstream output(options.output_dir / "level_allocation.csv");
        output << "level,euler_steps,samples,reference_cost_per_sample,"
                  "actual_path_steps_per_sample\n";
        for (int level = 0; level <= plan.final_level; ++level) {
            const int fine_steps = euler_steps_for_level(level);
            const int actual_steps = level == 0
                ? fine_steps : fine_steps + fine_steps / 2;
            output << level << ',' << fine_steps << ','
                   << plan.samples_per_level[level] << ','
                   << plan.cost_per_level[level] << ','
                   << actual_steps << '\n';
        }
    }

    const double baseline = median(methods.front().runtimes_ms);
    {
        std::ofstream output(options.output_dir / "timing_repetitions.csv");
        output << "method,repetition,runtime_ms\n";
        output << std::setprecision(17);
        for (const TimedMethod& method : methods) {
            for (std::size_t repetition = 0;
                 repetition < method.runtimes_ms.size(); ++repetition) {
                output << method.name << ',' << repetition + 1 << ','
                       << method.runtimes_ms[repetition] << '\n';
            }
        }
    }
    {
        std::ofstream output(options.output_dir / "benchmark_results.csv");
        output << "method,differentiation,sampling,runtime_ms,"
                  "speedup_vs_standard,samples,reference_cost,"
                  "actual_euler_steps,max_standard_error,"
                  "bias_bound,estimated_rmse,meets_eps,"
                  "max_error_vs_analytic_euler\n";
        output << std::setprecision(17);
        for (const TimedMethod& method : methods) {
            const double runtime = median(method.runtimes_ms);
            const bool uses_mlmc = method.sampling == "MLMC";
            const double bias_bound = uses_mlmc
                ? reconstructed_bias : analytic_bias;
            const double estimated_rmse = std::hypot(
                method.result.max_standard_error, bias_bound);
            output << method.name << ',' << method.differentiation << ','
                   << method.sampling << ',' << runtime << ','
                   << baseline / runtime << ',' << method.result.samples << ','
                   << method.result.reference_cost << ','
                   << method.result.actual_euler_steps << ','
                   << method.result.max_standard_error << ',' << bias_bound
                   << ',' << estimated_rmse << ','
                   << (estimated_rmse <= options.eps ? "true" : "false") << ','
                   << max_abs_difference(method.result.mean, analytic_euler)
                   << '\n';
        }
    }

    {
        std::ofstream output(options.output_dir / "greek_estimates.csv");
        output << "greek,analytic_euler";
        for (const TimedMethod& method : methods) {
            output << ',' << method.name;
        }
        output << '\n' << std::setprecision(17);
        for (std::size_t p = 0; p < model.pairs.size(); ++p) {
            output << "rho_" << (model.pairs[p].first + 1)
                   << (model.pairs[p].second + 1)
                   << ',' << analytic_euler[p];
            for (const TimedMethod& method : methods) {
                output << ',' << method.result.mean[p];
            }
            output << '\n';
        }
    }

    std::ostringstream report;
    report << std::fixed << std::setprecision(6);
    report << "# Correlation-Greek benchmark\n\n";
    report << "## Configuration\n\n";
    report << "- Dimension: " << model.n << "\n";
    report << "- Correlation directions: " << model.pairs.size() << "\n";
    report << "- Target RMSE: " << options.eps << "\n";
    report << "- Bias-squared MSE fraction: " << kBiasFraction << "\n";
    report << "- N0: " << options.N0 << "\n";
    report << "- Final level: " << plan.final_level << "\n";
    report << "- Euler steps on final level: "
           << euler_steps_for_level(plan.final_level) << "\n";
    report << "- Coarsest-level Euler steps: "
           << euler_steps_for_level(0) << "\n";
    report << "- Execution: single-threaded\n";
    report << "- Volatilities:";
    for (double sigma : model.params.sigma) {
        report << ' ' << sigma;
    }
    report << "\n";
    report << "- Timing repetitions: " << options.repeats << "\n";
    report << "- Standard-MC samples: " << standard_samples << "\n";
    report << "- Reconstructed MLMC bias estimate: "
           << reconstructed_bias << "\n";
    report << "- Analytic discretisation-bias check: "
           << analytic_bias << "\n";
    report << "- MLMC planning time (excluded from method timings): "
           << mlmc_plan_runtime_ms << " ms\n";
    report << "- Standard-MC planning time (excluded): "
           << standard_plan_runtime_ms << " ms\n\n";

    report << "## Numerical checks\n\n";
    report << "| Check | Maximum absolute error |\n|---|---:|\n";
    report << std::scientific << std::setprecision(3);
    report << "| Cholesky reconstruction | " << checks.cholesky << " |\n";
    report << "| Forward vs standard adjoint | "
           << checks.forward_adjoint << " |\n";
    report << "| Forward vs finite difference | "
           << checks.finite_difference << " |\n";
    report << "| Standard reverse vs reference reverse | "
           << checks.standard_reference_reverse << " |\n";
    report << "| Coupled correction: forward vs adjoint | "
           << checks.coupled_correction << " |\n\n";
    report << std::fixed << std::setprecision(6);

    report << "## MLMC allocation\n\n";
    report << "| Level | Euler steps | Samples | Reference cost/sample | "
              "Actual path steps/sample |\n";
    report << "|---:|---:|---:|---:|---:|\n";
    for (int level = 0; level <= plan.final_level; ++level) {
        const int fine_steps = euler_steps_for_level(level);
        const int actual_steps = level == 0
            ? fine_steps : fine_steps + fine_steps / 2;
        report << "| " << level << " | " << fine_steps << " | "
               << plan.samples_per_level[level] << " | "
               << plan.cost_per_level[level] << " | "
               << actual_steps << " |\n";
    }
    report << '\n';

    report << "## Fixed-work timing comparison\n\n";
    report << "| Method | Differentiation | Sampling | Runtime (ms) | "
              "Speedup | Samples | Actual Euler steps | Max SE | "
              "Bias bound | Est. RMSE | Meets eps | "
              "Max error vs analytic Euler |\n";
    report << "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|\n";
    for (const TimedMethod& method : methods) {
        const double runtime = median(method.runtimes_ms);
        const bool uses_mlmc = method.sampling == "MLMC";
        const double bias_bound = uses_mlmc
            ? reconstructed_bias : analytic_bias;
        const double estimated_rmse = std::hypot(
            method.result.max_standard_error, bias_bound);
        report << "| " << method.name << " | "
               << method.differentiation << " | "
               << method.sampling << " | "
               << runtime << " | " << baseline / runtime << " | "
               << method.result.samples << " | "
               << method.result.actual_euler_steps << " | "
               << method.result.max_standard_error << " | " << bias_bound
               << " | " << estimated_rmse << " | "
               << (estimated_rmse <= options.eps ? "yes" : "NO") << " | "
               << max_abs_difference(method.result.mean, analytic_euler)
               << " |\n";
    }
    report << "\nPlanning and file output are outside the timing region.\n";

    std::ofstream output(options.output_dir / "report.md");
    output << report.str();
    std::cout << report.str();
}
