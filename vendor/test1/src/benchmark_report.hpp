#ifndef BENCHMARK_REPORT_HPP
#define BENCHMARK_REPORT_HPP

#include "benchmark_model.hpp"
#include "mlmc_vector.hpp"
#include "options.hpp"
#include "statistics.hpp"
#include "validation.hpp"

#include <functional>
#include <string>
#include <vector>

struct TimedMethod {
    std::string name;
    std::string differentiation;
    std::string sampling;
    std::function<Estimate()> run;
    std::vector<double> runtimes_ms;
    Estimate result;
    bool has_result = false;
};

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
);

#endif
