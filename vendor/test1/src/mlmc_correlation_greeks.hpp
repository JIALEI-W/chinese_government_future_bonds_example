#ifndef BENCHMARK_MLMC_CORRELATION_GREEKS_HPP
#define BENCHMARK_MLMC_CORRELATION_GREEKS_HPP

#include "benchmark_model.hpp"
#include "mlmc_vector.hpp"
#include "path_sampling.hpp"
#include "statistics.hpp"

#include <cstdint>
#include <vector>

MlmcVectorStatistics plan_mlmc_correlation_greeks(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    int Lmin,
    int Lmax,
    int N0,
    float eps,
    float alpha,
    float beta,
    float gamma,
    std::uint32_t seed
);

Estimate fixed_mlmc_correlation_greeks(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    const std::vector<int>& samples_per_level,
    std::uint32_t seed
);

double reconstructed_mlmc_bias(
    const MlmcVectorStatistics& plan,
    float alpha
);

#endif
