#ifndef STANDARD_MONTE_CARLO_HPP
#define STANDARD_MONTE_CARLO_HPP

#include "path_sampling.hpp"
#include "statistics.hpp"

#include <cstdint>

std::uint64_t plan_standard_samples(
    const BenchmarkModel& model,
    int level,
    int N0,
    double eps,
    double bias_fraction,
    std::uint32_t seed
);

Estimate standard_monte_carlo(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    int level,
    std::uint64_t samples,
    std::uint32_t seed
);

#endif
