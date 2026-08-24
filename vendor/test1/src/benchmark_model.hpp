#ifndef BENCHMARK_MODEL_HPP
#define BENCHMARK_MODEL_HPP

#include "forward_sensitivity.hpp"
#include "standard_reverse_cholesky.hpp"

#include <cstdint>

struct BenchmarkModel {
    int n = 0;
    PathwiseParams params;
    Mat rho;
    Mat C;
    CorrelationPairs pairs;
    CholeskyTangents Cdot;
    std::vector<Vec> reverse_mapping;
};

BenchmarkModel make_benchmark_model(int n, std::uint32_t seed);
Vec analytic_euler_greeks(const BenchmarkModel& model, int steps);
Vec analytic_continuous_greeks(const BenchmarkModel& model, int observations);
int minimum_bias_controlled_level(
    const BenchmarkModel& model,
    double eps,
    double bias_fraction,
    int Lmin,
    int Lmax
);

#endif
