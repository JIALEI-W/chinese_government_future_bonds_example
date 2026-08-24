#include "benchmark_model.hpp"

#include "analytic_greeks.hpp"
#include "cholesky.hpp"
#include "options.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace {

Mat make_random_correlation(int n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    Mat factor(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            at(factor, n, i, j) = 0.45 * normal(rng);
        }
        at(factor, n, i, i) += 1.8;
    }

    Mat covariance(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                at(covariance, n, i, j) +=
                    at(factor, n, i, k) * at(factor, n, j, k);
            }
        }
    }

    Mat rho(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            at(rho, n, i, j) = at(covariance, n, i, j)
                / std::sqrt(
                    at(covariance, n, i, i)
                    * at(covariance, n, j, j));
        }
        at(rho, n, i, i) = 1.0;
    }
    return rho;
}

PathwiseParams make_params(int n) {
    PathwiseParams params;
    params.S0.assign(n, 100.0);
    params.drift.assign(n, 0.0);
    params.sigma.resize(n);
    params.T = 1.0;
    params.K = 100.0;
    for (int i = 0; i < n; ++i) {
        // Keep the volatility range fixed when d changes so the dimension
        // experiment isolates differentiation cost rather than changing the
        // stochastic model's scale.  For d=5 this is exactly
        // (0.03, 0.06, 0.09, 0.12, 0.15).
        params.sigma[i] = n == 1
            ? 0.09
            : 0.03 + 0.12 * static_cast<double>(i)
                / static_cast<double>(n - 1);
    }
    return params;
}

} // namespace

BenchmarkModel make_benchmark_model(int n, std::uint32_t seed) {
    BenchmarkModel model;
    model.n = n;
    model.params = make_params(n);
    model.rho = make_random_correlation(n, seed);
    model.C = cholesky_lower(model.rho, n);
    model.pairs = independent_correlation_pairs(n);
    model.Cdot = forward_cholesky_tangents(model.C, n, model.pairs);
    model.reverse_mapping = standard_reverse_mapping(model.C, n, model.pairs);
    return model;
}

Vec analytic_euler_greeks(const BenchmarkModel& model, int steps) {
    Vec result(model.pairs.size(), 0.0);
    for (std::size_t p = 0; p < model.pairs.size(); ++p) {
        result[p] = analytic_euler_correlation_greek(
            model.params,
            model.rho,
            model.n,
            model.pairs[p].first,
            model.pairs[p].second,
            steps);
    }
    return result;
}

Vec analytic_continuous_greeks(
    const BenchmarkModel& model,
    int observations
) {
    Vec result(model.pairs.size(), 0.0);
    for (std::size_t p = 0; p < model.pairs.size(); ++p) {
        result[p] = analytic_continuous_correlation_greek(
            model.params,
            model.rho,
            model.n,
            model.pairs[p].first,
            model.pairs[p].second,
            observations);
    }
    return result;
}

int minimum_bias_controlled_level(
    const BenchmarkModel& model,
    double eps,
    double bias_fraction,
    int Lmin,
    int Lmax
) {
    const Vec continuous = analytic_continuous_greeks(model, 65536);
    const double bias_limit = std::sqrt(bias_fraction) * eps;
    for (int level = Lmin; level <= Lmax; ++level) {
        const Vec euler = analytic_euler_greeks(
            model, euler_steps_for_level(level));
        double maximum = 0.0;
        for (std::size_t output = 0; output < euler.size(); ++output) {
            maximum = std::max(
                maximum, std::abs(euler[output] - continuous[output]));
        }
        if (maximum <= bias_limit) return level;
    }
    throw std::runtime_error("Lmax cannot satisfy the analytic bias budget");
}
