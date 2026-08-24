#include "benchmark_model.hpp"

#include "analytic_greeks.hpp"
#include "cholesky.hpp"
#include "options.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr int kEmpiricalDimension = 4;

PathwiseParams make_empirical_params() {
    PathwiseParams params;
    params.S0.assign(kEmpiricalDimension, 100.0);
    params.drift.assign(kEmpiricalDimension, 0.0);
    params.sigma = {
        0.015352426899092591, // TF
        0.049098909526934360, // TL
        0.018935566892335930, // T
        0.006949923708868145  // TS
    };
    params.T = 1.0;
    params.K = 100.0;
    return params;
}

Mat make_empirical_correlation() {
    return {
        1.0,                0.7559935956980006,
        0.9465752017878403, 0.9144216677058656,

        0.7559935956980006, 1.0,
        0.8469200885752329, 0.6528499745887023,

        0.9465752017878403, 0.8469200885752329,
        1.0,                0.8459608384447707,

        0.9144216677058656, 0.6528499745887023,
        0.8459608384447707, 1.0
    };
}

} // namespace

BenchmarkModel make_benchmark_model(int n, std::uint32_t /*seed*/) {
    if (n != kEmpiricalDimension) {
        throw std::invalid_argument(
            "the empirical benchmark requires --dimension 4"
        );
    }

    BenchmarkModel model;
    model.n = kEmpiricalDimension;
    model.params = make_empirical_params();
    model.rho = make_empirical_correlation();
    model.C = cholesky_lower(model.rho, model.n);
    model.pairs = independent_correlation_pairs(model.n);
    model.Cdot = forward_cholesky_tangents(
        model.C, model.n, model.pairs
    );
    model.reverse_mapping = standard_reverse_mapping(
        model.C, model.n, model.pairs
    );
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
            steps
        );
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
            observations
        );
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
            model, euler_steps_for_level(level)
        );
        double maximum = 0.0;
        for (std::size_t output = 0; output < euler.size(); ++output) {
            maximum = std::max(
                maximum,
                std::abs(euler[output] - continuous[output])
            );
        }
        if (maximum <= bias_limit) {
            return level;
        }
    }
    throw std::runtime_error("Lmax cannot satisfy the analytic bias budget");
}

