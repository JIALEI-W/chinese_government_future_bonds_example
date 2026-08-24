#include "analytic_greeks.hpp"

#include <cmath>
#include <stdexcept>

namespace {

double geometric_sum_from_one(double ratio, int count) {
    if (count <= 0) {
        return 0.0;
    }
    if (ratio == 1.0) {
        return static_cast<double>(count);
    }
    if (ratio > 0.0) {
        const double log_ratio = std::log(ratio);
        if (std::abs(log_ratio) < 1e-8) {
            return ratio * std::expm1(count * log_ratio) / std::expm1(log_ratio);
        }
    }
    return ratio * (1.0 - std::pow(ratio, count)) / (1.0 - ratio);
}

void validate_inputs(
    const PathwiseParams& params,
    const Mat& rho,
    int n,
    int i,
    int j,
    int steps
) {
    if (n <= 1 || i < 0 || j < 0 || i >= n || j >= n || i == j) {
        throw std::invalid_argument("analytic Greek requires distinct valid indices");
    }
    if (steps <= 0 || params.T <= 0.0) {
        throw std::invalid_argument("steps and T must be positive");
    }
    if (params.S0.size() != static_cast<std::size_t>(n)
        || params.drift.size() != static_cast<std::size_t>(n)
        || params.sigma.size() != static_cast<std::size_t>(n)
        || rho.size() != static_cast<std::size_t>(n * n)) {
        throw std::invalid_argument("analytic Greek input dimensions are inconsistent");
    }
}

double normalization(int n, int observations) {
    const double count = static_cast<double>(n) * observations;
    return 2.0 / (count * count);
}

} // namespace

double analytic_euler_correlation_greek(
    const PathwiseParams& params,
    const Mat& rho,
    int n,
    int i,
    int j,
    int steps
) {
    validate_inputs(params, rho, n, i, j, steps);
    const double h = params.T / static_cast<double>(steps);
    const double ai = 1.0 + params.drift[i] * h;
    const double aj = 1.0 + params.drift[j] * h;
    const double sigma_product = params.sigma[i] * params.sigma[j];
    const double b = ai * aj + at(rho, n, i, j) * sigma_product * h;

    double sum = 0.0;
    double b_to_q_minus_one = 1.0;
    for (int q = 1; q <= steps; ++q) {
        const int remaining = steps - q;
        const double extensions =
            1.0
            + geometric_sum_from_one(ai, remaining)
            + geometric_sum_from_one(aj, remaining);
        sum += static_cast<double>(q) * sigma_product * h
            * b_to_q_minus_one * extensions;
        b_to_q_minus_one *= b;
    }

    return normalization(n, steps)
        * params.S0[i] * params.S0[j] * sum;
}

double analytic_continuous_correlation_greek(
    const PathwiseParams& params,
    const Mat& rho,
    int n,
    int i,
    int j,
    int observations
) {
    validate_inputs(params, rho, n, i, j, observations);
    const double h = params.T / static_cast<double>(observations);
    const double sigma_product = params.sigma[i] * params.sigma[j];
    const double shared_rate =
        params.drift[i] + params.drift[j]
        + at(rho, n, i, j) * sigma_product;
    const double extension_i = std::exp(params.drift[i] * h);
    const double extension_j = std::exp(params.drift[j] * h);
    const double shared_step = std::exp(shared_rate * h);

    double sum = 0.0;
    double shared_moment = shared_step;
    for (int q = 1; q <= observations; ++q) {
        const int remaining = observations - q;
        const double extensions =
            1.0
            + geometric_sum_from_one(extension_i, remaining)
            + geometric_sum_from_one(extension_j, remaining);
        sum += static_cast<double>(q) * h * sigma_product
            * shared_moment * extensions;
        shared_moment *= shared_step;
    }

    return normalization(n, observations)
        * params.S0[i] * params.S0[j] * sum;
}
