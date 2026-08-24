#include "forward_sensitivity.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

CorrelationPairs independent_correlation_pairs(int n) {
    if (n < 2) {
        throw std::invalid_argument("at least two assets are required");
    }
    CorrelationPairs pairs;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            pairs.emplace_back(i, j);
        }
    }
    return pairs;
}

CholeskyTangents forward_cholesky_tangents(
    const Mat& C,
    int n,
    const CorrelationPairs& pairs
) {
    if (C.size() != static_cast<std::size_t>(n * n)) {
        throw std::invalid_argument("C must be an n x n matrix");
    }

    CholeskyTangents tangents(pairs.size(), Mat(n * n, 0.0));
    for (std::size_t direction = 0; direction < pairs.size(); ++direction) {
        Mat& Cdot = tangents[direction];
        const int parameter_i = pairs[direction].first;
        const int parameter_j = pairs[direction].second;
        // Cholesky reads the lower entry rho_{parameter_j,parameter_i}.

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double qdot =
                    (i == parameter_j && j == parameter_i) ? 1.0 : 0.0;
                for (int k = 0; k < j; ++k) {
                    qdot -=
                        at(Cdot, n, i, k) * at(C, n, j, k)
                        + at(C, n, i, k) * at(Cdot, n, j, k);
                }

                if (i == j) {
                    at(Cdot, n, i, j) =
                        qdot / (2.0 * at(C, n, j, j));
                } else {
                    at(Cdot, n, i, j) =
                        (qdot
                         - at(C, n, i, j) * at(Cdot, n, j, j))
                        / at(C, n, j, j);
                }
            }
        }
    }
    return tangents;
}

Vec pathwise_forward_correlation_greeks(
    const Mat& C,
    const CholeskyTangents& Cdot,
    const PathNormals& z_path,
    const PathwiseParams& params,
    double& price_out
) {
    const int steps = static_cast<int>(z_path.size());
    if (steps <= 0) {
        throw std::invalid_argument("a path must contain at least one step");
    }
    const int n = static_cast<int>(z_path.front().size());
    const int directions = static_cast<int>(Cdot.size());
    if (C.size() != static_cast<std::size_t>(n * n)
        || params.S0.size() != static_cast<std::size_t>(n)
        || params.drift.size() != static_cast<std::size_t>(n)
        || params.sigma.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("inconsistent pathwise dimensions");
    }

    const double h = params.T / static_cast<double>(steps);
    const double sqrt_h = std::sqrt(h);
    Vec state = params.S0;
    std::vector<Vec> state_dot(directions, Vec(n, 0.0));
    Vec next_state(n, 0.0);
    std::vector<Vec> correlated_dot(directions, Vec(n, 0.0));
    std::vector<Vec> next_state_dot(directions, Vec(n, 0.0));
    double average_sum = 0.0;
    Vec average_dot_sum(directions, 0.0);

    for (int t = 0; t < steps; ++t) {
        const Vec correlated = matvec_lower(C, z_path[t]);
        for (int p = 0; p < directions; ++p) {
            std::fill(correlated_dot[p].begin(), correlated_dot[p].end(), 0.0);
            std::fill(next_state_dot[p].begin(), next_state_dot[p].end(), 0.0);
        }
        for (int p = 0; p < directions; ++p) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    correlated_dot[p][i] +=
                        at(Cdot[p], n, i, j) * z_path[t][j];
                }
            }
        }

        for (int i = 0; i < n; ++i) {
            const double multiplier =
                1.0 + params.drift[i] * h
                + params.sigma[i] * sqrt_h * correlated[i];
            next_state[i] = state[i] * multiplier;
            average_sum += next_state[i];

            for (int p = 0; p < directions; ++p) {
                const double multiplier_dot =
                    params.sigma[i] * sqrt_h * correlated_dot[p][i];
                next_state_dot[p][i] =
                    state_dot[p][i] * multiplier
                    + state[i] * multiplier_dot;
                average_dot_sum[p] += next_state_dot[p][i];
            }
        }
        state.swap(next_state);
        state_dot.swap(next_state_dot);
    }

    const double normalization = 1.0 / static_cast<double>(steps * n);
    const double average = average_sum * normalization;
    const double payoff_argument = average - params.K;
    price_out = payoff_argument * payoff_argument;

    Vec greeks(directions, 0.0);
    for (int p = 0; p < directions; ++p) {
        greeks[p] =
            2.0 * payoff_argument * average_dot_sum[p] * normalization;
    }
    return greeks;
}
