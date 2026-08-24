#include "cholesky.hpp"

#include <cmath>
#include <stdexcept>

Mat cholesky_lower(const Mat& rho, int n) {
    if (n <= 0 || rho.size() != static_cast<std::size_t>(n * n)) {
        throw std::invalid_argument("rho must be a non-empty n x n matrix");
    }

    // rho_ij = rho_ji
    constexpr double symmetry_tolerance = 1e-12;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (std::abs(at(rho, n, i, j) - at(rho, n, j, i)) >
                symmetry_tolerance) {
                throw std::invalid_argument("rho must be symmetric");
            }
        }
    }

    // rho = C C^T
    // C_ii = sqrt(rho_ii - sum_{k=0}^{i-1} C_ik^2)
    // C_ij = (rho_ij - sum_{k=0}^{j-1} C_ik C_jk) / C_jj, i > j
    Mat C(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = at(rho, n, i, j);
            for (int k = 0; k < j; ++k) {
                value -= at(C, n, i, k) * at(C, n, j, k);
            }

            if (i == j) {
                if (value <= 0.0) {
                    throw std::domain_error(
                        "rho must be positive definite for Cholesky factorization"
                    );
                }
                at(C, n, i, j) = std::sqrt(value);
            } else {
                at(C, n, i, j) = value / at(C, n, j, j);
            }
        }
    }

    return C;
}
