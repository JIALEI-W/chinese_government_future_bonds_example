#include "standard_reverse_cholesky.hpp"

#include <stdexcept>

Mat standard_reverse_cholesky(
    const Mat& C,
    const Mat& Cbar,
    int n
) {
    if (C.size() != static_cast<std::size_t>(n * n)
        || Cbar.size() != static_cast<std::size_t>(n * n)) {
        throw std::invalid_argument("C and Cbar must be n x n matrices");
    }

    Mat active_cbar(n * n, 0.0);
    Mat rhobar_lower(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            at(active_cbar, n, i, j) = at(Cbar, n, i, j);
        }
    }

    for (int j = n - 1; j >= 0; --j) {
        for (int i = n - 1; i >= j; --i) {
            double qbar = 0.0;
            if (i == j) {
                qbar = at(active_cbar, n, j, j)
                    / (2.0 * at(C, n, j, j));
            } else {
                qbar = at(active_cbar, n, i, j) / at(C, n, j, j);
                at(active_cbar, n, j, j) -=
                    at(C, n, i, j) / at(C, n, j, j)
                    * at(active_cbar, n, i, j);
            }

            at(rhobar_lower, n, i, j) += qbar;
            for (int k = 0; k < j; ++k) {
                at(active_cbar, n, i, k) -= at(C, n, j, k) * qbar;
                at(active_cbar, n, j, k) -= at(C, n, i, k) * qbar;
            }
        }
    }
    return rhobar_lower;
}

Vec lower_rhobar_to_pair_greeks(
    const Mat& rhobar_lower,
    int n,
    const CorrelationPairs& pairs
) {
    Vec greeks(pairs.size(), 0.0);
    for (std::size_t p = 0; p < pairs.size(); ++p) {
        const int i = pairs[p].first;
        const int j = pairs[p].second;
        greeks[p] = at(rhobar_lower, n, j, i);
    }
    return greeks;
}

std::vector<Vec> standard_reverse_mapping(
    const Mat& C,
    int n,
    const CorrelationPairs& pairs
) {
    const int cbar_size = n * n;
    std::vector<Vec> mapping(
        pairs.size(), Vec(static_cast<std::size_t>(cbar_size), 0.0));

    for (int k = 0; k < cbar_size; ++k) {
        Mat basis(cbar_size, 0.0);
        basis[k] = 1.0;
        const Vec basis_greeks = lower_rhobar_to_pair_greeks(
            standard_reverse_cholesky(C, basis, n), n, pairs);
        for (std::size_t p = 0; p < pairs.size(); ++p) {
            mapping[p][k] = basis_greeks[p];
        }
    }
    return mapping;
}

Vec apply_reverse_mapping(
    const std::vector<Vec>& mapping,
    const Mat& Cbar
) {
    Vec greeks(mapping.size(), 0.0);
    for (std::size_t p = 0; p < mapping.size(); ++p) {
        if (mapping[p].size() != Cbar.size()) {
            throw std::invalid_argument("reverse mapping has the wrong size");
        }
        for (std::size_t k = 0; k < Cbar.size(); ++k) {
            greeks[p] += mapping[p][k] * Cbar[k];
        }
    }
    return greeks;
}
