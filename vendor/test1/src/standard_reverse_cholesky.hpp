#ifndef STANDARD_REVERSE_CHOLESKY_HPP
#define STANDARD_REVERSE_CHOLESKY_HPP

#include "forward_sensitivity.hpp"

// Reverse the elementwise lower-triangular Cholesky algorithm. The returned
// matrix stores active derivatives in its lower triangle.
Mat standard_reverse_cholesky(
    const Mat& C,
    const Mat& Cbar,
    int n
);

Vec lower_rhobar_to_pair_greeks(
    const Mat& rhobar_lower,
    int n,
    const CorrelationPairs& pairs
);

// Precompute the linear map Cbar -> independent symmetric-correlation Greeks.
// mapping[p][k] multiplies Cbar[k] for correlation direction p.
std::vector<Vec> standard_reverse_mapping(
    const Mat& C,
    int n,
    const CorrelationPairs& pairs
);

Vec apply_reverse_mapping(
    const std::vector<Vec>& mapping,
    const Mat& Cbar
);

#endif
