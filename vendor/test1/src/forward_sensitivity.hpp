#ifndef FORWARD_SENSITIVITY_HPP
#define FORWARD_SENSITIVITY_HPP

#include "pathwise_c_adjoint.hpp"

#include <utility>
#include <vector>

using CorrelationPairs = std::vector<std::pair<int, int>>;
using CholeskyTangents = std::vector<Mat>;

// Independent symmetric parameters are ordered
// (0,1), (0,2), ..., (n-2,n-1).
CorrelationPairs independent_correlation_pairs(int n);

// Differentiate the lower-triangular Cholesky algorithm once in every
// independent symmetric-correlation direction.
CholeskyTangents forward_cholesky_tangents(
    const Mat& C,
    int n,
    const CorrelationPairs& pairs
);

// Traditional forward-mode pathwise differentiation of
// rho -> C -> correlated normals -> Euler path -> payoff.
Vec pathwise_forward_correlation_greeks(
    const Mat& C,
    const CholeskyTangents& Cdot,
    const PathNormals& z_path,
    const PathwiseParams& params,
    double& price_out
);

#endif
