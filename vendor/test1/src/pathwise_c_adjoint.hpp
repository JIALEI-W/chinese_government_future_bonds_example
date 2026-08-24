#ifndef PATHWISE_C_ADJOINT_HPP
#define PATHWISE_C_ADJOINT_HPP

// pathwise_c_adjoint.hpp
//
// This file declares the GBM path calculation and the pathwise adjoint
// derivative with respect to the Cholesky factor C:
//
//   C -> Z -> S -> P
//
// Output:
//
//   price_out = P^(m)
//   Cbar      = dP^(m) / dC
//
// Componentwise:
//
//   Cbar_ij = dP^(m) / dC_ij
//
// Here m denotes one Monte Carlo path/sample. Cbar is stored as an n x n
// row-major matrix in Mat.

#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<double>;
using PathNormals = std::vector<Vec>;

struct PathwiseParams {
    Vec S0;
    Vec drift;
    Vec sigma;
    double T = 1.0;
    double K = 100.0;
};

// Store an n x n matrix in a 1D vector using row-major order:
//
//   A(i,j)  <->  A[i * n + j]
int idx(int n, int i, int j);
double& at(Mat& A, int n, int i, int j);
double at(const Mat& A, int n, int i, int j);

// Add the elements of source to target in place.
void add_in_place(Mat& target, const Mat& source);

// Scale the elements of target in place.
void scale_in_place(Mat& target, double scale);

Vec matvec_lower(const Mat& C, const Vec& ztilde);

Mat pathwise_cbar_gbm(
    const Mat& C,
    const PathNormals& z_path,
    const PathwiseParams& params,
    double& price_out
);

#endif
