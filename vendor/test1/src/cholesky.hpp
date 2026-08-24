#ifndef CHOLESKY_HPP
#define CHOLESKY_HPP

#include "pathwise_c_adjoint.hpp"

// rho = C C^T
Mat cholesky_lower(const Mat& rho, int n);

#endif
