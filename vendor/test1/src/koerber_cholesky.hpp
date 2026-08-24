#ifndef KOERBER_CHOLESKY_HPP
#define KOERBER_CHOLESKY_HPP

#include "pathwise_c_adjoint.hpp"

// Cbar = dP/dC
// rhobar = dP/drho
Mat koerber_cholesky_adjoint(const Mat& C, const Mat& Cbar, int n);

#endif
