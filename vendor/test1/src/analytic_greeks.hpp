#ifndef ANALYTIC_GREEKS_HPP
#define ANALYTIC_GREEKS_HPP

#include "pathwise_c_adjoint.hpp"

// Exact derivative of the expected squared Asian-basket payoff with respect
// to the single symmetric correlation parameter rho_ij = rho_ji.
double analytic_euler_correlation_greek(
    const PathwiseParams& params,
    const Mat& rho,
    int n,
    int i,
    int j,
    int steps
);

// Continuous-time GBM reference sampled at the same observation dates.
double analytic_continuous_correlation_greek(
    const PathwiseParams& params,
    const Mat& rho,
    int n,
    int i,
    int j,
    int observations
);

#endif
