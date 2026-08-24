#ifndef BENCHMARK_VALIDATION_HPP
#define BENCHMARK_VALIDATION_HPP

#include "benchmark_model.hpp"

struct ValidationChecks {
    double cholesky = 0.0;
    double forward_adjoint = 0.0;
    double finite_difference = 0.0;
    double standard_reference_reverse = 0.0;
    double coupled_correction = 0.0;
};

ValidationChecks run_validation_checks(const BenchmarkModel& model);

#endif
