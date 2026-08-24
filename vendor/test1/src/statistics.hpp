#ifndef BENCHMARK_STATISTICS_HPP
#define BENCHMARK_STATISTICS_HPP

#include "pathwise_c_adjoint.hpp"

#include <cstdint>
#include <vector>

struct Moments {
    std::uint64_t count = 0;
    Vec sum;
    Vec sum_squares;

    explicit Moments(int outputs = 0);
    void add(const Vec& value);
    void merge(const Moments& other);
    Vec mean() const;
    Vec population_variance() const;
    Vec sample_variance() const;
};

struct Estimate {
    Vec mean;
    Vec standard_error;
    double max_standard_error = 0.0;
    std::uint64_t samples = 0;
    std::uint64_t reference_cost = 0;
    std::uint64_t actual_euler_steps = 0;
};

double max_abs_difference(const Vec& left, const Vec& right);
double median(std::vector<double> values);

#endif
