#include "statistics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Moments::Moments(int outputs)
    : sum(outputs, 0.0), sum_squares(outputs, 0.0) {}

void Moments::add(const Vec& value) {
    if (value.size() != sum.size()) {
        throw std::invalid_argument("moment vector has the wrong size");
    }
    ++count;
    for (std::size_t j = 0; j < value.size(); ++j) {
        sum[j] += value[j];
        sum_squares[j] += value[j] * value[j];
    }
}

void Moments::merge(const Moments& other) {
    count += other.count;
    for (std::size_t j = 0; j < sum.size(); ++j) {
        sum[j] += other.sum[j];
        sum_squares[j] += other.sum_squares[j];
    }
}

Vec Moments::mean() const {
    Vec result(sum.size(), 0.0);
    for (std::size_t j = 0; j < sum.size(); ++j) {
        result[j] = sum[j] / static_cast<double>(count);
    }
    return result;
}

Vec Moments::population_variance() const {
    const Vec averages = mean();
    Vec result(sum.size(), 0.0);
    for (std::size_t j = 0; j < sum.size(); ++j) {
        result[j] = std::max(
            sum_squares[j] / static_cast<double>(count)
                - averages[j] * averages[j],
            0.0);
    }
    return result;
}

Vec Moments::sample_variance() const {
    Vec result(sum.size(), 0.0);
    if (count < 2) {
        return result;
    }
    for (std::size_t j = 0; j < sum.size(); ++j) {
        const double centered = sum_squares[j]
            - sum[j] * sum[j] / static_cast<double>(count);
        result[j] = std::max(
            centered / static_cast<double>(count - 1), 0.0);
    }
    return result;
}

double max_abs_difference(const Vec& left, const Vec& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("vectors have different sizes");
    }
    double maximum = 0.0;
    for (std::size_t j = 0; j < left.size(); ++j) {
        maximum = std::max(maximum, std::abs(left[j] - right[j]));
    }
    return maximum;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 1
        ? values[middle]
        : 0.5 * (values[middle - 1] + values[middle]);
}
