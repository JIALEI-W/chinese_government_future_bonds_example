#include "validation.hpp"

#include "cholesky.hpp"
#include "koerber_cholesky.hpp"
#include "path_sampling.hpp"
#include "statistics.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

ValidationChecks run_validation_checks(const BenchmarkModel& model) {
    ValidationChecks checks;
    for (int i = 0; i < model.n; ++i) {
        for (int j = 0; j < model.n; ++j) {
            double reconstructed = 0.0;
            for (int k = 0; k < model.n; ++k) {
                reconstructed +=
                    at(model.C, model.n, i, k)
                    * at(model.C, model.n, j, k);
            }
            checks.cholesky = std::max(
                checks.cholesky,
                std::abs(reconstructed - at(model.rho, model.n, i, j)));
        }
    }

    std::mt19937 rng(987654321U);
    const PathNormals normals = make_fine_normals(8, model.n, rng);
    double forward_price = 0.0;
    const Vec forward = pathwise_forward_correlation_greeks(
        model.C,
        model.Cdot,
        normals,
        model.params,
        forward_price);
    double adjoint_price = 0.0;
    const Mat Cbar = pathwise_cbar_gbm(
        model.C, normals, model.params, adjoint_price);
    const Vec adjoint = apply_reverse_mapping(model.reverse_mapping, Cbar);
    checks.forward_adjoint = max_abs_difference(forward, adjoint);

    const Mat reference_reverse =
        koerber_cholesky_adjoint(model.C, Cbar, model.n);
    Vec reference_pairs(model.pairs.size(), 0.0);
    for (std::size_t p = 0; p < model.pairs.size(); ++p) {
        reference_pairs[p] = 2.0 * at(
            reference_reverse,
            model.n,
            model.pairs[p].first,
            model.pairs[p].second);
    }
    checks.standard_reference_reverse =
        max_abs_difference(adjoint, reference_pairs);

    constexpr double h = 1.0e-6;
    Vec finite_difference(model.pairs.size(), 0.0);
    for (std::size_t p = 0; p < model.pairs.size(); ++p) {
        Mat plus = model.rho;
        Mat minus = model.rho;
        const int i = model.pairs[p].first;
        const int j = model.pairs[p].second;
        at(plus, model.n, i, j) += h;
        at(plus, model.n, j, i) += h;
        at(minus, model.n, i, j) -= h;
        at(minus, model.n, j, i) -= h;

        double plus_price = 0.0;
        double minus_price = 0.0;
        (void)pathwise_cbar_gbm(
            cholesky_lower(plus, model.n),
            normals,
            model.params,
            plus_price);
        (void)pathwise_cbar_gbm(
            cholesky_lower(minus, model.n),
            normals,
            model.params,
            minus_price);
        finite_difference[p] = (plus_price - minus_price) / (2.0 * h);
    }
    checks.finite_difference =
        max_abs_difference(forward, finite_difference);

    std::mt19937 coupled_rng(246813579U);
    const Vec forward_correction = make_level_sample(
        model, DifferentiationMode::Forward, 3, coupled_rng);
    coupled_rng.seed(246813579U);
    const Vec adjoint_correction = make_level_sample(
        model, DifferentiationMode::Adjoint, 3, coupled_rng);
    checks.coupled_correction =
        max_abs_difference(forward_correction, adjoint_correction);

    if (checks.cholesky > 1.0e-11
        || checks.forward_adjoint > 1.0e-9
        || checks.finite_difference > 1.0e-4
        || checks.standard_reference_reverse > 1.0e-9
        || checks.coupled_correction > 1.0e-9
        || std::abs(forward_price - adjoint_price) > 1.0e-12) {
        throw std::runtime_error("a numerical consistency check failed");
    }
    return checks;
}
