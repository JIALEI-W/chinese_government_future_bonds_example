#include "standard_monte_carlo.hpp"
#include "options.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

Moments fine_moments(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    int level,
    std::uint64_t first_sample,
    std::uint64_t samples,
    std::uint32_t seed
) {
    const int steps = euler_steps_for_level(level);
    Moments moments(static_cast<int>(model.pairs.size()));
    std::mt19937 rng(mixed_seed(seed, level, first_sample));
    for (std::uint64_t sample = 0; sample < samples; ++sample) {
        moments.add(evaluate_gradient(
            model, mode, make_fine_normals(steps, model.n, rng)));
    }
    return moments;
}

} // namespace

std::uint64_t plan_standard_samples(
    const BenchmarkModel& model,
    int level,
    int N0,
    double eps,
    double bias_fraction,
    std::uint32_t seed
) {
    Moments moments(static_cast<int>(model.pairs.size()));
    std::uint64_t additional = static_cast<std::uint64_t>(N0);
    constexpr std::uint64_t kSafetyLimit = 50000000ULL;

    while (true) {
        moments.merge(fine_moments(
            model,
            DifferentiationMode::Adjoint,
            level,
            moments.count,
            additional,
            seed));

        const Vec variance = moments.population_variance();
        std::uint64_t required = static_cast<std::uint64_t>(N0);
        for (double value : variance) {
            const double component_requirement =
                value / ((1.0 - bias_fraction) * eps * eps);
            required = std::max(
                required,
                static_cast<std::uint64_t>(
                    std::ceil(component_requirement)));
        }
        if (required > kSafetyLimit) {
            throw std::runtime_error(
                "standard-MC sample plan exceeds the safety limit");
        }

        additional = required > moments.count
            ? required - moments.count
            : 0;
        if (static_cast<double>(additional)
            <= 0.01 * static_cast<double>(moments.count)) {
            return moments.count;
        }
    }
}

Estimate standard_monte_carlo(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    int level,
    std::uint64_t samples,
    std::uint32_t seed
) {
    const Moments moments = fine_moments(
        model, mode, level, 0, samples, seed);

    Estimate result;
    result.mean = moments.mean();
    const Vec variance = moments.sample_variance();
    result.standard_error.resize(variance.size());
    for (std::size_t j = 0; j < variance.size(); ++j) {
        result.standard_error[j] =
            std::sqrt(variance[j] / static_cast<double>(samples));
        result.max_standard_error = std::max(
            result.max_standard_error, result.standard_error[j]);
    }
    result.samples = samples;
    result.reference_cost =
        samples * static_cast<std::uint64_t>(euler_steps_for_level(level));
    result.actual_euler_steps = result.reference_cost;
    return result;
}
