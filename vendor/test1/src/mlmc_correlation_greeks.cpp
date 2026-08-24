#include "mlmc_correlation_greeks.hpp"
#include "options.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace {

constexpr int kMaximumLevels = 21;

struct CallbackState {
    const BenchmarkModel* model = nullptr;
    DifferentiationMode mode = DifferentiationMode::Adjoint;
    std::uint32_t seed = 0;
    std::vector<std::uint64_t> level_offsets;
};

CallbackState g_callback;

void mlmc_level_callback(
    int level,
    int samples,
    int outputs,
    double* sums
) {
    if (g_callback.model == nullptr) {
        throw std::runtime_error("MLMC callback has not been initialized");
    }
    if (outputs != static_cast<int>(g_callback.model->pairs.size())) {
        throw std::invalid_argument("unexpected MLMC output count");
    }

    const std::uint64_t offset = g_callback.level_offsets[level];
    g_callback.level_offsets[level] += static_cast<std::uint64_t>(samples);
    std::mt19937 rng(mixed_seed(g_callback.seed, level, offset));

    for (int sample = 0; sample < samples; ++sample) {
        const Vec correction = make_level_sample(
            *g_callback.model, g_callback.mode, level, rng);
        for (int output = 0; output < outputs; ++output) {
            const double value = correction[output];
            sums[1 + 4 * output] += value;
            sums[1 + 4 * output + 1] += value * value;
            sums[1 + 4 * output + 2] += value;
            sums[1 + 4 * output + 3] += value * value;
        }
        sums[0] += level == 0
            ? 1.0
            : static_cast<double>(1 << level);
    }
}

Moments level_moments(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    int level,
    std::uint64_t samples,
    std::uint32_t seed
) {
    Moments moments(static_cast<int>(model.pairs.size()));
    std::mt19937 rng(mixed_seed(seed, level, 0));
    for (std::uint64_t sample = 0; sample < samples; ++sample) {
        moments.add(make_level_sample(model, mode, level, rng));
    }
    return moments;
}

} // namespace

MlmcVectorStatistics plan_mlmc_correlation_greeks(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    int Lmin,
    int Lmax,
    int N0,
    float eps,
    float alpha,
    float beta,
    float gamma,
    std::uint32_t seed
) {
    g_callback.model = &model;
    g_callback.mode = mode;
    g_callback.seed = seed;
    g_callback.level_offsets.assign(kMaximumLevels, 0);

    return mlmc_vector(
        static_cast<int>(model.pairs.size()),
        Lmin,
        Lmax,
        N0,
        eps,
        mlmc_level_callback,
        alpha,
        beta,
        gamma);
}

Estimate fixed_mlmc_correlation_greeks(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    const std::vector<int>& samples_per_level,
    std::uint32_t seed
) {
    const int outputs = static_cast<int>(model.pairs.size());
    Estimate result;
    result.mean.assign(outputs, 0.0);
    Vec variance_of_mean(outputs, 0.0);

    for (int level = 0;
         level < static_cast<int>(samples_per_level.size());
         ++level) {
        const int level_samples = samples_per_level[level];
        const Moments moments = level_moments(
            model, mode, level, level_samples, seed);
        const Vec level_mean = moments.mean();
        const Vec level_variance = moments.sample_variance();
        for (int output = 0; output < outputs; ++output) {
            result.mean[output] += level_mean[output];
            variance_of_mean[output] +=
                level_variance[output]
                / static_cast<double>(level_samples);
        }

        const std::uint64_t fine_steps = static_cast<std::uint64_t>(
            euler_steps_for_level(level));
        const std::uint64_t actual_steps = level == 0
            ? fine_steps
            : fine_steps + fine_steps / 2ULL;
        result.samples += static_cast<std::uint64_t>(level_samples);
        result.reference_cost +=
            static_cast<std::uint64_t>(level_samples) * fine_steps;
        result.actual_euler_steps +=
            static_cast<std::uint64_t>(level_samples) * actual_steps;
    }

    result.standard_error.resize(outputs);
    for (int output = 0; output < outputs; ++output) {
        result.standard_error[output] = std::sqrt(variance_of_mean[output]);
        result.max_standard_error = std::max(
            result.max_standard_error, result.standard_error[output]);
    }
    return result;
}

double reconstructed_mlmc_bias(
    const MlmcVectorStatistics& plan,
    float alpha
) {
    double maximum = 0.0;
    for (std::size_t output = 0;
         output < plan.level_mean.size();
         ++output) {
        double adjusted = 0.0;
        for (int level = 0; level <= plan.final_level; ++level) {
            const double raw = std::abs(plan.level_mean[output][level]);
            adjusted = level > 1
                ? std::max(raw, 0.25 * adjusted)
                : raw;
        }
        maximum = std::max(maximum, adjusted);
    }
    return maximum / (std::pow(2.0, alpha) - 1.0);
}
