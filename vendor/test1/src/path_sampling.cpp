#include "path_sampling.hpp"

#include "pathwise_c_adjoint.hpp"
#include "standard_reverse_cholesky.hpp"
#include "options.hpp"

#include <cmath>

std::uint32_t mixed_seed(
    std::uint32_t base_seed,
    int level,
    std::uint64_t offset
) {
    std::uint64_t value = offset
        ^ (static_cast<std::uint64_t>(base_seed) << 32)
        ^ (0x9e3779b97f4a7c15ULL
           * static_cast<std::uint64_t>(level + 1));
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<std::uint32_t>(value ^ (value >> 32));
}

PathNormals make_fine_normals(int steps, int n, std::mt19937& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    PathNormals values(steps, Vec(n, 0.0));
    for (Vec& value : values) {
        for (double& component : value) {
            component = normal(rng);
        }
    }
    return values;
}

PathNormals make_coarse_normals(const PathNormals& fine) {
    const int fine_steps = static_cast<int>(fine.size());
    const int n = static_cast<int>(fine.front().size());
    PathNormals coarse(fine_steps / 2, Vec(n, 0.0));
    const double inverse_sqrt_two = 1.0 / std::sqrt(2.0);
    for (int t = 0; t < fine_steps / 2; ++t) {
        for (int i = 0; i < n; ++i) {
            coarse[t][i] =
                (fine[2 * t][i] + fine[2 * t + 1][i])
                * inverse_sqrt_two;
        }
    }
    return coarse;
}

Vec evaluate_gradient(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    const PathNormals& normals
) {
    double price = 0.0;
    if (mode == DifferentiationMode::Forward) {
        return pathwise_forward_correlation_greeks(
            model.C, model.Cdot, normals, model.params, price);
    }
    const Mat Cbar = pathwise_cbar_gbm(
        model.C, normals, model.params, price);
    return apply_reverse_mapping(model.reverse_mapping, Cbar);
}

Vec make_level_sample(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    int level,
    std::mt19937& rng
) {
    const int fine_steps = euler_steps_for_level(level);
    const PathNormals fine = make_fine_normals(fine_steps, model.n, rng);
    if (level == 0) {
        return evaluate_gradient(model, mode, fine);
    }
    const PathNormals coarse = make_coarse_normals(fine);
    const Vec fine_greek = evaluate_gradient(model, mode, fine);
    const Vec coarse_greek = evaluate_gradient(model, mode, coarse);

    Vec correction(fine_greek.size(), 0.0);
    for (std::size_t j = 0; j < correction.size(); ++j) {
        correction[j] = fine_greek[j] - coarse_greek[j];
    }
    return correction;
}
