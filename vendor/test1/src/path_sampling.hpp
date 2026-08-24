#ifndef PATH_SAMPLING_HPP
#define PATH_SAMPLING_HPP

#include "benchmark_model.hpp"

#include <cstdint>
#include <random>

enum class DifferentiationMode {
    Forward,
    Adjoint
};

std::uint32_t mixed_seed(
    std::uint32_t base_seed,
    int level,
    std::uint64_t offset
);

PathNormals make_fine_normals(int steps, int n, std::mt19937& rng);
PathNormals make_coarse_normals(const PathNormals& fine);

Vec evaluate_gradient(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    const PathNormals& normals
);

Vec make_level_sample(
    const BenchmarkModel& model,
    DifferentiationMode mode,
    int level,
    std::mt19937& rng
);

#endif
