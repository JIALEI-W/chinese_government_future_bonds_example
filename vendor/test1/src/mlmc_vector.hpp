#ifndef MLMC_VECTOR_HPP
#define MLMC_VECTOR_HPP

#include <vector>

using MlmcLevelFunction =
    void (*)(int level, int samples, int outputs, double* sums);

struct MlmcVectorStatistics {
    int final_level = 0;
    std::vector<double> estimate;
    std::vector<double> standard_error;
    std::vector<int> samples_per_level;
    std::vector<double> cost_per_level;

    // Indexed as [output][level]. Variances are unbiased sample variances of
    // the MLMC correction Y_l, before division by N_l.
    std::vector<std::vector<double>> level_mean;
    std::vector<std::vector<double>> level_variance;
};

// Generic vector-output MLMC driver. N0 is also the minimum pilot sample
// count used whenever a new level is introduced, so every active level has a
// meaningful variance estimate.
MlmcVectorStatistics mlmc_vector(
    int outputs,
    int Lmin,
    int Lmax,
    int N0,
    float eps,
    MlmcLevelFunction level_function,
    float alpha_0,
    float beta_0,
    float gamma_0
);

#endif
