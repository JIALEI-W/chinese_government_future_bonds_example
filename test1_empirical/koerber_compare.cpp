#include "benchmark_model.hpp"
#include "mlmc_correlation_greeks.hpp"
#include "mlmc_vector.hpp"
#include "options.hpp"
#include "path_sampling.hpp"
#include "statistics.hpp"
#include "standard_reverse_cholesky.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kMaximumLevels = 21;

struct LocalOptions {
    double eps = 0.01;
    int N0 = 200;
    int repeats = 5;
    int Lmin = 2;
    int Lmax = 20;
    std::uint32_t seed = 20260820U;
    fs::path output_dir = "../outputs/test1_empirical_koerber_eps001_optimized";
};

struct CallbackState {
    const BenchmarkModel* model = nullptr;
    std::uint32_t seed = 0;
    std::vector<std::uint64_t> level_offsets;
};

CallbackState g_callback;

Vec pathwise_optimized_koerber_gbm(
    const BenchmarkModel& model,
    const PathNormals& normals,
    bool full_matrix
) {
    const int steps = static_cast<int>(normals.size());
    const int n = model.n;
    const PathwiseParams& params = model.params;
    const double h = params.T / static_cast<double>(steps);
    const double sqrt_h = std::sqrt(h);

    std::vector<Vec> correlated(steps, Vec(n, 0.0));
    std::vector<Vec> paths(steps + 1, Vec(n, 0.0));
    paths[0] = params.S0;
    for (int t = 0; t < steps; ++t) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                correlated[t][i] +=
                    at(model.C, n, i, j) * normals[t][j];
            }
            const double multiplier = 1.0 + params.drift[i] * h
                + params.sigma[i] * sqrt_h * correlated[t][i];
            paths[t + 1][i] = paths[t][i] * multiplier;
        }
    }

    double average = 0.0;
    for (int t = 1; t <= steps; ++t) {
        for (int i = 0; i < n; ++i) {
            average += paths[t][i];
        }
    }
    average /= static_cast<double>(steps * n);
    const double payoff_sbar =
        2.0 * (average - params.K) / static_cast<double>(steps * n);

    std::vector<Vec> path_bar(steps + 1, Vec(n, 0.0));
    for (int t = 1; t <= steps; ++t) {
        for (int i = 0; i < n; ++i) {
            path_bar[t][i] = payoff_sbar;
        }
    }

    Vec rhobar(
        full_matrix ? static_cast<std::size_t>(n * n)
                    : model.pairs.size(),
        0.0
    );
    Vec wbar(n, 0.0);
    Vec transformed_normal(n, 0.0);
    for (int t = steps - 1; t >= 0; --t) {
        for (int i = 0; i < n; ++i) {
            wbar[i] = path_bar[t + 1][i] * paths[t][i]
                * params.sigma[i] * sqrt_h;
            const double multiplier = 1.0 + params.drift[i] * h
                + params.sigma[i] * sqrt_h * correlated[t][i];
            path_bar[t][i] += path_bar[t + 1][i] * multiplier;
        }

        // Apply the reordered Koerber formula directly at this Euler step:
        //
        //   C^T v_t = Z_t,
        //   H_t     = 0.5 Wbar_t v_t^T.
        //
        // This retains the rank-one structure and avoids first accumulating
        // a dense Cbar followed by the matrix solve H C = 0.5 Cbar.
        for (int i = n - 1; i >= 0; --i) {
            double value = normals[t][i];
            for (int k = i + 1; k < n; ++k) {
                value -= at(model.C, n, k, i)
                    * transformed_normal[k];
            }
            transformed_normal[i] = value / at(model.C, n, i, i);
        }

        if (full_matrix) {
            for (int i = 0; i < n; ++i) {
                at(rhobar, n, i, i) +=
                    0.5 * wbar[i] * transformed_normal[i];
                for (int j = 0; j < i; ++j) {
                    const double value = 0.5 * (
                        wbar[i] * transformed_normal[j]
                        + wbar[j] * transformed_normal[i]
                    );
                    at(rhobar, n, i, j) += value;
                    at(rhobar, n, j, i) += value;
                }
            }
        } else {
            for (std::size_t p = 0; p < model.pairs.size(); ++p) {
                const int i = model.pairs[p].first;
                const int j = model.pairs[p].second;
                rhobar[p] += 0.5 * (
                    wbar[i] * transformed_normal[j]
                    + wbar[j] * transformed_normal[i]
                );
            }
        }
    }
    return rhobar;
}

std::string require_value(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            std::string("missing value after ") + argv[index]
        );
    }
    return argv[++index];
}

LocalOptions parse_local_options(int argc, char** argv) {
    LocalOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--eps") {
            options.eps = std::stod(require_value(argc, argv, i));
        } else if (argument == "--N0") {
            options.N0 = std::stoi(require_value(argc, argv, i));
        } else if (argument == "--repeats") {
            options.repeats = std::stoi(require_value(argc, argv, i));
        } else if (argument == "--Lmin") {
            options.Lmin = std::stoi(require_value(argc, argv, i));
        } else if (argument == "--Lmax") {
            options.Lmax = std::stoi(require_value(argc, argv, i));
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(
                std::stoul(require_value(argc, argv, i))
            );
        } else if (argument == "--output-dir") {
            options.output_dir = require_value(argc, argv, i);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.eps <= 0.0 || options.N0 < 2 || options.repeats < 1
        || options.Lmin < 2 || options.Lmax < options.Lmin
        || options.Lmax > 20) {
        throw std::invalid_argument("invalid option");
    }
    return options;
}

Mat pathwise_full_cbar_gbm(
    const Mat& C,
    const PathNormals& normals,
    const PathwiseParams& params
) {
    const int steps = static_cast<int>(normals.size());
    const int n = static_cast<int>(params.S0.size());
    const double h = params.T / static_cast<double>(steps);
    const double sqrt_h = std::sqrt(h);

    std::vector<Vec> correlated(steps, Vec(n, 0.0));
    std::vector<Vec> paths(steps + 1, Vec(n, 0.0));
    paths[0] = params.S0;
    for (int t = 0; t < steps; ++t) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                correlated[t][i] += at(C, n, i, j) * normals[t][j];
            }
            const double multiplier = 1.0 + params.drift[i] * h
                + params.sigma[i] * sqrt_h * correlated[t][i];
            paths[t + 1][i] = paths[t][i] * multiplier;
        }
    }

    double average = 0.0;
    for (int t = 1; t <= steps; ++t) {
        for (int i = 0; i < n; ++i) {
            average += paths[t][i];
        }
    }
    average /= static_cast<double>(steps * n);
    const double payoff_sbar =
        2.0 * (average - params.K) / static_cast<double>(steps * n);

    std::vector<Vec> path_bar(steps + 1, Vec(n, 0.0));
    for (int t = 1; t <= steps; ++t) {
        for (int i = 0; i < n; ++i) {
            path_bar[t][i] = payoff_sbar;
        }
    }

    Mat cbar(n * n, 0.0);
    for (int t = steps - 1; t >= 0; --t) {
        for (int i = 0; i < n; ++i) {
            const double zbar = path_bar[t + 1][i] * paths[t][i]
                * params.sigma[i] * sqrt_h;
            const double multiplier = 1.0 + params.drift[i] * h
                + params.sigma[i] * sqrt_h * correlated[t][i];
            path_bar[t][i] += path_bar[t + 1][i] * multiplier;
            for (int j = 0; j < n; ++j) {
                at(cbar, n, i, j) += zbar * normals[t][j];
            }
        }
    }
    return cbar;
}

Vec koerber_gradient(
    const BenchmarkModel& model,
    const PathNormals& normals
) {
    return pathwise_optimized_koerber_gbm(model, normals, false);
}

Vec koerber_level_sample(
    const BenchmarkModel& model,
    int level,
    std::mt19937& rng
) {
    const int fine_steps = euler_steps_for_level(level);
    const PathNormals fine = make_fine_normals(fine_steps, model.n, rng);
    const Vec fine_greek = koerber_gradient(model, fine);
    if (level == 0) {
        return fine_greek;
    }
    const Vec coarse_greek = koerber_gradient(
        model, make_coarse_normals(fine)
    );
    Vec correction(fine_greek.size(), 0.0);
    for (std::size_t p = 0; p < correction.size(); ++p) {
        correction[p] = fine_greek[p] - coarse_greek[p];
    }
    return correction;
}

void koerber_level_callback(
    int level,
    int samples,
    int outputs,
    double* sums
) {
    const std::uint64_t offset = g_callback.level_offsets[level];
    g_callback.level_offsets[level] += static_cast<std::uint64_t>(samples);
    std::mt19937 rng(mixed_seed(g_callback.seed, level, offset));
    for (int sample = 0; sample < samples; ++sample) {
        const Vec correction = koerber_level_sample(
            *g_callback.model, level, rng
        );
        for (int output = 0; output < outputs; ++output) {
            const double value = correction[output];
            sums[1 + 4 * output] += value;
            sums[1 + 4 * output + 1] += value * value;
            sums[1 + 4 * output + 2] += value;
            sums[1 + 4 * output + 3] += value * value;
        }
        sums[0] += level == 0
            ? 1.0 : static_cast<double>(1 << level);
    }
}

MlmcVectorStatistics plan_koerber(
    const BenchmarkModel& model,
    int Lmin,
    int Lmax,
    int N0,
    double eps,
    std::uint32_t seed
) {
    g_callback.model = &model;
    g_callback.seed = seed;
    g_callback.level_offsets.assign(kMaximumLevels, 0);
    return mlmc_vector(
        static_cast<int>(model.pairs.size()),
        Lmin,
        Lmax,
        N0,
        static_cast<float>(eps),
        koerber_level_callback,
        kMlmcAlpha,
        kMlmcBeta,
        kMlmcGamma
    );
}

Estimate fixed_koerber(
    const BenchmarkModel& model,
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
        const int samples = samples_per_level[level];
        Moments moments(outputs);
        std::mt19937 rng(mixed_seed(seed, level, 0));
        for (int sample = 0; sample < samples; ++sample) {
            moments.add(koerber_level_sample(model, level, rng));
        }
        const Vec mean = moments.mean();
        const Vec variance = moments.sample_variance();
        for (int output = 0; output < outputs; ++output) {
            result.mean[output] += mean[output];
            variance_of_mean[output] +=
                variance[output] / static_cast<double>(samples);
        }
        const std::uint64_t fine_steps = euler_steps_for_level(level);
        const std::uint64_t actual_steps = level == 0
            ? fine_steps : fine_steps + fine_steps / 2ULL;
        result.samples += static_cast<std::uint64_t>(samples);
        result.reference_cost += static_cast<std::uint64_t>(samples)
            * fine_steps;
        result.actual_euler_steps += static_cast<std::uint64_t>(samples)
            * actual_steps;
    }

    result.standard_error.resize(outputs);
    for (int output = 0; output < outputs; ++output) {
        result.standard_error[output] = std::sqrt(variance_of_mean[output]);
        result.max_standard_error = std::max(
            result.max_standard_error, result.standard_error[output]
        );
    }
    return result;
}

Mat demo_standard_rhobar(
    const BenchmarkModel& model,
    const Mat& cbar
) {
    const Mat lower = standard_reverse_cholesky(
        model.C, cbar, model.n
    );
    Mat full(model.n * model.n, 0.0);
    for (int i = 0; i < model.n; ++i) {
        at(full, model.n, i, i) = at(lower, model.n, i, i);
        for (int j = 0; j < i; ++j) {
            const double value = at(lower, model.n, i, j);
            at(full, model.n, i, j) = value;
            at(full, model.n, j, i) = value;
        }
    }
    return full;
}

Mat demo_koerber_rhobar(
    const BenchmarkModel& model,
    const Mat& cbar
) {
    const int n = model.n;
    Mat B(n * n, 0.0);
    // Exact implementation from /Users/km/demo/koerber_cholesky.cpp:
    // B C = Cbar, hence B = Cbar C^{-1}.
    for (int row = 0; row < n; ++row) {
        for (int j = n - 1; j >= 0; --j) {
            double value = at(cbar, n, row, j);
            for (int k = j + 1; k < n; ++k) {
                value -= at(model.C, n, k, j) * at(B, n, row, k);
            }
            at(B, n, row, j) = value / at(model.C, n, j, j);
        }
    }
    Mat rhobar(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        at(rhobar, n, i, i) = 0.5 * at(B, n, i, i);
        for (int j = 0; j < i; ++j) {
            const double value =
                0.5 * (at(B, n, i, j) + at(B, n, j, i));
            at(rhobar, n, i, j) = value;
            at(rhobar, n, j, i) = value;
        }
    }
    return rhobar;
}

Vec full_rhobar_level_sample(
    const BenchmarkModel& model,
    int level,
    std::mt19937& rng,
    bool use_koerber
) {
    const int fine_steps = euler_steps_for_level(level);
    const PathNormals fine = make_fine_normals(fine_steps, model.n, rng);
    const Mat fine_rhobar = use_koerber
        ? pathwise_optimized_koerber_gbm(model, fine, true)
        : demo_standard_rhobar(
            model,
            pathwise_full_cbar_gbm(model.C, fine, model.params)
        );
    if (level == 0) {
        return fine_rhobar;
    }
    const PathNormals coarse = make_coarse_normals(fine);
    const Mat coarse_rhobar = use_koerber
        ? pathwise_optimized_koerber_gbm(model, coarse, true)
        : demo_standard_rhobar(
            model,
            pathwise_full_cbar_gbm(model.C, coarse, model.params)
        );
    Vec correction(fine_rhobar.size(), 0.0);
    for (std::size_t k = 0; k < correction.size(); ++k) {
        correction[k] = fine_rhobar[k] - coarse_rhobar[k];
    }
    return correction;
}

void validate_optimized_koerber(const BenchmarkModel& model) {
    std::mt19937 rng(0x6b6f6572U);
    const PathNormals normals = make_fine_normals(8, model.n, rng);
    const Mat reference = demo_koerber_rhobar(
        model,
        pathwise_full_cbar_gbm(model.C, normals, model.params)
    );
    const Mat optimized = pathwise_optimized_koerber_gbm(
        model, normals, true
    );
    double maximum_error = 0.0;
    for (std::size_t k = 0; k < reference.size(); ++k) {
        maximum_error = std::max(
            maximum_error,
            std::abs(reference[k] - optimized[k])
        );
    }
    if (maximum_error > 1.0e-11) {
        throw std::runtime_error(
            "optimized Koerber calculation failed equivalence check"
        );
    }
}

Estimate fixed_full_rhobar(
    const BenchmarkModel& model,
    const std::vector<int>& samples_per_level,
    std::uint32_t seed,
    bool use_koerber
) {
    const int outputs = model.n * model.n;
    Estimate result;
    result.mean.assign(outputs, 0.0);
    Vec variance_of_mean(outputs, 0.0);
    for (int level = 0;
         level < static_cast<int>(samples_per_level.size());
         ++level) {
        const int samples = samples_per_level[level];
        Moments moments(outputs);
        std::mt19937 rng(mixed_seed(seed, level, 0));
        for (int sample = 0; sample < samples; ++sample) {
            moments.add(full_rhobar_level_sample(
                model, level, rng, use_koerber
            ));
        }
        const Vec mean = moments.mean();
        const Vec variance = moments.sample_variance();
        for (int output = 0; output < outputs; ++output) {
            result.mean[output] += mean[output];
            variance_of_mean[output] +=
                variance[output] / static_cast<double>(samples);
        }
    }
    result.standard_error.resize(outputs);
    for (int output = 0; output < outputs; ++output) {
        result.standard_error[output] = std::sqrt(variance_of_mean[output]);
    }
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? 0.5 * (values[middle - 1] + values[middle])
        : values[middle];
}

void write_full_matrix(
    const fs::path& path,
    const BenchmarkModel& model,
    const Vec& values
) {
    const std::vector<std::string> names = {"TF", "TL", "T", "TS"};
    std::ofstream output(path);
    output << "instrument";
    for (const std::string& name : names) {
        output << ',' << name;
    }
    output << '\n' << std::setprecision(17);
    for (int i = 0; i < model.n; ++i) {
        output << names[i];
        for (int j = 0; j < model.n; ++j) {
            output << ',' << at(values, model.n, i, j);
        }
        output << '\n';
    }
}

int run(int argc, char** argv) {
    const LocalOptions options = parse_local_options(argc, argv);
    const BenchmarkModel model = make_benchmark_model(4, options.seed);
    validate_optimized_koerber(model);
    const int controlled_Lmin = minimum_bias_controlled_level(
        model,
        options.eps,
        kBiasFraction,
        options.Lmin,
        options.Lmax
    );
    const std::uint32_t plan_seed = options.seed ^ 0x11111111U;
    const std::uint32_t sample_seed_value = 0x327b23c6U;

    MlmcVectorStatistics standard_plan =
        plan_mlmc_correlation_greeks(
            model,
            DifferentiationMode::Adjoint,
            controlled_Lmin,
            options.Lmax,
            options.N0,
            static_cast<float>(options.eps),
            kMlmcAlpha,
            kMlmcBeta,
            kMlmcGamma,
            plan_seed
        );
    MlmcVectorStatistics koerber_plan = plan_koerber(
        model,
        controlled_Lmin,
        options.Lmax,
        options.N0,
        options.eps,
        plan_seed
    );
    // Same fixed MLMC work for an apples-to-apples end-to-end comparison.
    // These are the exact daily-11:30 counts from real_data_mlmc.
    const std::vector<int> authoritative_samples = {
        5323139, 1942505, 686779, 287018,
        109414, 40606, 14837, 5366
    };
    standard_plan.final_level = 7;
    standard_plan.samples_per_level = authoritative_samples;
    koerber_plan.final_level = 7;
    koerber_plan.samples_per_level = authoritative_samples;

    std::vector<double> standard_times;
    std::vector<double> koerber_times;
    Estimate standard_result;
    Estimate koerber_result;
    for (int repetition = 0; repetition < options.repeats; ++repetition) {
        if (repetition % 2 == 0) {
            auto start = Clock::now();
            standard_result = fixed_mlmc_correlation_greeks(
                model,
                DifferentiationMode::Adjoint,
                standard_plan.samples_per_level,
                sample_seed_value
            );
            auto stop = Clock::now();
            standard_times.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count()
            );

            start = Clock::now();
            koerber_result = fixed_koerber(
                model, koerber_plan.samples_per_level, sample_seed_value
            );
            stop = Clock::now();
            koerber_times.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count()
            );
        } else {
            auto start = Clock::now();
            koerber_result = fixed_koerber(
                model, koerber_plan.samples_per_level, sample_seed_value
            );
            auto stop = Clock::now();
            koerber_times.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count()
            );

            start = Clock::now();
            standard_result = fixed_mlmc_correlation_greeks(
                model,
                DifferentiationMode::Adjoint,
                standard_plan.samples_per_level,
                sample_seed_value
            );
            stop = Clock::now();
            standard_times.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count()
            );
        }
    }

    const double standard_bias = reconstructed_mlmc_bias(
        standard_plan, kMlmcAlpha
    );
    const double koerber_bias = reconstructed_mlmc_bias(
        koerber_plan, kMlmcAlpha
    );
    const double standard_rmse = std::hypot(
        standard_result.max_standard_error, standard_bias
    );
    const double koerber_rmse = std::hypot(
        koerber_result.max_standard_error, koerber_bias
    );
    if (standard_rmse > options.eps || koerber_rmse > options.eps) {
        throw std::runtime_error("a reverse method failed the common RMSE target");
    }

    fs::create_directories(options.output_dir);
    std::ofstream repetitions(options.output_dir / "koerber_timing_repetitions.csv");
    repetitions << "method,repetition,runtime_ms\n";
    for (int i = 0; i < options.repeats; ++i) {
        repetitions << "standard," << i + 1 << ','
                    << std::setprecision(17) << standard_times[i] << '\n'
                    << "koerber," << i + 1 << ','
                    << koerber_times[i] << '\n';
    }

    std::ofstream summary(options.output_dir / "koerber_comparison.csv");
    summary << "method,runtime_ms,samples,actual_euler_steps,final_level,"
            << "max_standard_error,bias_bound,estimated_rmse\n"
            << std::setprecision(17)
            << "standard," << median(standard_times) << ','
            << standard_result.samples << ','
            << standard_result.actual_euler_steps << ','
            << standard_plan.final_level << ','
            << standard_result.max_standard_error << ','
            << standard_bias << ',' << standard_rmse << '\n'
            << "koerber," << median(koerber_times) << ','
            << koerber_result.samples << ','
            << koerber_result.actual_euler_steps << ','
            << koerber_plan.final_level << ','
            << koerber_result.max_standard_error << ','
            << koerber_bias << ',' << koerber_rmse << '\n';

    const Estimate standard_full = fixed_full_rhobar(
        model, authoritative_samples, sample_seed_value, false
    );
    const Estimate koerber_full = fixed_full_rhobar(
        model, authoritative_samples, sample_seed_value, true
    );
    Vec common_expectation(standard_full.mean.size(), 0.0);
    Vec standard_estimator_variance(standard_full.standard_error.size());
    Vec koerber_estimator_variance(koerber_full.standard_error.size());
    for (std::size_t p = 0; p < common_expectation.size(); ++p) {
        // The theorem gives equal expectations.  Pool the two estimates to
        // report one common finite-sample estimate for both methods.
        common_expectation[p] =
            0.5 * (standard_full.mean[p] + koerber_full.mean[p]);
        standard_estimator_variance[p] =
            standard_full.standard_error[p]
            * standard_full.standard_error[p];
        koerber_estimator_variance[p] =
            koerber_full.standard_error[p]
            * koerber_full.standard_error[p];
    }
    write_full_matrix(
        options.output_dir / "standard_rhobar_expectation.csv",
        model,
        common_expectation
    );
    write_full_matrix(
        options.output_dir / "standard_rhobar_estimator_variance.csv",
        model,
        standard_estimator_variance
    );
    write_full_matrix(
        options.output_dir / "koerber_rhobar_expectation.csv",
        model,
        common_expectation
    );
    write_full_matrix(
        options.output_dir / "koerber_rhobar_estimator_variance.csv",
        model,
        koerber_estimator_variance
    );

    std::cout << "standard median ms: " << median(standard_times) << '\n'
              << "koerber median ms: " << median(koerber_times) << '\n'
              << "standard RMSE: " << standard_rmse << '\n'
              << "koerber RMSE: " << koerber_rmse << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
