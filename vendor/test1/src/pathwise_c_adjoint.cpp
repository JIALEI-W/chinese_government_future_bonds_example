#include "pathwise_c_adjoint.hpp"

#include <cmath>

int idx(int n, int i, int j) {
    return i * n + j;
}

double& at(Mat& A, int n, int i, int j) {
    return A[idx(n, i, j)];
}

double at(const Mat& A, int n, int i, int j) {
    return A[idx(n, i, j)];
}

void add_in_place(Mat& target, const Mat& source) {
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] += source[i];
    }
}

void scale_in_place(Mat& target, double scale) {
    for (double& x : target) {
        x *= scale;
    }
}

// Compute Z = C * Ztilde 
Vec matvec_lower(const Mat& C, const Vec& ztilde) {
    // Correlated-normal step:
    //
    //   Z_tilde ~ N(0, I)
    //   Z       = C Z_tilde
    //
    // Component form:
    //
    //   Z_i = sum_{j <= i} C_ij Z_tilde_j
    //
    // Therefore:
    //
    //   Cov(Z) = C I C^T = C C^T = rho
    int N = ztilde.size();
    Vec Z(N, 0.0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            Z[i] += at(C, N, i, j) * ztilde[j];
        }
    }

    return Z;
}



Mat pathwise_cbar_gbm(
    const Mat& C, 
    const PathNormals& z_path,    // matrix of standard normals for one path, size = steps x n
    const PathwiseParams& params, // model parameters, including S0, r, sigma, T, K
    double& price_out             // output: P^(m) = payoff for this path
) {
    const int steps = z_path.size();
    const int n = z_path[0].size();
    const double h = params.T / double(steps);
    const double sqrt_h = std::sqrt(h);

    // Euler step for:
    //
    //   dS_t = r S_t dt + sigma S_t dW_t
    //
    // with:
    //
    //   dW_t ~= sqrt(h) Z_t
    //
    // Therefore:
    //
    //   S_{t+1,i}
    //     = S_{t,i} + r S_{t,i} h + sigma S_{t,i} sqrt(h) Z_{t,i}
    //     = S_{t,i} (1 + r h + sigma sqrt(h) Z_{t,i})
    //
    // The two constants below are the drift and diffusion coefficients in
    // this one-step multiplier.
    Vec drift_h(n, 0.0);
    Vec vol_sqrt_h(n, 0.0);
    for (int i = 0; i < n; ++i) {
        drift_h[i] = params.drift[i] * h;
        vol_sqrt_h[i] = params.sigma[i] * sqrt_h;
    }

    std::vector<Vec> Z_path(steps, Vec(n, 0.0));
    std::vector<Vec> S_path(steps + 1, Vec(n, 0.0));
    S_path[0] = params.S0;

    // Forward sweep:
    //
    //   z_t      ~ N(0, I)
    //   Z_t      = C z_t
    //   S_{t+1,i}= S_{t,i} (1 + r h + sigma sqrt(h) Z_{t,i})
    //
    // This is the C -> Z -> S part of the path calculation.
    for (int t = 0; t < steps; ++t) {
        Z_path[t] = matvec_lower(C, z_path[t]);

        for (int i = 0; i < n; ++i) {
            const double multiplier =
                1.0 + drift_h[i] + vol_sqrt_h[i] * Z_path[t][i];
            S_path[t + 1][i] = S_path[t][i] * multiplier;
        }
    }

    // Smooth Asian basket payoff:
    //
    //   A = (1/(steps*n)) sum_{t=1}^{steps} sum_i S_{t,i}
    //   P = (A - K)^2
    //
    // This is intentionally path-dependent. If the payoff only used S_T and
    // the payoff only used S_T, the fine/coarse correction would be much less
    // informative for this demo.
    double path_average = 0.0;
    for (int t = 1; t <= steps; ++t) {
        for (int i = 0; i < n; ++i) {
            path_average += S_path[t][i];
        }
    }
    path_average /= double(steps * n);

    const double payoff_arg = path_average - params.K;
    price_out = payoff_arg * payoff_arg; // P = (A - K)^2

    // Backward sweep:
    //
    //   dP/dS_{t,i}
    //     = 2(A - K) * dA/dS_{t,i}
    //     = 2(A - K) / (steps*n)
    //
    // for every observation date t = 1,...,steps.
    const double payoff_sbar =
        2.0 * payoff_arg / double(steps * n);

    std::vector<Vec> Sbar_path(steps + 1, Vec(n, 0.0));
    for (int t = 1; t <= steps; ++t) {
        for (int i = 0; i < n; ++i) {
            Sbar_path[t][i] += payoff_sbar;
        }
    }

    Mat Cbar(n * n, 0.0);

    for (int t = steps - 1; t >= 0; --t) {
        for (int i = 0; i < n; ++i) {
            // S_{t+1,i} = S_{t,i} (1 + r h + sigma sqrt(h) Z_{t,i})
            //
            // Hence:
            //
            // dS_{t+1,i}/dZ_{t,i}
            // = S_{t,i} sigma sqrt(h)
            //
            // and:
            //
            // Zbar_{t,i}
            // = Sbar_{t+1,i} S_{t,i} sigma sqrt(h)
            const double Zbar =
                Sbar_path[t + 1][i] * S_path[t][i] * vol_sqrt_h[i];

            // Sbar propagation:
            //
            //   dS_{t+1,i}/dS_{t,i}
            //     = 1 + r h + sigma sqrt(h) Z_{t,i}
            const double multiplier =
                1.0 + drift_h[i] + vol_sqrt_h[i] * Z_path[t][i];
            Sbar_path[t][i] +=
                Sbar_path[t + 1][i] * multiplier;

            for (int j = 0; j <= i; ++j) {
                // Cbar_{i,j}
                // = dP/dC_{i,j}
                // = sum_t Zbar_{t,i} z_{t,j}
                at(Cbar, n, i, j) += Zbar * z_path[t][j];
            }
        }
    }

    return Cbar;
}
