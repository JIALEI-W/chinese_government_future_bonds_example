#include "koerber_cholesky.hpp"

Mat koerber_cholesky_adjoint(const Mat& C, const Mat& Cbar, int n) {
    // For rho = C C^T with lower-triangular C, define
    //
    //   G = C^T Cbar.
    //
    // If Phi keeps the strict lower triangle and halves the diagonal, then
    // the symmetric middle factor is
    //
    //   H = sym(Phi(G)),
    //
    // where sym places half of a strict-lower entry in both symmetric
    // positions. The reverse Cholesky derivative is
    //
    //   rhobar = C^{-T} H C^{-1}.
    //
    // This convention satisfies
    //
    //   df = sum_ij rhobar_ij d rho_ij.
    //
    // Therefore a single symmetric off-diagonal parameter has derivative
    // rhobar_ij + rhobar_ji = 2*rhobar_ij.

    Mat G(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                at(G, n, i, j) += at(C, n, k, i) * at(Cbar, n, k, j);
            }
        }
    }

    Mat H(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        at(H, n, i, i) = 0.5 * at(G, n, i, i);
        for (int j = 0; j < i; ++j) {
            const double value = 0.5 * at(G, n, i, j);
            at(H, n, i, j) = value;
            at(H, n, j, i) = value;
        }
    }

    // X = H C^{-1}, evaluated as the right-side triangular solve X C = H.
    Mat X(n * n, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int j = n - 1; j >= 0; --j) {
            double value = at(H, n, row, j);
            for (int k = j + 1; k < n; ++k) {
                value -= at(X, n, row, k) * at(C, n, k, j);
            }
            at(X, n, row, j) = value / at(C, n, j, j);
        }
    }

    // rhobar = C^{-T} X, evaluated as C^T rhobar = X.
    Mat rhobar(n * n, 0.0);
    for (int column = 0; column < n; ++column) {
        for (int i = n - 1; i >= 0; --i) {
            double value = at(X, n, i, column);
            for (int k = i + 1; k < n; ++k) {
                value -= at(C, n, k, i) * at(rhobar, n, k, column);
            }
            at(rhobar, n, i, column) = value / at(C, n, i, i);
        }
    }

    // Roundoff can leave tiny asymmetry; return an explicitly symmetric
    // gradient because rho is a symmetric matrix input.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            const double value =
                0.5 * (at(rhobar, n, i, j) + at(rhobar, n, j, i));
            at(rhobar, n, i, j) = value;
            at(rhobar, n, j, i) = value;
        }
    }
    return rhobar;
}
