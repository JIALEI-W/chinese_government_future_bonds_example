#include "mlmc_vector.hpp"

/*
   P = mlmc_vector(M,Lmin,Lmax,N0,eps, mlmc_l, alpha,beta,gamma, Nl,Cl)

   vector-output multilevel Monte Carlo control routine

   This file is intentionally written to look like mlmc.cpp.
   The scalar code has one output Y.
   This vector code has M outputs:

       Y_j,  j = 0,...,M-1

   Scalar mlmc_l:

       mlmc_l(l,N,sums)

       sums[0] = sum(cost)
       sums[1] = sum(Y)
       sums[2] = sum(Y^2)

   Vector mlmc_l:

       mlmc_l(l,N,M,sums)

       sums[0]           = sum(cost)
       sums[1 + 4*j]     = sum(Y_j)
       sums[1 + 4*j + 1] = sum(Y_j^2)
       sums[1 + 4*j + 2] = sum(Pf_j)
       sums[1 + 4*j + 3] = sum(Pf_j^2)

   For each component j:

       E[P_0^{(j)}]            on level 0
       E[P_l^{(j)}-P_{l-1}^{(j)}] on level l>0

   The same N_l is used for all outputs on one level.
   Therefore, after computing the optimal N_l required by each component,
   we take the maximum over j.
*/

#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

void regression_vector(int, float *, float *, float &a, float &b);

MlmcVectorStatistics mlmc_vector(int M, int Lmin, int Lmax, int N0,
           float eps,
           MlmcLevelFunction mlmc_l,
           float alpha_0, float beta_0, float gamma_0) {

  const int max_levels = 21;
  int Nl[max_levels] = {};
  float Cl[max_levels] = {};

  std::vector<double> sums(1 + 4*M, 0.0); // here M is output dimension
                                          // sums[0] = sum(cost)
                                          // each output stores:
                                          // sum(Y_j), sum(Y_j^2), sum(Pf_j), sum(Pf_j^2)，
                                          //j = 0,...,M-1

  // suml[n][j][l]: , n is index of sum type, j = 0,...,M-1, l = 0,...,Lmax
  //
  //   suml[0][j][l] = total number of samples at level l
  //                   same value for every j
  //   suml[1][j][l] = sum(Y_j)   for level l
  //   suml[2][j][l] = sum(Y_j^2) for level l
  //
  // This is the vector-output analogue of scalar:
  //
  //   suml[0][l] = total number of samples at level l
  //   suml[1][l] = sum(Y)
  //   suml[2][l] = sum(Y^2)
  //
  std::vector<std::vector<std::vector<double> > > suml(
      3, std::vector<std::vector<double> >(
             M, std::vector<double>(max_levels, 0.0)));

  // ml[j][l] = |E[P_l^{(j)} - P_{l-1}^{(j)}]|
  // Vl[j][l] = Var[P_l^{(j)} - P_{l-1}^{(j)}]
  std::vector<std::vector<float> > ml(M, std::vector<float>(max_levels, 0.0f));
  std::vector<std::vector<float> > Vl(M, std::vector<float>(max_levels, 0.0f));

  std::vector<float> alpha(M), beta(M);
  // 21 is the maximum number of levels allowed
  float NlCl[21], x[21], y[21],
        gamma, sum, theta;
  int   dNl[21], L, converged;

  int   diag = 0;  // diagnostics, set to 0 for none

  //
  // check input parameters
  //

  if (M<=0) {
    fprintf(stderr,"error: needs M > 0 \n");
    exit(1);
  }
  if (Lmin<2) {
    fprintf(stderr,"error: needs Lmin >= 2 \n");
    exit(1);
  }
  if (Lmax<Lmin) {
    fprintf(stderr,"error: needs Lmax >= Lmin \n");
    exit(1);
  }
  if (Lmax>=max_levels) {
    fprintf(stderr,"error: needs Lmax <= 20 \n");
    exit(1);
  }
  if (N0<=0 || eps<=0.0f) {
    fprintf(stderr,"error: needs N0>0 and eps>0 \n");
    exit(1);
  }
  //
  // initialisation
  //

  for (int j=0; j<M; j++) {
    alpha[j] = fmaxf(0.0f, alpha_0);
    beta[j]  = fmaxf(0.0f, beta_0);
  }
  gamma = fmaxf(0.0f, gamma_0); // gamma is for cost C_l = O(2^{gamma l}), thus not vector
  theta = 0.25f;             // MSE split between bias^2 and variance

  L = Lmin;
  converged = 0;

  for(int l=0; l<=Lmax; l++) {
    Nl[l]   = 0;
    Cl[l]   = powf(2.0f,(float)l*gamma);
    NlCl[l] = 0.0f;
    dNl[l]  = 0;

    for(int j=0; j<M; j++) {
      for(int n=0; n<3; n++) suml[n][j][l] = 0.0;
    }
  }

  for(int l=0; l<=Lmin; l++) dNl[l] = N0;

  //
  // main loop
  //

  while (!converged) {

    //
    // update sample sums
    //

    if (diag) {
      for (int l=0; l<=L; l++) printf(" %d ",dNl[l]);
      printf(" \n");
    }

    for (int l=0; l<=L; l++) {
      if (dNl[l]>0) {
        for(int n=0; n<1+4*M; n++) sums[n] = 0.0;

        // Low-level simulation returns sums for all components:
        //
        //   Y_0^{(j)} = P_0^{(j)}
        //   Y_l^{(j)} = P_l^{(j)} - P_{l-1}^{(j)},  l > 0
        //
        mlmc_l(l,dNl[l],M,sums.data());

        for (int j=0; j<M; j++) {
          suml[0][j][l] += (float) dNl[l];
          suml[1][j][l] += sums[1 + 4*j];
          suml[2][j][l] += sums[1 + 4*j + 1];
        }

        NlCl[l] += sums[0];
      }
    }

    //
    // compute absolute average, variance and cost,
    // correct for possible under-sampling,
    // and set optimal number of new samples
    //

    for (int l=0; l<=L; l++) {
      if (gamma_0 <= 0.0f) Cl[l] = NlCl[l] / suml[0][0][l]; // if gamma not provided (then initially gamma = 0), 
                                                            // use average cost from data
                                                            // here suml[0][j][l] is same = Nl
      for (int j=0; j<M; j++) {
        ml[j][l] = fabs(suml[1][j][l]/suml[0][j][l]); // ml[j][l] = |E[P_l^{(j)} - P_{l-1}^{(j)}]|, 
                                                      // estimated by |sum(Y_j)/Nl|
        Vl[j][l] = fmaxf(suml[2][j][l]/suml[0][j][l]
                         - ml[j][l]*ml[j][l], 0.0f); // V_{j,l} = Var(Y_{j,l})
                                                     //         = E[Y_{j,l}^2] - (E[Y_{j,l}])^2

        if (l>1) {
          ml[j][l] = fmaxf(ml[j][l],
                           0.5f*ml[j][l-1]/powf(2.0f,alpha[j]));
          Vl[j][l] = fmaxf(Vl[j][l],
                           0.5f*Vl[j][l-1]/powf(2.0f,beta[j]));
        }
      }
    }

    for (int l=0; l<=L; l++) dNl[l] = 0;

    for (int j=0; j<M; j++) {
      sum = 0.0f;

      for (int l=0; l<=L; l++)
        sum += sqrtf(Vl[j][l]*Cl[l]); // (optimally) how many samples needed for level l

      for (int l=0; l<=L; l++) {
        int dNlj = ceilf( fmaxf( 0.0f,
                         sqrtf(Vl[j][l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                       - suml[0][j][l] ) );

        // One common N_l is used for all vector components, so choose the
        // largest sample requirement among all outputs j.
        dNl[l] = std::max(dNl[l], dNlj);
      }
    }

    //
    // use linear regression to estimate alpha, beta, gamma if not given
    //

    if (alpha_0 <= 0.0f) {
      for (int j=0; j<M; j++) {
        for (int l=1; l<=L; l++) {
          x[l-1] = l;
          y[l-1] = - log2f(ml[j][l]);
        }
        regression_vector(L,x,y,alpha[j],sum);
        alpha[j] = fmaxf(alpha[j],0.5f);

        if (diag) printf(" alpha[%d] = %f \n",j,alpha[j]);
      }
    }

    if (beta_0 <= 0.0f) {
      for (int j=0; j<M; j++) {
        for (int l=1; l<=L; l++) {
          x[l-1] = l;
          y[l-1] = - log2f(Vl[j][l]);
        }
        regression_vector(L,x,y,beta[j],sum);
        beta[j] = fmaxf(beta[j],0.5f);

        if (diag) printf(" beta[%d] = %f \n",j,beta[j]);
      }
    }

    if (gamma_0 <= 0.0f) {
      for (int l=1; l<=L; l++) {
        x[l-1] = l;
        y[l-1] = log2f(Cl[l]);
      }
      regression_vector(L,x,y,gamma,sum);
      gamma = fmaxf(gamma,0.5f);

      if (diag) printf(" gamma = %f \n",gamma);
    }

    //
    // if (almost) converged, estimate remaining error and decide
    // whether a new level is required
    //

    sum = 0.0f;
    for (int l=0; l<=L; l++)
      sum += fmaxf(0.0f, (float)dNl[l]-0.01f*suml[0][0][l]); 

    if (sum==0) {
      if (diag) printf(" achieved variance target \n");

      converged = 1;

      int needs_more_levels = 0;
      for (int j=0; j<M; j++) {
        // Geometric tail estimate:
        //        |E[P^{(j)} - P_L^{(j)}]|
        //     ~= |E[Y_L^{(j)}]| * (2^{-alpha[j]} + 2^{-2 alpha[j]} + ...)
        //     =  |E[Y_L^{(j)}]| / (2^{alpha[j]} - 1)
        float rem = ml[j][L] / (powf(2.0f,alpha[j])-1.0f);
        if (rem > sqrtf(theta)*eps) needs_more_levels = 1;
      }

      if (needs_more_levels) {
        if (L==Lmax)
          printf("*** failed to achieve weak convergence *** \n");
        else {
          converged = 0;
          L++;

          Cl[L] = Cl[L-1]*powf(2.0f,gamma);
          for (int j=0; j<M; j++)
            Vl[j][L] = Vl[j][L-1]/powf(2.0f,beta[j]);

          if (diag) printf(" L = %d \n",L);

          for (int l=0; l<=L; l++) dNl[l] = 0;

          for (int j=0; j<M; j++) {
            sum = 0.0f;
            for (int l=0; l<=L; l++)
              sum += sqrtf(Vl[j][l]*Cl[l]);

            for (int l=0; l<=L; l++) {
              int dNlj = ceilf( fmaxf( 0.0f,
                               sqrtf(Vl[j][l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                             - suml[0][j][l] ) );
              dNl[l] = std::max(dNl[l], dNlj); // waiting for discussion: this is not a matrix as we assume all samples uses same path.
                                               // but what if each j uses different path? shall we use a matrix?
            }
          }

          // A newly introduced level must receive a full pilot sample. This
          // avoids unreliable last-level variances based on only a handful
          // of paths, which was possible in the old binned implementation.
          dNl[L] = std::max(dNl[L], N0);
        }
      }
    }
  }

  //
  // finally, evaluate multilevel estimator and set outputs
  //

  MlmcVectorStatistics result;
  result.final_level = L;
  result.estimate.assign(M, 0.0);
  result.standard_error.assign(M, 0.0);
  result.level_mean.assign(M, std::vector<double>(L + 1, 0.0));
  result.level_variance.assign(M, std::vector<double>(L + 1, 0.0));
  result.samples_per_level.resize(L + 1);
  result.cost_per_level.resize(L + 1);

  for (int l=0; l<=L; l++) {
    Nl[l] = static_cast<int>(suml[0][0][l]);
    Cl[l] = NlCl[l] / Nl[l];
    result.samples_per_level[l] = Nl[l];
    result.cost_per_level[l] = Cl[l];
  }

  for (int j=0; j<M; j++) {
    double estimator_variance = 0.0;
    for (int l=0; l<=L; l++) {
      const double count = suml[0][j][l];
      const double raw_sum = suml[1][j][l];
      const double mean = raw_sum / count;
      const double centered_sum_squares = fmax(
          suml[2][j][l] - raw_sum * raw_sum / count, 0.0);
      const double variance = count > 1.0
          ? centered_sum_squares / (count - 1.0)
          : 0.0;
      result.level_mean[j][l] = mean;
      result.level_variance[j][l] = variance;
      result.estimate[j] += mean;
      estimator_variance += variance / count;
    }
    result.standard_error[j] = sqrt(fmax(estimator_variance, 0.0));
  }

  return result;
}

//
// linear regression routine
//

void regression_vector(int N, float *x, float *y, float &a, float &b){

  float sum0=0.0f, sum1=0.0f, sum2=0.0f, sumy0=0.0f, sumy1=0.0f;

  for (int i=0; i<N; i++) {
    sum0  += 1.0f;
    sum1  += x[i];
    sum2  += x[i]*x[i];

    sumy0 += y[i];
    sumy1 += y[i]*x[i];
  }

  a = (sum0*sumy1 - sum1*sumy0) / (sum0*sum2 - sum1*sum1);
  b = (sum2*sumy0 - sum1*sumy1) / (sum0*sum2 - sum1*sum1);
}
