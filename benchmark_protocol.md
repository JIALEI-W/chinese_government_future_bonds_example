# Benchmark protocol

## Methods

Use the same payoff, calibration and six off-diagonal correlation Greeks in all
four cases.

| Label | Derivative engine | Sampling engine |
|---|---|---|
| Traditional | Forward differentiation | Standard Monte Carlo |
| Adjoint only | Adjoint differentiation | Standard Monte Carlo |
| MLMC only | Forward differentiation | Multilevel Monte Carlo |
| MLMC + adjoint | Adjoint differentiation | Multilevel Monte Carlo |

This terminology matches the logical development of the paper.  In particular,
"MLMC only" must use forward pathwise derivatives inside the level correction;
it must not be a payoff MLMC calculation followed by finite differences of two
independently allocated runs.

## Accuracy and work normalization

Wall-clock times are meaningful only if every method targets the same complete
six-dimensional Greek vector at the same accuracy.

1. Use the daily 11:30 calibration, the same payoff, `T`, `K`, random-number
   family and compiler flags.
2. Use common random numbers for the fine/coarse pair and, where applicable,
   across derivative implementations.
3. Apply one stated vector accuracy rule to all methods.  The recommended rule
   is that every component satisfies estimated RMSE <= epsilon.
4. Standard MC must use a finest Euler level selected by the same bias rule as
   MLMC, not an arbitrarily coarser grid.
5. Do not compare a serial method with an OpenMP method.  Report one-thread
   timings first; a separate parallel-scaling table may be added later.
6. Exclude compilation and CSV writing from the timed region.  Include random
   number generation, path simulation, derivative propagation and adaptive
   pilot/allocation work.
7. Run each completed method at least five times.  Report the median elapsed
   time and the interquartile range; retain the seed and per-level sample counts.
8. Verify estimates against a common high-accuracy reference before comparing
   speed.

For a stronger empirical result, repeat the benchmark for at least three target
accuracies (for example, epsilon = 0.10, 0.075 and 0.05) and plot time against
achieved RMSE.  A single timing table is easier to read, but it cannot establish
the expected complexity advantage by itself.

## Körber comparison

Compare standard reverse Cholesky and the optimized Körber transformation using
the same random-number seeds, payoff, MLMC levels and sample counts.  The timed
Körber path must retain the rank-one pathwise structure.  At each Euler step it
solves `C^T v = Z` and accumulates the symmetric entries of
`0.5 * Wbar * v^T`; it must not first accumulate a dense `Cbar` and then solve
the matrix equation `H C = 0.5 Cbar`.

Report the end-to-end time for the complete MLMC plus adjoint estimate.  The
timed region includes random-number generation, fine/coarse path construction,
payoff differentiation and the correlation reverse calculation.  Because the
two estimators have different pathwise variances, also report the achieved RMSE
for both methods and verify that both remain below the common target.

## Required raw outputs

Each run should append one row to a machine-readable CSV containing at least:

```text
method,cholesky_reverse,epsilon,seed,threads,repeat,elapsed_seconds,
final_level,total_samples,achieved_max_standard_error,estimate_1,...,estimate_6
```

MLMC runs should additionally write a level CSV with sample count, correction
mean, correction variance and measured cost per sample at every level.
