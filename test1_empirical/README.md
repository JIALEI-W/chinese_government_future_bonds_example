# Empirical-input adapter for `/Users/km/test1`

This directory links the original `test1` numerical sources and replaces only
the benchmark model inputs with the daily 11:30 empirical calibration:

- dimension: 4, ordered `(TF, TL, T, TS)`;
- normalized initial values: `(100, 100, 100, 100)`;
- annualized volatilities: `(0.0153524269, 0.0490989095, 0.0189355669,
  0.0069499237)`;
- zero drift and `T=1`, `K=100`;
- the empirical 4-by-4 Pearson correlation matrix.

The level convention and published allocation are taken from
`/Users/km/real_data_mlmc/cpp/empirical_mlmc`: level `l` uses exactly `2^l`
Euler steps, including one step at level zero.  The local benchmark forces this
convention through `options_override.hpp`; it must not use the later
`kBaseTimeLevel=3` convention from `/Users/km/test1/src/options.hpp`.

The local `empirical_main.cpp` differs from `test1/src/main.cpp` only by turning
the synthetic example's hard-coded `>=1.5x` speedup assertion into a warning.
The common-RMSE checks remain mandatory.

Run the publication benchmark with:

```bash
make run
make koerber
```

The Körber executable uses the reordered pathwise calculation.  For each
standard-normal vector `Z` and correlated-normal adjoint `Wbar`, it solves
`C^T v = Z` and directly accumulates the symmetric entries of
`0.5 * Wbar * v^T`.  A startup equivalence check compares this calculation
against the earlier `Cbar C^{-1}` implementation before any timing begins.

At `eps=0.05`, `N0=200`, and seed `12345`, the authoritative daily-11:30 run
reaches level 5 and allocates `(201017, 73292, 25913, 11554, 4364, 1620)`
samples.  The plotting code reads that allocation directly from the reproduced
original output, rather than substituting an allocation produced under a
different time-grid convention.
