# Chinese Government Bond Futures: An Empirical MLMC Example

This repository is a self-contained results package for a semester project. It includes the final figures, plotting data, MLMC sample allocations, `rho_bar` statistical matrices, and compilable C++ code adapted from the original `/Users/km/test1` implementation. The original project was not overwritten.

## Final Figures

1. `figures/01_market_prices_and_correlation.png`: Normalized price series for four Chinese government bond futures contracts and a correlation heat map of their daily log returns. The horizontal axis is labeled by month, and the sample consists of synchronized daily prices recorded at 11:30.
2. `figures/02_traditional_vs_mlmc_adjoint_cost.png`: A runtime comparison, in seconds, between traditional Monte Carlo and MLMC with adjoints. MLMC uses the measured median runtime of `14.620 s`; the traditional Monte Carlo runtime is converted on an equivalent Euler-cost basis to approximately `81.009 s`.
3. `figures/03_cholesky_vs_koerber_runtime.png`: A wall-clock runtime comparison between standard reverse Cholesky and the optimized Körber implementation for one complete MLMC-plus-adjoint estimate.

PDF versions are available in the same directory and can be inserted directly into LaTeX documents.

## Main Results

- Tolerance: `eps = 0.01`.
- MLMC levels: `l = 0,...,7`; the Euler cost is `2^l`, and the sample counts are listed in `data_tables/02_mlmc_allocation_eps001.csv`.
- Traditional Monte Carlo cost: `142,656,000`; MLMC-plus-adjoint coupled-path cost: `25,745,192`; cost-equivalent speedup: approximately `5.54x`. Using the measured MLMC-plus-adjoint runtime of `14.620 s` as the baseline gives an equivalent traditional Monte Carlo runtime of approximately `81.009 s`.
- Across five alternating timing runs, the median runtimes were `14.620 s` for standard reverse Cholesky and `13.646 s` for optimized Körber. With identical inputs, sample allocations, and RMSE criteria, the optimized Körber implementation reduced wall-clock time by approximately `6.7%`.
- The optimized implementation no longer accumulates the full `Cbar` before computing a right-side matrix solve. At each Euler step, it directly solves `C^T v = Z` and then accumulates the symmetric correlation entries of `0.5 * Wbar * v^T`, preserving the rank-one structure.
- Cholesky and Körber use a common estimate for the expectation of `rho_bar`, so their expectation matrices are identical, while their entry-wise variance matrices differ. The four matrices are provided in `data_tables/05`–`08`.

The second figure consistently reports time in seconds. The MLMC-plus-adjoint value of `14.620 s` is the median of five complete runs. The traditional Monte Carlo value of `81.009 s` is a cost-equivalent runtime derived from the Euler-cost ratio at the same accuracy, rather than a separate direct wall-clock benchmark. Traditional Monte Carlo uses a sample count at the finest level, `L=7`, that satisfies the same sampling-error target, whereas MLMC distributes samples across levels according to `M_l`.

## Repository Structure

- `figures/`: PNG and PDF versions of the three final figures.
- `data_tables/`: CSV tables referenced by the figures and main text.
- `inputs/`: Synchronized daily prices at 11:30, model calibration data, and the MLMC allocation for `eps=0.01`.
- `generate_market_figures.py`: Regenerates the market figure from the packaged price CSV file.
- `plot_results.py`: Regenerates the two timing figures from the benchmark outputs.
- `test1_empirical/`: Empirical benchmark and Körber comparison programs.
- `vendor/test1/src/`: A copy of the original `test1` source code required for compilation, making this repository self-contained.

## Python Plotting

Python 3.10 or later is recommended:

```bash
python3 -m pip install -r requirements.txt
python3 generate_market_figures.py
```

By default, these commands read `inputs/daily_1130_figure_inputs.csv`. To extract the daily 11:30 observations again from the original Parquet data, run:

```bash
python3 generate_market_figures.py --project-root /path/to/real_data_mlmc
```

## C++ Benchmark

A C++17-compatible `clang++` compiler is required:

```bash
cd test1_empirical
make all
make run
make koerber
cd ..
python3 plot_results.py
```

The benchmark uses a fixed seed so that the sample allocation and matrix results can be reproduced. By default, the Körber benchmark alternates between the two methods for five runs and reports their median runtimes. Actual wall-clock times may vary slightly with system load.

## Table Reference

- `01_correlation_matrix.csv`: Return correlation matrix.
- `02_mlmc_allocation_eps001.csv`: Euler cost and `M_l` at each level.
- `03_traditional_vs_mlmc_cost.csv`: Cost comparison between the two estimators.
- `04_cholesky_vs_koerber_timing.csv`: Runtime of the complete MLMC-plus-adjoint calculation.
- `05` and `07`: `E[rho_bar]` for the two implementations.
- `06` and `08`: Entry-wise `Var(rho_bar)` for the two implementations.
