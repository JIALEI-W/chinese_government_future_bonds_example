# Chinese Government Bond Futures: empirical MLMC example

这是学期 project 的独立结果包。它包含最终采用的图片、作图数据、MLMC 样本分配、`rho_bar` 统计矩阵，以及基于原 `/Users/km/test1` 实现整理出的可编译 C++ 代码。原始项目没有被覆盖。

## 最终图片

1. `figures/01_market_prices_and_correlation.png`：四种国债期货的标准化价格走势与日对数收益 correlation heat map。横轴按月标注，样本为每日 11:30 的同步价格。
2. `figures/02_traditional_vs_mlmc_adjoint_cost.png`：traditional Monte Carlo 与 MLMC + adjoint 的秒数对比。MLMC 使用实测中位数 `14.620 s`；traditional time 按相同 Euler-cost 口径换算为约 `81.009 s`。
3. `figures/03_cholesky_vs_koerber_runtime.png`：完成一次完整 MLMC + adjoint 估计时，standard reverse Cholesky 与 optimized Körber 实现的 wall-clock time 对比。

PDF 版本也放在同一目录中，可直接插入 LaTeX。

## 采用的主要结果

- 容差：`eps = 0.01`。
- MLMC 层数：`l = 0,...,7`；Euler cost 为 `2^l`；样本数见 `data_tables/02_mlmc_allocation_eps001.csv`。
- Traditional MC cost：`142,656,000`；MLMC + adjoint coupled-path cost：`25,745,192`；cost-equivalent speedup 约 `5.54x`。以 MLMC + adjoint 的实测 `14.620 s` 为基准，对应的 traditional time 约为 `81.009 s`。
- 五次交替计时的中位数为：standard reverse Cholesky `14.620 s`，optimized Körber `13.646 s`。在相同输入、样本分配和 RMSE 口径下，优化后的 Körber wall-clock time 低约 `6.7%`。
- 优化后的实现不再先累积完整的 `Cbar` 再计算矩阵右解。它在每个 Euler step 直接求解 `C^T v = Z`，随后累积 `0.5 * Wbar * v^T` 的对称 correlation entries，从而保留 rank-one 结构。
- Cholesky 与 Koerber 的 `rho_bar` expectation 使用共同估计值，因此两个 expectation matrix 完全相同；各自的 entry-wise variance matrix 不同。四个矩阵在 `data_tables/05`–`08`。

第二张图现在统一使用 seconds。MLMC + adjoint 的 `14.620 s` 是五次完整运行的中位数；traditional MC 的 `81.009 s` 是根据同精度 Euler cost ratio 换算的 cost-equivalent time，并非另一次直接 wall-clock benchmark。Traditional MC 在最细层 `L=7` 上使用满足同一采样误差目标的样本量；MLMC 则按照 `M_l` 在各层分配样本。

## 文件结构

- `figures/`：三张最终图片的 PNG 和 PDF。
- `data_tables/`：图片及正文可引用的 CSV 表格。
- `inputs/`：每日 11:30 同步价格、模型 calibration，以及 `eps=0.01` 的 MLMC allocation。
- `generate_market_figures.py`：从包内价格 CSV 重画市场图。
- `plot_results.py`：根据 benchmark 输出重画两张时间图。
- `test1_empirical/`：empirical benchmark 和 Koerber comparison 程序。
- `vendor/test1/src/`：编译所需的原 test1 源码副本，使本目录可以独立编译。

## Python 作图

建议使用 Python 3.10 或更新版本：

```bash
python3 -m pip install -r requirements.txt
python3 generate_market_figures.py
```

以上命令默认读取 `inputs/daily_1130_figure_inputs.csv`。若要从原始 Parquet 重新抽取每日 11:30 数据，可运行：

```bash
python3 generate_market_figures.py --project-root /path/to/real_data_mlmc
```

## C++ benchmark

需要支持 C++17 的 `clang++`：

```bash
cd test1_empirical
make all
make run
make koerber
cd ..
python3 plot_results.py
```

benchmark 使用固定 seed，以便复核 sample allocation 和矩阵结果。Körber benchmark 默认交替运行两种方法五次并报告中位数；实际 wall-clock time 会随电脑负载略有变化。

## 表格对应关系

- `01_correlation_matrix.csv`：收益率相关矩阵。
- `02_mlmc_allocation_eps001.csv`：每层 Euler cost 与 `M_l`。
- `03_traditional_vs_mlmc_cost.csv`：两种估计器的 cost comparison。
- `04_cholesky_vs_koerber_timing.csv`：完整 MLMC + adjoint 的时间。
- `05`、`07`：两种实现的 `E[rho_bar]`。
- `06`、`08`：两种实现的 entry-wise `Var(rho_bar)`。
