#!/usr/bin/env python3
"""Plot the empirical traditional-MC and MLMC+adjoint comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["standard", "mlmc_adjoint"]
LABELS = [
    "Traditional",
    "MLMC + adjoint",
]
COLORS = ["#0072B2", "#009E73"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "test1_empirical_eps001_original",
    )
    parser.add_argument(
        "--koerber-result-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "test1_empirical_koerber_eps001_optimized",
    )
    parser.add_argument(
        "--allocation-csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "inputs"
        / "original_mlmc_levels_eps001.csv",
        help="MLMC level allocation CSV (the bundled default uses eps=0.01).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    allocation = pd.read_csv(args.allocation_csv)
    pd.DataFrame(
        {
            "level": allocation["level"].astype(int),
            "euler_cost_2_to_l": (2 ** allocation["level"]).astype(int),
            "samples_M_l": allocation["samples_per_level"].astype(int),
        }
    ).to_csv(args.result_dir / "table5_mlmc_allocation.csv", index=False)
    final_level = int(allocation["level"].max())
    traditional_samples = 185_750
    correlation_directions = 6
    traditional_cost = (
        traditional_samples * correlation_directions * (2**final_level)
    )
    mlmc_cost = 0
    for row in allocation.itertuples(index=False):
        level = int(row.level)
        samples = int(row.samples_per_level)
        steps = 2**level
        mlmc_cost += samples * (steps if level == 0 else steps + steps // 2)

    koerber_results = pd.read_csv(
        args.koerber_result_dir / "koerber_comparison.csv"
    ).set_index("method").loc[["standard", "koerber"]]
    koerber_timings = pd.read_csv(
        args.koerber_result_dir / "koerber_timing_repetitions.csv"
    )
    koerber_summary = koerber_timings.groupby("method")["runtime_ms"].agg(
        median="median",
        q1=lambda values: values.quantile(0.25),
        q3=lambda values: values.quantile(0.75),
    ).loc[["standard", "koerber"]]
    koerber_summary["estimated_rmse"] = koerber_results["estimated_rmse"]
    koerber_summary.to_csv(
        args.koerber_result_dir / "koerber_timing_summary.csv"
    )

    mlmc_seconds = koerber_summary.loc["standard", "median"] / 1000.0
    traditional_seconds = mlmc_seconds * traditional_cost / mlmc_cost
    cost_summary = pd.DataFrame(
        {
            "method": ORDER,
            "euler_step_cost": [traditional_cost, mlmc_cost],
            "normalized_time": [1.0, mlmc_cost / traditional_cost],
            "time_seconds": [traditional_seconds, mlmc_seconds],
        }
    ).set_index("method")
    cost_summary.to_csv(args.result_dir / "cost_equivalent_timing_summary.csv")

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(ORDER))
    time_seconds = cost_summary["time_seconds"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    bars = ax.bar(
        x,
        time_seconds,
        width=0.68,
        color=COLORS,
    )
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Traditional versus MLMC + Adjoint, $\\epsilon=0.01$")
    ax.set_ylim(0, time_seconds.max() * 1.18)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, time_seconds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + time_seconds.max() * 0.025,
            f"{value:.3f} s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    base = args.result_dir / "four_method_timing_test1_empirical"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    k_median = koerber_summary["median"].to_numpy() / 1000.0
    k_lower = (
        koerber_summary["median"] - koerber_summary["q1"]
    ).to_numpy() / 1000.0
    k_upper = (
        koerber_summary["q3"] - koerber_summary["median"]
    ).to_numpy() / 1000.0
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    k_bars = ax.bar(
        np.arange(2),
        k_median,
        width=0.62,
        color=["#0072B2", "#E69F00"],
        yerr=np.vstack([k_lower, k_upper]),
        capsize=4,
        error_kw={"linewidth": 0.9},
    )
    ax.set_xticks(
        np.arange(2),
        ["Standard reverse\nCholesky", "Optimized Körber"],
    )
    ax.set_ylabel("Wall-clock time (seconds)")
    ax.set_title("Full MLMC + Adjoint Runtime")
    ax.set_ylim(0, k_median.max() * 1.20)
    ax.set_axisbelow(True)
    for bar, seconds in zip(k_bars, k_median):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            seconds + k_median.max() * 0.025,
            f"{seconds:.3f} s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    reduction = 100.0 * (k_median[0] - k_median[1]) / k_median[0]
    ax.text(
        0.5,
        0.96,
        f"Körber: {reduction:.1f}% lower median runtime",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    koerber_base = (
        args.koerber_result_dir / "standard_vs_koerber_time"
    )
    fig.savefig(koerber_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(koerber_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {base.with_suffix('.png').resolve()}")
    print(f"Wrote {koerber_base.with_suffix('.png').resolve()}")


if __name__ == "__main__":
    main()
