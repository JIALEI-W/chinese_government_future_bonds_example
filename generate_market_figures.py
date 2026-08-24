#!/usr/bin/env python3
"""Generate the price-path and correlation figures for the case study.

By default this script reads the synchronized 11:30 panel bundled in ``inputs``.
It can alternatively rebuild that panel from the original Parquet files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


INSTRUMENT_FILES = {
    "TF": "TFM01.parquet",
    "TL": "TLM01.parquet",
    "T": "TM01.parquet",
    "TS": "TSM01.parquet",
}

SERIES_COLORS = {
    "TF": "#0072B2",
    "TL": "#D55E00",
    "T": "#009E73",
    "TS": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "inputs"
        / "daily_1130_figure_inputs.csv",
        help="Bundled synchronized price panel; pass an empty value only when rebuilding from Parquet.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional real_data_mlmc root used to rebuild inputs from raw Parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "figures",
        help="Directory for PDF, PNG and audit CSV outputs.",
    )
    parser.add_argument(
        "--observation-time",
        default="11:30",
        help="Exact synchronized observation time in HH:MM format.",
    )
    return parser.parse_args()


def load_daily_prices(project_root: Path, observation_time: str) -> pd.DataFrame:
    raw_dir = project_root / "data" / "raw"
    series: list[pd.Series] = []

    for instrument, filename in INSTRUMENT_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing raw data file: {path}")

        frame = pd.read_parquet(path, columns=["close_adj_2"])
        frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
        frame = frame.sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"{instrument} contains duplicate timestamps")

        price = frame["close_adj_2"].astype(float)
        selected = price.between_time(observation_time, observation_time)
        if selected.empty:
            raise ValueError(
                f"{instrument} has no observations at {observation_time}"
            )

        trading_dates = selected.index.normalize()
        if trading_dates.duplicated().any():
            raise ValueError(
                f"{instrument} has multiple observations at "
                f"{observation_time} on one trading day"
            )
        selected.index = pd.DatetimeIndex(
            trading_dates, name="trading_date"
        )
        selected.name = instrument
        series.append(selected)

    prices = pd.concat(series, axis=1, join="inner").dropna().sort_index()
    if prices.empty:
        raise ValueError("The four instruments have no common observations")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite and strictly positive")
    return prices


def load_bundled_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing synchronized input CSV: {path}")
    prices = pd.read_csv(path, parse_dates=["trading_date"])
    prices = prices.set_index("trading_date")[list(INSTRUMENT_FILES)]
    prices.index = pd.DatetimeIndex(prices.index, name="trading_date")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Input trading dates must be unique and increasing")
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Input prices must be finite and complete")
    if (prices <= 0).any().any():
        raise ValueError("Input prices must be strictly positive")
    return prices


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: mpl.figure.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_normalized_prices(prices: pd.DataFrame, output_dir: Path) -> None:
    normalized = 100.0 * prices.divide(prices.iloc[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for instrument in INSTRUMENT_FILES:
        ax.plot(
            normalized.index,
            normalized[instrument],
            label=instrument,
            color=SERIES_COLORS[instrument],
            linewidth=1.35,
        )

    ax.axhline(100.0, color="0.35", linewidth=0.7, linestyle="--")
    ax.set_title("Normalized Chinese Government Bond Futures Prices")
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Normalized price (first common date = 100)")
    apply_monthly_axis(ax, prices.index)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    save_figure(fig, output_dir / "normalized_daily_prices")


def apply_monthly_axis(ax: mpl.axes.Axes, dates: pd.DatetimeIndex) -> None:
    """Label every calendar month and show the year only at boundaries."""
    locator = mdates.MonthLocator(interval=1, bymonthday=1)

    def label_month(value: float, position: int) -> str:
        date = mdates.num2date(value)
        if position == 0 or date.month == 1:
            return date.strftime("%b\n%Y")
        return date.strftime("%b")

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(FuncFormatter(label_month))
    month_start = dates.min().to_period("M").start_time
    ax.set_xlim(month_start, dates.max())


def annotation_color(value: float) -> str:
    return "white" if abs(value) >= 0.82 else "black"


def plot_correlation_heatmap(rho: pd.DataFrame, output_dir: Path) -> None:
    values = rho.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.3, 4.6), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    labels = list(rho.columns)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                color=annotation_color(values[row, column]),
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")
    ax.set_title("Correlation of Synchronized Daily Log Returns", pad=12)
    save_figure(fig, output_dir / "daily_return_correlation_heatmap")


def plot_combined_market_figure(
    prices: pd.DataFrame,
    rho: pd.DataFrame,
    output_dir: Path,
) -> None:
    normalized = 100.0 * prices.divide(prices.iloc[0])
    fig = plt.figure(figsize=(11.4, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
    price_ax = fig.add_subplot(grid[0, 0])
    heat_ax = fig.add_subplot(grid[0, 1])

    for instrument in INSTRUMENT_FILES:
        price_ax.plot(
            normalized.index,
            normalized[instrument],
            label=instrument,
            color=SERIES_COLORS[instrument],
            linewidth=1.3,
        )
    price_ax.axhline(100.0, color="0.35", linewidth=0.7, linestyle="--")
    price_ax.set_title("(a) Normalized daily prices")
    price_ax.set_xlabel("Trading date")
    price_ax.set_ylabel("Normalized price (first common date = 100)")
    apply_monthly_axis(price_ax, normalized.index)
    price_ax.legend(ncol=4, frameon=False, loc="upper left")

    values = rho.to_numpy(dtype=float)
    image = heat_ax.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    labels = list(rho.columns)
    heat_ax.set_xticks(np.arange(len(labels)), labels=labels)
    heat_ax.set_yticks(np.arange(len(labels)), labels=labels)
    heat_ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    heat_ax.grid(False)
    for spine in heat_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            heat_ax.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                color=annotation_color(values[row, column]),
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")
    heat_ax.set_title("(b) Daily-return correlation", pad=12)
    save_figure(fig, output_dir / "prices_and_correlation")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.project_root is not None:
        prices = load_daily_prices(args.project_root, args.observation_time)
    else:
        prices = load_bundled_prices(args.input_csv)
    returns = np.log(prices).diff().dropna()
    rho = returns.corr(method="pearson")

    prices.to_csv(args.output_dir / "figure_inputs.csv", index_label="trading_date")
    rho.to_csv(args.output_dir / "figure_correlation.csv", index_label="instrument")

    configure_style()
    plot_normalized_prices(prices, args.output_dir)
    plot_correlation_heatmap(rho, args.output_dir)
    plot_combined_market_figure(prices, rho, args.output_dir)

    print(
        f"Wrote figures from {len(prices)} synchronized prices and "
        f"{len(returns)} daily returns to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
