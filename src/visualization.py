"""Publication-quality figures for the research report.

Every function saves a PNG into reports/figures/ and returns the path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, ensure_directories

plt.rcParams.update(
    {
        "figure.figsize": (11, 5.5),
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)

COLORS = {
    "price": "#1f4e79",
    "vol": "#c0392b",
    "vix": "#7d3c98",
    "accent": "#e67e22",
    "neutral": "#566573",
}


def _save(fig: plt.Figure, name: str) -> Path:
    ensure_directories()
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _format_dates(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_price_history(data: pd.DataFrame) -> Path:
    fig, ax = plt.subplots()
    ax.plot(data.index, data["adj_close"], color=COLORS["price"], lw=1.0)
    ax.set_title("SPY Adjusted Close, 2010–Present")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted close (USD)")
    _format_dates(ax)
    return _save(fig, "01_spy_price_history.png")


def plot_realized_volatility(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots()
    ax.plot(df.index, df["rv_21d"] * 100, color=COLORS["vol"], lw=0.9, label="21-day realized vol")
    ax.plot(df.index, df["rv_63d"] * 100, color=COLORS["neutral"], lw=0.9, alpha=0.8, label="63-day realized vol")
    ax.set_title("SPY Annualized Realized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.legend(loc="upper right")
    _format_dates(ax)
    return _save(fig, "02_realized_volatility.png")


def plot_vix_vs_realized(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(df.index, df["vix"], color=COLORS["vix"], lw=0.8, label="VIX (implied)")
    ax.plot(df.index, df["rv_21d"] * 100, color=COLORS["vol"], lw=0.8, alpha=0.8, label="21-day realized vol")
    ax.set_title("VIX vs Realized Volatility Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.legend(loc="upper right")
    _format_dates(ax)

    ax = axes[1]
    ax.scatter(df["vix"], df["rv_5d_forward"] * 100, s=4, alpha=0.25, color=COLORS["price"])
    corr = df["vix"].corr(df["rv_5d_forward"])
    ax.set_title(f"VIX vs Forward 5-Day Realized Vol (corr = {corr:.2f})")
    ax.set_xlabel("VIX level")
    ax.set_ylabel("Forward 5-day realized vol (%)")
    return _save(fig, "03_vix_vs_realized_vol.png")


def plot_return_distribution(returns: pd.Series) -> Path:
    from scipy import stats

    r = returns.dropna()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(r * 100, bins=120, density=True, color=COLORS["price"], alpha=0.75, label="Daily log returns")
    x = np.linspace(r.min(), r.max(), 400)
    ax.plot(x * 100, stats.norm.pdf(x, r.mean(), r.std()) / 100, color=COLORS["vol"], lw=1.5, label="Normal fit")
    ax.set_title(
        f"SPY Daily Log Return Distribution "
        f"(skew = {stats.skew(r):.2f}, excess kurtosis = {stats.kurtosis(r):.1f})"
    )
    ax.set_xlabel("Daily log return (%)")
    ax.set_ylabel("Density")
    ax.legend()
    return _save(fig, "04_return_distribution.png")


def plot_volatility_clustering(returns: pd.Series) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(returns.index, returns * 100, color=COLORS["price"], lw=0.5)
    axes[0].set_title("Daily Log Returns — Large Moves Cluster in Time")
    axes[0].set_ylabel("Return (%)")
    axes[1].plot(returns.index, returns.abs() * 100, color=COLORS["vol"], lw=0.5)
    axes[1].set_title("Absolute Daily Returns (Volatility Clustering)")
    axes[1].set_ylabel("|Return| (%)")
    axes[1].set_xlabel("Date")
    _format_dates(axes[1])
    return _save(fig, "05_volatility_clustering.png")


def plot_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> Path:
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    ax.set_title("Correlation Matrix: Volatility, VIX, Returns, and Drawdowns")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, "06_correlation_matrix.png")


def plot_actual_vs_predicted(
    y_true: pd.Series, predictions: dict[str, pd.Series], target_name: str
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(y_true.index, y_true * 100, color="black", lw=1.1, label="Actual", zorder=5)
    palette = ["#c0392b", "#2471a3", "#229954", "#e67e22", "#7d3c98", "#566573"]
    for (name, pred), color in zip(predictions.items(), palette):
        common = y_true.index.intersection(pred.index)
        ax.plot(common, pred.loc[common] * 100, lw=0.9, alpha=0.8, label=name, color=color)
    ax.set_title(f"Actual vs Predicted {target_name} (Test Period, Walk-Forward)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, "07_actual_vs_predicted.png")


def plot_model_comparison(results: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = results.sort_values("RMSE")
    for ax, metric in zip(axes, ("RMSE", "MAE")):
        vals = order[metric] * 100
        bars = ax.barh(order["Model"], vals, color=COLORS["price"], alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(f"Test {metric} by Model (vol %, lower is better)")
        ax.set_xlabel(f"{metric} (annualized vol %)")
        ax.invert_yaxis()
        ax.set_xlim(0, vals.max() * 1.15)
    return _save(fig, "08_model_comparison.png")


def plot_feature_importance(importance: pd.Series, top_n: int = 15) -> Path:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values, color=COLORS["accent"], alpha=0.9)
    ax.set_title(f"XGBoost Feature Importance (Top {top_n}, gain-based)")
    ax.set_xlabel("Importance (normalized gain)")
    return _save(fig, "09_xgb_feature_importance.png")


def plot_error_by_regime(regime_metrics: pd.DataFrame) -> Path:
    pivot = regime_metrics.pivot(index="Model", columns="Regime", values="RMSE") * 100
    pivot = pivot[["low", "medium", "high"]].sort_values("high")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(pivot))
    width = 0.26
    colors = {"low": "#229954", "medium": "#e67e22", "high": "#c0392b"}
    for i, regime in enumerate(("low", "medium", "high")):
        ax.bar(x + (i - 1) * width, pivot[regime], width, label=f"{regime} vol regime", color=colors[regime], alpha=0.9)
    ax.set_xticks(x, pivot.index, rotation=15, ha="right")
    ax.set_title("Forecast RMSE by Volatility Regime (Test Period)")
    ax.set_ylabel("RMSE (annualized vol %)")
    ax.legend()
    return _save(fig, "10_error_by_regime.png")
