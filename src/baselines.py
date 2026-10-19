"""Simple baselines. Any model has to beat these to be worth its complexity.

Run: python src/baselines.py  (after src/prepare.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

ROOT = Path(__file__).resolve().parents[1]


def naive(history: pd.DataFrame, horizon: int) -> np.ndarray:
    """Next weeks = last week."""
    return np.repeat(history.iloc[:, [-1]].to_numpy(), horizon, axis=1)


def moving_average(window: int):
    def f(history: pd.DataFrame, horizon: int) -> np.ndarray:
        mean = history.iloc[:, -window:].mean(axis=1).to_numpy()[:, None]
        return np.repeat(mean, horizon, axis=1)
    return f


def seasonal_naive(history: pd.DataFrame, horizon: int) -> np.ndarray:
    """Same week last year (52 weeks back)."""
    t = history.shape[1]
    return history.iloc[:, t - 52 : t - 52 + horizon].to_numpy()


def seasonal_naive_scaled(history: pd.DataFrame, horizon: int) -> np.ndarray:
    """Last year's same weeks, scaled by this year's level vs. last year's (last 8 weeks each).

    Keeps the seasonal shape while correcting for growth or decline of each SKU.
    """
    t = history.shape[1]
    recent = history.iloc[:, t - 8 : t].sum(axis=1)
    year_ago = history.iloc[:, t - 60 : t - 52].sum(axis=1)
    ratio = (recent / year_ago.replace(0, np.nan)).fillna(1).clip(0.25, 4).to_numpy()[:, None]
    return seasonal_naive(history, horizon) * ratio


BASELINES = {
    "naive (last week)": naive,
    "moving average 4w": moving_average(4),
    "moving average 12w": moving_average(12),
    "seasonal naive": seasonal_naive,
    "seasonal naive scaled": seasonal_naive_scaled,
}


def main() -> None:
    wide = bt.to_wide(pd.read_parquet(ROOT / "data" / "interim" / "weekly_panel.parquet"))
    results = pd.concat([bt.run(wide, f, name) for name, f in BASELINES.items()], ignore_index=True)
    out = ROOT / "data" / "processed" / "baseline_results.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out, index=False)

    first, last = results["target_week"].min().date(), results["target_week"].max().date()
    print(f"test weeks {first} to {last}, {wide.shape[0]} SKUs, {bt.N_ORIGINS} origins, h=1..{bt.HORIZON}\n")
    print(bt.summary(results).sort_values("WAPE").round(3).to_string())
    by_h = results.groupby(["model", "h"]).apply(
        lambda g: bt.wape(g.actual, g.forecast), include_groups=False).unstack().round(3)
    print("\nWAPE by horizon (weeks ahead):\n" + by_h.to_string())


if __name__ == "__main__":
    main()
