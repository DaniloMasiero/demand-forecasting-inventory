"""Rolling-origin backtest shared by baselines and models, plus the error metrics.

At each origin we pretend today is that week: the forecaster only sees data up to the origin
and predicts the next HORIZON weeks. Origins move one week at a time over the test period.
"""
from collections.abc import Callable

import numpy as np
import pandas as pd

HORIZON = 4        # weeks ahead: a typical replenishment lead time for planning
N_ORIGINS = 16     # test period: the last 16 origins (Aug-Nov 2011, includes the Q4 peak)

# forecaster(history, horizon) -> array (n_skus, horizon). history is SKU x week (wide).
Forecaster = Callable[[pd.DataFrame, int], np.ndarray]


def to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot(index="stock_code", columns="week", values="units").sort_index(axis=1)


def origins(wide: pd.DataFrame) -> list[int]:
    """Column positions of the last observed week at each test origin."""
    last = wide.shape[1] - 1 - HORIZON
    return list(range(last - N_ORIGINS + 1, last + 1))


def validation_origins(wide: pd.DataFrame) -> list[int]:
    """The N_ORIGINS origins right before the test ones, whose targets end before the test starts.

    Model choices are made here; the test period is used once, to report the chosen model.
    """
    first_test = origins(wide)[0]
    last = first_test - HORIZON
    return list(range(last - N_ORIGINS + 1, last + 1))


def run(wide: pd.DataFrame, forecaster: Forecaster, name: str, at: list[int] | None = None) -> pd.DataFrame:
    rows = []
    for t in (origins(wide) if at is None else at):
        history = wide.iloc[:, : t + 1]
        actual = wide.iloc[:, t + 1 : t + 1 + HORIZON].to_numpy()
        pred = np.clip(forecaster(history, HORIZON), 0, None)   # demand is never negative
        for h in range(HORIZON):
            rows.append(pd.DataFrame({
                "model": name, "origin": wide.columns[t], "h": h + 1,
                "target_week": wide.columns[t + 1 + h], "stock_code": wide.index,
                "actual": actual[:, h], "forecast": pred[:, h],
            }))
    return pd.concat(rows, ignore_index=True)


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: total absolute error / total demand.

    Preferred over MAPE because MAPE explodes on weeks with zero or tiny sales and treats
    a 1-unit SKU like a 1,000-unit SKU.
    """
    return np.abs(np.asarray(actual) - np.asarray(forecast)).sum() / np.asarray(actual).sum()


def bias(actual, forecast) -> float:
    """Total over (+) or under (-) forecast relative to demand. WAPE hides direction; inventory cares."""
    return (np.asarray(forecast) - np.asarray(actual)).sum() / np.asarray(actual).sum()


def summary(results: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        return pd.Series({"WAPE": wape(g.actual, g.forecast), "bias": bias(g.actual, g.forecast)})
    return results.groupby("model").apply(agg, include_groups=False)
