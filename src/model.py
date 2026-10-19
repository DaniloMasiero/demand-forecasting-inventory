"""Global LightGBM forecaster, evaluated with the same rolling-origin backtest as the baselines.

Run: python src/model.py  (after src/baselines.py)

Design choices (see README for the reasoning):
- One global model for all SKUs, not one per SKU: 526 noisy series with 2 years each are too
  little data per series, but together they share patterns (Q4 ramp-up, life-cycle decline).
- Direct multi-horizon: the horizon h (1-4 weeks) is a feature, so one model predicts all four.
- Scale-free target: units divided by the SKU's recent level (12-week mean), so a 10-unit SKU
  and a 1,000-unit SKU look alike to the model. Trained with L1 loss weighted by that level,
  which minimizes total absolute error in units, i.e. exactly WAPE.
- Retrained at every origin using only targets already observed at that origin (no leakage).
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import backtest as bt

ROOT = Path(__file__).resolve().parents[1]
MIN_HISTORY = 26   # weeks of history needed before an origin can be used for training
PARAMS = dict(objective="l1", learning_rate=0.05, n_estimators=400, num_leaves=15,
              min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1, random_state=42)


def features(y: np.ndarray, t: int, week0: pd.Timestamp, h: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Features known at origin t (last observed column) to predict column t + h.

    y is the SKU x week matrix. Returns the features and the scale used to normalize them.
    """
    scale = y[:, t - 11 : t + 1].mean(axis=1) + 1.0          # +1 avoids division by ~0
    f = {}
    for lag in (1, 2, 3, 4, 8, 12):
        f[f"lag_{lag}"] = y[:, t - lag + 1] / scale
    for w in (4, 26):
        f[f"mean_{w}"] = y[:, t - w + 1 : t + 1].mean(axis=1) / scale
    f["std_12"] = y[:, t - 11 : t + 1].std(axis=1) / scale
    f["zero_weeks_12"] = (y[:, t - 11 : t + 1] == 0).mean(axis=1)
    # Trend: last 4 weeks vs the 8 before
    f["trend_4_vs_8"] = y[:, t - 3 : t + 1].mean(axis=1) / (y[:, t - 11 : t - 3].mean(axis=1) + 1)

    # Last year around the target week, relative to last year's level then (NaN before week 52)
    target = t + h
    if target - 52 - 1 >= 0 and t - 52 - 11 >= 0:
        ly_level = y[:, t - 52 - 11 : t - 52 + 1].mean(axis=1) + 1.0
        f["ly_target_rel"] = y[:, target - 52 - 1 : target - 52 + 2].mean(axis=1) / ly_level
    else:
        f["ly_target_rel"] = np.full(y.shape[0], np.nan)

    date = week0 + pd.Timedelta(weeks=target)
    f["h"] = np.full(y.shape[0], h)
    f["target_weekofyear"] = np.full(y.shape[0], date.isocalendar().week)
    xmas = pd.Timestamp(year=date.year, month=12, day=25)
    f["weeks_to_xmas"] = np.full(y.shape[0], (xmas - date).days // 7)
    f["log_scale"] = np.log(scale)
    return pd.DataFrame(f), scale


def training_set(y: np.ndarray, last: int, week0: pd.Timestamp):
    """All (origin, h) pairs whose target week is <= last, i.e. already observed."""
    X, target, weight = [], [], []
    for t in range(MIN_HISTORY - 1, last):
        for h in range(1, bt.HORIZON + 1):
            if t + h > last:
                break
            fx, scale = features(y, t, week0, h)
            X.append(fx)
            target.append(y[:, t + h] / scale)
            weight.append(scale)
    return pd.concat(X, ignore_index=True), np.concatenate(target), np.concatenate(weight)


def make_forecaster(objective: str = "l1", drop: tuple[str, ...] = (), bias_correct: bool = False,
                    alpha: float | None = None, blend_ma12: float = 0.0):
    """Build a forecaster for bt.run.

    bias_correct: L1 loss learns the median, which for right-skewed demand sits below the mean
    and makes the model under-forecast. The fix rescales predictions by total actual / total
    predicted over the last 12 weeks of training targets (in units, so big SKUs count more).
    alpha: with objective="quantile", the quantile to learn (0.5 = median; higher = less under-forecast).
    blend_ma12: weight of the 12-week moving average in a blend with the model's forecast.
    """
    def forecaster(history: pd.DataFrame, horizon: int) -> np.ndarray:
        y = history.to_numpy(dtype=float)
        week0 = history.columns[0]
        t = y.shape[1] - 1
        X, target, weight = training_set(y, t, week0)
        X = X.drop(columns=list(drop))
        params = {**PARAMS, "objective": objective, **({"alpha": alpha} if alpha else {})}
        model = lgb.LGBMRegressor(**params).fit(X, target, sample_weight=weight)
        factor = 1.0
        if bias_correct:
            recent = slice(len(X) - 12 * y.shape[0] * bt.HORIZON, None)
            fit = model.predict(X.iloc[recent]) * weight[recent]
            factor = (target[recent] * weight[recent]).sum() / fit.sum()
        forecaster.last_model, forecaster.last_factor = model, factor
        preds = []
        for h in range(1, horizon + 1):
            fx, scale = features(y, t, week0, h)
            preds.append(model.predict(fx.drop(columns=list(drop))) * scale * factor)
        pred = np.column_stack(preds)
        if blend_ma12:
            ma12 = y[:, t - 11 : t + 1].mean(axis=1)[:, None]
            pred = (1 - blend_ma12) * pred + blend_ma12 * ma12
        return pred
    return forecaster


# Variants compared on the validation period (see validate.py); FINAL is the one chosen there.
VARIANTS = {
    "L1": dict(objective="l1"),
    "L1 + bias correction": dict(objective="l1", bias_correct=True),
    "L2 (mean)": dict(objective="l2"),
    "L1 + bias corr, no week-of-year": dict(objective="l1", bias_correct=True, drop=("target_weekofyear",)),
    "L2, no week-of-year": dict(objective="l2", drop=("target_weekofyear",)),
    "quantile 0.55": dict(objective="quantile", alpha=0.55),
    "quantile 0.60": dict(objective="quantile", alpha=0.60),
    "L1 50/50 blend with MA12": dict(objective="l1", blend_ma12=0.5),
}
# Second round, after the first five: only these are run by validate.py --round 2
ROUND_2 = ("quantile 0.55", "quantile 0.60", "L1 50/50 blend with MA12")
FINAL = "quantile 0.60"   # chosen on validation: lowest WAPE among variants with |bias| <= 10%


def main() -> None:
    wide = bt.to_wide(pd.read_parquet(ROOT / "data" / "interim" / "weekly_panel.parquet"))
    forecaster = make_forecaster(**VARIANTS[FINAL])
    results = bt.run(wide, forecaster, f"LightGBM ({FINAL})")
    out = ROOT / "data" / "processed"
    results.to_parquet(out / "lightgbm_results.parquet", index=False)

    base = pd.read_parquet(out / "baseline_results.parquet")
    allr = pd.concat([base[base.model == "moving average 12w"], results], ignore_index=True)
    print(bt.summary(allr).round(3).to_string())

    model = forecaster.last_model
    imp = pd.Series(model.booster_.feature_importance("gain"), index=model.feature_name_)
    print("\nfeature importance (gain share, last origin):")
    print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
