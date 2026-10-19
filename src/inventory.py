"""What the better forecast is worth: safety stock needed for the same service level.

Run: python src/inventory.py  (after src/validate.py --round 2 and src/model.py)

The buyer orders forecast + safety stock to cover the 4-week lead time. A more accurate forecast
needs less safety stock for the same protection against stockouts. Safety stock is sized on the
validation period (Mar-Jun 2011) and checked on the test period (Jul-Nov 2011), so the service
level reported is out of sample.

No cost assumptions are needed: inventory is valued at each SKU's median selling price.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import clean

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TARGET_SERVICE = 0.95          # share of 4-week cycles with no stockout
Z = 1.645                      # normal quantile for 95%
MODELS = {"moving average 12w": "moving average 12w", "quantile 0.60": "LightGBM (quantile 0.60)"}


def lead_time_demand(results: pd.DataFrame) -> pd.DataFrame:
    """Sum of the 4 weekly targets per SKU and origin: demand during one replenishment cycle."""
    return results.groupby(["model", "stock_code", "origin"])[["actual", "forecast"]].sum().reset_index()


def unit_prices() -> pd.Series:
    df = clean(pd.read_parquet(ROOT / "data" / "interim" / "transactions.parquet"))
    df = df[df["invoice_date"].dt.year == 2011]
    return df.groupby("stock_code")["price"].median().rename("price")


def error_stats(val: pd.DataFrame) -> pd.DataFrame:
    """Per model and SKU: mean (bias) and std of the lead-time forecast error on validation."""
    err = val["actual"] - val["forecast"]
    return err.groupby([val["model"], val["stock_code"]]).agg(["mean", "std"]).reset_index()


def evaluate(test: pd.DataFrame, z: float) -> pd.DataFrame:
    """Service level, fill rate and safety stock value per model for safety stock = mean + z * std."""
    t = test.assign(safety_stock=(test["mean"] + z * test["std"]).clip(lower=0))
    t["covered"] = t["actual"] <= t["forecast"] + t["safety_stock"]
    t["short_units"] = (t["actual"] - t["forecast"] - t["safety_stock"]).clip(lower=0)
    return t.groupby("model").apply(lambda g: pd.Series({
        "z": z,
        "service level": g["covered"].mean(),
        "fill rate": 1 - g["short_units"].sum() / g["actual"].sum(),
        # Safety stock is a per-SKU constant; count it once per SKU
        "safety stock value": g.groupby("stock_code")[["safety_stock", "price"]].first().prod(axis=1).sum(),
    }), include_groups=False).reset_index()


def main() -> None:
    val = lead_time_demand(pd.read_parquet(PROCESSED / "validation_results_round2.parquet"))
    test = pd.concat([pd.read_parquet(PROCESSED / f) for f in
                      ("baseline_results.parquet", "lightgbm_results.parquet")])
    test = lead_time_demand(test)

    val["model"] = val["model"].map(MODELS)
    val = val.dropna(subset=["model"])
    test = test[test["model"].isin(MODELS.values())]

    prices = unit_prices()
    test = test.merge(error_stats(val), on=["model", "stock_code"]).merge(prices, on="stock_code")

    print(f"textbook sizing, z = {Z} (aims at {TARGET_SERVICE:.0%} if errors were normal and stable):")
    print(evaluate(test, Z).round(3).to_string(index=False))

    # Trade-off curve: safety stock value needed for each service level actually achieved on test
    curve = pd.concat([evaluate(test, z) for z in np.arange(0, 6.01, 0.05)], ignore_index=True)
    curve.to_parquet(PROCESSED / "inventory_curve.parquet", index=False)
    rows = []
    for level in (0.80, 0.85, 0.90, 0.95):
        row = {"service level": level}
        for model, g in curve.groupby("model"):
            g = g.sort_values("service level")
            row[model] = np.interp(level, g["service level"], g["safety stock value"])
        rows.append(row)
    table = pd.DataFrame(rows).set_index("service level")
    table["saved by model"] = table["moving average 12w"] - table["LightGBM (quantile 0.60)"]
    table["saved %"] = table["saved by model"] / table["moving average 12w"]
    print("\nsafety stock value (GBP, at median selling price) for the same service level on test:")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
