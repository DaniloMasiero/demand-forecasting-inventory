"""Clean the transactions and build a weekly demand panel (one row per SKU per week).

Run: python src/prepare.py  (after src/load_data.py)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"

# A SKU enters the panel only if it sold in at least this share of weeks: forecasting needs history.
MIN_ACTIVE_SHARE = 0.85


def drop_cancelled_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove sales that were later cancelled, plus the cancellation lines themselves.

    A cancellation (invoice starting with "C") cancels an earlier line with the same customer,
    SKU and quantity. Removing only the "C" line would keep phantom demand, e.g. an order of
    80,995 units cancelled 12 minutes later.
    """
    is_cancel = df["invoice"].str.startswith("C")
    cancels = df.loc[is_cancel & df["customer_id"].notna(),
                     ["customer_id", "stock_code", "quantity", "invoice_date"]]
    cancels = cancels.assign(quantity=-cancels["quantity"]).rename(columns={"invoice_date": "cancel_date"})

    sales = df[~is_cancel].reset_index(names="row")
    matched = sales.merge(cancels, on=["customer_id", "stock_code", "quantity"])
    matched = matched[matched["cancel_date"] >= matched["invoice_date"]]
    # For each cancellation, take only the latest sale before it (one cancel removes one sale)
    matched = matched.sort_values("invoice_date").groupby(
        ["customer_id", "stock_code", "quantity", "cancel_date"]).tail(1)
    kept = sales[~sales["row"].isin(matched["row"])].drop(columns="row")
    return kept, matched["row"].nunique()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates()
    print(f"exact duplicates removed: {n0 - len(df):,}")

    df, n_matched = drop_cancelled_orders(df)
    print(f"sales removed because they were cancelled later: {n_matched:,}")

    # Negative quantities outside cancellations are stock adjustments (damages, missing), not sales.
    # Price 0 lines are free samples or adjustments.
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    # Real products have a 5-digit code (optionally with a letter suffix for variants).
    # POST, DOT, M, BANK CHARGES, AMAZONFEE... are fees and manual entries.
    df = df[df["stock_code"].str.upper().str.match(r"^\d{5}")]
    df = df.assign(stock_code=df["stock_code"].str.upper())
    print(f"clean sales lines: {len(df):,} ({len(df) / n0:.1%} of raw)")
    return df


def weekly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """SKU x week units, with zeros for weeks without sales."""
    df = df.assign(week=df["invoice_date"].dt.to_period("W-SUN").dt.start_time)
    # The first and last weeks are partial (data starts on a Tuesday and ends on a Friday)
    first, last = df["week"].min(), df["week"].max()
    df = df[(df["week"] > first) & (df["week"] < last)]

    units = df.groupby(["stock_code", "week"])["quantity"].sum()
    weeks = pd.date_range(units.index.get_level_values("week").min(),
                          units.index.get_level_values("week").max(), freq="7D")
    n_weeks = len(weeks)

    active = units.groupby("stock_code").size()
    keep = active[active >= MIN_ACTIVE_SHARE * n_weeks].index
    print(f"weeks: {n_weeks}, SKUs kept: {len(keep)} of {active.size} "
          f"(sold in >= {MIN_ACTIVE_SHARE:.0%} of weeks)")

    full = pd.MultiIndex.from_product([keep, weeks], names=["stock_code", "week"])
    panel = units.reindex(full, fill_value=0).rename("units").reset_index()
    share = panel["units"].sum() / units.sum()
    print(f"kept SKUs cover {share:.1%} of all units sold")
    return panel


def main() -> None:
    df = pd.read_parquet(INTERIM / "transactions.parquet")
    panel = weekly_panel(clean(df))
    out = INTERIM / "weekly_panel.parquet"
    panel.to_parquet(out, index=False)
    print(f"{len(panel):,} rows -> {out}")


if __name__ == "__main__":
    main()
