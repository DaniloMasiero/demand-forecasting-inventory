"""Pick the model variant on the validation period (Mar-Jul 2011), before touching the test period.

Run: python src/validate.py
"""
from pathlib import Path

import pandas as pd

import backtest as bt
from baselines import moving_average
import sys

from model import ROUND_2, VARIANTS, make_forecaster

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    wide = bt.to_wide(pd.read_parquet(ROOT / "data" / "interim" / "weekly_panel.parquet"))
    at = bt.validation_origins(wide)
    print(f"validation origins: {wide.columns[at[0]].date()} to {wide.columns[at[-1]].date()}")
    runs = [bt.run(wide, moving_average(12), "moving average 12w", at)]
    second = "--round" in sys.argv and sys.argv[-1] == "2"
    names = ROUND_2 if second else [n for n in VARIANTS if n not in ROUND_2]
    for name in names:
        cfg = VARIANTS[name]
        runs.append(bt.run(wide, make_forecaster(**cfg), name, at))
        print(f"done: {name}", flush=True)
    results = pd.concat(runs, ignore_index=True)
    out = ROOT / "data" / "processed" / f"validation_results{'_round2' if second else ''}.parquet"
    results.to_parquet(out, index=False)
    print(bt.summary(results).sort_values("WAPE").round(3).to_string())


if __name__ == "__main__":
    main()
