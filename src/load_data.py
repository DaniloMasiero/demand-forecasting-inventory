"""Read the raw UCI Online Retail II Excel file (two yearly sheets) and save one parquet file."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail_II.xlsx"
OUT = ROOT / "data" / "interim" / "transactions.parquet"


def main() -> None:
    sheets = pd.read_excel(RAW, sheet_name=None, dtype={"Invoice": str, "StockCode": str, "Description": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = ["invoice", "stock_code", "description", "quantity", "invoice_date",
                  "price", "customer_id", "country"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"{len(df):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
