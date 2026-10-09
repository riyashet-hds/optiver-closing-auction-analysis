"""Turn Optiver 'Trading at the Close' train.csv into small, dashboard-ready tables for Power BI.

Input : data/raw/train.csv  (download from Kaggle after accepting the competition rules)
Output: data/processed/*.csv

Tables
- stock_day.csv       one row per stock per day: imbalance, spread, target volatility
- auction_second.csv  one row per second-in-auction (all stocks): how the auction evolves
- stock_summary.csv   one row per stock: overall predictability metrics for the decision page
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train.csv"
OUT = ROOT / "data" / "processed"

EXPECTED = {
    "stock_id", "date_id", "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
    "reference_price", "matched_size", "bid_price", "bid_size", "ask_price", "ask_size", "wap", "target",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    missing = EXPECTED - set(df.columns)
    if missing:
        raise SystemExit(f"train.csv is missing expected columns: {sorted(missing)}")
    df = df.dropna(subset=["target"])
    # Derived row-level features (all interpretable, so the dashboard can explain them)
    df["spread_bps"] = (df["ask_price"] - df["bid_price"]) / df["wap"] * 1e4
    df["imbalance_ratio"] = df["imbalance_size"] / (df["imbalance_size"] + df["matched_size"])
    df["signed_imbalance_ratio"] = df["imbalance_ratio"] * df["imbalance_buy_sell_flag"]
    df["abs_target"] = df["target"].abs()
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"rows {len(df):,} | stocks {df.stock_id.nunique()} | days {df.date_id.nunique()}")

    stock_day = df.groupby(["stock_id", "date_id"]).agg(
        mean_spread_bps=("spread_bps", "mean"),
        mean_imbalance_ratio=("imbalance_ratio", "mean"),
        mean_signed_imbalance=("signed_imbalance_ratio", "mean"),
        target_std=("target", "std"),
        mean_abs_target=("abs_target", "mean"),
        n_snapshots=("target", "size"),
    ).reset_index()
    stock_day.to_csv(OUT / "stock_day.csv", index=False)

    auction_second = df.groupby("seconds_in_bucket").agg(
        mean_spread_bps=("spread_bps", "mean"),
        mean_imbalance_ratio=("imbalance_ratio", "mean"),
        target_std=("target", "std"),
        mean_abs_target=("abs_target", "mean"),
    ).reset_index()
    auction_second.to_csv(OUT / "auction_second.csv", index=False)

    stock_summary = df.groupby("stock_id").agg(
        target_std=("target", "std"),
        mean_abs_target=("abs_target", "mean"),
        mean_spread_bps=("spread_bps", "mean"),
        mean_imbalance_ratio=("imbalance_ratio", "mean"),
        days=("date_id", "nunique"),
    ).reset_index()
    stock_summary["predictability_rank"] = stock_summary["target_std"].rank(ascending=False).astype(int)
    stock_summary.to_csv(OUT / "stock_summary.csv", index=False)

    for name in ["stock_day", "auction_second", "stock_summary"]:
        print("wrote", OUT / f"{name}.csv")


if __name__ == "__main__":
    main()
