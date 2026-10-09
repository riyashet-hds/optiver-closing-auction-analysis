"""Simple, honest baseline for Optiver "Trading at the Close", plus error tables for Power BI.

Goal: not to compete with the winners, but to see WHERE a reasonable model fails.

- Time-based split: train on earlier days, validate on the last 20% of days (no look-ahead).
  Early stopping uses the last 10% of TRAINING days, so the validation days stay untouched.
- Model: LightGBM with an L1 (mean absolute error) objective, the competition metric.
- Comparison: a naive "predict zero" baseline, so the model's gain is measured, not assumed.
- Fixed seed; deterministic settings.

Input : data/raw/train.csv
Output: data/processed/
    metrics.json                  validation MAE: model vs naive, split details
    feature_importance.csv        gain importance per feature
    error_by_stock.csv            MAE per stock (model and naive)
    error_by_second.csv           MAE by seconds_in_bucket (time in the auction)
    error_by_imbalance.csv        MAE by imbalance-ratio decile
    error_by_date.csv             MAE per validation day (does error drift over time?)
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train.csv"
OUT = ROOT / "data" / "processed"
SEED = 42
VALID_FRACTION = 0.2


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spread_bps"] = (df["ask_price"] - df["bid_price"]) / df["wap"] * 1e4
    df["imbalance_ratio"] = df["imbalance_size"] / (df["imbalance_size"] + df["matched_size"])
    df["signed_imbalance_ratio"] = df["imbalance_ratio"] * df["imbalance_buy_sell_flag"]
    df["near_far_gap"] = df["near_price"] - df["far_price"]
    df["ref_wap_gap"] = df["reference_price"] - df["wap"]
    df["book_pressure"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"])
    return df


FEATURES = [
    "stock_id", "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag", "reference_price",
    "matched_size", "far_price", "near_price", "bid_price", "bid_size", "ask_price", "ask_size", "wap",
    "spread_bps", "imbalance_ratio", "signed_imbalance_ratio", "near_far_gap", "ref_wap_gap",
    "book_pressure",
]


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW).dropna(subset=["target"])
    df = add_features(df)

    days = np.sort(df["date_id"].unique())
    cut = days[int(len(days) * (1 - VALID_FRACTION))]
    train, valid = df[df.date_id < cut], df[df.date_id >= cut]
    train_days = days[days < cut]
    es_cut = train_days[int(len(train_days) * 0.9)]
    fit, early = train[train.date_id < es_cut], train[train.date_id >= es_cut]
    print(f"train days {days[0]}-{cut - 1} ({len(train):,} rows) | valid days {cut}-{days[-1]} ({len(valid):,} rows)")

    model = lgb.LGBMRegressor(
        objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, deterministic=True,
        force_row_wise=True, verbose=-1,
    )
    model.fit(
        fit[FEATURES], fit["target"],
        eval_set=[(early[FEATURES], early["target"])], eval_metric="l1",
        categorical_feature=["stock_id"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    valid = valid.assign(pred=model.predict(valid[FEATURES]))
    valid["abs_err"] = (valid["target"] - valid["pred"]).abs()
    valid["abs_err_naive"] = valid["target"].abs()

    metrics = {
        "valid_mae_model": mae(valid.target, valid.pred),
        "valid_mae_naive_zero": mae(valid.target, 0.0),
        "improvement_pct": None,
        "train_days": [int(days[0]), int(cut - 1)],
        "valid_days": [int(cut), int(days[-1])],
        "n_train_rows": int(len(fit)), "n_early_stop_rows": int(len(early)), "n_valid_rows": int(len(valid)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators), "seed": SEED,
    }
    metrics["improvement_pct"] = 100 * (1 - metrics["valid_mae_model"] / metrics["valid_mae_naive_zero"])
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")}) \
        .sort_values("gain", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)

    agg = {"mae_model": ("abs_err", "mean"), "mae_naive": ("abs_err_naive", "mean"), "rows": ("abs_err", "size")}
    valid.groupby("stock_id").agg(**agg).reset_index().to_csv(OUT / "error_by_stock.csv", index=False)
    valid.groupby("seconds_in_bucket").agg(**agg).reset_index().to_csv(OUT / "error_by_second.csv", index=False)
    valid.groupby("date_id").agg(**agg).reset_index().to_csv(OUT / "error_by_date.csv", index=False)
    valid["imbalance_decile"] = pd.qcut(valid["imbalance_ratio"], 10, labels=False, duplicates="drop") + 1
    valid.groupby("imbalance_decile").agg(**agg).reset_index().to_csv(OUT / "error_by_imbalance.csv", index=False)
    model.booster_.save_model(str(OUT / "model.txt"))
    print("wrote error tables and model.txt to", OUT)


if __name__ == "__main__":
    main()
