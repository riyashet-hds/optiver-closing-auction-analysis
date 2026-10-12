"""Predictive skill of the baseline in the terms finance uses: correlation (information
coefficient), direction hit rate, out-of-sample R-squared, and MAE against predict-zero.

Input : data/raw/train.csv, data/processed/model.txt
Output: data/processed/skill_summary.json   overall skill metrics
        data/processed/skill_by_day.csv     per test day: correlation, hit rate, MAE model and zero
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_baseline import FEATURES, OUT, RAW, VALID_FRACTION, add_features


def main() -> None:
    df = add_features(pd.read_csv(RAW).dropna(subset=["target"]))
    days = np.sort(df["date_id"].unique())
    cut = days[int(len(days) * (1 - VALID_FRACTION))]
    v = df[df.date_id >= cut].copy()
    v["pred"] = lgb.Booster(model_file=str(OUT / "model.txt")).predict(v[FEATURES])
    v["hit"] = (np.sign(v["pred"]) == np.sign(v["target"])).astype(float)

    by_day = v.groupby("date_id").apply(lambda g: pd.Series({
        "correlation": g["pred"].corr(g["target"]),
        "hit_rate": g["hit"].mean(),
        "mae_model": (g["target"] - g["pred"]).abs().mean(),
        "mae_zero": g["target"].abs().mean(),
    })).reset_index()
    by_day.to_csv(OUT / "skill_by_day.csv", index=False)

    r2 = 1 - ((v.target - v.pred) ** 2).sum() / ((v.target - v.target.mean()) ** 2).sum()
    summary = {
        "correlation_pearson": float(v["pred"].corr(v["target"])),
        "correlation_spearman": float(v["pred"].corr(v["target"], method="spearman")),
        "mean_daily_correlation": float(by_day["correlation"].mean()),
        "days_positive_correlation": int((by_day["correlation"] > 0).sum()),
        "test_days": int(len(by_day)),
        "direction_hit_rate": float(v["hit"].mean()),
        "r2_out_of_sample": float(r2),
        "mae_model": float((v.target - v.pred).abs().mean()),
        "mae_zero": float(v.target.abs().mean()),
    }
    summary["mae_reduction_pct"] = 100 * (1 - summary["mae_model"] / summary["mae_zero"])
    (OUT / "skill_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
