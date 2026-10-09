"""Explainability for the baseline: SHAP values from LightGBM's built-in TreeSHAP.

Why: the error tables show WHERE the model fails; SHAP shows WHAT drives its predictions there.
Uses `pred_contrib=True` (exact TreeSHAP inside LightGBM), so no extra package is needed.

Input : data/raw/train.csv, data/processed/model.txt (from train_baseline.py)
Output: data/processed/
    shap_global.csv          mean |SHAP| and mean SHAP per feature (overall importance + direction)
    shap_by_second.csv       mean |SHAP| per feature at each second of the auction (long format)
    shap_big_errors.csv      mean |SHAP| per feature for the worst 10% of errors vs the rest
    shap_dependence.csv      feature decile vs mean SHAP for the top numeric features (long format)
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_baseline import FEATURES, OUT, RAW, SEED, VALID_FRACTION, add_features

SAMPLE_ROWS = 200_000


def main() -> None:
    booster = lgb.Booster(model_file=str(OUT / "model.txt"))
    df = add_features(pd.read_csv(RAW).dropna(subset=["target"]))
    days = np.sort(df["date_id"].unique())
    cut = days[int(len(days) * (1 - VALID_FRACTION))]
    valid = df[df.date_id >= cut].sample(n=SAMPLE_ROWS, random_state=SEED).reset_index(drop=True)
    valid["stock_id"] = valid["stock_id"].astype("int64")

    contrib = booster.predict(valid[FEATURES], pred_contrib=True)
    shap = pd.DataFrame(contrib[:, :-1], columns=FEATURES)
    valid["pred"] = contrib.sum(axis=1)
    valid["abs_err"] = (valid["target"] - valid["pred"]).abs()
    print(f"SHAP on {len(valid):,} validation rows; check: max |sum(SHAP)+base - predict| =",
          float(np.abs(valid["pred"] - booster.predict(valid[FEATURES])).max()))

    glob = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_shap": shap.abs().mean().values,
        "mean_shap": shap.mean().values,
    }).sort_values("mean_abs_shap", ascending=False)
    glob["share_pct"] = 100 * glob["mean_abs_shap"] / glob["mean_abs_shap"].sum()
    glob.to_csv(OUT / "shap_global.csv", index=False)
    print(glob.round(4).to_string(index=False))

    # key renamed so it does not overwrite the SHAP column of the feature "seconds_in_bucket"
    by_sec = shap.abs().assign(auction_second=valid["seconds_in_bucket"].values).groupby("auction_second").mean()
    by_sec.reset_index().melt(id_vars="auction_second", var_name="feature", value_name="mean_abs_shap") \
        .to_csv(OUT / "shap_by_second.csv", index=False)

    big = valid["abs_err"] >= valid["abs_err"].quantile(0.9)
    big_tbl = pd.DataFrame({
        "feature": FEATURES,
        "worst_10pct_errors": shap[big].abs().mean().values,
        "other_90pct": shap[~big].abs().mean().values,
    })
    big_tbl["ratio"] = big_tbl["worst_10pct_errors"] / big_tbl["other_90pct"]
    big_tbl.sort_values("ratio", ascending=False).to_csv(OUT / "shap_big_errors.csv", index=False)

    numeric_top = [f for f in glob["feature"] if f != "stock_id"][:4]
    rows = []
    for f in numeric_top:
        dec = pd.qcut(valid[f], 10, labels=False, duplicates="drop")
        t = pd.DataFrame({"decile": dec + 1, "value": valid[f], "shap": shap[f]}).groupby("decile") \
            .agg(feature_value_mean=("value", "mean"), mean_shap=("shap", "mean")).reset_index()
        t.insert(0, "feature", f)
        rows.append(t)
    pd.concat(rows).to_csv(OUT / "shap_dependence.csv", index=False)
    print("wrote shap tables to", OUT)


if __name__ == "__main__":
    main()
