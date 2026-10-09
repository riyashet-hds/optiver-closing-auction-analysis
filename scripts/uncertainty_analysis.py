"""When should the baseline be trusted? Calibrated uncertainty with conformalized quantile regression.

Root issue: most of the target is noise, and how much noise varies a lot by stock, spread and time in
the auction (aleatoric uncertainty). Seed-ensemble disagreement mainly measures model uncertainty
(epistemic), which is small for gradient boosting trained on millions of rows, so it is the wrong
tool here. Instead:

1. Quantile regression (LightGBM, alpha 0.1 and 0.9) predicts an 80% range for each row.
2. Split conformal calibration (CQR, Romano et al. 2019) widens the range using held-out days, so
   coverage is guaranteed on exchangeable data. Coverage is then checked over the validation days
   to see whether it holds as time moves on (drift).
3. Risk-coverage: rank rows by interval width; if we only act on the narrowest X%, how much does
   the point model's MAE drop? Compared against two simple rankings (per-stock historical volatility,
   spread) and random order, so the method has to beat cheap heuristics to be worth it.

Same day split as train_baseline.py: fit on days < es_cut, calibrate on [es_cut, cut), validate on
days >= cut.

Input : data/raw/train.csv, data/processed/model.txt
Output: data/processed/
    uncertainty_summary.json         coverage, width, rank correlation, AURC per ranking
    coverage_by_date.csv             coverage and width per validation day (drift check)
    coverage_by_second.csv           coverage and width by time in the auction
    coverage_by_stock.csv            coverage and width per stock
    risk_coverage.csv                MAE kept vs fraction kept, for each ranking (long format)
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_baseline import FEATURES, OUT, RAW, SEED, VALID_FRACTION, add_features

ALPHA = 0.2  # target 80% intervals
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, random_state=SEED, deterministic=True,
              force_row_wise=True, verbose=-1)


def quantile_model(alpha, X, y):
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **PARAMS)
    m.fit(X, y, categorical_feature=["stock_id"])
    return m


def aurc(order_values, abs_err, fractions):
    """Mean of the risk-coverage curve: MAE of the kept rows, keeping the lowest-uncertainty first."""
    idx = np.argsort(order_values, kind="stable")
    err = abs_err[idx]
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    curve = [cum[max(int(f * len(err)) - 1, 0)] for f in fractions]
    return curve, float(np.mean(curve))


def main() -> None:
    df = add_features(pd.read_csv(RAW).dropna(subset=["target"]))
    days = np.sort(df["date_id"].unique())
    cut = days[int(len(days) * (1 - VALID_FRACTION))]
    train_days = days[days < cut]
    es_cut = train_days[int(len(train_days) * 0.9)]
    fit = df[df.date_id < es_cut]
    calib = df[(df.date_id >= es_cut) & (df.date_id < cut)]
    valid = df[df.date_id >= cut].copy()
    print(f"fit {len(fit):,} | calibrate {len(calib):,} | validate {len(valid):,}")

    lo = quantile_model(ALPHA / 2, fit[FEATURES], fit["target"])
    hi = quantile_model(1 - ALPHA / 2, fit[FEATURES], fit["target"])

    # Conformal calibration on held-out training days
    c_lo, c_hi = lo.predict(calib[FEATURES]), hi.predict(calib[FEATURES])
    scores = np.maximum(c_lo - calib["target"].values, calib["target"].values - c_hi)
    n = len(scores)
    qhat = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))

    v_lo = lo.predict(valid[FEATURES]) - qhat
    v_hi = hi.predict(valid[FEATURES]) + qhat
    valid["covered"] = ((valid["target"] >= v_lo) & (valid["target"] <= v_hi)).astype(int)
    valid["width"] = v_hi - v_lo
    raw_cov = float(np.mean((valid["target"] >= v_lo + qhat) & (valid["target"] <= v_hi - qhat)))

    point = lgb.Booster(model_file=str(OUT / "model.txt"))
    valid["pred"] = point.predict(valid[FEATURES])
    valid["abs_err"] = (valid["target"] - valid["pred"]).abs()

    # Cheap heuristic rankings to beat
    stock_vol = df[df.date_id < cut].groupby("stock_id")["target"].std()
    valid["stock_hist_vol"] = valid["stock_id"].map(stock_vol)
    rng = np.random.default_rng(SEED)
    rankings = {
        "conformal_width": valid["width"].values,
        "stock_historical_volatility": valid["stock_hist_vol"].values,
        "spread_bps": valid["spread_bps"].fillna(valid["spread_bps"].median()).values,
        "random": rng.random(len(valid)),
    }
    fractions = np.round(np.arange(0.1, 1.0001, 0.05), 2)
    abs_err = valid["abs_err"].values
    rc_rows, aurcs = [], {}
    for name, vals in rankings.items():
        curve, a = aurc(vals, abs_err, fractions)
        aurcs[name] = a
        rc_rows += [{"ranking": name, "fraction_kept": f, "mae_kept": m} for f, m in zip(fractions, curve)]
    pd.DataFrame(rc_rows).to_csv(OUT / "risk_coverage.csv", index=False)

    summary = {
        "target_coverage": 1 - ALPHA,
        "raw_quantile_coverage_valid": raw_cov,
        "conformal_qhat_bps": qhat,
        "conformal_coverage_valid": float(valid["covered"].mean()),
        "mean_width_bps": float(valid["width"].mean()),
        "spearman_width_vs_abs_error": float(valid["width"].rank().corr(valid["abs_err"].rank())),
        "aurc": aurcs,
        "mae_all_rows": float(abs_err.mean()),
        "mae_narrowest_50pct": float(np.sort(abs_err[np.argsort(valid["width"].values)][: len(valid) // 2]).mean()),
    }
    (OUT / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    agg = {"coverage": ("covered", "mean"), "mean_width": ("width", "mean"), "mae": ("abs_err", "mean"),
           "rows": ("covered", "size")}
    valid.groupby("date_id").agg(**agg).reset_index().to_csv(OUT / "coverage_by_date.csv", index=False)
    valid.groupby("seconds_in_bucket").agg(**agg).reset_index().to_csv(OUT / "coverage_by_second.csv", index=False)
    valid.groupby("stock_id").agg(**agg).reset_index().to_csv(OUT / "coverage_by_stock.csv", index=False)
    print("wrote uncertainty tables to", OUT)


if __name__ == "__main__":
    main()
