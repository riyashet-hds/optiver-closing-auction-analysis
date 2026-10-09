"""Volatility-scaled conformal ranges: fixing per-stock regime change.

Diagnosis from group_conformal.py: stocks whose volatility rose after the calibration days lost
coverage (corr -0.69 between volatility ratio and coverage; stock 93: 2.4x, 62% coverage).
A correction learned on calm days cannot anticipate a stock turning wild.

Fix: rescale each prediction range by the stock's RECENT volatility relative to its usual level.
    ratio(stock, day) = recent_vol / reference_vol
    recent_vol    = sqrt(mean of daily target variance over the previous K revealed days)
    reference_vol = the stock's target std over all training days
    range         = centre +/- (half_width * ratio + q * ratio)
The conformal correction q is calibrated on the same scaled scores, so the 80% guarantee is kept.
Only past, already revealed days are used for recent_vol (no look-ahead).

Variants compared with the best previous one:
  mondrian_aci             (from group_conformal.py, re-run here for a like-for-like table)
  volscaled_aci            volatility-scaled ranges + global q + daily ACI
  volscaled_mondrian_aci   volatility-scaled ranges + per (stock x phase) q + daily ACI

Input : data/raw/train.csv, data/processed/quantile_lo.txt, quantile_hi.txt (from group_conformal.py)
Output: data/processed/vol_conformal_summary.csv, vol_conformal_by_stock.csv,
        vol_conformal_by_second.csv, vol_conformal_by_date.csv
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from group_conformal import GAMMA, PHASE_SPLIT, conformal_q
from train_baseline import FEATURES, OUT, RAW, VALID_FRACTION, add_features
from uncertainty_analysis import ALPHA

K = 5  # days of recent volatility


def recent_vol_table(df: pd.DataFrame, train_last_day: int) -> pd.DataFrame:
    """Per (stock, day): sqrt(mean daily variance over the previous K days) / training-period std."""
    daily_var = df.groupby(["stock_id", "date_id"])["target"].var().rename("var").reset_index()
    daily_var = daily_var.sort_values(["stock_id", "date_id"])
    daily_var["recent_vol"] = (
        daily_var.groupby("stock_id")["var"].transform(lambda v: v.shift(1).rolling(K, min_periods=2).mean()) ** 0.5
    )
    ref = df[df.date_id <= train_last_day].groupby("stock_id")["target"].std().rename("ref_vol")
    out = daily_var.merge(ref, on="stock_id")
    out["ratio"] = (out["recent_vol"] / out["ref_vol"]).clip(0.25, 4.0).fillna(1.0)
    return out[["stock_id", "date_id", "ratio"]]


def evaluate(name, valid, lo_v, hi_v, danger):
    y = valid["target"].values
    cov = ((y >= lo_v) & (y <= hi_v)).astype(float)
    width = hi_v - lo_v
    t = valid.assign(covered=cov, width=width)
    s = t.groupby("stock_id")["covered"].mean()
    dd = t.groupby("date_id")["covered"].mean()
    summary = {
        "variant": name, "coverage": cov.mean(), "mean_width": width.mean(),
        "stock_cov_min": s.min(), "stock_cov_max": s.max(), "stock_cov_std": s.std(),
        "stocks_below_75pct": int((s < 0.75).sum()),
        "danger_zone_coverage_220_290s": cov[danger].mean(),
        "corr_day_vs_coverage": float(pd.Series(dd.index).corr(pd.Series(dd.values))),
    }
    return summary, s.rename("coverage").reset_index().assign(variant=name), \
        t.groupby("seconds_in_bucket").agg(coverage=("covered", "mean"), mean_width=("width", "mean")) \
         .reset_index().assign(variant=name), dd.rename("coverage").reset_index().assign(variant=name)


def run_aci(valid, centre, half, scale, group_scores, global_scores, use_groups):
    """Daily adaptive conformal with optional per-group q; scale multiplies half-width and q."""
    alpha_t = ALPHA
    lo, hi = np.empty(len(valid)), np.empty(len(valid))
    for d in np.sort(valid["date_id"].unique()):
        idx = np.where(valid["date_id"].values == d)[0]
        level = float(np.clip(1 - alpha_t, 0.0, 1.0))
        qg = conformal_q(global_scores, level)
        if use_groups:
            qd = {g: conformal_q(s, level) for g, s in group_scores.items()}
            q = valid["group"].iloc[idx].map(qd).fillna(qg).values
        else:
            q = np.full(len(idx), qg)
        lo[idx] = centre[idx] - (half[idx] + q) * scale[idx]
        hi[idx] = centre[idx] + (half[idx] + q) * scale[idx]
        y = valid["target"].values[idx]
        alpha_t += GAMMA * (ALPHA - float(np.mean((y < lo[idx]) | (y > hi[idx]))))
    return lo, hi


def main() -> None:
    df = add_features(pd.read_csv(RAW).dropna(subset=["target"]))
    df["phase"] = (df["seconds_in_bucket"] >= PHASE_SPLIT).astype(int)
    days = np.sort(df["date_id"].unique())
    cut = days[int(len(days) * (1 - VALID_FRACTION))]
    train_days = days[days < cut]
    es_cut = train_days[int(len(train_days) * 0.9)]

    vol = recent_vol_table(df, train_last_day=int(es_cut) - 1)
    df = df.merge(vol, on=["stock_id", "date_id"], how="left")
    df["ratio"] = df["ratio"].fillna(1.0)

    lo_b = lgb.Booster(model_file=str(OUT / "quantile_lo.txt"))
    hi_b = lgb.Booster(model_file=str(OUT / "quantile_hi.txt"))
    calib = df[(df.date_id >= es_cut) & (df.date_id < cut)].copy()
    valid = df[df.date_id >= cut].sort_values(["date_id", "stock_id", "seconds_in_bucket"]).reset_index(drop=True)
    for t in (calib, valid):
        t["lo"], t["hi"] = lo_b.predict(t[FEATURES]), hi_b.predict(t[FEATURES])
        t["centre"], t["half"] = (t["lo"] + t["hi"]) / 2, (t["hi"] - t["lo"]) / 2
        t["group"] = list(zip(t["stock_id"], t["phase"]))

    y_c = calib["target"].values
    plain = np.abs(y_c - calib["centre"].values) - calib["half"].values               # unscaled score
    scaled = np.abs(y_c - calib["centre"].values) / calib["ratio"].values - calib["half"].values
    calib["plain"], calib["scaled"] = plain, scaled
    groups_plain = {g: s.values for g, s in calib.groupby("group")["plain"]}
    groups_scaled = {g: s.values for g, s in calib.groupby("group")["scaled"]}

    danger = valid["seconds_in_bucket"].between(220, 290).values
    one = np.ones(len(valid))
    c, h, r = valid["centre"].values, valid["half"].values, valid["ratio"].values
    variants = {
        "mondrian_aci": run_aci(valid, c, h, one, groups_plain, plain, True),
        "volscaled_aci": run_aci(valid, c, h, r, groups_scaled, scaled, False),
        "volscaled_mondrian_aci": run_aci(valid, c, h, r, groups_scaled, scaled, True),
    }
    rows = [evaluate(n, valid, lo, hi, danger) for n, (lo, hi) in variants.items()]
    summ = pd.DataFrame([x[0] for x in rows])
    summ.to_csv(OUT / "vol_conformal_summary.csv", index=False)
    pd.concat([x[1] for x in rows]).to_csv(OUT / "vol_conformal_by_stock.csv", index=False)
    pd.concat([x[2] for x in rows]).to_csv(OUT / "vol_conformal_by_second.csv", index=False)
    pd.concat([x[3] for x in rows]).to_csv(OUT / "vol_conformal_by_date.csv", index=False)
    print(summ.round(4).to_string(index=False))
    s93 = pd.concat([x[1] for x in rows]).query("stock_id == 93")
    print("stock 93:", s93[["variant", "coverage"]].round(3).to_dict("records"))


if __name__ == "__main__":
    main()
