"""Fixing where the uncertainty guarantee broke: group-conditional and adaptive conformal calibration.

Problem found in uncertainty_analysis.py: global conformal calibration hits 80% coverage on average,
but per stock it ranged from 59% to 91%, it fell to ~75% at 240-270 s, and it drifted too wide
over time. A single correction is shared by groups whose noise levels differ.

Variants compared on the same validation days:
  A. global      one conformal correction for everyone (as before)
  B. mondrian    a separate correction per (stock x auction phase), phase = before/after 300 s
                 when near/far prices are first published (Vovk's Mondrian conformal prediction)
  C. mondrian+ACI  B, plus adaptive conformal inference (Gibbs & Candes, 2021): after each validation
                 day, the target miscoverage is nudged up or down depending on that day's coverage,
                 so the ranges track calmer or wilder periods. Uses only targets already revealed
                 (like the competition's day-by-day test API).

Input : data/raw/train.csv
Output: data/processed/
    group_conformal_summary.csv      one row per variant: coverage, width, per-stock spread, danger zone
    group_conformal_by_stock.csv     per-stock coverage for each variant (long format)
    group_conformal_by_second.csv    coverage by time in auction for each variant (long format)
    group_conformal_by_date.csv      coverage per day for each variant (long format)
    quantile_lo.txt, quantile_hi.txt saved quantile models (git-ignored)
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from train_baseline import FEATURES, OUT, RAW, SEED, VALID_FRACTION, add_features
from uncertainty_analysis import ALPHA, quantile_model

GAMMA = 0.05          # ACI step size per day
PHASE_SPLIT = 300     # seconds: near/far prices first published


def conformal_q(scores: np.ndarray, level: float) -> float:
    n = len(scores)
    q = min(1.0, max(0.0, np.ceil((n + 1) * level) / n))
    return float(np.quantile(scores, q))


def main() -> None:
    df = add_features(pd.read_csv(RAW).dropna(subset=["target"]))
    df["phase"] = (df["seconds_in_bucket"] >= PHASE_SPLIT).astype(int)
    days = np.sort(df["date_id"].unique())
    cut = days[int(len(days) * (1 - VALID_FRACTION))]
    train_days = days[days < cut]
    es_cut = train_days[int(len(train_days) * 0.9)]
    fit = df[df.date_id < es_cut]
    calib = df[(df.date_id >= es_cut) & (df.date_id < cut)].copy()
    valid = df[df.date_id >= cut].copy()

    lo_path, hi_path = OUT / "quantile_lo.txt", OUT / "quantile_hi.txt"
    if lo_path.exists() and hi_path.exists():
        lo, hi = lgb.Booster(model_file=str(lo_path)), lgb.Booster(model_file=str(hi_path))
        pred_lo, pred_hi = lo.predict, hi.predict
    else:
        lo_m = quantile_model(ALPHA / 2, fit[FEATURES], fit["target"])
        hi_m = quantile_model(1 - ALPHA / 2, fit[FEATURES], fit["target"])
        lo_m.booster_.save_model(str(lo_path)); hi_m.booster_.save_model(str(hi_path))
        pred_lo, pred_hi = lo_m.predict, hi_m.predict

    calib["lo"], calib["hi"] = pred_lo(calib[FEATURES]), pred_hi(calib[FEATURES])
    calib["score"] = np.maximum(calib["lo"] - calib["target"], calib["target"] - calib["hi"])
    valid["lo"], valid["hi"] = pred_lo(valid[FEATURES]), pred_hi(valid[FEATURES])
    valid["group"] = list(zip(valid["stock_id"], valid["phase"]))
    calib["group"] = list(zip(calib["stock_id"], calib["phase"]))

    global_scores = calib["score"].values
    group_scores = {g: s.values for g, s in calib.groupby("group")["score"]}

    results = {}
    # A. global
    q = conformal_q(global_scores, 1 - ALPHA)
    results["global"] = valid["lo"] - q, valid["hi"] + q
    # B. mondrian
    qg = {g: conformal_q(s, 1 - ALPHA) for g, s in group_scores.items()}
    q_row = valid["group"].map(qg).fillna(q).values
    results["mondrian"] = valid["lo"] - q_row, valid["hi"] + q_row
    # C. mondrian + adaptive (ACI) across validation days, in time order
    alpha_t, lo_c, hi_c = ALPHA, np.empty(len(valid)), np.empty(len(valid))
    valid = valid.reset_index(drop=True)
    for d in np.sort(valid["date_id"].unique()):
        idx = np.where(valid["date_id"].values == d)[0]
        level = float(np.clip(1 - alpha_t, 0.0, 1.0))
        qd = {g: conformal_q(s, level) for g, s in group_scores.items()}
        qrow = valid.loc[idx, "group"].map(qd).fillna(conformal_q(global_scores, level)).values
        lo_c[idx], hi_c[idx] = valid.loc[idx, "lo"].values - qrow, valid.loc[idx, "hi"].values + qrow
        y = valid.loc[idx, "target"].values
        miss = float(np.mean((y < lo_c[idx]) | (y > hi_c[idx])))
        alpha_t = alpha_t + GAMMA * (ALPHA - miss)   # day's targets revealed -> update
    results["mondrian_aci"] = lo_c, hi_c

    summary, by_stock, by_sec, by_date = [], [], [], []
    danger = valid["seconds_in_bucket"].between(220, 290)
    for name, (lo_v, hi_v) in results.items():
        lo_v, hi_v = np.asarray(lo_v), np.asarray(hi_v)
        cov = ((valid["target"].values >= lo_v) & (valid["target"].values <= hi_v)).astype(float)
        width = hi_v - lo_v
        t = valid.assign(covered=cov, width=width)
        s = t.groupby("stock_id")["covered"].mean()
        dd = t.groupby("date_id")["covered"].mean()
        summary.append({
            "variant": name, "coverage": cov.mean(), "mean_width": width.mean(),
            "stock_cov_min": s.min(), "stock_cov_max": s.max(), "stock_cov_std": s.std(),
            "stocks_below_75pct": int((s < 0.75).sum()),
            "danger_zone_coverage_220_290s": cov[danger.values].mean(),
            "daily_cov_min": dd.min(), "daily_cov_max": dd.max(),
            "corr_day_vs_coverage": float(pd.Series(dd.index).corr(pd.Series(dd.values))),
        })
        by_stock.append(s.rename("coverage").reset_index().assign(variant=name))
        by_sec.append(t.groupby("seconds_in_bucket").agg(coverage=("covered", "mean"), mean_width=("width", "mean"))
                      .reset_index().assign(variant=name))
        by_date.append(dd.rename("coverage").reset_index().assign(variant=name))

    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "group_conformal_summary.csv", index=False)
    pd.concat(by_stock).to_csv(OUT / "group_conformal_by_stock.csv", index=False)
    pd.concat(by_sec).to_csv(OUT / "group_conformal_by_second.csv", index=False)
    pd.concat(by_date).to_csv(OUT / "group_conformal_by_date.csv", index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
