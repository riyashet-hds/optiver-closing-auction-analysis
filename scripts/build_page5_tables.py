"""Combine the conformal results into labelled tables for the Power BI 'When to trust' page.
Run after uncertainty_analysis.py, group_conformal.py and vol_conformal.py (inside scripts/)."""
import pandas as pd

from train_baseline import OUT

NAMES = {"global": "1. One shared correction", "mondrian": "2. Per stock & phase",
         "mondrian_aci": "3. Per stock & phase + daily update",
         "volscaled_mondrian_aci": "4. + Volatility scaling (final)"}


def main() -> None:
    g, v = pd.read_csv(OUT / "group_conformal_summary.csv"), pd.read_csv(OUT / "vol_conformal_summary.csv")
    s = pd.concat([g[g.variant.isin(["global", "mondrian", "mondrian_aci"])], v[v.variant == "volscaled_mondrian_aci"]])
    s["method"] = s.variant.map(NAMES)
    s = s[["method", "coverage", "mean_width", "stock_cov_min", "stocks_below_75pct",
           "danger_zone_coverage_220_290s", "corr_day_vs_coverage"]]
    for c in ["coverage", "stock_cov_min", "danger_zone_coverage_220_290s"]:
        s[c] = 100 * s[c]
    s.round(3).to_csv(OUT / "conformal_methods.csv", index=False)

    gs, vs = pd.read_csv(OUT / "group_conformal_by_stock.csv"), pd.read_csv(OUT / "vol_conformal_by_stock.csv")
    st = gs[gs.variant == "global"][["stock_id", "coverage"]].rename(columns={"coverage": "coverage_shared"}).merge(
        vs[vs.variant == "volscaled_mondrian_aci"][["stock_id", "coverage"]].rename(columns={"coverage": "coverage_final"}),
        on="stock_id")
    st[["coverage_shared", "coverage_final"]] *= 100
    st.round(2).to_csv(OUT / "conformal_by_stock.csv", index=False)

    gsec, vsec = pd.read_csv(OUT / "group_conformal_by_second.csv"), pd.read_csv(OUT / "vol_conformal_by_second.csv")
    sec = pd.concat([gsec[gsec.variant == "global"], vsec[vsec.variant == "volscaled_mondrian_aci"]])
    sec["method"] = sec.variant.map(NAMES)
    sec["coverage"] *= 100
    sec[["seconds_in_bucket", "method", "coverage", "mean_width"]].round(3).to_csv(OUT / "conformal_by_second.csv", index=False)

    r = pd.read_csv(OUT / "risk_coverage.csv")
    r["ranking"] = r.ranking.map({"conformal_width": "Model uncertainty (calibrated range)",
                                  "stock_historical_volatility": "Stock's past volatility",
                                  "spread_bps": "Spread", "random": "Random"})
    r["fraction_kept"] *= 100
    r.round(3).to_csv(OUT / "risk_coverage_labelled.csv", index=False)
    print("wrote page 5 tables")


if __name__ == "__main__":
    main()
