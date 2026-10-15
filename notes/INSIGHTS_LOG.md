# Insights log

Running log of my own thinking on the Optiver "Trading at the Close" competition, written before
reading the winning solutions, so I can see later how my understanding changed.

---

## 2026-10-08: Initial intentions (my words)

Since it's a completed competition and similar to the current Wunder Fund one, I should study the
top 2 winners' approaches and also the 5th one, to see what factor stopped the 5th from winning.

From the overview (kaggle.com/competitions/optiver-trading-at-the-close), there may be domain
knowledge I have to fill in to come up with different combinations of strategies: adjusting prices,
assessing supply and demand from auctions. Is it a good strategy to have a self-adjusting price model
that changes and tests which price leads to the most correct prediction?

Should trading opportunities be ranked by which audience they suit, or is it objective?

I know the model has to be very flexible and always testing itself to optimise towards the most
correct prediction, while working out which factor most affected a prediction that went off course.
Even a correct prediction can't be taken as a simple x + y = z, because there is always noise, so it
needs a strategy. I don't know which one yet.

---

## 2026-10-08: What I clarified before modelling

- **I don't set or adjust prices in this task.** The prices (reference, near, far, bid/ask, WAP) are
  inputs. The task is to *predict* the target: each stock's 60-second future move relative to a
  synthetic index, in basis points, scored by mean absolute error. The auction itself is the
  "self-adjusting price" mechanism: it keeps updating the price at which supply and demand would
  clear. My model reads those updates as signals.
- **My "always testing itself" instinct maps to two real ideas:** (1) *online learning*: updating
  the model as new days of data arrive (the competition's test API revealed targets over time, so
  check whether top solutions retrained during the test period); (2) *honest validation*: testing
  on later days than the model trained on, so the score reflects the future, not the past.
- **"Which factor caused the miss" = error analysis.** Group the largest errors and check which
  features (imbalance, spread, time in auction, stock) explain them, e.g. with SHAP on the errors.
  This is my strongest skill from the DR audit and the MICCAI paper.
- **Noise is the core problem, not a side issue.** The target is mostly noise with a small signal.
  Strategies against noise: robust loss functions, many weak features, ensembles of models, and
  never trusting a single good-looking validation score.
- **The 1st vs 5th question:** on Kaggle the final gap between 1st and 5th is often tiny and partly
  luck on the private test set. So "what stopped 5th" may be noise rather than a missing idea. Worth
  checking: how far apart were their scores, and did they differ in validation method, features or
  online retraining?
- **Ranking opportunities by audience:** not part of this competition. The score is objective (MAE).
  In real trading it is partly subjective, because a market maker, a hedge fund and a pension fund
  have different risk limits and time horizons. Park this for later.

## Next reading
1. The competition's Data tab: column definitions (imbalance, reference/near/far price, WAP).
2. Write-ups from 1st, 2nd and 5th place (Kaggle Discussion tab, filter by "place solution").
3. For each: validation method, key features, model type, online retraining or not, final score gap.

---

## 2026-10-09: First baseline results (LightGBM, MAE objective, time split)

- Train days 0-383 (early stopping on days ~345-383), validate on days 384-480 (1.07M rows).
- **Validation MAE 5.963 vs 6.062 for "predict zero": a 1.6% improvement.** Small, and that is the
  honest picture: the target is mostly noise. (For scale: Kaggle leaderboard scores sat in the
  5.3-5.5 range on a different, later period, so they are not directly comparable.)
- **Most important features:** stock_id (31% of gain), book pressure (bid vs ask size, 20%),
  reference-vs-WAP gap (10%), time in auction (7%), spread (7%). The imbalance features matter less
  than I expected.
- **Error over the auction is not smooth.** Errors are highest between 220 and 290 seconds (MAE ~9.5
  vs ~5.5 elsewhere), just before near/far prices start being published at 300 s. Hypothesis to
  check: the 60-second target window then spans the moment new auction information is released, so
  moves are larger and harder to predict. The model also helps least there (<1% gain).
- **The model is most useful at the very start and the very end** (6% gain at 0 s, 7.7% at 540 s).
- **For 20 of 200 stocks, the model is worse than predicting zero.** The noisiest stocks (31, 82, 86)
  have MAE 13-18 and gains under 1%.
- **Error falls over the validation period** (correlation -0.27 between day and MAE): later days
  were calmer, not necessarily better predicted (gain stays ~1-2%).

Questions to take to the winners' write-ups: how did they handle time in auction, stock-level
differences, and the 300-second information release? Did online retraining help?

---

## 2026-10-09: SHAP (explainability) on the baseline

TreeSHAP via LightGBM `pred_contrib`, 200,000 random validation rows (seed 42). Check: SHAP values
sum to the prediction (max difference 8e-14).

- **Top drivers (share of mean |SHAP|):** book pressure 22%, reference-vs-WAP gap 16%, stock 11%,
  time in auction 7%, spread 7%. (LightGBM's own "gain" importance had stock first; SHAP ranks it
  third. Gain importance over-credits features used in many splits.)
- **Direction, book pressure:** when bid size outweighs ask size, the model predicts a LOWER 60-second
  move (decile 10: -1.04 bps; decile 1: +0.98 bps). Counter-intuitive at first. Likely reason: WAP
  already leans towards the ask when bids dominate, so the "good news" is priced in and the move
  reverts. Hypothesis: check against the winners' write-ups.
- **Direction, reference-vs-WAP gap:** when the auction's reference price sits above the current WAP,
  the model predicts an upward move (decile 10: +1.06 bps). Sensible: the price is pulled towards
  where the auction is clearing.
- **When features matter:** near price is almost irrelevant until 300 s, then jumps to one of the
  strongest signals exactly when it is first published (mean |SHAP| 0.03 to 0.43). In the final
  seconds (540 s), time in auction and the reference-vs-WAP gap dominate.
- **Worst 10% of errors:** the reference-vs-WAP gap is 1.66x more active there than in normal
  predictions; raw price levels (WAP, bid, ask) are ~1.4-1.5x. The near/far features are *less*
  involved in big errors. So big misses happen when the model leans hard on the reference-WAP gap.

---

## 2026-10-09: When should the model be trusted? (conformalized quantile regression)

Why this method: the error here is mostly market noise whose size varies by stock, spread and time
(aleatoric). Seed-ensemble disagreement measures model uncertainty (epistemic), which is small for
gradient boosting on millions of rows, so it was rejected. Quantile regression (10th/90th
percentiles) + split-conformal calibration on held-out training days (CQR) gives 80% ranges.

- **Coverage:** 81.8% on validation days vs the 80% target. Raw quantiles alone: 81.0%. The
  conformal adjustment was tiny (0.19 bps).
- **Usefulness:** ranking rows by range width and acting only on the narrowest 50% cuts MAE from
  5.96 to 4.16 bps (-30%). This beats two cheap rules: stock's historical volatility (4.63) and
  spread (4.75). Random order gives no gain (5.96). Area under the risk-coverage curve: 4.35 vs 4.77
  / 4.93 / 5.96. Rank correlation between width and actual error: 0.32.
- **Over time:** coverage drifts UP (80% to 86% across the validation days; correlation 0.55) as
  later days were calmer. The ranges become over-cautious rather than failing.
- **Danger zone confirmed:** at 240-270 s coverage drops to 75-77% even though ranges are widest
  there. The model underestimates the noise just before near/far prices are released.
- **Per stock, the guarantee breaks:** coverage ranges from 59% to 91%; 9 stocks fall below 75%.
  Conformal gives an average guarantee, not a per-stock one.

Next improvement to propose: group-conditional ("Mondrian") conformal calibration, per stock and per
auction phase (before/after 300 s), so each group gets its own correction; and an adaptive conformal
update across days to stop the ranges drifting wide.

---

## 2026-10-09: Group-conditional (Mondrian) + adaptive conformal: results

| Variant | Coverage | Mean width | Worst stock | Stocks <75% | 220-290 s | Day-vs-coverage drift |
|---|---|---|---|---|---|---|
| Global | 81.8% | 19.76 | 59% | 9 | 78.0% | 0.55 |
| Mondrian (stock x phase) | 81.8% | 20.00 | 63% | 6 | 79.2% | 0.56 |
| Mondrian + ACI (daily) | **80.8%** | **19.47** | 62% | 6 | 78.4% | **0.21** |

- Mondrian narrowed the per-stock spread (std 0.040 to 0.032) and lifted the danger zone, for +1%
  width. ACI fixed the drift (0.55 to 0.21) and brought coverage to target with *narrower* ranges.
  Mondrian + ACI is the best overall, but the per-stock failures only partly improved.
- **Root cause of the remaining failures (diagnosed, not guessed):** stocks whose volatility jumped
  after the calibration days lose coverage. Correlation between a stock's volatility ratio
  (validation std / calibration std) and its coverage: **-0.69**. Stock 93's volatility rose 2.4x and
  its coverage fell to 62%. So the issue is per-stock regime change, not the grouping.
- **Next fix to propose:** volatility-normalised conformal scores, dividing each error by the stock's
  recent realised volatility (rolling over revealed days), so ranges scale up automatically when a
  stock gets wilder. Optionally per-stock ACI.

---

## 2026-10-09: Volatility-scaled conformal: results

| Variant | Coverage | Width | Worst stock | Stocks <75% | Stock spread (std) | 220-290 s | Drift |
|---|---|---|---|---|---|---|---|
| Mondrian + ACI | 80.8% | 19.47 | 62% | 6 | 0.032 | 78.4% | 0.21 |
| Vol-scaled + ACI | 80.2% | 19.59 | 64% | 20 | 0.037 | 74.9% | 0.14 |
| **Vol-scaled + Mondrian + ACI** | **80.2%** | 19.79 | **74%** | **5** | **0.022** | 76.4% | **0.15** |

- **The targeted cause is fixed:** stock 93 (volatility up 2.4x) went from 62% to 85% coverage. The
  worst stock overall rose from 62% to 74%, and coverage across stocks is the most even so far
  (std 0.022). Cost: ranges 1.6% wider than Mondrian + ACI.
- **Scaling alone is not enough:** without per-group corrections, 20 stocks fall below 75%. The two
  ideas work together: groups fix the stock's level, scaling fixes its changes over time.
- **Trade-off:** the 220-290 s danger zone got slightly worse (78.4% to 76.4%). Scaling works per
  stock per day, so it cannot see the within-day spike before near/far prices are published.
- **Final choice for the dashboard:** vol-scaled + Mondrian + ACI. Known limitation, stated
  honestly: the window just before the 300 s information release is still under-covered. Fixing it
  needs a signal that anticipates the release (a modelling change), not more calibration.

---

## 2026-10-09: Correction (spotted on the Power BI chart)
An early reading of the summary table described target volatility across the auction as "U-shaped:
hardest at the start and the end". The Power BI chart showed otherwise: it **peaks at 270 s (13.3 bps)**, just before near/far prices
are published at 300 s, then drops sharply (11.0 at 300 s, 10.2 at 540 s; 11.6 at 0 s). The same
220-290 s window is where the model's errors spike and the calibrated ranges under-cover. One story
across three analyses: the moments before new auction information is released are the most uncertain.

---

## 2026-10-12: Feedback from a finance practitioner, checked against the literature

Feedback: pages 1-2 are basic knowledge in finance; where is the model's accuracy headlined?

- Baseline skill in finance terms (validation days 384-480): **correlation with actual moves 0.167
  (Spearman 0.154), positive on every one of the 97 test days; direction correct 54.9% of the time;
  out-of-sample R-squared 2.8%.** MAE gain vs predict-zero is only 1.6% because the target is
  fat-tailed and MAE barely rewards a weak but real signal.
- Context: Gu, Kelly and Xiu (2020, RFS) report monthly out-of-sample R-squared of roughly 0.1-0.4% for
  individual stocks and still find economically large portfolio returns, so small R-squared is normal.
  Horizons differ, so this is context, not a comparison.
- Honest gap: Kaggle top private scores sat near MAE 5.40 against an unofficial predict-zero figure of
  ~6.4 (about 15% better), on a different test period. The baseline is well below competitive level.
- Conformal intervals for position sizing exist in the literature (e.g. Conformal Kelly, arXiv
  2608.01494), but a pre-registered follow-up found calibration transferred while the economic value
  did not. Uncertainty is a second act, not the headline.
- Lesson: lead with predictive skill (correlation, hit rate, R-squared, MAE vs baseline), then where
  it fails, then how confidence should change position size.
