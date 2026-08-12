# Research Summary — Equity Volatility Forecasting & Market Regime Research

*All numbers in this summary are produced by `run_pipeline.py` (see
`reports/run_summary.json` for the structured record of the run).*

## Hypothesis

Short-horizon S&P 500 realized volatility is forecastable beyond naive
persistence: features capturing volatility dynamics (multi-horizon trailing
RV, range-based estimators), market-implied expectations (VIX), market
stress (drawdowns, downside vol), and volume should reduce out-of-sample
forecast error relative to "the next week will look like the recent past."

## Dataset

Daily SPY OHLCV and adjusted close, VIX index, and Treasury-yield proxies
(^TNX, ^IRX) from Yahoo Finance — 4,177 raw daily observations from
2010-01-04 to 2026-08-12. After rolling-window warm-up the modeling sample
is 3,957 observations (2010-10-18 onward) with 35 engineered features. The
sample covers the post-GFC recovery, the 2015–16 and 2018 corrections, the
2020 COVID crash, the 2022 bear market, and the 2025 volatility spike.

## Methodology

- **Target:** annualized 5-day forward realized volatility,
  RV = sqrt(252/5 × Σ r²) over returns t+1..t+5, from daily log returns on
  adjusted close. Alignment is unit-tested against manual calculations.
- **Features (35):** trailing RV (5/10/21/63d), Parkinson and Garman–Klass
  estimators, momentum and rolling return statistics, moving-average
  distances, VIX level/changes/rolling stats and the VIX−RV spread,
  drawdown and downside-volatility measures, relative volume, and optional
  yield features.
- **Validation:** chronological 70/15/15 split (test: 2024-02-29 to
  2026-07-14, 594 obs). Final evaluation uses expanding-window walk-forward
  refits (10 folds × 63 trading days) with the last 5 training observations
  of each fold purged, since their forward targets overlap the test window.
  Hyperparameters were selected on the validation block only; the test set
  was used once for final reporting.

## Models

Naive persistence (trailing 5d and 21d RV), OLS on all standardized
features, a HAR-style regression (RV at 5/21/63d horizons), XGBoost (small
grid, 24 combinations), Random Forest, and rolling-refit GARCH(1,1).

## Results (test set, walk-forward)

| Model | MAE | RMSE | R² | QLIKE |
|---|---|---|---|---|
| Random Forest | 0.0417 | 0.0706 | 0.411 | 0.359 |
| Linear Regression | 0.0478 | 0.0759 | 0.319 | 0.398 |
| XGBoost | 0.0454 | 0.0785 | 0.272 | 0.374 |
| HAR | 0.0495 | 0.0824 | 0.198 | 0.515 |
| GARCH(1,1) | 0.0535 | 0.0851 | 0.144 | 0.449 |
| Naive (last 5d RV) | 0.0558 | 0.0940 | −0.044 | 0.971 |
| Naive (last 21d RV) | 0.0573 | 0.0970 | −0.111 | 0.599 |

## Key Findings

1. Feature-based models beat naive persistence on this window: the best
   model cut RMSE 25% versus the best naive baseline. Naive forecasts even
   posted negative out-of-sample R², because trailing volatility lags
   around spikes.
2. Model sophistication added little beyond the features themselves: plain
   OLS (RMSE 0.0759) slightly beat tuned XGBoost (0.0785) and nearly
   matched the Random Forest — most predictable variation is linear
   persistence structure.
3. VIX is the strongest single linear correlate of future 5-day realized
   volatility (≈0.71) and ranks second in XGBoost gain importance; implied
   volatility appears to add forward-looking information that backward-
   looking features cannot. Predictive, not causal.
4. The most important XGBoost features were trailing 5-day RV, VIX level,
   drawdown depth, trailing 10-day RV, and 21-day mean return.
5. Every model's RMSE is ~2–3× larger in the high-volatility regime than
   the low regime (regimes = training-period terciles of trailing 21d RV).
   Forecast uncertainty should be reported regime-conditionally.

## Limitations

Daily public data only (no intraday RV); no option-chain/implied-surface
data beyond the VIX index; overlapping 5-day targets reduce the effective
sample size, so small metric gaps are not statistically decisive; a single
2024–2026 test window; a limited macro feature set; and no trading strategy
or costs — this is a forecasting study.

## Future Extensions

Intraday realized volatility, option-implied surfaces and the volatility
risk premium, single-name volatility and dispersion, Diebold–Mariano tests
on forecast differences, and carefully regularized deep-learning
comparisons.
