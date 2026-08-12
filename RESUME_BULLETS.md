# Resume Bullets

Suggested bullets based strictly on what this repository implements and the
metrics it actually produced (see `reports/model_results.csv` and
`reports/run_summary.json`). Re-run the pipeline before reusing numbers —
results shift as new market data arrives.

---

- Built a Python volatility-forecasting pipeline analyzing 4,100+ daily
  S&P 500 (SPY) and VIX observations (2010–2026), forecasting 5-day forward
  realized volatility with naive, OLS, HAR-style, GARCH(1,1), Random Forest,
  and XGBoost models.

- Engineered 35 leakage-safe time-series and market-risk features (multi-
  horizon realized volatility, Parkinson/Garman–Klass range estimators, VIX
  dynamics, drawdown and downside-volatility measures) with unit tests
  verifying forward-target alignment and absence of look-ahead bias.

- Evaluated all models with purged, expanding-window walk-forward validation
  on a held-out 2024–2026 test period; the best model (Random Forest) reduced
  forecast RMSE by 25% versus the naive persistence benchmark
  (0.071 vs 0.094 annualized vol, out-of-sample R² 0.41).

- Conducted volatility-regime analysis showing forecast RMSE rises 2–3×
  in high-volatility regimes across all models, and identified trailing
  short-horizon realized volatility and the VIX as the strongest predictors
  of future realized volatility.

---

### Notes on usage

- The 25% RMSE reduction is specific to the 2024–2026 test window; quote it
  with the caveat "on the held-out test period" if pressed for detail.
- Do not claim trading profitability — this project forecasts volatility
  and does not implement or backtest a strategy.
