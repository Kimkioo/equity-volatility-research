# Research Brief (auto-generated)

**Sample:** 3,957 daily observations, 2010-10-18 to 2026-07-14, 35 engineered features.
**Target:** rv_5d_forward (annualized 5-day forward realized volatility of SPY).
**Validation:** chronological 70/15/15 split with 10 expanding walk-forward folds over the test period.

## Model ranking (test RMSE, lower is better)

| Model | MAE | RMSE | R2 | QLIKE |
|---|---|---|---|---|
| Random Forest | 0.0417 | 0.0706 | 0.411 | 0.359 |
| Linear Regression | 0.0478 | 0.0759 | 0.319 | 0.398 |
| XGBoost | 0.0454 | 0.0785 | 0.271 | 0.374 |
| HAR | 0.0495 | 0.0824 | 0.198 | 0.515 |
| GARCH(1,1) | 0.0535 | 0.0851 | 0.144 | 0.449 |
| Naive (last 5d RV) | 0.0558 | 0.0940 | -0.043 | 0.971 |
| Naive (last 21d RV) | 0.0573 | 0.0970 | -0.111 | 0.599 |

**Best model:** Random Forest (RMSE 0.0706, R2 0.411).
**Best naive benchmark:** Naive (last 5d RV) (RMSE 0.0940).
The best model outperformed the naive benchmark by 24.9% in RMSE.

## Most important predictors (XGBoost gain)

`rv_5d`, `vix`, `drawdown`, `rv_10d`, `ret_mean_21d`.

_Feature importance is predictive, not causal._