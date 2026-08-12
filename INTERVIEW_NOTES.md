# Interview Notes — Equity Volatility Forecasting & Market Regime Research

Concise explanations for discussing this project in a quantitative research
interview. Numbers referenced here come from `reports/model_results.csv` and
`reports/run_summary.json` — re-run the pipeline to refresh them.

---

## Explain this project in 30 seconds

> I built a research pipeline that forecasts short-horizon S&P 500 realized
> volatility. I engineered ~35 features from daily SPY prices, VIX, and
> Treasury yields — trailing realized vol at several horizons, range-based
> estimators like Parkinson, drawdowns, and volume — and compared naive
> persistence benchmarks against linear regression, a HAR-style model,
> GARCH(1,1), random forest, and XGBoost, all evaluated with walk-forward
> expanding-window validation to avoid look-ahead bias. On the held-out
> 2024–2026 test period the feature-based models beat naive persistence —
> the best model cut RMSE by about 25% — but every model's errors grew
> two-to-three-fold in the high-volatility regime.

## Explain this project in 2 minutes

> The research question is whether market, volatility, and macro features can
> improve 5-day-ahead realized volatility forecasts for SPY relative to
> simple statistical benchmarks.
>
> **Data.** Daily SPY OHLCV and VIX from 2010 to the present via Yahoo
> Finance, plus optional Treasury-yield proxies — roughly 3,900 daily
> observations spanning several distinct regimes: the post-GFC recovery, the
> 2015–16 and 2018 corrections, the 2020 COVID crash, and the 2022 rate-hike
> bear market.
>
> **Target.** Annualized forward realized volatility: the square root of
> 252/5 times the sum of squared log returns over the next five trading
> days. Constructing this carefully matters — the target at time *t* must
> use only returns from *t+1* onward, which I verify with unit tests.
>
> **Features.** About 35, all computable at the close of day *t*: trailing
> realized vol at 5/10/21/63 days, Parkinson and Garman–Klass range
> estimators, VIX level and dynamics, the VIX–realized-vol spread, momentum,
> drawdown depth, downside volatility, and relative volume.
>
> **Models.** Two naive persistence baselines, OLS on standardized features,
> a HAR-style regression on multi-horizon realized vol, GARCH(1,1),
> random forest, and XGBoost tuned with a small grid on a chronologically
> separate validation block.
>
> **Validation.** Strictly chronological 70/15/15 split, then expanding-window
> walk-forward evaluation over the test period, refitting roughly quarterly.
> Random cross-validation would let models train on the future of their own
> test points, which inflates scores badly for persistent series.
>
> **Findings.** On the 2024–2026 test window the feature-based models beat
> naive persistence: the best model (Random Forest, RMSE 0.071, R² 0.41)
> reduced RMSE ~25% versus the best naive baseline, which actually posted
> negative out-of-sample R² because trailing vol lags around spikes. Plain
> OLS on the same features was competitive with the tree ensembles — it
> slightly beat tuned XGBoost — VIX was the strongest single linear
> predictor, and every model's errors grew 2–3× in the high-vol regime.

---

## Core concepts

### What is volatility?
The standard deviation of returns — a measure of the magnitude, not
direction, of price moves. Usually annualized. It's the central quantity in
options pricing, risk management, and position sizing.

### Realized vs implied volatility
- **Realized (historical)**: computed from actual returns that occurred,
  e.g. the annualized std-dev of daily returns over a window. Backward-looking
  and measurable, which makes it a clean forecasting target.
- **Implied**: backed out of option prices via a pricing model — the
  market's risk-neutral expectation of future volatility plus a risk premium.
  The VIX is a model-free 30-day implied vol index for the S&P 500.
- Implied usually exceeds subsequent realized (the volatility risk premium),
  because option sellers demand compensation for bearing vol risk.

### Why log returns?
They are time-additive (a 5-day log return is the sum of five daily log
returns), which makes multi-period volatility aggregation clean; they're
approximately equal to simple returns for small moves; and they make
compounded prices a sum of increments, which fits standard time-series
models.

### Volatility clustering
Large moves tend to follow large moves and calm periods follow calm periods —
returns themselves are close to serially uncorrelated, but their *squares*
are strongly autocorrelated. This persistence is why even a naive "recent vol
predicts future vol" forecast works, and why ARCH-family models exist.

### Why does the VIX matter here?
It's a forward-looking, market-priced 30-day volatility expectation
aggregated from S&P 500 option prices. It embeds information about scheduled
events (elections, FOMC meetings) that no backward-looking feature can see.
The spread between VIX and trailing realized vol is a natural feature: a
large positive spread means the market expects vol to rise.

### What is the HAR model?
Corsi's Heterogeneous Autoregressive model: regress future realized vol on
trailing realized vol at daily/weekly/monthly horizons (here 5/21/63 days).
The intuition is that market participants operate at heterogeneous
frequencies — intraday traders, weekly rebalancers, monthly institutions —
so each horizon carries distinct information. Despite being plain OLS on
three regressors, HAR is a famously strong realized-vol benchmark.

### What is GARCH?
GARCH(1,1) models conditional variance recursively:
σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁ — today's variance is a weighted combination of
a long-run level, yesterday's squared shock, and yesterday's variance.
α + β close to 1 means shocks decay slowly (high persistence). In this
project it serves as the classical econometric comparison; multi-day
forecasts come from iterating the recursion forward and aggregating.

### Why does time-series cross-validation matter?
Financial data is ordered and autocorrelated. Random K-fold shuffling puts
observations from the future into the training set of a model tested on the
past — with a persistent target like volatility, the model effectively
memorizes the neighborhood of its test points. Walk-forward validation
(train on everything up to *t*, predict the next block, expand, repeat)
mirrors deployment and gives honest error estimates.

### Look-ahead bias
Using any information at time *t* that would not have been available at
time *t*. Subtle sources this project explicitly guards against: forward
targets misaligned by one day, scalers fit on the full sample, regime
breakpoints computed with test data, and hyperparameters tuned on the test
set. There are unit tests that recompute features on truncated data to prove
feature values don't change when the future is removed.

### Overfitting
A model capturing noise specific to the training sample rather than
generalizable structure. Defenses used here: strong simple baselines, a
small hyperparameter grid, conservative tree depths, chronological
validation, and reporting results honestly even when complex models lose.

### Why does a naive benchmark matter?
Because volatility persistence makes "the next 5 days will look like the
last 21 days" surprisingly accurate. Any claimed ML improvement is
meaningless unless it beats that free benchmark out-of-sample. Skipping this
comparison is one of the most common ways quant research overstates results.

### Why might XGBoost help?
Gradient-boosted trees capture non-linearities and interactions
automatically — e.g. "VIX spread matters more when drawdown is deep" — and
are robust to feature scaling and outliers. The risk is overfitting a small,
highly autocorrelated sample, which is why it's tuned on a separate
chronological validation block and compared against the naive baseline.

### Limitations of ML in financial markets
Low signal-to-noise ratios; non-stationarity (relationships drift across
regimes); one historical path — we can't rerun 2020; effective sample sizes
much smaller than row counts because of overlapping windows and
autocorrelation; and the ease of leaking future information. ML shines with
stable relationships and lots of independent data — financial markets offer
neither.

### How does this relate to equity options trading?
Realized-vol forecasts are directly relevant to volatility trading: an
options desk compares implied vol (what the market charges) against a
forecast of realized vol (what delta-hedging a position is likely to cost).
If forecast realized vol is well below implied, selling options and
delta-hedging captures the spread — this is the volatility risk premium.
Better short-horizon RV forecasts also improve hedge-ratio timing, margin
and risk estimates, and market-making quotes. This project forecasts
volatility only — it does not implement or backtest a trading strategy.

---

## Honest answers to hard questions

**"Did your ML model beat the baseline?"** Yes, on this test window — the
Random Forest cut RMSE ~25% versus the best naive baseline (0.071 vs 0.094),
and the naive forecasts had negative out-of-sample R². But be precise about
*why*: the 2024–2026 window contained sharp vol spikes where persistence
forecasts lag badly (they stay elevated after the spike passes). Also note
that plain OLS on the same features slightly beat tuned XGBoost (RMSE
0.0759 vs 0.0785) — most of the predictable variation is linear persistence
structure, so the honest claim is "features help; model sophistication
helps only modestly."

**"What would you do next?"** Intraday (5-minute) realized volatility for a
cleaner target and less noisy features; option-chain data to model the
volatility risk premium directly; testing whether the VIX-spread feature adds
value after controlling for HAR terms; longer samples including 2008.

**"What's the effective sample size?"** Much less than the row count —
5-day forward windows overlap, so adjacent targets share returns. Standard
errors on performance differences are wide; that's a reason to be humble
about small RMSE gaps.
