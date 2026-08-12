"""Feature engineering and realized-volatility target construction.

Every feature at date t uses information available at the close of date t
only. Forward realized volatility targets are the only columns that look
into the future, and they are used exclusively as prediction targets.

Realized volatility convention (annualized):

    RV(t, n) = sqrt( (252 / n) * sum_{i=t-n+1..t} r_i^2 )        (trailing)
    RV_fwd(t, h) = sqrt( (252 / h) * sum_{i=t+1..t+h} r_i^2 )    (forward)

where r_i = ln(P_i / P_{i-1}) are daily log returns on adjusted close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_HORIZONS, TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Core quantities
# ---------------------------------------------------------------------------

def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Trailing annualized realized volatility over `window` days ending at t."""
    return np.sqrt(
        (TRADING_DAYS_PER_YEAR / window) * returns.pow(2).rolling(window).sum()
    )


def forward_realized_volatility(returns: pd.Series, horizon: int) -> pd.Series:
    """Forward annualized realized volatility over days t+1 .. t+horizon.

    The rolling sum of squared returns ending at t+horizon covers exactly
    r_{t+1}..r_{t+horizon}; shifting it back by `horizon` aligns that value
    with date t without using any information available before t+horizon.
    """
    fwd_sum_sq = returns.pow(2).rolling(horizon).sum().shift(-horizon)
    return np.sqrt((TRADING_DAYS_PER_YEAR / horizon) * fwd_sum_sq)


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Parkinson range-based volatility estimator (annualized).

    sigma_P = sqrt( 252 / (4 ln 2 * n) * sum ln(H_i / L_i)^2 )
    """
    log_hl_sq = np.log(high / low).pow(2)
    return np.sqrt(
        (TRADING_DAYS_PER_YEAR / (4.0 * np.log(2.0) * window))
        * log_hl_sq.rolling(window).sum()
    )


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Garman-Klass OHLC volatility estimator (annualized)."""
    term = 0.5 * np.log(high / low).pow(2) - (2.0 * np.log(2.0) - 1.0) * np.log(
        close / open_
    ).pow(2)
    return np.sqrt((TRADING_DAYS_PER_YEAR / window) * term.rolling(window).sum())


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix + targets from the merged market frame.

    Expects columns: open, high, low, close, adj_close, volume, vix and
    optionally yield_10y, yield_3m. Returns a frame with feature columns,
    target columns (rv_{h}d_forward), and drops rows with missing values
    created by rolling windows.
    """
    px = data["adj_close"]
    r = log_returns(px)

    f = pd.DataFrame(index=data.index)
    f["log_return_1d"] = r

    # --- Return features -------------------------------------------------
    for n in (5, 10, 21, 63):
        f[f"return_{n}d"] = px.pct_change(n)

    # --- Trailing realized volatility ------------------------------------
    for n in (5, 10, 21, 63):
        f[f"rv_{n}d"] = realized_volatility(r, n)

    # --- Rolling return statistics ----------------------------------------
    f["ret_mean_21d"] = r.rolling(21).mean()
    f["ret_std_21d"] = r.rolling(21).std()
    f["ret_skew_63d"] = r.rolling(63).skew()

    # --- Distance from moving averages ------------------------------------
    for n in (21, 63, 200):
        f[f"dist_ma_{n}d"] = px / px.rolling(n).mean() - 1.0

    # --- Range-based volatility -------------------------------------------
    f["parkinson_10d"] = parkinson_volatility(data["high"], data["low"], 10)
    f["parkinson_21d"] = parkinson_volatility(data["high"], data["low"], 21)
    f["garman_klass_21d"] = garman_klass_volatility(
        data["open"], data["high"], data["low"], data["close"], 21
    )

    # --- VIX features -------------------------------------------------------
    vix = data["vix"]
    f["vix"] = vix
    f["vix_change_1d"] = vix.diff(1)
    f["vix_change_5d"] = vix.diff(5)
    f["vix_pct_change_1d"] = vix.pct_change(1)
    f["vix_mean_21d"] = vix.rolling(21).mean()
    f["vix_std_21d"] = vix.rolling(21).std()
    # VIX is quoted in annualized % points; rv_21d is annualized decimal.
    f["vix_rv_spread"] = vix / 100.0 - f["rv_21d"]

    # --- Drawdown / market-stress features ----------------------------------
    rolling_max = px.rolling(252, min_periods=63).max()
    f["drawdown"] = px / rolling_max - 1.0
    f["max_drawdown_63d"] = f["drawdown"].rolling(63).min()
    f["neg_return_frac_21d"] = (r < 0).rolling(21).mean()
    downside = r.where(r < 0, 0.0)
    f["downside_vol_21d"] = np.sqrt(
        (TRADING_DAYS_PER_YEAR / 21) * downside.pow(2).rolling(21).sum()
    )

    # --- Volume features -----------------------------------------------------
    vol = data["volume"].astype(float)
    f["volume_change_1d"] = vol.pct_change(1)
    vol_mean_21 = vol.rolling(21).mean()
    f["relative_volume_21d"] = vol / vol_mean_21
    f["volume_trend_5_21"] = vol.rolling(5).mean() / vol_mean_21

    # --- Optional macro features ---------------------------------------------
    if "yield_10y" in data.columns:
        f["yield_10y"] = data["yield_10y"]
        f["yield_10y_change_21d"] = data["yield_10y"].diff(21)
    if "yield_3m" in data.columns and "yield_10y" in data.columns:
        f["term_spread"] = data["yield_10y"] - data["yield_3m"]

    # --- Targets (forward-looking, prediction targets only) -------------------
    for h in TARGET_HORIZONS:
        f[f"rv_{h}d_forward"] = forward_realized_volatility(r, h)

    return f.dropna()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model input columns (everything that is not a forward target)."""
    return [c for c in df.columns if not c.endswith("_forward")]


def target_column(horizon: int) -> str:
    """Name of the forward realized-volatility target for a horizon."""
    return f"rv_{horizon}d_forward"
