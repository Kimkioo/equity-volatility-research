"""Tests for return, volatility, and feature calculations — including
explicit checks that no feature uses future information."""

import numpy as np
import pandas as pd
import pytest

from src.config import TRADING_DAYS_PER_YEAR
from src.features import (
    build_features,
    feature_columns,
    forward_realized_volatility,
    log_returns,
    parkinson_volatility,
    realized_volatility,
)


@pytest.fixture
def synthetic_market() -> pd.DataFrame:
    """Random-walk OHLCV + VIX frame long enough for all rolling windows."""
    rng = np.random.default_rng(7)
    n = 700
    dates = pd.bdate_range("2015-01-01", periods=n)
    ret = rng.normal(0.0003, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            "vix": 15 + 5 * np.abs(rng.normal(0, 1, n)),
        },
        index=dates,
    )


def test_log_returns_match_manual_calculation():
    prices = pd.Series([100.0, 105.0, 102.0])
    r = log_returns(prices)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(105 / 100))
    assert r.iloc[2] == pytest.approx(np.log(102 / 105))


def test_realized_volatility_formula():
    r = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    rv = realized_volatility(r, window=5)
    expected = np.sqrt((TRADING_DAYS_PER_YEAR / 5) * np.sum(np.square(r.values)))
    assert rv.iloc[-1] == pytest.approx(expected)
    assert rv.iloc[:-1].isna().all()  # window not yet full


def test_forward_target_alignment_uses_only_future_returns():
    """The forward target at t must equal RV computed from r_{t+1}..r_{t+h}."""
    r = pd.Series(np.arange(1, 11) / 1000.0)  # r_0 .. r_9, all distinct
    h = 3
    fwd = forward_realized_volatility(r, horizon=h)
    t = 2
    expected = np.sqrt(
        (TRADING_DAYS_PER_YEAR / h) * np.sum(np.square(r.iloc[t + 1 : t + 1 + h].values))
    )
    assert fwd.iloc[t] == pytest.approx(expected)
    # Last h observations have no complete future window.
    assert fwd.iloc[-h:].isna().all()


def test_forward_target_excludes_current_return():
    """Changing r_t must not change the forward target at t."""
    r = pd.Series(np.full(20, 0.01))
    base = forward_realized_volatility(r, horizon=5)
    bumped = r.copy()
    bumped.iloc[10] = 0.5  # huge shock at t=10
    fwd = forward_realized_volatility(bumped, horizon=5)
    assert fwd.iloc[10] == pytest.approx(base.iloc[10])  # target at t unaffected
    assert fwd.iloc[9] > base.iloc[9]  # but it enters the window of t=9


def test_parkinson_positive_and_scales_with_range():
    high = pd.Series(np.full(30, 102.0))
    low = pd.Series(np.full(30, 100.0))
    wide = parkinson_volatility(high * 1.05, low, 10)
    narrow = parkinson_volatility(high, low, 10)
    assert (narrow.dropna() > 0).all()
    assert (wide.dropna() > narrow.dropna()).all()


def test_features_have_no_lookahead(synthetic_market):
    """Feature values at time t must be identical whether or not the future
    exists: truncate the data after t and recompute."""
    full = build_features(synthetic_market)
    cutoff = full.index[len(full) // 2]
    truncated_market = synthetic_market.loc[:cutoff]
    truncated = build_features(truncated_market)

    common_dates = truncated.index.intersection(full.index)
    assert len(common_dates) > 100
    for col in feature_columns(full):
        pd.testing.assert_series_equal(
            full.loc[common_dates, col],
            truncated.loc[common_dates, col],
            check_exact=False,
            rtol=1e-10,
            obj=f"feature {col}",
        )


def test_rolling_features_match_manual_windows(synthetic_market):
    df = build_features(synthetic_market)
    t = df.index[400]
    pos = synthetic_market.index.get_loc(t)
    window = synthetic_market["adj_close"].iloc[pos - 20 : pos + 1]
    r = log_returns(synthetic_market["adj_close"]).iloc[pos - 20 : pos + 1]
    assert df.loc[t, "ret_mean_21d"] == pytest.approx(r.mean())
    assert df.loc[t, "dist_ma_21d"] == pytest.approx(window.iloc[-1] / window.mean() - 1)


def test_targets_dropped_from_feature_list(synthetic_market):
    df = build_features(synthetic_market)
    cols = feature_columns(df)
    assert not any(c.endswith("_forward") for c in cols)
    assert "rv_5d_forward" in df.columns
