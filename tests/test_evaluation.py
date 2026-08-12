"""Tests for chronological validation, metrics, and regime assignment."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    assign_regimes,
    chronological_split,
    forecast_metrics,
    qlike,
    walk_forward_folds,
)


def test_chronological_split_preserves_order():
    n = 1000
    split = chronological_split(n)
    df = pd.DataFrame({"x": range(n)})
    train, val, test = split.slices(df)
    assert len(train) == 700 and len(val) == 150 and len(test) == 150
    assert train.index.max() < val.index.min() < test.index.min()


def test_walk_forward_folds_never_train_on_future():
    folds = walk_forward_folds(n=500, test_start=350, fold_size=63, purge=5)
    covered = []
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()  # strictly chronological
        covered.extend(test_idx.tolist())
    assert covered == list(range(350, 500))  # full test region, no overlap


def test_walk_forward_folds_purge_overlapping_targets():
    """Training samples whose forward targets overlap the test window must
    be removed: a gap of `purge` observations before each test block."""
    folds = walk_forward_folds(n=500, test_start=350, fold_size=63, purge=5)
    for train_idx, test_idx in folds:
        assert test_idx.min() - train_idx.max() > 5


def test_forecast_metrics_perfect_prediction():
    y = pd.Series([0.1, 0.2, 0.3, 0.25])
    m = forecast_metrics(y, y.to_numpy())
    assert m["MAE"] == pytest.approx(0.0)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["R2"] == pytest.approx(1.0)
    assert m["QLIKE"] == pytest.approx(0.0)


def test_qlike_penalizes_underprediction_more():
    y = np.array([0.20])
    under = qlike(y, np.array([0.10]))  # forecast half the true vol
    over = qlike(y, np.array([0.40]))  # forecast double the true vol
    assert under > over > 0


def test_regime_breakpoints_use_training_data_only():
    idx = pd.date_range("2020-01-01", periods=300)
    indicator = pd.Series(np.linspace(0.1, 0.5, 300), index=idx)
    train = indicator.iloc[:200]
    regimes = assign_regimes(indicator, train)
    # Breakpoints come from the train distribution, so the trending test
    # portion should be classified almost entirely as "high".
    assert (regimes.iloc[220:] == "high").all()
    assert set(regimes.unique()) <= {"low", "medium", "high"}
