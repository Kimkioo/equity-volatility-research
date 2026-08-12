"""Chronological validation, forecast metrics, and regime analysis.

Random K-fold cross-validation is deliberately avoided: shuffling breaks
the temporal ordering of the data, so a model would be trained on
observations from the future of its own test points. With persistent,
autocorrelated series like volatility this leaks information and inflates
scores. All evaluation here is strictly chronological.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PRIMARY_HORIZON, TRAIN_FRAC, VAL_FRAC


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChronologicalSplit:
    """Index boundaries of a train / validation / test split."""

    train_end: int
    val_end: int

    def slices(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return df.iloc[: self.train_end], df.iloc[self.train_end : self.val_end], df.iloc[self.val_end :]


def chronological_split(
    n: int, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC
) -> ChronologicalSplit:
    """Earliest `train_frac` for training, next `val_frac` for validation
    (hyperparameter selection), final block for testing."""
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return ChronologicalSplit(train_end=train_end, val_end=val_end)


def walk_forward_folds(
    n: int, test_start: int, fold_size: int = 63, purge: int = PRIMARY_HORIZON
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over the test region, with purging.

    Each fold trains on all observations before the fold and predicts the
    next `fold_size` observations, mimicking how a model would be refit
    and deployed in production (roughly quarterly refits).

    The last `purge` training observations are dropped from every fold:
    their forward targets are computed from returns that fall *inside* the
    fold's test window, so training on them would leak test-period
    information (overlapping-sample leakage).
    """
    folds = []
    start = test_start
    while start < n:
        stop = min(start + fold_size, n)
        folds.append((np.arange(0, max(start - purge, 0)), np.arange(start, stop)))
        start = stop
    return folds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss on variances: mean( s2/h2 - ln(s2/h2) - 1 ).

    s2 = realized variance, h2 = forecast variance. QLIKE is a robust loss
    for volatility forecasts that penalizes under-prediction of variance
    more heavily than over-prediction; 0 is a perfect forecast.
    """
    s2 = np.asarray(y_true, dtype=float) ** 2
    h2 = np.asarray(y_pred, dtype=float) ** 2
    if np.any(h2 <= 0) or np.any(s2 <= 0):
        return float("nan")
    ratio = s2 / h2
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def forecast_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, out-of-sample R^2, and QLIKE for a volatility forecast."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    err = y - p
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": 1.0 - ss_res / ss_tot,
        "QLIKE": qlike(y, p),
    }


def walk_forward_evaluate(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> pd.Series:
    """Refit a model on each expanding-window fold and collect out-of-sample
    predictions over the whole test region."""
    preds = []
    for train_idx, test_idx in folds:
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = model.predict(X.iloc[test_idx])
        preds.append(pd.Series(np.asarray(fold_pred), index=X.index[test_idx]))
    return pd.concat(preds)


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def assign_regimes(
    indicator: pd.Series, train_indicator: pd.Series, labels: tuple[str, str, str] = ("low", "medium", "high")
) -> pd.Series:
    """Classify each observation into volatility terciles.

    Tercile breakpoints are estimated on the *training* period only, then
    applied out-of-sample, so regime labels never use future information.
    """
    lo, hi = train_indicator.quantile([1 / 3, 2 / 3])
    return pd.Series(
        np.select(
            [indicator <= lo, indicator <= hi],
            [labels[0], labels[1]],
            default=labels[2],
        ),
        index=indicator.index,
        name="regime",
    )


def metrics_by_regime(
    y_true: pd.Series, predictions: dict[str, pd.Series], regimes: pd.Series
) -> pd.DataFrame:
    """Forecast metrics per model per volatility regime."""
    rows = []
    for name, pred in predictions.items():
        common = y_true.index.intersection(pred.index)
        for regime in ("low", "medium", "high"):
            mask = regimes.loc[common] == regime
            if mask.sum() == 0:
                continue
            idx = common[mask]
            m = forecast_metrics(y_true.loc[idx], pred.loc[idx].to_numpy())
            rows.append({"Model": name, "Regime": regime, "N": int(mask.sum()), **m})
    return pd.DataFrame(rows)
