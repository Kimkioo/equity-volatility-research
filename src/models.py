"""Volatility forecasting models.

All models share a minimal fit/predict interface operating on a feature
matrix X (DataFrame) and target vector y (Series) so they can be evaluated
identically under walk-forward validation. The GARCH model is the exception:
it consumes the raw daily return series and is handled by a dedicated
forecasting routine.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import PRIMARY_HORIZON, RANDOM_SEED, TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

HAR_PREDICTORS = ["rv_5d", "rv_21d", "rv_63d"]


class NaiveForecaster:
    """Predicts forward volatility with a single trailing realized-vol column.

    The simplest defensible benchmark: tomorrow's volatility environment
    looks like the recent past (volatility is highly persistent).
    """

    def __init__(self, source_column: str = "rv_21d") -> None:
        self.source_column = source_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveForecaster":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.source_column].to_numpy()


class LinearModel:
    """OLS on standardized features. Scaler is fit on training data only."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("ols", LinearRegression())]
        )

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns] if self.columns is not None else X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
        self.pipeline.fit(self._select(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(self._select(X))

    def coefficients(self, X: pd.DataFrame) -> pd.Series:
        """Standardized coefficients, sorted by absolute magnitude."""
        cols = self._select(X).columns
        coefs = pd.Series(self.pipeline.named_steps["ols"].coef_, index=cols)
        return coefs.reindex(coefs.abs().sort_values(ascending=False).index)


class HARModel(LinearModel):
    """HAR-style model: OLS of forward RV on trailing RV at three horizons.

    Corsi's Heterogeneous Autoregressive model captures the idea that
    market participants operate at different frequencies (day traders,
    weekly rebalancers, monthly institutions), so volatility at short,
    medium, and long horizons all carry independent information.
    """

    def __init__(self) -> None:
        super().__init__(columns=list(HAR_PREDICTORS))


@dataclass
class XGBoostModel:
    """XGBoost regressor with a small, chronological hyperparameter search."""

    params: dict = field(default_factory=dict)
    model: XGBRegressor | None = None

    DEFAULT_PARAMS = {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    SEARCH_GRID = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.1],
        "n_estimators": [200, 400],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        params = {**self.DEFAULT_PARAMS, **self.params}
        self.model = XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_SEED, n_jobs=-1, **params
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "fit() must be called before predict()"
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        assert self.model is not None, "fit() must be called before predict()"
        imp = pd.Series(
            self.model.feature_importances_, index=self.model.feature_names_in_
        )
        return imp.sort_values(ascending=False)

    @classmethod
    def tune(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Grid search over a small parameter grid, scored on the
        chronologically later validation block (never the test set)."""
        keys = list(cls.SEARCH_GRID)
        best_params, best_rmse = {}, np.inf
        for values in itertools.product(*cls.SEARCH_GRID.values()):
            params = dict(zip(keys, values))
            model = cls(params=params).fit(X_train, y_train)
            rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))
            if rmse < best_rmse:
                best_rmse, best_params = rmse, params
        logger.info("XGBoost tuning: best val RMSE %.4f with %s", best_rmse, best_params)
        return best_params


class RandomForestModel:
    """Random forest with conservative depth to limit overfitting."""

    def __init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def garch_forecast(
    returns: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    horizon: int = PRIMARY_HORIZON,
    refit_every: int = 21,
) -> pd.Series:
    """Rolling GARCH(1,1) forecasts of forward `horizon`-day volatility.

    For each forecast date t, the model is estimated on all returns up to
    and including t (refit every `refit_every` days for speed; between
    refits, the last fitted parameters filter the newest returns). The
    h-step-ahead conditional variances are aggregated to an annualized
    volatility comparable with the realized-vol targets.
    """
    from arch import arch_model  # imported lazily; optional dependency

    scaled = returns * 100.0  # arch is numerically happier with % returns
    preds: dict[pd.Timestamp, float] = {}
    fitted = None
    positions = returns.index.get_indexer(forecast_dates)

    for i, (date, pos) in enumerate(zip(forecast_dates, positions)):
        if pos < 0:
            continue
        history = scaled.iloc[: pos + 1]
        if fitted is None or i % refit_every == 0:
            model = arch_model(history, vol="GARCH", p=1, q=1, mean="Constant")
            fitted = model.fit(disp="off", show_warning=False)
            params = fitted.params
        # Re-apply the last estimated parameters to the updated history so
        # the conditional variance reflects the newest observed returns.
        model = arch_model(history, vol="GARCH", p=1, q=1, mean="Constant")
        fixed = model.fix(params)
        fc = fixed.forecast(horizon=horizon, reindex=False)
        # Mean daily variance over the next `horizon` days, in %^2 units.
        var_daily = float(fc.variance.iloc[-1].mean())
        preds[date] = np.sqrt(var_daily * TRADING_DAYS_PER_YEAR) / 100.0

    return pd.Series(preds, name="garch")
