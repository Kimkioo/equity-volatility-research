"""End-to-end research pipeline.

Downloads data, engineers features, tunes and evaluates all models with
walk-forward validation, runs the regime analysis, and writes every
figure and results table used in the README and research summary.

Usage:
    python run_pipeline.py            # full run (uses cached data if present)
    python run_pipeline.py --refresh  # force re-download of market data
    python run_pipeline.py --no-garch # skip the (slower) GARCH comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from src import visualization as viz
from src.config import (
    PRIMARY_HORIZON,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    ensure_directories,
)
from src.data import build_market_dataset, summarize_dataset
from src.evaluation import (
    assign_regimes,
    chronological_split,
    forecast_metrics,
    metrics_by_regime,
    walk_forward_evaluate,
    walk_forward_folds,
)
from src.features import build_features, feature_columns, log_returns, target_column
from src.models import (
    HARModel,
    LinearModel,
    NaiveForecaster,
    RandomForestModel,
    XGBoostModel,
    garch_forecast,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")


def main(refresh: bool = False, include_garch: bool = True) -> None:
    np.random.seed(RANDOM_SEED)
    ensure_directories()
    t0 = time.time()

    # ------------------------------------------------------------------ data
    logger.info("Building market dataset...")
    market = build_market_dataset(force=refresh)
    print("\n=== Dataset summary ===")
    print(summarize_dataset(market))

    df = build_features(market)
    features = feature_columns(df)
    target = target_column(PRIMARY_HORIZON)
    X, y = df[features], df[target]
    df.to_csv(PROCESSED_DATA_DIR / "features.csv")
    print(f"\nFeature matrix: {len(df):,} rows x {len(features)} features")
    print(f"Feature window drops rows before {df.index[0].date()}")

    # ---------------------------------------------------------------- splits
    split = chronological_split(len(df))
    train_df, val_df, test_df = split.slices(df)
    print(
        f"\nChronological split: train {len(train_df):,} "
        f"({train_df.index[0].date()}–{train_df.index[-1].date()}), "
        f"val {len(val_df):,} ({val_df.index[0].date()}–{val_df.index[-1].date()}), "
        f"test {len(test_df):,} ({test_df.index[0].date()}–{test_df.index[-1].date()})"
    )

    # ------------------------------------------------- hyperparameter tuning
    # The last PRIMARY_HORIZON training rows are purged: their forward
    # targets overlap the validation window (overlapping-sample leakage).
    logger.info("Tuning XGBoost on train/validation blocks (test set untouched)...")
    tune_train = train_df.iloc[:-PRIMARY_HORIZON]
    best_xgb_params = XGBoostModel.tune(
        tune_train[features], tune_train[target], val_df[features], val_df[target]
    )

    # -------------------------------------------- walk-forward test evaluation
    folds = walk_forward_folds(len(df), test_start=split.val_end, fold_size=63)
    logger.info("Walk-forward evaluation over %d expanding-window folds...", len(folds))

    model_factories = {
        "Naive (last 5d RV)": lambda: NaiveForecaster("rv_5d"),
        "Naive (last 21d RV)": lambda: NaiveForecaster("rv_21d"),
        "Linear Regression": lambda: LinearModel(),
        "HAR": lambda: HARModel(),
        "Random Forest": lambda: RandomForestModel(),
        "XGBoost": lambda: XGBoostModel(params=best_xgb_params),
    }

    predictions: dict[str, pd.Series] = {}
    for name, factory in model_factories.items():
        logger.info("  evaluating %s", name)
        predictions[name] = walk_forward_evaluate(factory, X, y, folds)

    if include_garch:
        logger.info("  evaluating GARCH(1,1) (rolling refit)...")
        returns = log_returns(market["adj_close"]).dropna()
        predictions["GARCH(1,1)"] = garch_forecast(
            returns, test_df.index, horizon=PRIMARY_HORIZON
        )

    y_test = y.loc[test_df.index]

    # ---------------------------------------------------------------- metrics
    rows = []
    for name, pred in predictions.items():
        common = y_test.index.intersection(pred.index)
        rows.append({"Model": name, **forecast_metrics(y_test.loc[common], pred.loc[common].to_numpy())})
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    results.to_csv(REPORTS_DIR / "model_results.csv", index=False)
    print("\n=== Test-set model comparison (walk-forward) ===")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # -------------------------------------------------------- regime analysis
    regimes = assign_regimes(df["rv_21d"], train_df["rv_21d"])
    regime_results = metrics_by_regime(y_test, predictions, regimes.loc[test_df.index])
    regime_results.to_csv(REPORTS_DIR / "regime_results.csv", index=False)
    print("\n=== RMSE by volatility regime (test set) ===")
    print(
        regime_results.pivot(index="Model", columns="Regime", values="RMSE")[
            ["low", "medium", "high"]
        ].to_string(float_format=lambda v: f"{v:.4f}")
    )

    # ------------------------------------------- coefficients & importances
    har = HARModel().fit(train_df[features], train_df[target])
    har_coefs = har.coefficients(train_df[features])
    xgb_full = XGBoostModel(params=best_xgb_params).fit(
        pd.concat([train_df, val_df])[features], pd.concat([train_df, val_df])[target]
    )
    importance = xgb_full.feature_importance()
    importance.rename("importance").to_csv(REPORTS_DIR / "xgb_feature_importance.csv")
    print("\n=== HAR coefficients (standardized, train set) ===")
    print(har_coefs.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n=== XGBoost top-10 feature importance ===")
    print(importance.head(10).to_string(float_format=lambda v: f"{v:.4f}"))

    # ---------------------------------------------------------------- figures
    logger.info("Generating figures...")
    returns_full = log_returns(market["adj_close"])
    corr_cols = [
        "rv_5d", "rv_21d", "rv_63d", "vix", "log_return_1d", "return_21d",
        "drawdown", "downside_vol_21d", "relative_volume_21d", target,
    ]
    figure_paths = [
        viz.plot_price_history(market),
        viz.plot_realized_volatility(df),
        viz.plot_vix_vs_realized(df),
        viz.plot_return_distribution(returns_full),
        viz.plot_volatility_clustering(returns_full),
        viz.plot_correlation_matrix(df, corr_cols),
        viz.plot_actual_vs_predicted(
            y_test,
            {k: predictions[k] for k in ("Naive (last 21d RV)", "HAR", "XGBoost") if k in predictions},
            "5-Day Forward Realized Volatility",
        ),
        viz.plot_model_comparison(results),
        viz.plot_feature_importance(importance),
        viz.plot_error_by_regime(regime_results),
    ]
    for p in figure_paths:
        logger.info("  saved %s", p.name)

    # ------------------------------------------------------------- run summary
    summary = {
        "observations_raw": len(market),
        "observations_model": len(df),
        "date_start": str(df.index[0].date()),
        "date_end": str(df.index[-1].date()),
        "n_features": len(features),
        "features": features,
        "target": target,
        "split": {
            "train": [str(train_df.index[0].date()), str(train_df.index[-1].date()), len(train_df)],
            "val": [str(val_df.index[0].date()), str(val_df.index[-1].date()), len(val_df)],
            "test": [str(test_df.index[0].date()), str(test_df.index[-1].date()), len(test_df)],
        },
        "xgb_params": best_xgb_params,
        "n_walk_forward_folds": len(folds),
        "results": results.to_dict(orient="records"),
        "har_coefficients": har_coefs.round(6).to_dict(),
        "top_features": importance.head(10).round(6).to_dict(),
    }
    with open(REPORTS_DIR / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nPipeline complete in {time.time() - t0:.1f}s. Outputs in {REPORTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the volatility research pipeline")
    parser.add_argument("--refresh", action="store_true", help="force re-download of market data")
    parser.add_argument("--no-garch", action="store_true", help="skip the GARCH(1,1) comparison")
    args = parser.parse_args()
    main(refresh=args.refresh, include_garch=not args.no_garch)
