"""End-to-end pipeline for FRED-MD preprocessing, regime detection, and backtesting.

This script stitches together the individual components implemented in ``src`` to
run the full workflow described in the reference paper:

1. FRED-MD preprocessing with t-code transformations and PCA.
2. Regime detection via layered k-means and transition matrix estimation.
3. Forecasting with both a naive Sharpe-based model and ridge regression.
4. Position sizing strategies (long-only and long-short examples).
5. Walk-forward backtesting and metric summarization.

The script expects CSV inputs for FRED-MD data, t-code metadata, and ETF daily
prices. Index columns must be parseable as dates. All outputs are printed to the
console, and the function ``run_full_pipeline`` returns intermediate results for
programmatic use.
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.data_loader import build_monthly_returns
from src.forecasting_naive import NaiveRegimeForecaster
from src.forecasting_ridge import RegimeRidgeForecaster
from src.fred_preprocess import preprocess_fred_data, transform_series
from src.metrics import summary_table
from src.position_sizing import long_only, long_short
from src.regime_detection import LayeredKMeansRegimeDetector
from src.regime_transition import compute_transition_matrix


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _load_fred_dataset(fred_path: Path, tcode_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load FRED-MD raw data and t-code metadata from CSV files."""

    fred_df = pd.read_csv(fred_path, index_col=0)
    fred_df.index = pd.to_datetime(fred_df.index)
    fred_df.index = fred_df.index.to_period("M").to_timestamp("MS")

    tcode_info = pd.read_csv(tcode_path)
    return fred_df, tcode_info


def _load_etf_prices(etf_path: Path) -> pd.DataFrame:
    """Load daily ETF prices from a CSV file."""

    prices = pd.read_csv(etf_path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


# ---------------------------------------------------------------------------
# Transformation and alignment
# ---------------------------------------------------------------------------
def _apply_tcodes(raw_data: pd.DataFrame, tcode_info: pd.DataFrame) -> pd.DataFrame:
    """Apply t-code transformations to FRED-MD series and drop missing rows."""

    if not {"variable", "tcode"}.issubset(tcode_info.columns):
        raise ValueError("tcode_info must contain 'variable' and 'tcode' columns")

    tcode_map = tcode_info.set_index("variable")["tcode"].to_dict()
    missing = set(raw_data.columns) - set(tcode_map.keys())
    if missing:
        raise KeyError(f"Missing tcode entries for variables: {sorted(missing)}")

    transformed = {
        column: transform_series(raw_data[column], int(tcode_map[column]))
        for column in raw_data.columns
    }
    return pd.DataFrame(transformed, index=raw_data.index).dropna()


def _align_features_and_returns(
    features: np.ndarray, feature_index: pd.DatetimeIndex, returns: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame, pd.DatetimeIndex]:
    """Align feature matrix and returns on common monthly dates."""

    feature_df = pd.DataFrame(features, index=feature_index)
    common_months = feature_df.index.intersection(returns.index).sort_values()
    if common_months.empty:
        raise ValueError("No overlapping months between FRED features and ETF returns")

    aligned_features = feature_df.loc[common_months].values
    aligned_returns = returns.loc[common_months]
    return aligned_features, aligned_returns, common_months


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_full_pipeline(
    fred_path: Path,
    tcode_path: Path,
    etf_price_path: Path,
    ridge_alpha: float = 1.0,
    top_k: int = 3,
    window: int = 48,
) -> Dict[str, object]:
    """Execute the full regime-based forecasting and backtesting pipeline."""

    fred_df, tcode_info = _load_fred_dataset(fred_path, tcode_path)
    transformed_fred = _apply_tcodes(fred_df, tcode_info)
    X_pca, pca, scaler = preprocess_fred_data(fred_df, tcode_info)

    if transformed_fred.shape[0] != X_pca.shape[0]:
        raise RuntimeError("Mismatch between transformed data and PCA output lengths")

    etf_prices = _load_etf_prices(etf_price_path)
    monthly_returns = build_monthly_returns(etf_prices, transformed_fred.index)

    X_aligned, returns_aligned, common_index = _align_features_and_returns(
        X_pca, transformed_fred.index, monthly_returns
    )

    if X_aligned.shape[0] <= window:
        raise ValueError("Not enough observations to run walk-forward backtest with the given window")

    regime_detector = LayeredKMeansRegimeDetector()
    regime_detector.fit(X_aligned)
    regimes = regime_detector.predict(X_aligned)
    transition_matrix = compute_transition_matrix(regimes)

    n_assets = returns_aligned.shape[1]
    asset_top_k = min(top_k, n_assets)

    naive_forecaster = NaiveRegimeForecaster()
    ridge_forecaster = RegimeRidgeForecaster(alpha=ridge_alpha)

    strategies = [
        ("naive_long_only", naive_forecaster, partial(long_only, l=asset_top_k)),
        ("ridge_long_short", ridge_forecaster, partial(long_short, l=asset_top_k)),
    ]

    strategy_frames = []
    for name, model, sizer in strategies:
        results = run_walk_forward_backtest(
            X_aligned, returns_aligned, regimes, regime_detector, model, sizer, window=window
        )
        strategy_frames.append(results.rename(columns={"strategy": name}))

    results_df = pd.concat(strategy_frames, axis=1)
    metrics_df = summary_table(results_df)

    return {
        "features": X_aligned,
        "feature_index": common_index,
        "pca": pca,
        "scaler": scaler,
        "regimes": regimes,
        "transition": transition_matrix,
        "strategy_returns": results_df,
        "metrics": metrics_df,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full FRED-MD regime pipeline")
    parser.add_argument("fred_path", type=Path, help="Path to FRED-MD CSV with monthly data")
    parser.add_argument("tcode_path", type=Path, help="Path to t-code CSV with 'variable' and 'tcode' columns")
    parser.add_argument("etf_price_path", type=Path, help="Path to ETF daily price CSV")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--top-k", type=int, default=3, help="Number of assets to long/short in sizing")
    parser.add_argument("--window", type=int, default=48, help="Rolling window length for walk-forward backtest")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_full_pipeline(
        fred_path=args.fred_path,
        tcode_path=args.tcode_path,
        etf_price_path=args.etf_price_path,
        ridge_alpha=args.ridge_alpha,
        top_k=args.top_k,
        window=args.window,
    )

    print("Transition matrix (full sample):")
    print(pd.DataFrame(outputs["transition"]))
    print("\nStrategy returns (tail):")
    print(outputs["strategy_returns"].tail())
    print("\nPerformance summary:")
    print(outputs["metrics"])


if __name__ == "__main__":
    main()
