"""Performance and evaluation metrics for forecasts and strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 12) -> float:
    """Compute the annualized Sharpe ratio.

    Args:
        series: Periodic return series.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of periods per year (12 for monthly, 252 for daily).

    Returns:
        Annualized Sharpe ratio. Returns ``np.nan`` when volatility is zero or data is empty.
    """

    if series.empty:
        return np.nan

    rf_per_period = risk_free_rate / periods_per_year
    excess = series - rf_per_period
    std = excess.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.nan

    return excess.mean() / std * np.sqrt(periods_per_year)


def sortino_ratio(series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 12) -> float:
    """Compute the annualized Sortino ratio using downside deviation."""

    if series.empty:
        return np.nan

    rf_per_period = risk_free_rate / periods_per_year
    excess = series - rf_per_period
    downside = excess[excess < 0]
    downside_std = downside.pow(2).mean() ** 0.5
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan

    return excess.mean() / downside_std * np.sqrt(periods_per_year)


def _drawdown_series(series: pd.Series) -> pd.Series:
    """Return the drawdown series from periodic returns."""

    equity_curve = (1 + series.fillna(0)).cumprod()
    running_max = equity_curve.cummax()
    return equity_curve.divide(running_max).subtract(1)


def max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown (most negative peak-to-trough return)."""

    if series.empty:
        return np.nan

    drawdowns = _drawdown_series(series)
    return drawdowns.min()


def avg_drawdown(series: pd.Series) -> float:
    """Average drawdown magnitude over drawdown periods."""

    if series.empty:
        return np.nan

    drawdowns = _drawdown_series(series)
    negative = drawdowns[drawdowns < 0]
    if negative.empty:
        return 0.0
    return float((-negative).mean())


def positive_ratio(series: pd.Series) -> float:
    """Fraction of periods with positive returns."""

    if series.empty:
        return np.nan
    return float((series > 0).mean())


def summary_table(results_df: pd.DataFrame, risk_free_rate: float = 0.0, periods_per_year: int = 12) -> pd.DataFrame:
    """Create a metrics summary table for multiple strategies.

    Args:
        results_df: DataFrame of strategy returns with one column per strategy.
        risk_free_rate: Annualized risk-free rate for Sharpe/Sortino calculations.
        periods_per_year: Number of return periods per year.

    Returns:
        DataFrame indexed by metric with a column for each strategy.
    """

    metrics = {
        "Sharpe": sharpe_ratio,
        "Sortino": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Avg Drawdown": avg_drawdown,
        "Positive Ratio": positive_ratio,
    }

    summary = {}
    for name in results_df.columns:
        col = results_df[name].dropna()
        summary[name] = {
            metric_name: func(col, risk_free_rate, periods_per_year)
            if func in (sharpe_ratio, sortino_ratio)
            else func(col)
            for metric_name, func in metrics.items()
        }

    return pd.DataFrame(summary)


# Backwards-compatibility helpers -------------------------------------------------
def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 12) -> float:
    """Alias for :func:`sharpe_ratio` for compatibility."""

    return sharpe_ratio(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)


def compute_drawdowns(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown series from an equity curve."""

    running_max = equity_curve.cummax()
    return equity_curve.divide(running_max).subtract(1)


def regime_confusion_matrix(true_regimes: pd.Series, predicted_regimes: pd.Series) -> pd.DataFrame:
    """Generate confusion matrix for regime classification."""

    return pd.crosstab(true_regimes, predicted_regimes, dropna=False)
