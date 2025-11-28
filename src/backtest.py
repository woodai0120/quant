"""Walk-forward backtesting engine following Sections 5.3 and 5.4."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .regime_transition import compute_transition_matrix


def _fit_forecaster(
    forecasting_model, X_train: np.ndarray, returns_train: pd.DataFrame, regimes_train: np.ndarray
) -> None:
    """Fit forecasting model with flexible signatures."""

    try:
        forecasting_model.fit(X_train, returns_train.values, regimes_train)
    except TypeError:
        forecasting_model.fit(regimes_train, returns_train)


def _predict_forecast(forecasting_model, x_next: np.ndarray, p_next: np.ndarray) -> np.ndarray:
    """Call predict with either (x, p) or (p) depending on model signature."""

    try:
        return forecasting_model.predict(x_next.reshape(1, -1), p_next)
    except TypeError:
        return forecasting_model.predict(p_next)


def run_walk_forward_backtest(
    X: np.ndarray,
    returns: pd.DataFrame,
    regimes: Iterable[int],
    regime_model,
    forecasting_model,
    position_sizer: Callable[[np.ndarray], np.ndarray],
    window: int = 48,
) -> pd.DataFrame:
    """Run walk-forward evaluation with rolling re-estimation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (e.g., PCA scores) with shape ``(T, m)``.
    returns : pd.DataFrame
        Asset returns aligned with ``X`` (shape: ``(T, d)``).
    regimes : Iterable[int]
        Regime labels for each observation (length ``T``).
    regime_model : object
        Regime detector providing ``predict_proba``; assumed pre-fitted or
        fit externally.
    forecasting_model : object
        Model supporting ``fit`` and ``predict``. The ``fit`` method should
        accept either ``(X_train, returns_train, regimes_train)`` or
        ``(regimes_train, returns_train)``. The ``predict`` method should
        accept ``(x_next, p_next)`` or ``(p_next)``.
    position_sizer : Callable[[np.ndarray], np.ndarray]
        Function that converts forecasts into portfolio weights. The callable
        should be pre-configured (e.g., via ``functools.partial``) with any
        required hyperparameters.
    window : int, default 48
        Rolling training window size in months.

    Returns
    -------
    pd.DataFrame
        Time series of strategy returns indexed by evaluation dates.
    """

    X_arr = np.asarray(X)
    regimes_arr = np.asarray(list(regimes), dtype=int)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X_arr.shape[0] != len(returns) or X_arr.shape[0] != regimes_arr.size:
        raise ValueError("X, returns, and regimes must have matching lengths")
    if window <= 1 or window >= X_arr.shape[0]:
        raise ValueError("window must be between 2 and the number of observations - 1")

    n_periods = X_arr.shape[0]
    unique_regimes = np.unique(regimes_arr)
    regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
    n_regimes = unique_regimes.size
    strategy_returns = []
    strategy_index = []

    for test_idx in range(window, n_periods):
        train_slice = slice(test_idx - window, test_idx)
        X_train = X_arr[train_slice]
        returns_train = returns.iloc[train_slice]
        regimes_train = regimes_arr[train_slice]

        _fit_forecaster(forecasting_model, X_train, returns_train, regimes_train)

        transition = compute_transition_matrix(regimes_train)
        last_regime = regimes_arr[test_idx - 1]
        x_next = X_arr[test_idx - 1]

        if hasattr(regime_model, "predict_proba"):
            p_curr = np.asarray(regime_model.predict_proba(x_next.reshape(1, -1))).reshape(-1)
        else:
            p_curr = np.zeros(n_regimes)
            p_curr[regime_to_idx[last_regime]] = 1.0

        if p_curr.shape[0] != n_regimes:
            p_curr = np.zeros(n_regimes)
            p_curr[regime_to_idx[last_regime]] = 1.0

        p_curr = p_curr / max(p_curr.sum(), 1e-12)
        p_next = p_curr @ transition

        forecast = _predict_forecast(forecasting_model, x_next, p_next)

        weights = position_sizer(np.asarray(forecast, dtype=float))
        period_return = float(np.dot(weights, returns.iloc[test_idx].values))

        strategy_returns.append(period_return)
        strategy_index.append(returns.index[test_idx])

    return pd.DataFrame({"strategy": pd.Series(strategy_returns, index=strategy_index)})


__all__ = ["run_walk_forward_backtest"]
