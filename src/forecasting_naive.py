"""Naive forecasting baselines."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def forecast_naive(series: pd.Series, horizon: int = 1) -> pd.Series:
    """Forecast future values by repeating the last observed value."""

    last_value = series.iloc[-1]
    index = pd.RangeIndex(start=1, stop=horizon + 1, step=1)
    return pd.Series(np.repeat(last_value, horizon), index=index)


class NaiveRegimeForecaster:
    """Regime-conditional Sharpe ratio forecaster from 2503.11499v2 equations (8)-(10)."""

    def __init__(self) -> None:
        self.regime_sharpes: Optional[pd.DataFrame] = None
        self.regimes_: Optional[np.ndarray] = None

    def fit(self, regimes: Iterable[int], returns: pd.DataFrame) -> "NaiveRegimeForecaster":
        """Fit regime-conditional Sharpe ratios.

        Parameters
        ----------
        regimes : Iterable[int]
            Sequence of integer regime labels aligned with ``returns`` rows.
        returns : pd.DataFrame
            Asset returns with shape (T, n_assets).

        Returns
        -------
        NaiveRegimeForecaster
            Fitted forecaster.
        """

        regimes_arr = np.asarray(list(regimes))
        if len(regimes_arr) != len(returns):
            raise ValueError("Length of regimes must match number of return observations")

        sharpes = {}
        for regime in np.unique(regimes_arr):
            regime_mask = regimes_arr == regime
            regime_returns = returns.loc[regime_mask]
            mean = regime_returns.mean()
            std = regime_returns.std(ddof=1)
            sharpe = mean.divide(std.replace(0, np.nan)).fillna(0.0)
            sharpes[regime] = sharpe

        self.regime_sharpes = pd.DataFrame.from_dict(sharpes, orient="index")
        self.regimes_ = np.array(sorted(sharpes.keys()))
        return self

    def predict(self, X) -> pd.Series:
        """Predict using most likely regime probabilities.

        Parameters
        ----------
        X : array-like
            Regime probabilities for the next period with shape (n_regimes,).
        """

        return self.predict_proba(X)

    def predict_proba(self, p_next: Iterable[float]) -> pd.Series:
        """Return Sharpe ratios of the most likely next regime.

        Parameters
        ----------
        p_next : Iterable[float]
            Probabilities over regimes ordered to match the fitted regime set.
        """

        if self.regime_sharpes is None or self.regimes_ is None:
            raise RuntimeError("Forecaster has not been fitted yet")

        p_arr = np.asarray(list(p_next), dtype=float)
        if p_arr.ndim != 1 or p_arr.shape[0] != len(self.regimes_):
            raise ValueError(
                "p_next must be a 1D array with length equal to the number of fitted regimes"
            )

        best_idx = int(np.argmax(p_arr))
        regime_label = self.regimes_[best_idx]
        return self.regime_sharpes.loc[regime_label]
