"""Regime-conditional ridge regression forecaster (equations 12–14)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import Ridge


class RegimeRidgeForecaster:
    """
    Train ridge regression models per regime and form predictions
    weighted by the next-period regime probabilities.

    Parameters
    ----------
    alpha : float, default 1.0
        Regularization strength passed to ``sklearn.linear_model.Ridge``.
    ridge_kwargs : dict, optional
        Additional keyword arguments forwarded to the Ridge constructor.
    """

    def __init__(self, alpha: float = 1.0, **ridge_kwargs) -> None:
        self.alpha = alpha
        self.ridge_kwargs = ridge_kwargs
        self.regime_order_: np.ndarray | None = None
        self.models_: Dict[int, List[Ridge]] = {}

    def fit(self, X: np.ndarray, Y: np.ndarray, regimes: Sequence[int]) -> None:
        """
        Fit ridge models for each regime/asset pair.

        Parameters
        ----------
        X : array-like of shape (T, m)
            PCA-transformed features.
        Y : array-like of shape (T, d)
            Asset returns aligned to ``X``.
        regimes : sequence of shape (T,)
            Integer regime labels for each observation.
        """

        X_arr = np.asarray(X)
        Y_arr = np.asarray(Y)
        regimes_arr = np.asarray(regimes)

        if X_arr.shape[0] != Y_arr.shape[0] or X_arr.shape[0] != regimes_arr.shape[0]:
            raise ValueError("X, Y, and regimes must share the same number of observations")

        if Y_arr.ndim != 2:
            raise ValueError("Y must be 2D with shape (T, d)")

        unique_regimes = np.unique(regimes_arr)
        if unique_regimes.size == 0:
            raise ValueError("At least one regime label is required to fit models")

        self.regime_order_ = unique_regimes
        self.models_ = {}

        for regime in unique_regimes:
            mask = regimes_arr == regime
            if not np.any(mask):
                raise ValueError(f"No samples found for regime {regime}")

            X_r = X_arr[mask]
            Y_r = Y_arr[mask]

            models_for_regime: List[Ridge] = []
            for asset_idx in range(Y_arr.shape[1]):
                model = Ridge(alpha=self.alpha, **self.ridge_kwargs)
                model.fit(X_r, Y_r[:, asset_idx])
                models_for_regime.append(model)

            self.models_[int(regime)] = models_for_regime

    def predict(self, x_new: np.ndarray, p_next: Sequence[float]) -> np.ndarray:
        """
        Predict asset returns using regime-conditional ridge models.

        Parameters
        ----------
        x_new : array-like of shape (m,) or (1, m)
            Feature vector for the next period.
        p_next : array-like of shape (n_regimes,)
            Probability distribution over regimes for the next period.

        Returns
        -------
        np.ndarray
            Weighted average prediction across regimes (shape: d,).
        """

        if self.regime_order_ is None or not self.models_:
            raise RuntimeError("The forecaster must be fitted before calling predict")

        x_arr = np.asarray(x_new)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.shape[0] != 1:
            raise ValueError("x_new must represent a single observation")

        probs = np.asarray(p_next, dtype=float)
        if probs.shape[0] != self.regime_order_.size:
            raise ValueError("p_next length must match the number of fitted regimes")

        prob_sum = probs.sum()
        if prob_sum <= 0:
            raise ValueError("Regime probabilities must sum to a positive value")
        probs = probs / prob_sum

        n_assets = len(next(iter(self.models_.values())))
        regime_predictions = np.zeros((self.regime_order_.size, n_assets))

        for idx, regime in enumerate(self.regime_order_):
            models = self.models_[int(regime)]
            for asset_idx, model in enumerate(models):
                regime_predictions[idx, asset_idx] = model.predict(x_arr)[0]

        weighted_prediction = probs @ regime_predictions
        return weighted_prediction


__all__ = ["RegimeRidgeForecaster"]
