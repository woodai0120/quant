"""Black-Litterman posterior computation for regime-based modeling."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class BlackLittermanRegimeModel:
    """Compute Black-Litterman posterior returns and weights.

    Implements the posterior mean and covariance from equations (11) and (19)
    of the referenced paper, supporting regime-conditioned priors and user
    specified views.
    """

    def __init__(self, tau: float = 0.05) -> None:
        self.tau = tau
        self.prior_mean: Optional[np.ndarray] = None
        self.prior_cov: Optional[np.ndarray] = None
        self.asset_names: Optional[pd.Index] = None
        self.view_vector: Optional[np.ndarray] = None
        self.pick_matrix: Optional[np.ndarray] = None
        self.view_uncertainty: Optional[np.ndarray] = None
        self.posterior_mean: Optional[np.ndarray] = None
        self.posterior_cov: Optional[np.ndarray] = None

    def fit(self, prior_mean: pd.Series | np.ndarray, prior_cov: pd.DataFrame | np.ndarray) -> None:
        """Store the prior mean (mu) and covariance (Sigma)."""

        if isinstance(prior_mean, pd.Series):
            self.asset_names = prior_mean.index
            self.prior_mean = prior_mean.to_numpy(dtype=float)
        else:
            self.prior_mean = np.asarray(prior_mean, dtype=float)

        if isinstance(prior_cov, pd.DataFrame):
            self.prior_cov = prior_cov.to_numpy(dtype=float)
            if self.asset_names is None:
                self.asset_names = prior_cov.index
        else:
            self.prior_cov = np.asarray(prior_cov, dtype=float)

        if self.prior_mean is None or self.prior_cov is None:
            raise ValueError("Prior mean and covariance must be provided")
        if self.prior_cov.shape[0] != self.prior_cov.shape[1]:
            raise ValueError("Prior covariance must be square")
        if self.prior_mean.shape[0] != self.prior_cov.shape[0]:
            raise ValueError("Prior mean dimension must match covariance")

    def set_view(self, view_vector: pd.Series | np.ndarray, pick_matrix: np.ndarray, omega: Optional[np.ndarray] = None) -> None:
        """Set the view (q), pick matrix (P), and optional uncertainty (Omega)."""

        if self.prior_cov is None:
            raise RuntimeError("Fit must be called before setting views")

        if isinstance(view_vector, pd.Series):
            self.view_vector = view_vector.to_numpy(dtype=float)
        else:
            self.view_vector = np.asarray(view_vector, dtype=float)

        self.pick_matrix = np.asarray(pick_matrix, dtype=float)
        if self.pick_matrix.ndim != 2:
            raise ValueError("Pick matrix must be two-dimensional")

        if omega is None:
            # Default view uncertainty proportional to variance of picked assets.
            projected_cov = self.pick_matrix @ (self.tau * self.prior_cov) @ self.pick_matrix.T
            self.view_uncertainty = np.diag(np.diag(projected_cov))
        else:
            self.view_uncertainty = np.asarray(omega, dtype=float)

        if self.view_vector.shape[0] != self.pick_matrix.shape[0]:
            raise ValueError("View vector dimension must match number of views in pick matrix")
        if self.view_uncertainty.shape[0] != self.view_uncertainty.shape[1]:
            raise ValueError("View uncertainty must be square")
        if self.view_uncertainty.shape[0] != self.view_vector.shape[0]:
            raise ValueError("View uncertainty dimension must align with views")

    def compute_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute posterior mean and covariance following equations (11) and (19)."""

        if any(v is None for v in (self.prior_mean, self.prior_cov, self.view_vector, self.pick_matrix, self.view_uncertainty)):
            raise RuntimeError("Prior and views must be set before computing posterior")

        tau_sigma = self.tau * self.prior_cov
        p_mat = self.pick_matrix
        omega = self.view_uncertainty

        middle = np.linalg.inv(p_mat @ tau_sigma @ p_mat.T + omega)
        adjustment = tau_sigma @ p_mat.T @ middle @ (self.view_vector - p_mat @ self.prior_mean)

        self.posterior_mean = self.prior_mean + adjustment
        self.posterior_cov = self.prior_cov + tau_sigma @ p_mat.T @ middle @ p_mat @ tau_sigma

        return self.posterior_mean, self.posterior_cov

    def get_weights(self) -> pd.Series:
        """Compute portfolio weights proportional to posterior mean-variance trade-off."""

        if self.posterior_mean is None or self.posterior_cov is None:
            raise RuntimeError("Posterior must be computed before deriving weights")

        weights = np.linalg.solve(self.posterior_cov, self.posterior_mean)
        if self.asset_names is not None:
            return pd.Series(weights, index=self.asset_names)
        return pd.Series(weights)
