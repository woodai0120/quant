"""Regime detection models for macroeconomic time series."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


class LayeredKMeansRegimeDetector:
    """Two-step regime detector following sections 3.2 and 3.3 of the paper.

    Stage 1 separates outlier (rare) regimes from the bulk using Euclidean k-means
    with ``k=2``. The smaller cluster is labeled Regime 0 (outlier) and the larger
    cluster is retained for a second-layer clustering on cosine distance.

    Stage 2 applies cosine k-means on the bulk cluster. The number of regimes is
    chosen automatically via an inertia-based elbow heuristic up to ``k_max``.

    Membership probabilities are derived from soft assignments based on the
    distance to each centroid, mirroring equations (1) and (4) in the paper.
    """

    def __init__(
        self,
        k_max: int = 8,
        random_state: Optional[int] = 0,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
    ) -> None:
        self.k_max = k_max
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol

        self.stage1_model: Optional[KMeans] = None
        self.stage2_model: Optional[KMeans] = None
        self.outlier_label: Optional[int] = None
        self.bulk_label: Optional[int] = None
        self.n_regimes_: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "LayeredKMeansRegimeDetector":
        """Fit the two-layer k-means regime detector.

        Parameters
        ----------
        X : np.ndarray
            PCA-transformed feature matrix with shape ``(T, m)``.
        """

        if X.ndim != 2:
            raise ValueError("Input X must be 2-dimensional (T, m)")

        # Stage 1: Euclidean k-means with k=2
        self.stage1_model = KMeans(
            n_clusters=2, random_state=self.random_state, n_init=self.n_init, max_iter=self.max_iter
        )
        stage1_labels = self.stage1_model.fit_predict(X)
        counts = np.bincount(stage1_labels)
        if counts.size < 2:
            raise RuntimeError("Stage 1 k-means failed to form two clusters")

        self.outlier_label = int(np.argmin(counts))
        self.bulk_label = 1 - self.outlier_label

        bulk_mask = stage1_labels == self.bulk_label
        bulk_X = X[bulk_mask]
        if bulk_X.shape[0] == 0:
            raise RuntimeError("No observations assigned to bulk cluster in stage 1")

        # Stage 2: cosine k-means on bulk cluster with automatic k selection
        best_k, stage2_model = self._fit_cosine_kmeans_elbow(bulk_X)
        self.stage2_model = stage2_model
        self.n_regimes_ = 1 + best_k  # regime 0 (outlier) + bulk regimes

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate membership probabilities for each regime.

        Parameters
        ----------
        X : np.ndarray
            Observations with shape ``(T, m)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(T, n_regimes)`` with regime membership probabilities.
        """

        self._check_is_fitted()
        if X.ndim != 2:
            raise ValueError("Input X must be 2-dimensional (T, m)")

        # Stage 1 probabilities (outlier vs. bulk)
        dists_stage1 = self._euclidean_distances(X, self.stage1_model.cluster_centers_)
        p_stage1 = self._soft_assign(dists_stage1)
        p_outlier = p_stage1[:, self.outlier_label]
        p_bulk = p_stage1[:, self.bulk_label]

        # Stage 2 probabilities across bulk regimes
        X_norm = self._safe_row_normalize(X)
        bulk_centers = self.stage2_model.cluster_centers_
        dists_stage2 = self._cosine_distances(X_norm, self._safe_row_normalize(bulk_centers))
        p_stage2 = self._soft_assign(dists_stage2)

        # Combine probabilities
        proba = np.zeros((X.shape[0], self.n_regimes_), dtype=float)
        proba[:, 0] = p_outlier
        proba[:, 1:] = (p_bulk[:, None]) * p_stage2
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most probable regime for each observation."""

        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        if self.stage1_model is None or self.stage2_model is None or self.n_regimes_ is None:
            raise RuntimeError("The model must be fitted before calling this method")

    def _fit_cosine_kmeans_elbow(self, X: np.ndarray) -> Tuple[int, KMeans]:
        """Fit cosine k-means and pick k via an inertia-based elbow heuristic."""

        n_samples = X.shape[0]
        max_k = min(self.k_max, n_samples)
        if max_k < 1:
            raise RuntimeError("Not enough samples to form clusters in stage 2")

        inertias: List[float] = []
        models: List[KMeans] = []
        X_norm = self._safe_row_normalize(X)

        for k in range(1, max_k + 1):
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            model.fit(X_norm)
            models.append(model)
            # For cosine k-means, inertia is measured using cosine distance
            centroids = self._safe_row_normalize(model.cluster_centers_)
            dists = self._cosine_distances(X_norm, centroids)
            inertias.append(float(np.sum(np.min(dists, axis=1))))

        best_k = self._select_elbow(inertias)
        return best_k, models[best_k - 1]

    def _select_elbow(self, inertias: Iterable[float]) -> int:
        """Pick the elbow point using maximum curvature on the inertia curve."""

        inertia_list = list(inertias)
        if len(inertia_list) == 1:
            return 1

        second_diffs = []
        for i in range(1, len(inertia_list) - 1):
            second_diff = inertia_list[i - 1] - 2 * inertia_list[i] + inertia_list[i + 1]
            second_diffs.append(second_diff)

        if second_diffs:
            best_idx = int(np.argmax(second_diffs)) + 1  # +1 to offset for center index
            return best_idx + 1  # convert to k (1-based)

        # Fallback: choose the smallest inertia (largest k)
        return len(inertia_list)

    def _soft_assign(self, distances: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Convert distances to soft membership probabilities."""

        scaled = -distances / max(temperature, 1e-8)
        scaled -= np.max(scaled, axis=1, keepdims=True)
        weights = np.exp(scaled)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        return weights / weights_sum

    @staticmethod
    def _euclidean_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances between rows of X and centers."""

        return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    @staticmethod
    def _cosine_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute cosine distance matrix between normalized rows of X and centers."""

        similarity = np.clip(np.dot(X, centers.T), -1.0, 1.0)
        return 1.0 - similarity

    @staticmethod
    def _safe_row_normalize(X: np.ndarray) -> np.ndarray:
        """Row-normalize vectors to unit length, guarding against zero vectors."""

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return X / norms
