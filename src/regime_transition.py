"""Regime transition modeling utilities."""

from typing import Iterable

import numpy as np


def compute_transition_matrix(regime_sequence: Iterable[int], epsilon: float = 1e-6) -> np.ndarray:
    """Compute regime transition probabilities using counts from a label sequence.

    The implementation follows equations (6) and (7) in the referenced paper:

    - Eq. (6): :math:`n_{ij} = \sum_t \mathbb{1}(r_{t-1} = i, r_t = j)`
    - Eq. (7): :math:`E_{ij} = n_{ij} / \sum_j n_{ij}`

    Args:
        regime_sequence: Iterable of integer regime labels ordered in time.
        epsilon: Small constant added to each transition count to avoid zero-probability
            rows for rare or missing transitions.

    Returns:
        Transition probability matrix ``E`` with shape ``(n_regimes, n_regimes)``.
        Rows correspond to source regimes and columns to destination regimes.
    """

    labels = np.asarray(regime_sequence, dtype=int)
    if labels.ndim != 1:
        raise ValueError("regime_sequence must be a 1D iterable of integer labels")
    if labels.size < 2:
        raise ValueError("regime_sequence must contain at least two observations")

    unique_labels = np.unique(labels)
    n_regimes = unique_labels.size
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    counts = np.zeros((n_regimes, n_regimes), dtype=float)
    for prev, curr in zip(labels[:-1], labels[1:]):
        i = label_to_index[prev]
        j = label_to_index[curr]
        counts[i, j] += 1.0

    counts += epsilon
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0  # safeguard against divide-by-zero

    transition_matrix = counts / row_sums
    return transition_matrix
