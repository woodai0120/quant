"""Position sizing strategies based on regime-conditioned forecasts.

The strategies here follow equations (15)–(18) of the reference paper and
return weight vectors whose absolute weights sum to one. Negative weights
represent short positions.
"""

from __future__ import annotations

import numpy as np


def _normalize(weights: np.ndarray) -> np.ndarray:
    """Normalize weights so that the L1 norm equals one."""

    scale = np.sum(np.abs(weights))
    if scale == 0:
        return np.zeros_like(weights)
    return weights / scale


def long_short(y_hat: np.ndarray, l: int) -> np.ndarray:
    """Allocate equally to top/bottom ``l`` forecasts with long/short legs.

    Long and short legs each take half of the exposure; absolute weights sum to
    one after normalization.
    """

    y_hat = np.asarray(y_hat)
    weights = np.zeros_like(y_hat, dtype=float)
    if y_hat.size == 0 or l <= 0:
        return weights

    longs = np.argpartition(-y_hat, min(l, y_hat.size) - 1)[:l]
    shorts = np.argpartition(y_hat, min(l, y_hat.size) - 1)[:l]

    long_weight = 0.5 / max(len(longs), 1)
    short_weight = 0.5 / max(len(shorts), 1)

    weights[longs] = long_weight
    weights[shorts] = -short_weight
    return _normalize(weights)


def long_only(y_hat: np.ndarray, l: int) -> np.ndarray:
    """Invest in the top ``l`` forecasts on a long-only basis."""

    y_hat = np.asarray(y_hat)
    weights = np.zeros_like(y_hat, dtype=float)
    if y_hat.size == 0 or l <= 0:
        return weights

    longs = np.argpartition(-y_hat, min(l, y_hat.size) - 1)[:l]
    positive_forecasts = np.clip(y_hat[longs], a_min=0, a_max=None)
    if positive_forecasts.sum() == 0:
        weights[longs] = 1.0 / len(longs)
    else:
        weights[longs] = positive_forecasts / positive_forecasts.sum()
    return _normalize(weights)


def long_or_short(y_hat: np.ndarray, l: int) -> np.ndarray:
    """Choose long-only if forecasts are mostly positive, otherwise short-only."""

    y_hat = np.asarray(y_hat)
    if y_hat.size == 0 or l <= 0:
        return np.zeros_like(y_hat, dtype=float)

    if np.mean(y_hat) >= 0:
        return long_only(y_hat, l)

    # Short-only: mirror long-only on negative forecasts
    weights = np.zeros_like(y_hat, dtype=float)
    shorts = np.argpartition(y_hat, min(l, y_hat.size) - 1)[:l]
    negative_forecasts = np.clip(-y_hat[shorts], a_min=0, a_max=None)
    if negative_forecasts.sum() == 0:
        weights[shorts] = -1.0 / len(shorts)
    else:
        weights[shorts] = -negative_forecasts / negative_forecasts.sum()
    return _normalize(weights)


def mixed_strategy(y_hat: np.ndarray, l: int, crisis_prob: float) -> np.ndarray:
    """Blend long-short and long-only based on crisis probability.

    ``crisis_prob`` in [0, 1] increases the weight of the defensive long-only
    stance; otherwise a balanced long-short allocation is preferred.
    """

    crisis_prob = float(np.clip(crisis_prob, 0.0, 1.0))
    ws_ls = long_short(y_hat, l)
    ws_lo = long_only(y_hat, l)
    blended = (1.0 - crisis_prob) * ws_ls + crisis_prob * ws_lo
    return _normalize(blended)
