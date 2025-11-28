"""Preprocessing utilities for FRED-MD data following the paper's Section 3.1."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def transform_series(series: pd.Series, tcode: int) -> pd.Series:
    """Transform an individual time series according to FRED-MD t-code rules.

    The transformations follow the standard seven-code scheme:
        1: level (no transformation)
        2: first difference
        3: second difference
        4: natural log
        5: first difference of log
        6: second difference of log
        7: first difference of growth rate (ratio difference)

    Args:
        series: Input time series as a pandas Series.
        tcode: Transformation code in the range 1-7.

    Returns:
        Transformed pandas Series aligned with the original index.
    """

    if tcode == 1:
        return series
    if tcode == 2:
        return series.diff()
    if tcode == 3:
        return series.diff().diff()
    if tcode == 4:
        return np.log(series)
    if tcode == 5:
        return np.log(series).diff()
    if tcode == 6:
        return np.log(series).diff().diff()
    if tcode == 7:
        return (series / series.shift(1)) - 1

    raise ValueError(f"Unsupported tcode {tcode}. Expected integer in 1-7 range.")


def preprocess_fred_data(
    raw_data: pd.DataFrame, tcode_info: pd.DataFrame
) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """Transform, standardize, and apply PCA to FRED-MD data.

    Args:
        raw_data: Monthly FRED-MD raw data with variables as columns.
        tcode_info: DataFrame containing columns 'variable' and 'tcode'.

    Returns:
        X_pca: Numpy array of PCA-transformed features (T, m).
        pca: Fitted PCA object with components covering 95% variance.
        scaler: Fitted StandardScaler used for z-score normalization.
    """

    if not {"variable", "tcode"}.issubset(tcode_info.columns):
        raise ValueError("tcode_info must contain 'variable' and 'tcode' columns.")

    tcode_map = tcode_info.set_index("variable")["tcode"].to_dict()
    transformed_cols = {}

    for column in raw_data.columns:
        if column not in tcode_map:
            raise KeyError(f"No tcode provided for variable '{column}'.")
        transformed_cols[column] = transform_series(raw_data[column], int(tcode_map[column]))

    transformed_df = pd.DataFrame(transformed_cols, index=raw_data.index).dropna()

    if transformed_df.empty:
        raise ValueError("Transformed data is empty after applying tcodes and dropna().")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(transformed_df)

    full_pca = PCA()
    full_pca.fit(X_scaled)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative_variance, 0.95) + 1)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca, scaler


def impute_missing_values(data: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Impute missing values using forward/backward fill or mean strategies."""

    if method == "ffill":
        return data.ffill().bfill()
    if method == "bfill":
        return data.bfill().ffill()
    if method == "mean":
        return data.fillna(data.mean())

    raise ValueError("Unsupported imputation method. Choose from 'ffill', 'bfill', or 'mean'.")
