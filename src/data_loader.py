"""Data loading utilities for FRED-MD and ETF datasets."""

from typing import Optional

import pandas as pd


class DataLoader:
    """Load and cache raw datasets."""

    def __init__(self, fred_path: Optional[str] = None, etf_path: Optional[str] = None) -> None:
        self.fred_path = fred_path
        self.etf_path = etf_path

    def load_fred_data(self) -> pd.DataFrame:
        """Load FRED-MD macroeconomic data."""
        raise NotImplementedError("Implement FRED data loading logic")

    def load_etf_data(self) -> pd.DataFrame:
        """Load ETF price or return data."""
        raise NotImplementedError("Implement ETF data loading logic")


def build_monthly_returns(price_df: pd.DataFrame, fred_month_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Convert daily ETF prices to month-start returns and align to FRED-MD months.

    Parameters
    ----------
    price_df : pd.DataFrame
        Daily ETF prices with a DatetimeIndex and tickers as columns.
    fred_month_index : pd.DatetimeIndex
        Monthly DatetimeIndex used by the FRED-MD dataset.

    Returns
    -------
    pd.DataFrame
        Monthly ETF returns computed from month-start prices, aligned via an
        inner join to ``fred_month_index``.
    """

    if not isinstance(price_df.index, pd.DatetimeIndex):
        raise ValueError("price_df must have a DatetimeIndex")

    # Ensure chronological order and compute month-start prices.
    price_df = price_df.sort_index()
    month_start_prices = price_df.resample("MS").first()

    # Compute month-over-month returns.
    monthly_returns = month_start_prices.pct_change()

    # Align to the FRED-MD monthly index using an inner join.
    common_months = monthly_returns.index.intersection(pd.DatetimeIndex(fred_month_index))
    common_months = common_months.sort_values()
    aligned_returns = monthly_returns.loc[common_months]

    # Drop the first row if it is NaN due to differencing.
    return aligned_returns.dropna(how="all")
