"""
Data download and preprocessing utilities.

Download daily adjusted close prices via the Massive REST API with a yfinance
fallback, compute causal feature vectors, and split into train/OOS samples.
"""

import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def download_prices(tickers: Dict[str, str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily adjusted close prices for given tickers.

    Parameters
    ----------
    tickers : dict mapping asset_label -> yahoo_ticker (e.g. {"SPX": "SPY", ...})
    start   : start date string "YYYY-MM-DD"
    end     : end date string "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame
        DatetimeIndex rows, asset labels as columns.  Rows where *any* ticker
        has a missing price are dropped (intersection join semantics).
    """
    prices = _try_massive(tickers, start, end)
    if prices is None:
        prices = _try_yfinance(tickers, start, end)
    return prices


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _try_massive(tickers: Dict[str, str], start: str, end: str) -> pd.DataFrame | None:
    """Attempt download via Massive REST API; return None on any failure."""
    token = os.getenv("MASSIVE_TOKEN")
    if not token:
        logger.info("MASSIVE_TOKEN not set; skipping Massive download.")
        return None
    try:
        from massive import RESTClient  # noqa: PLC0415
        client = RESTClient(api_key=token)
        frames: dict[str, pd.Series] = {}
        for label, ticker in tickers.items():
            aggs = list(
                client.list_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start,
                    to=end,
                    adjusted=True,
                    limit=50000,
                )
            )
            if not aggs:
                raise ValueError(f"No Massive data returned for {ticker}")
            dates = (
                pd.to_datetime([a.timestamp for a in aggs], unit="ms", utc=True)
                .normalize()
                .tz_localize(None)
            )
            frames[label] = pd.Series([a.close for a in aggs], index=dates, name=label)

        df = pd.DataFrame(frames).sort_index()
        df.index.name = "Date"
        df = df.dropna(how="any")          # intersection join
        logger.info("Massive download succeeded: %d rows.", len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Massive download failed (%s); falling back to yfinance.", exc)
        return None


def _try_yfinance(tickers: Dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Download via yfinance (fallback)."""
    import yfinance as yf  # noqa: PLC0415

    yahoo_tickers = list(tickers.values())
    label_map = {v: k for k, v in tickers.items()}

    raw = yf.download(
        yahoo_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # Extract close prices; handle MultiIndex (multi-ticker) and flat (single-ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        if "Close" in level0:
            close = raw["Close"].copy()
        else:
            # Older yfinance: tickers on level 0, field on level 1
            close = raw.xs("Close", axis=1, level=1).copy()
    else:
        # Single ticker (rare, since we always pass a list)
        close = raw[["Close"]].copy()
        close.columns = yahoo_tickers

    # Rename ticker symbols -> asset labels
    close = close.rename(columns=label_map)
    # Keep only the labels we requested (drop any extra tickers returned)
    ordered = [lbl for lbl in tickers if lbl in close.columns]
    df = close[ordered].copy().sort_index()
    df.index.name = "Date"
    df = df.dropna(how="any")              # intersection join
    logger.info("yfinance download succeeded: %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(
    prices: pd.DataFrame,
    vol_window: int = 60,
    mom_window: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute log returns and a 15-dimensional causal feature matrix.

    Feature vector at date t = [ret_{t-1}, vol_{t-1}, mean_{t-1}] where:
      - ret   : raw log return
      - vol   : rolling annualised volatility (vol_window days)
      - mean  : rolling mean log return (mom_window days)
    All three blocks are lagged by 1 day so feature_t contains *no*
    information from day t (strict causality).

    Parameters
    ----------
    prices     : DataFrame of adjusted close prices (DatetimeIndex, asset cols)
    vol_window : rolling window for volatility (default 60)
    mom_window : rolling window for mean return  (default 20)

    Returns
    -------
    log_returns : pd.DataFrame  — aligned to same dates as features
    features    : pd.DataFrame  — 15-dim feature vectors, NaN rows dropped
    """
    assets = prices.columns.tolist()
    n = len(assets)  # should be 5

    log_ret = np.log(prices) - np.log(prices.shift(1))

    rolling_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)
    rolling_mean = log_ret.rolling(mom_window).mean()

    # Strict causality: lag every block by 1 trading day
    ret_lagged = log_ret.shift(1)
    vol_lagged = rolling_vol.shift(1)
    mean_lagged = rolling_mean.shift(1)

    ret_lagged.columns = [f"ret_{c}" for c in assets]
    vol_lagged.columns = [f"vol_{c}" for c in assets]
    mean_lagged.columns = [f"mean_{c}" for c in assets]

    features = pd.concat([ret_lagged, vol_lagged, mean_lagged], axis=1)

    # Drop any row with a NaN in the feature matrix
    features = features.dropna(how="any")

    # Align log_returns to the valid feature dates (inner join on index)
    common = log_ret.index.intersection(features.index)
    features = features.loc[common]
    log_returns = log_ret.loc[common]

    assert features.shape[1] == n * 3, (
        f"Expected {n * 3} feature columns, got {features.shape[1]}"
    )
    return log_returns, features


# ---------------------------------------------------------------------------
# Train / OOS split
# ---------------------------------------------------------------------------

def align_to_oos(
    log_returns: pd.DataFrame,
    features: pd.DataFrame,
    oos_start: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split log_returns and features into in-sample (train) and out-of-sample
    (OOS) portions at *oos_start*.

    Parameters
    ----------
    log_returns : aligned log-return DataFrame
    features    : aligned feature DataFrame (same index as log_returns)
    oos_start   : first OOS date string "YYYY-MM-DD"

    Returns
    -------
    (train_returns, train_features, oos_returns, oos_features)
    """
    cutoff = pd.Timestamp(oos_start)
    train_mask = features.index < cutoff
    oos_mask = features.index >= cutoff

    train_returns = log_returns.loc[train_mask]
    train_features = features.loc[train_mask]
    oos_returns = log_returns.loc[oos_mask]
    oos_features = features.loc[oos_mask]

    return train_returns, train_features, oos_returns, oos_features
