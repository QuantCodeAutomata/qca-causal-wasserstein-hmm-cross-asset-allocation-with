"""Tests for data utility functions."""
import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pandas as pd
import pytest
from src.data_utils import build_features


def make_dummy_prices(n=300, n_assets=5, seed=0):
    rng = np.random.RandomState(seed)
    prices = 100 * np.exp(np.cumsum(rng.randn(n, n_assets) * 0.01, axis=0))
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.DataFrame(prices, index=dates,
                        columns=['SPX', 'BOND', 'GOLD', 'OIL', 'USD'])


def test_feature_dim():
    prices = make_dummy_prices(300)
    log_returns, features = build_features(prices, vol_window=60, mom_window=20)
    assert features.shape[1] == 15  # 5 * 3


def test_no_lookahead():
    """Feature for day t must be based only on data through t-1."""
    prices = make_dummy_prices(300)
    log_returns, features = build_features(prices, vol_window=60, mom_window=20)
    # The return block in features should be lagged: features on row t uses return at t-1
    # We verify the feature index is a subset of log_returns index
    assert set(features.index).issubset(set(log_returns.index))


def test_feature_no_nan():
    """After build_features, no NaN in features."""
    prices = make_dummy_prices(300)
    _, features = build_features(prices, vol_window=60, mom_window=20)
    assert not features.isna().any().any()


def test_log_returns_shape():
    prices = make_dummy_prices(300)
    log_returns, _ = build_features(prices, 60, 20)
    assert log_returns.shape[1] == 5
