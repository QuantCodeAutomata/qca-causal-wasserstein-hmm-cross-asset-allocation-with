"""Tests for performance metrics."""
import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pandas as pd
import pytest
from src.metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, compute_all_metrics
)


def make_returns(seed=0, n=252):
    rng = np.random.RandomState(seed)
    returns = pd.Series(rng.randn(n) * 0.01 + 0.0005,
                        index=pd.date_range('2023-01-01', periods=n, freq='B'))
    return returns


def test_annualized_return_positive():
    rets = pd.Series([0.001] * 252)
    ann_ret = annualized_return(rets, 252)
    assert abs(ann_ret - 0.252) < 0.01


def test_max_drawdown_nonpositive():
    rets = make_returns()
    mdd = max_drawdown(rets)
    assert mdd <= 0


def test_sharpe_ratio_positive_for_positive_returns():
    rets = pd.Series([0.001] * 252)
    sr = sharpe_ratio(rets, periods=252)
    assert sr > 0


def test_compute_all_metrics_keys():
    rets = make_returns()
    metrics = compute_all_metrics(rets, periods=252)
    required = ['annualized_return', 'annualized_vol', 'sharpe', 'sortino', 'max_drawdown']
    for k in required:
        assert k in metrics, f'Missing key: {k}'


def test_sharpe_zero_vol():
    """Zero-volatility returns: Sharpe should be handled gracefully."""
    rets = pd.Series([0.001] * 100)
    sr = sharpe_ratio(rets, periods=252)
    assert np.isfinite(sr) or np.isinf(sr)  # either inf or finite is acceptable


def test_max_drawdown_no_drawdown():
    """Monotonically increasing returns should have zero drawdown."""
    rets = pd.Series([0.001] * 100)
    mdd = max_drawdown(rets)
    assert abs(mdd) < 1e-10
