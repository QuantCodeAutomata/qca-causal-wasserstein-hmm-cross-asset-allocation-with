import numpy as np
import pandas as pd
from typing import Optional


def equal_weight_returns(oos_returns: pd.DataFrame,
                         rebalance: str = 'daily') -> pd.Series:
    """
    Equal-weight benchmark (1/N_assets per asset).

    rebalance='daily': each day portfolio is rebalanced to 1/N weights.
        daily return = (1/N) * sum_i r_{t,i}  (using log returns approx)
    rebalance='bah': buy-and-hold initialized at equal weight at OOS start.
        Uses arithmetic returns for proper portfolio compounding.

    Documented choice: 'daily' rebalancing used as primary benchmark.
    Returns pd.Series of daily portfolio returns, same index as oos_returns.
    """
    N = oos_returns.shape[1]
    if rebalance == 'daily':
        # Simple equal-weight daily return (log return approximation)
        port_returns = oos_returns.mean(axis=1)
    else:  # buy-and-hold
        # Convert log returns to arithmetic, compound, then convert back
        arith = np.expm1(oos_returns.values)  # exp(r) - 1
        weights = np.ones(N) / N
        port_values = np.cumprod(1 + arith @ weights)
        port_arith_returns = np.diff(port_values, prepend=1.0) / np.concatenate([[1.0], port_values[:-1]])
        port_returns = pd.Series(np.log1p(port_arith_returns), index=oos_returns.index)
    return port_returns


def spx_bah_returns(oos_returns: pd.DataFrame, spx_col: str = 'SPX') -> pd.Series:
    """
    SPX buy-and-hold benchmark: 100% weight in SPX proxy.
    Returns pd.Series of daily SPX log returns.
    """
    return oos_returns[spx_col].copy()
