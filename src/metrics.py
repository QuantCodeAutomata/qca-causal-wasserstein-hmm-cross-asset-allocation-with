import numpy as np
import pandas as pd
from typing import Dict, Optional


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    """Annualized mean return."""
    return float(returns.mean() * periods)


def annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    """Annualized volatility."""
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0,
                 periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    ann_vol = annualized_volatility(returns, periods)
    if ann_vol == 0.0:
        return 0.0
    return float((annualized_return(returns, periods) - risk_free) / ann_vol)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0,
                  periods: int = 252) -> float:
    """
    Annualized Sortino ratio (downside deviation).

    Downside deviation uses all observation days in the denominator
    (standard MAR = risk_free formulation):
        DD = sqrt(mean(min(r_t - rf/T, 0)^2)) * sqrt(T)
    """
    daily_rf = risk_free / periods
    excess = returns - daily_rf
    downside = np.minimum(excess, 0.0)
    mean_sq_downside = (downside ** 2).mean()
    if mean_sq_downside == 0.0:
        return np.inf
    downside_dev = np.sqrt(mean_sq_downside) * np.sqrt(periods)
    return float((annualized_return(returns, periods) - risk_free) / downside_dev)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from peak (negative number)."""
    cum = returns.cumsum()  # cumulative log returns
    running_max = cum.cummax()
    dd = cum - running_max
    return dd.min()


def compute_all_metrics(returns: pd.Series, turnovers: Optional[pd.Series] = None,
                        n_effs: Optional[pd.Series] = None,
                        periods: int = 252, risk_free: float = 0.0) -> Dict:
    """
    Compute comprehensive performance metrics.
    Returns dict with: annualized_return, annualized_vol, sharpe, sortino,
    max_drawdown, avg_turnover, p95_turnover, frac_to_gt1pct, frac_to_gt5pct,
    avg_n_eff, median_n_eff.
    """
    metrics = {}
    metrics['annualized_return'] = annualized_return(returns, periods)
    metrics['annualized_vol'] = annualized_volatility(returns, periods)
    metrics['sharpe'] = sharpe_ratio(returns, risk_free, periods)
    metrics['sortino'] = sortino_ratio(returns, risk_free, periods)
    metrics['max_drawdown'] = max_drawdown(returns)
    metrics['total_return'] = returns.sum()
    metrics['n_days'] = len(returns)

    if turnovers is not None:
        metrics['avg_turnover'] = turnovers.mean()
        metrics['p95_turnover'] = turnovers.quantile(0.95)
        metrics['frac_to_gt1pct'] = (turnovers > 0.01).mean()
        metrics['frac_to_gt5pct'] = (turnovers > 0.05).mean()

    if n_effs is not None:
        metrics['avg_n_eff'] = n_effs.mean()
        metrics['median_n_eff'] = n_effs.median()

    return metrics


def regime_conditional_metrics(returns: pd.Series, regime_labels: pd.Series,
                                periods: int = 252) -> pd.DataFrame:
    """
    Compute per-regime performance metrics.
    Returns DataFrame with regime as index and metrics as columns.
    """
    results = []
    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        ret_slice = returns[mask]
        if len(ret_slice) < 5:
            continue
        row = {
            'regime': regime,
            'n_days': len(ret_slice),
            'fraction': mask.mean(),
            'mean_return_ann': annualized_return(ret_slice, periods),
            'vol_ann': annualized_volatility(ret_slice, periods),
            'sharpe': sharpe_ratio(ret_slice, periods=periods),
            'max_drawdown': max_drawdown(ret_slice),
        }
        results.append(row)
    return pd.DataFrame(results).set_index('regime')


def asset_sharpe_by_regime(asset_returns: pd.DataFrame, regime_labels: pd.Series,
                            periods: int = 252) -> pd.DataFrame:
    """
    Compute per-asset Sharpe ratio for each dominant regime.
    Returns DataFrame with regime as index, assets as columns.
    """
    asset_returns, regime_labels = asset_returns.align(regime_labels, join='inner', axis=0)
    results = {}
    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        ret_slice = asset_returns[mask]
        sharpes = {}
        for col in asset_returns.columns:
            sharpes[col] = sharpe_ratio(ret_slice[col], periods=periods)
        results[regime] = sharpes
    return pd.DataFrame(results).T
