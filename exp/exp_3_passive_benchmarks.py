"""
Exp 3: Passive Benchmark Evaluation.
"""
import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import (
    ASSET_TICKERS,
    GLOBAL_START, GLOBAL_END, OOS_START,
    VOL_WINDOW, MOM_WINDOW,
    EW_REBALANCE,
    ANNUALIZATION, RISK_FREE,
)
from src.data_utils import download_prices, build_features, align_to_oos
from src.benchmarks import equal_weight_returns, spx_bah_returns
from src.metrics import compute_all_metrics

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp3() -> tuple:
    """Run passive benchmark evaluation. Returns (metrics_dict, ew_returns, spx_returns)."""
    print('=== EXP 3: Passive Benchmarks ===')

    prices = download_prices(ASSET_TICKERS, GLOBAL_START, GLOBAL_END)
    log_returns, features = build_features(prices, VOL_WINDOW, MOM_WINDOW)
    _, _, oos_ret, _ = align_to_oos(log_returns, features, OOS_START)

    ew_returns = equal_weight_returns(oos_ret, rebalance=EW_REBALANCE)
    spx_returns = spx_bah_returns(oos_ret, spx_col='SPX')

    ew_metrics = compute_all_metrics(ew_returns, periods=ANNUALIZATION, risk_free=RISK_FREE)
    spx_metrics = compute_all_metrics(spx_returns, periods=ANNUALIZATION, risk_free=RISK_FREE)

    print('\n=== EW METRICS ===')
    for k, v in ew_metrics.items():
        print(f'  {k}: {v:.4f}')
    print('\n=== SPX BaH METRICS ===')
    for k, v in spx_metrics.items():
        print(f'  {k}: {v:.4f}')

    # Save daily returns
    results_df = pd.DataFrame({'ew': ew_returns, 'spx': spx_returns})
    results_df.to_csv(RESULTS_DIR / 'exp3_benchmark_returns.csv')

    return {'ew': ew_metrics, 'spx': spx_metrics}, ew_returns, spx_returns


if __name__ == '__main__':
    results, ew_ret, spx_ret = run_exp3()
