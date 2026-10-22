"""
Exp 2: KNN Nonparametric Conditional-Moment Allocation Baseline.
"""
import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import (
    ASSET_TICKERS, ASSET_LABELS, N_ASSETS,
    GLOBAL_START, GLOBAL_END, OOS_START,
    VOL_WINDOW, MOM_WINDOW,
    K_NN, SCALE_KNN,
    GAMMA, TAU, W_MAX,
    ANNUALIZATION, RISK_FREE,
)
from src.data_utils import download_prices, build_features, align_to_oos
from src.knn_model import KNNMomentEstimator
from src.portfolio import solve_mvo, compute_turnover, compute_n_eff
from src.metrics import compute_all_metrics

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp2() -> list:
    """Run KNN OOS experiment. Returns list of daily record dicts."""
    print('=== EXP 2: KNN Nonparametric Baseline ===')

    prices = download_prices(ASSET_TICKERS, GLOBAL_START, GLOBAL_END)
    log_returns, features = build_features(prices, VOL_WINDOW, MOM_WINDOW)
    train_ret, train_feat, oos_ret, oos_feat = align_to_oos(log_returns, features, OOS_START)

    all_ret = pd.concat([train_ret, oos_ret])
    all_feat = pd.concat([train_feat, oos_feat])
    X_all = all_feat.values
    R_all = all_ret.values
    oos_start_idx = len(train_feat)

    knn = KNNMomentEstimator(k_nn=K_NN, scale=SCALE_KNN)
    w_prev = np.ones(N_ASSETS) / N_ASSETS
    records = []

    print(f'Running OOS loop over {len(oos_ret)} days...')
    for oos_i, date in enumerate(oos_ret.index):
        global_i = oos_start_idx + oos_i

        # Align: feature[s] = lagged info through s-1 → paired with return[s]
        # (return from s-1 to s).  Both query and historical pairs share the
        # same 1-step-ahead structure: features[global_i] (info through t-1)
        # predicts R_all[global_i] (return from t-1 to t).
        X_for_nn = X_all[:global_i]      # features for historical days 0..t-1
        R_for_nn = R_all[:global_i]      # returns for historical days 0..t-1
        x_query  = X_all[global_i]       # current day's feature (info through t-1)

        if len(X_for_nn) < K_NN + 5:
            w_t = np.ones(N_ASSETS) / N_ASSETS
        else:
            try:
                mu_t, Sigma_t = knn.estimate(X_for_nn, R_for_nn, x_query)
                w_t = solve_mvo(mu_t, Sigma_t, w_prev, GAMMA, TAU, W_MAX)
            except Exception as e:
                print(f'  KNN/MVO failed at {date}: {e}')
                w_t = w_prev.copy()

        r_realized = R_all[global_i]
        port_return = w_t @ r_realized

        rec = {
            'date': date,
            'port_return': port_return,
            'turnover': compute_turnover(w_t, w_prev),
            'n_eff': compute_n_eff(w_t),
        }
        for i, lab in enumerate(ASSET_LABELS):
            rec[f'w_{lab}'] = w_t[i]
            rec[f'r_{lab}'] = r_realized[i]
        records.append(rec)
        w_prev = w_t

        if oos_i % 100 == 0:
            print(f'  Day {oos_i}/{len(oos_ret)}')

    return records


def save_results(records: list):
    """Persist results and print metrics. Returns (metrics_dict, df)."""
    df = pd.DataFrame(records).set_index('date')
    df.to_csv(RESULTS_DIR / 'exp2_artifacts.csv')

    metrics = compute_all_metrics(
        df['port_return'], df['turnover'], df['n_eff'], ANNUALIZATION, RISK_FREE
    )
    print('\n=== EXP 2 METRICS ===')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')
    return metrics, df


if __name__ == '__main__':
    records = run_exp2()
    metrics, df = save_results(records)
