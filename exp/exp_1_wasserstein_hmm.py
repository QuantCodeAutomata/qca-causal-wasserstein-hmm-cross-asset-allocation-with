"""
Exp 1: Causal Wasserstein-HMM Cross-Asset Allocation.
Strict OOS sequential loop with adaptive order selection, template tracking,
and long-only MVO with turnover penalty.
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
    K_MIN, K_MAX, F_K, V_LEN, LAMBDA_K,
    HMM_LOOKBACK, HMM_N_ITER, HMM_TOL, HMM_N_RESTARTS,
    G, ETA,
    GAMMA, TAU, W_MAX,
    ANNUALIZATION, RISK_FREE,
    STRESS_START, STRESS_END,
)
from src.data_utils import download_prices, build_features, align_to_oos
from src.hmm_model import fit_hmm, select_hmm_order, compute_predictive_probs
from src.wasserstein import TemplateBank, real_psd_sqrt
from src.portfolio import solve_mvo, compute_turnover, compute_n_eff
from src.metrics import compute_all_metrics, regime_conditional_metrics, asset_sharpe_by_regime
from src.regime_analysis import (
    plot_template_probabilities, plot_weights_stacked,
    plot_cumulative_returns, plot_turnover_series, plot_n_eff_series,
    plot_regime_weights, plot_asset_sharpe_by_regime, stress_event_analysis
)

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)


def initialize_templates(X_init: np.ndarray, G: int) -> tuple:
    """Initialize G templates from a K=G HMM fit on initial data."""
    result = fit_hmm(X_init, n_components=G, n_restarts=3, n_iter=200,
                     tol=1e-4, random_state=42)
    if result.model is not None:
        return result.means, result.covars
    # Fallback: random init from data points
    D = X_init.shape[1]
    means = X_init[np.random.choice(len(X_init), G, replace=False)]
    covars = np.stack([np.eye(D) * X_init.var() for _ in range(G)])
    return means, covars


def _make_record(date, w_t, w_prev, asset_returns, K_t, tmpl_probs,
                 assignments, comp_probs):
    rec = {'date': date, 'K_t': K_t}
    for i, lab in enumerate(ASSET_LABELS):
        rec[f'w_{lab}'] = w_t[i]
        rec[f'r_{lab}'] = asset_returns[i] if asset_returns is not None else np.nan
    rec['port_return'] = float(w_t @ asset_returns) if asset_returns is not None else np.nan
    rec['turnover'] = float(compute_turnover(w_t, w_prev))
    rec['n_eff'] = float(compute_n_eff(w_t))
    if tmpl_probs is not None:
        for g in range(len(tmpl_probs)):
            rec[f'p_g_{g}'] = float(tmpl_probs[g])
        rec['dominant_regime'] = int(np.argmax(tmpl_probs))
    else:
        for g in range(G):
            rec[f'p_g_{g}'] = np.nan
        rec['dominant_regime'] = -1
    return rec


def run_exp1() -> list:
    """Run the full Wasserstein-HMM OOS experiment. Returns list of daily record dicts."""
    print('=== EXP 1: Wasserstein-HMM Cross-Asset Allocation ===')

    # 1. Data
    print('Downloading data...')
    prices = download_prices(ASSET_TICKERS, GLOBAL_START, GLOBAL_END)
    log_returns, features = build_features(prices, VOL_WINDOW, MOM_WINDOW)
    train_ret, train_feat, oos_ret, oos_feat = align_to_oos(log_returns, features, OOS_START)

    print(f'Train: {train_ret.index[0].date()} to {train_ret.index[-1].date()} ({len(train_ret)} days)')
    print(f'OOS:   {oos_ret.index[0].date()} to {oos_ret.index[-1].date()} ({len(oos_ret)} days)')

    # Build full history arrays
    all_ret = pd.concat([train_ret, oos_ret])
    all_feat = pd.concat([train_feat, oos_feat])

    X_all = all_feat.values
    R_all = all_ret.values

    oos_start_idx = len(train_feat)  # index in all_feat where OOS begins

    # 2. Initialize templates from initial training data
    print(f'Initializing {G} templates...')
    init_data = X_all[:min(HMM_LOOKBACK, oos_start_idx)]
    tmpl_means, tmpl_covars = initialize_templates(init_data, G)
    template_bank = TemplateBank(G, tmpl_means, tmpl_covars, ETA)

    # 3. Initial portfolio weight (documented choice: equal-weight)
    w_prev = np.ones(N_ASSETS) / N_ASSETS

    # 4. State for order selection
    current_K = K_MIN + 1  # start with K=3

    # 5. Containers for daily artifacts
    records = []

    print(f'Running OOS loop over {len(oos_ret)} days...')
    for oos_i, date in enumerate(oos_ret.index):
        # Global index into all arrays
        global_i = oos_start_idx + oos_i

        # History available through t-1 (strictly causal)
        hist_start = max(0, global_i - HMM_LOOKBACK)
        X_hist = X_all[hist_start:global_i]

        if len(X_hist) < max(K_MAX * 3, V_LEN + 20):
            # Not enough history yet; use equal weight
            w_t = np.ones(N_ASSETS) / N_ASSETS
            rec = _make_record(date, w_t, w_prev, R_all[global_i],
                               current_K, None, None, None)
            records.append(rec)
            w_prev = w_t
            continue

        # 5a. Adaptive order selection every F_K days
        if oos_i % F_K == 0:
            X_train_k = X_hist[:-V_LEN]
            X_val_k = X_hist[-V_LEN:]
            if len(X_train_k) >= K_MAX * 3:
                current_K, k_scores = select_hmm_order(
                    X_train_k, X_val_k, K_MIN, K_MAX,
                    LAMBDA_K, HMM_N_RESTARTS, HMM_N_ITER, HMM_TOL
                )
                if oos_i % (F_K * 5) == 0:
                    print(f'  Day {oos_i}: K_t={current_K}, scores={k_scores}')

        # 5b. Fit HMM on full history window
        hmm_res = fit_hmm(X_hist, current_K, HMM_N_RESTARTS, HMM_N_ITER,
                          HMM_TOL, random_state=oos_i)

        if hmm_res.model is None:
            # HMM fitting failed; hold previous weights
            w_t = w_prev.copy()
            rec = _make_record(date, w_t, w_prev, R_all[global_i],
                               current_K, None, None, None)
            records.append(rec)
            w_prev = w_t
            continue

        # 5c. Predictive regime probabilities P(z_t = k | F_{t-1})
        pred_probs = compute_predictive_probs(hmm_res.model, X_hist)  # shape (K,)

        # 5d. Template assignment via Wasserstein distance
        assignments = template_bank.assign_components(hmm_res.means, hmm_res.covars)

        # 5e. Aggregate template probabilities and moments
        tmpl_probs, bar_means, bar_covars = template_bank.aggregate_template_moments(
            hmm_res.means, hmm_res.covars, pred_probs, assignments
        )

        # 5f. Update templates via EMA
        template_bank.update(bar_means, bar_covars, assignments)

        # 5g. Portfolio optimization inputs
        mu_t, Sigma_t = template_bank.compute_portfolio_moments(tmpl_probs, N_ASSETS)

        # 5h. Solve MVO
        w_t = solve_mvo(mu_t, Sigma_t, w_prev, GAMMA, TAU, W_MAX)

        # 5i. Record
        rec = _make_record(date, w_t, w_prev, R_all[global_i],
                           current_K, tmpl_probs, assignments, pred_probs)
        records.append(rec)
        w_prev = w_t

    print(f'OOS loop complete. {len(records)} records.')
    return records


def save_results(records: list, df_oos_ret: pd.DataFrame):
    """Save results to /workspace/project/results/ and return computed artifacts."""
    df = pd.DataFrame(records).set_index('date')
    df.to_csv(RESULTS_DIR / 'exp1_artifacts.csv')

    port_returns = df['port_return']
    turnovers = df['turnover']
    n_effs = df['n_eff']

    metrics = compute_all_metrics(port_returns, turnovers, n_effs, ANNUALIZATION, RISK_FREE)
    print('\n=== EXP 1 METRICS ===')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')

    # Regime-conditional metrics
    if 'dominant_regime' in df.columns and df['dominant_regime'].notna().any():
        valid_mask = df['dominant_regime'] >= 0
        regime_metrics = regime_conditional_metrics(
            port_returns[valid_mask], df['dominant_regime'][valid_mask], ANNUALIZATION
        )
        asset_ret_df = df[[f'r_{a}' for a in ASSET_LABELS]].rename(
            columns={f'r_{a}': a for a in ASSET_LABELS})
        asset_sharpe_df = asset_sharpe_by_regime(
            asset_ret_df[valid_mask], df['dominant_regime'][valid_mask], ANNUALIZATION
        )
    else:
        regime_metrics = pd.DataFrame()
        asset_sharpe_df = pd.DataFrame()

    # --- Plots ---
    # Build a weights-only DataFrame with plain asset-label column names
    weights_df = df[[f'w_{a}' for a in ASSET_LABELS]].rename(
        columns={f'w_{a}': a for a in ASSET_LABELS}
    )
    plot_weights_stacked(weights_df, ASSET_LABELS, str(RESULTS_DIR / 'exp1_weights.png'))

    to_series = {'HMM-Wasserstein': turnovers}
    plot_turnover_series(to_series, str(RESULTS_DIR / 'exp1_turnover.png'))

    neff_series = {'HMM-Wasserstein': n_effs}
    plot_n_eff_series(neff_series, str(RESULTS_DIR / 'exp1_neff.png'))

    prob_cols = [c for c in df.columns if c.startswith('p_g_')]
    if prob_cols and 'dominant_regime' in df.columns:
        valid_df = df[df['dominant_regime'] >= 0]
        if len(valid_df) > 0:
            plot_template_probabilities(
                valid_df[prob_cols].rename(columns={f'p_g_{g}': g for g in range(G)}),
                valid_df['dominant_regime'],
                str(RESULTS_DIR / 'exp1_template_probs.png')
            )

    if not regime_metrics.empty:
        avg_weights_by_regime = (
            df.groupby('dominant_regime')[[f'w_{a}' for a in ASSET_LABELS]]
            .mean()
            .rename(columns={f'w_{a}': a for a in ASSET_LABELS})
        )
        plot_regime_weights(avg_weights_by_regime, ASSET_LABELS,
                            str(RESULTS_DIR / 'exp1_regime_weights.png'))
        if not asset_sharpe_df.empty:
            plot_asset_sharpe_by_regime(asset_sharpe_df,
                                        str(RESULTS_DIR / 'exp1_asset_sharpe_by_regime.png'))

    # Stress event
    stress_df = stress_event_analysis(
        df, STRESS_START, STRESS_END, ASSET_LABELS,
        str(RESULTS_DIR / 'exp1_stress_event.png')
    )

    return metrics, regime_metrics, asset_sharpe_df, df


if __name__ == '__main__':
    records = run_exp1()
    # Reload OOS returns for benchmark comparison reference
    prices = download_prices(ASSET_TICKERS, GLOBAL_START, GLOBAL_END)
    log_returns, features = build_features(prices, VOL_WINDOW, MOM_WINDOW)
    _, _, oos_ret, _ = align_to_oos(log_returns, features, OOS_START)
    metrics, regime_metrics, asset_sharpe_df, df = save_results(records, oos_ret)
    print('Results saved to', RESULTS_DIR)
