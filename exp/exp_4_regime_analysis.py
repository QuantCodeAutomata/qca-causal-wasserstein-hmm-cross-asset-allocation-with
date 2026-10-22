"""
Exp 4: Regime Stability, Template Identity Preservation, and Stress-Episode Attribution.
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
    ASSET_LABELS,
    ANNUALIZATION,
    STRESS_START, STRESS_END,
    G,
)
from src.metrics import regime_conditional_metrics, asset_sharpe_by_regime, compute_all_metrics
from src.regime_analysis import (
    plot_template_probabilities, plot_weights_stacked, plot_regime_weights,
    plot_asset_sharpe_by_regime, stress_event_analysis, plot_cumulative_returns
)

RESULTS_DIR = Path('/workspace/project/results')


def run_exp4() -> None:
    """Run regime stability and attribution analysis on exp1 artifacts."""
    print('=== EXP 4: Regime Stability and Attribution ===')

    # Load exp1 artifacts
    artifacts_path = RESULTS_DIR / 'exp1_artifacts.csv'
    if not artifacts_path.exists():
        print('EXP 1 artifacts not found. Run exp_1 first.')
        return

    df = pd.read_csv(artifacts_path, index_col='date', parse_dates=True)

    # Template usage analysis
    prob_cols = [c for c in df.columns if c.startswith('p_g_')]
    if prob_cols:
        avg_template_mass = df[prob_cols].mean()
        print('\nAverage template posterior mass:')
        for col, val in avg_template_mass.items():
            print(f'  {col}: {val:.4f}')

        dominant_counts = df['dominant_regime'].value_counts(normalize=True).sort_index()
        print('\nFraction of days dominant:')
        print(dominant_counts.to_string())

    # Regime-conditional performance
    valid = df['dominant_regime'] >= 0
    if valid.sum() > 10:
        port_ret = df.loc[valid, 'port_return']
        dom_reg = df.loc[valid, 'dominant_regime']

        reg_metrics = regime_conditional_metrics(port_ret, dom_reg, ANNUALIZATION)
        print('\nRegime-conditional performance:')
        print(reg_metrics.to_string())
        reg_metrics.to_csv(RESULTS_DIR / 'exp4_regime_metrics.csv')

        asset_ret_df = df.loc[valid, [f'r_{a}' for a in ASSET_LABELS]].rename(
            columns={f'r_{a}': a for a in ASSET_LABELS})
        asset_sharpe = asset_sharpe_by_regime(asset_ret_df, dom_reg, ANNUALIZATION)
        print('\nAsset Sharpe by regime:')
        print(asset_sharpe.to_string())
        asset_sharpe.to_csv(RESULTS_DIR / 'exp4_asset_sharpe_by_regime.csv')

        # Regime weight analysis
        avg_weights = (
            df.loc[valid]
            .groupby('dominant_regime')[[f'w_{a}' for a in ASSET_LABELS]]
            .mean()
            .rename(columns={f'w_{a}': a for a in ASSET_LABELS})
        )
        avg_weights.to_csv(RESULTS_DIR / 'exp4_avg_weights_by_regime.csv')

        plot_regime_weights(avg_weights, ASSET_LABELS,
                            str(RESULTS_DIR / 'exp4_regime_weights.png'))
        plot_asset_sharpe_by_regime(asset_sharpe,
                                    str(RESULTS_DIR / 'exp4_asset_sharpe_heatmap.png'))

    # Stress-event analysis
    stress_df = stress_event_analysis(
        df, STRESS_START, STRESS_END, ASSET_LABELS,
        str(RESULTS_DIR / 'exp4_stress_episode.png')
    )
    if len(stress_df) > 0:
        print(f'\nStress episode ({STRESS_START} to {STRESS_END}):')
        print(f'  Mean SPX weight:  {stress_df["w_SPX"].mean():.4f}')
        print(f'  Mean BOND weight: {stress_df["w_BOND"].mean():.4f}')
        print(f'  Mean GOLD weight: {stress_df["w_GOLD"].mean():.4f}')
        stress_df.to_csv(RESULTS_DIR / 'exp4_stress_episode.csv')

    # Template stability: run-length distribution of dominant regime
    if valid.sum() > 10:
        dom_series = df.loc[valid, 'dominant_regime']
        runs = (dom_series != dom_series.shift()).cumsum()
        run_lengths = dom_series.groupby(runs).count()
        print(f'\nDominant regime run-length statistics:')
        print(f'  Mean:   {run_lengths.mean():.1f} days')
        print(f'  Median: {run_lengths.median():.1f} days')
        print(f'  Max:    {run_lengths.max()} days')

    print('EXP 4 complete.')


if __name__ == '__main__':
    run_exp4()
