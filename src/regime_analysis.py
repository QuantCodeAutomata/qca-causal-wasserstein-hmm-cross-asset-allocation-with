import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


def plot_template_probabilities(template_probs_df: pd.DataFrame,
                                dominant_regimes: pd.Series,
                                save_path: Optional[str] = None) -> None:
    """
    Plot template probability time series stacked area chart.
    template_probs_df: DataFrame with date index, columns = template indices
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.stackplot(template_probs_df.index, template_probs_df.T.values,
                  labels=[f'Template {g}' for g in template_probs_df.columns],
                  alpha=0.7)
    ax1.set_ylabel('Template Probability')
    ax1.set_title('Template Posterior Probabilities Over Time')
    ax1.legend(loc='upper left', fontsize=8)
    ax2.plot(dominant_regimes.index, dominant_regimes.values, drawstyle='steps-post')
    ax2.set_ylabel('Dominant Regime')
    ax2.set_title('Dominant Template Label Over Time')
    ax2.set_yticks(sorted(dominant_regimes.unique()))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_weights_stacked(weights_df: pd.DataFrame, asset_labels: List[str],
                         save_path: Optional[str] = None) -> None:
    """
    Stacked area chart of portfolio weights over time.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(weights_df.index, weights_df[asset_labels].T.values,
                 labels=asset_labels, alpha=0.75)
    ax.set_ylabel('Portfolio Weight')
    ax.set_title('Portfolio Allocation Over Time')
    ax.legend(loc='upper left', fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cumulative_returns(returns_dict: Dict[str, pd.Series],
                            save_path: Optional[str] = None) -> None:
    """
    Plot cumulative log returns for multiple strategies.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, ret in returns_dict.items():
        cum = ret.cumsum()
        ax.plot(cum.index, cum.values, label=name, linewidth=1.5)
    ax.set_title('Cumulative OOS Log Returns')
    ax.set_ylabel('Cumulative Log Return')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_turnover_series(turnovers_dict: Dict[str, pd.Series],
                         save_path: Optional[str] = None) -> None:
    """
    Plot daily turnover time series.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    for name, to in turnovers_dict.items():
        ax.plot(to.index, to.values, label=name, linewidth=0.8, alpha=0.8)
    ax.set_title('Daily Turnover')
    ax.set_ylabel('Turnover (|Δw|/2)')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_n_eff_series(n_effs_dict: Dict[str, pd.Series],
                     save_path: Optional[str] = None) -> None:
    """
    Plot N_eff time series for multiple strategies.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    for name, neff in n_effs_dict.items():
        ax.plot(neff.index, neff.values, label=name, linewidth=0.8)
    ax.set_title('Effective Number of Positions (N_eff)')
    ax.set_ylabel('N_eff')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_regime_weights(avg_weights_by_regime: pd.DataFrame,
                        asset_labels: List[str],
                        save_path: Optional[str] = None) -> None:
    """
    Bar chart of average weights per regime.
    avg_weights_by_regime: DataFrame with regime as index, assets as columns
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    avg_weights_by_regime[asset_labels].plot(kind='bar', ax=ax, width=0.75)
    ax.set_title('Average Portfolio Weights by Dominant Regime')
    ax.set_ylabel('Average Weight')
    ax.set_xlabel('Regime')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_asset_sharpe_by_regime(asset_sharpe_df: pd.DataFrame,
                                save_path: Optional[str] = None) -> None:
    """
    Heatmap of asset Sharpe ratios by regime.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(asset_sharpe_df, annot=True, fmt='.2f', center=0,
                cmap='RdYlGn', ax=ax)
    ax.set_title('Asset Sharpe Ratio by Dominant Regime')
    ax.set_xlabel('Asset')
    ax.set_ylabel('Regime')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def stress_event_analysis(artifacts: pd.DataFrame,
                          stress_start: str, stress_end: str,
                          asset_labels: List[str],
                          save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Isolate stress-event window and plot key diagnostics.
    artifacts: DataFrame with columns: date, dominant_regime, spx_weight,
               template probabilities (p_g_0..p_g_G), and asset weights.
    Returns sub-DataFrame for the stress window.
    """
    mask = (artifacts.index >= stress_start) & (artifacts.index <= stress_end)
    stress = artifacts[mask].copy()
    if len(stress) == 0:
        return stress

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: dominant regime
    axes[0].step(stress.index, stress['dominant_regime'], where='post')
    axes[0].set_ylabel('Dominant Template')
    axes[0].set_title(f'Stress Episode ({stress_start} — {stress_end})')

    # Panel 2: template probabilities
    prob_cols = [c for c in stress.columns if c.startswith('p_g_')]
    if prob_cols:
        for col in prob_cols:
            axes[1].plot(stress.index, stress[col], label=col.replace('p_g_', 'Template '))
        axes[1].set_ylabel('Template Prob.')
        axes[1].legend(fontsize=7)

    # Panel 3: asset weights
    weight_cols = [f'w_{a}' for a in asset_labels if f'w_{a}' in stress.columns]
    for col in weight_cols:
        axes[2].plot(stress.index, stress[col], label=col.replace('w_', ''))
    axes[2].set_ylabel('Portfolio Weight')
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return stress
