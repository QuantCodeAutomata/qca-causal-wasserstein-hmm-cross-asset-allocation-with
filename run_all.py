"""
Master pipeline script — runs all four experiments sequentially.

Usage
-----
    python run_all.py

Experiments
-----------
    exp_1_wasserstein_hmm.py   — Parametric HMM-Wasserstein strategy
    exp_2_knn_baseline.py      — KNN nonparametric baseline
    exp_3_passive_benchmarks.py — Equal-weight and SPX buy-and-hold
    exp_4_regime_analysis.py   — Regime stability & stress attribution

All artefacts (CSVs, plots, RESULTS.md) are written to results/.
"""

import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    "exp/exp_1_wasserstein_hmm.py",
    "exp/exp_2_knn_baseline.py",
    "exp/exp_3_passive_benchmarks.py",
    "exp/exp_4_regime_analysis.py",
]


def run_experiment(script: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Running: {script}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=Path(__file__).parent,
        check=True,
    )
    return result


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    for script in EXPERIMENTS:
        run_experiment(script)

    print("\n" + "=" * 60)
    print("  All experiments complete.  Results in results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
