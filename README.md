# Causal Wasserstein-HMM Cross-Asset Allocation

A strictly-causal daily cross-asset allocation system implementing:

- **Gaussian HMM** regime inference with periodic adaptive order selection via penalised predictive log-likelihood
- **Gaussian 2-Wasserstein** distance for template-based regime identity stabilisation (many-to-one nearest-template matching)
- **Template-smoothed** portfolio moment aggregation (EMA tracking)
- **Long-only MVO** with L1 turnover penalty (cvxpy / CLARABEL)
- **KNN nonparametric** conditional moment estimator baseline
- Passive benchmarks: daily-rebalanced equal-weight and SPX buy-and-hold

## Asset Universe

| Label | Ticker | Description |
|-------|--------|-------------|
| SPX   | SPY    | SPDR S&P 500 ETF |
| BOND  | TLT    | iShares 20+ Year Treasury |
| GOLD  | GLD    | SPDR Gold Shares |
| OIL   | USO    | United States Oil Fund |
| USD   | UUP    | Invesco DB US Dollar Index |

## Repository Structure

```
├── config.py            # All hyperparameters and documented choices
├── data/
│   └── download.py      # Standalone data download script
├── src/
│   ├── data_utils.py    # Download + feature construction
│   ├── hmm_model.py     # Gaussian HMM + order selection
│   ├── wasserstein.py   # W2 distance + TemplateBank
│   ├── portfolio.py     # Long-only MVO with turnover penalty
│   ├── knn_model.py     # KNN conditional moment estimator
│   ├── benchmarks.py    # Passive benchmarks
│   ├── metrics.py       # Performance metrics
│   └── regime_analysis.py  # Plotting utilities
├── exp/
│   ├── exp_1_wasserstein_hmm.py   # Main parametric strategy
│   ├── exp_2_knn_baseline.py      # KNN baseline
│   ├── exp_3_passive_benchmarks.py
│   └── exp_4_regime_analysis.py   # Regime attribution
├── tests/               # pytest test suite
├── results/             # Generated outputs (RESULTS.md, plots, CSVs)
└── run_all.py           # Master script: all experiments + RESULTS.md
```

## Setup

```bash
pip install -r requirements.txt
```

Data is downloaded automatically (Massive API if `MASSIVE_TOKEN` is set, yfinance fallback).

## Run

```bash
# Run all experiments sequentially
python run_all.py

# Run individual experiments
python exp/exp_1_wasserstein_hmm.py
python exp/exp_2_knn_baseline.py
python exp/exp_3_passive_benchmarks.py
python exp/exp_4_regime_analysis.py   # requires exp_1 artifacts

# Run tests
pytest tests/ -v
```

## Key Hyperparameters (see `config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| OOS_START | 2023-05-01 | OOS evaluation start |
| G | 6 | Number of regime templates |
| K_MIN / K_MAX | 2 / 7 | HMM order selection range |
| F_K | 20 | Order-selection frequency (trading days) |
| eta | 0.05 | Template EMA smoothing rate |
| gamma | 1.0 | MVO risk aversion |
| tau | 0.01 | L1 turnover penalty |
| w_max | 0.60 | Maximum single-asset weight |
| K_NN | 50 | KNN neighbours |
| HMM_LOOKBACK | 750 | Rolling HMM fitting window (practical deviation from full expanding history) |

## Expected Results (from paper)

| Strategy | Sharpe | Max DD |
|----------|--------|--------|
| HMM-Wasserstein | ~2.18 | ~-5.4% |
| KNN | ~1.81 | ~-12.5% |
| Equal-weight | ~1.59 | ~-9.9% |
| SPX BaH | ~1.18 | ~-14.6% |
