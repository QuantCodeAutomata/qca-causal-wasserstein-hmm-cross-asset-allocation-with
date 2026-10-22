# Project: Causal Wasserstein-HMM Cross-Asset Allocation

## Structure
- `config.py` — central hyperparameters (asset universe, HMM, template, MVO, KNN settings)
- `src/` — library modules (data_utils, hmm_model, wasserstein, portfolio, knn_model, benchmarks, metrics, regime_analysis)
- `exp/` — experiment scripts (exp_1 through exp_4)
- `tests/` — pytest test files for each src module
- `results/` — output CSVs and PNGs written by experiments
- `data/` — local data cache

## Experiments
| Script | Purpose |
|---|---|
| `exp/exp_1_wasserstein_hmm.py` | Main OOS strategy: HMM + Wasserstein template bank + long-only MVO |
| `exp/exp_2_knn_baseline.py` | KNN nonparametric conditional-moment baseline |
| `exp/exp_3_passive_benchmarks.py` | Equal-weight and SPX buy-and-hold benchmarks |
| `exp/exp_4_regime_analysis.py` | Regime stability, attribution, stress analysis (requires exp1 artifacts) |

Run exp scripts in order: 1 → 2 → 3 → 4 (exp4 depends on exp1_artifacts.csv).

## src Module Contracts
- `compute_all_metrics(returns, turnovers=None, n_effs=None, periods=252, risk_free=0.0)` — `turnovers` and `n_effs` are optional; always called with keyword args `periods=` and `risk_free=`.
- `solve_mvo(mu, Sigma, w_prev, gamma=..., tau=..., w_max=...)` — `gamma`, `tau`, `w_max` have defaults matching config.
- `select_hmm_order(X_train, X_val, k_min, k_max, lambda_k, n_restarts, n_iter, tol=...)` — `tol` has a default.
- `src/metrics.py` exports individual helpers: `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown` (used in tests).

## Key Config Values
- Assets: SPX (SPY), BOND (TLT), GOLD (GLD), OIL (USO), USD (UUP)
- OOS start: 2023-05-01; full period: 2005-01-01 – 2026-12-31
- G=6 templates, ETA=0.05; HMM K_MIN=2, K_MAX=7
- MVO: GAMMA=1.0, TAU=0.01, W_MAX=0.60
- Features: 15-dim (5 assets × {log-return, rolling-vol, rolling-mean}), strictly lagged

## Causality Convention
Features used on OOS day `t` are computed exclusively from data through `t-1`.
In the KNN baseline: `X_for_nn[s]` (feature at time `s`) pairs with `R_for_nn[s]` (return at `s+1`).

## Implementation Notes (src/ modules)

### data_utils.py
- `download_prices`: tries Massive REST API (adjusted=True, limit=50000) if `MASSIVE_TOKEN` env
  var is set; falls back to yfinance with `auto_adjust=True`. MultiIndex yfinance columns handled
  by checking level-0 for 'Close'; falls back to `xs('Close', level=1)` for old column layout.
  Intersection join: `dropna(how='any')` after combining all tickers.
- `build_features`: lags log-returns, rolling-vol (60d × √252), rolling-mean (20d) each by 1
  trading day via `.shift(1)`. Feature columns named `ret_X`, `vol_X`, `mean_X`. NaN rows dropped
  and `log_returns` realigned to the surviving index. Assertion checks feature width = N_ASSETS×3.
- `align_to_oos`: hard cutoff at `pd.Timestamp(oos_start)`; train is strictly `< cutoff`.

### hmm_model.py
- `HMMResult`: plain container, all arrays are copies from the fitted hmmlearn model.
- `fit_hmm`: uses `GaussianHMM(covariance_type='full')`; seeds are `random_state + i` (or `i`
  when `random_state is None`). `model.score(X) * len(X)` gives total log-likelihood.
- Forward algorithm (`_forward_log`): log-space, normalised at every step; emission probs via
  `scipy.stats.multivariate_normal.logpdf`; transition via
  `logsumexp(log_alpha[t-1,:,None] + log_transmat, axis=0)`.
- `compute_predictive_loglik`: accumulates `logsumexp(log_pred + log_emiss[t])` for t=1..T-1.
- `select_hmm_order`: complexity = K×D + K×D×(D+1)//2 + K×K (integer division already matches
  float formula for whole-number K,D).
- `compute_predictive_probs`: runs forward on `X[:-1]`, then one more transition step.

### wasserstein.py
- `real_psd_sqrt`: eigendecomposition of symmetrised matrix; clips eigenvalues to `eps=1e-8`.
  More robust than `scipy.linalg.sqrtm` for near-singular matrices.
- `gaussian_w2_sq`: Bures metric; always clips result to ≥ 0.
- `TemplateBank.assign_components`: independent argmin over templates for each component.
- `aggregate_template_moments`: probability-weighted average of component (mean, cov) within each
  template group; templates with no assigned component keep current values.
- `update`: EMA applied only to templates present in `np.unique(assignments)`.
- `compute_portfolio_moments`: simple weighted sum — NOT full GMM covariance.

### portfolio.py
- `solve_mvo`: symmetrises Sigma and adds `eps_ridge×I`; CLARABEL solver with warm-start; clips
  result to [0, w_max] and renormalises; returns equal-weight on any solver failure.
- `compute_turnover`: half-sum of absolute weight changes (one-way).
- `compute_n_eff`: inverse Herfindahl; returns `len(w)` when sum-of-squares < 1e-12.
