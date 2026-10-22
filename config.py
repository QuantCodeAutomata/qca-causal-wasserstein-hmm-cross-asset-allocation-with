"""
Central configuration for Causal Wasserstein-HMM Cross-Asset Allocation.

All hyperparameters and implementation choices that are ambiguous in the paper
are documented here with explicit rationale.
"""

# ---------------------------------------------------------------------------
# Asset universe (documented choice: liquid US-listed ETF proxies)
# ---------------------------------------------------------------------------
ASSET_LABELS = ["SPX", "BOND", "GOLD", "OIL", "USD"]
ASSET_TICKERS = {
    "SPX":  "SPY",   # SPDR S&P 500 ETF Trust
    "BOND": "TLT",   # iShares 20+ Year Treasury Bond ETF
    "GOLD": "GLD",   # SPDR Gold Shares
    "OIL":  "USO",   # United States Oil Fund
    "USD":  "UUP",   # Invesco DB US Dollar Index Bullish Fund
}
N_ASSETS = 5

# ---------------------------------------------------------------------------
# Sample period
# ---------------------------------------------------------------------------
GLOBAL_START = "2005-01-01"
GLOBAL_END   = "2026-12-31"

# OOS split date (documented choice: ~680 OOS trading days as implied by paper)
OOS_START = "2023-05-01"

# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
VOL_WINDOW  = 60   # rolling volatility window (days)
MOM_WINDOW  = 20   # rolling mean-return window (days)
FEATURE_DIM = 15   # N_ASSETS * 3 (returns, volatilities, mean returns)

# Strict causality: feature used for day t is computed from data through t-1
# (i.e., the return block is lagged by 1 day before stacking)

# ---------------------------------------------------------------------------
# HMM hyperparameters
# ---------------------------------------------------------------------------
K_MIN         = 2     # minimum HMM state count for order selection
K_MAX         = 7     # maximum HMM state count
F_K           = 20    # order-selection frequency (every F_K OOS trading days)
V_LEN         = 60    # predictive validation slice length (trading days)
LAMBDA_K      = 1.0   # complexity penalty coefficient for order selection
HMM_LOOKBACK  = 750   # rolling window for HMM fitting (documented deviation:
                      # full expanding history is impractical; 750-day window
                      # preserves recent regime structure)
HMM_N_ITER    = 50    # EM maximum iterations (reduced for tractability; ~0.14 s/fit)
HMM_TOL       = 1e-3  # EM convergence tolerance
HMM_N_RESTARTS = 1    # random restarts per candidate K (1 for speed; convergence
                      # checked by monitoring log-likelihood across days)

# ---------------------------------------------------------------------------
# Template hyperparameters
# ---------------------------------------------------------------------------
G   = 6      # number of persistent regime templates
ETA = 0.05   # exponential smoothing rate for template updates

# Template initialization: K=G-state HMM fit on first HMM_LOOKBACK observations
# of training data; component means/covs seed the G templates.

# Within-template moment aggregation: probability-weighted average of all
# components assigned to the template on each day.
# If no component maps to a template on a day, that template is unchanged.

# ---------------------------------------------------------------------------
# Portfolio optimization
# ---------------------------------------------------------------------------
GAMMA  = 1.0      # risk-aversion coefficient
TAU    = 0.0001   # L1 turnover penalty (~1 bp proportional transaction cost):
                  # daily log-return scale ~0.001; cost of 10% tilt = 0.0001*0.2
                  # = 0.00002, well below expected return gain ~0.001*0.1 = 0.0001
W_MAX  = 0.60  # maximum single-asset weight
W_INIT = None  # initial weights at t0: None → equal weight (1/N_ASSETS each)

# MVO objective: maximize mu'w - gamma * w'Sigma w - tau * ||w - w_prev||_1
# Subject to: 1'w = 1, w >= 0, w <= W_MAX
# Realized returns are gross (cost-awareness only in optimizer, not applied
# to realized return accounting, consistent with paper pseudocode).

# ---------------------------------------------------------------------------
# KNN baseline
# ---------------------------------------------------------------------------
K_NN        = 50      # number of nearest neighbors
DISTANCE    = "euclidean"   # documented choice: Euclidean on standardized features
SCALE_KNN   = True    # standardize features before KNN (documented choice)
KNN_LOOKBACK = None   # None → use full expanding history up to each day

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
EW_REBALANCE = "daily"   # documented choice: daily-rebalanced equal-weight

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
ANNUALIZATION = 252   # trading days per year
RISK_FREE     = 0.0   # risk-free rate for Sharpe/Sortino (annualized)

# ---------------------------------------------------------------------------
# Stress event window (Liberation Day / early-2025 selloff)
# ---------------------------------------------------------------------------
STRESS_START = "2025-03-01"
STRESS_END   = "2025-05-31"
