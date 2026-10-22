"""
Long-only mean-variance portfolio optimisation with L1 turnover penalty.

The MVO problem is solved with CVXPY (CLARABEL solver).  If the solver fails
for any reason, an equal-weight fallback is returned so the strategy never
goes to cash due to numerical issues.
"""

import logging
from typing import Optional

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


def solve_mvo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    gamma: float = 1.0,
    tau: float = 0.01,
    w_max: float = 0.60,
    eps_ridge: float = 1e-6,
) -> np.ndarray:
    """
    Solve a long-only mean-variance optimisation with an L1 turnover penalty.

    Objective (maximise):
        mu' w  -  gamma * w' Sigma w  -  tau * ||w - w_prev||_1

    Subject to:
        1' w = 1,   w >= 0,   w <= w_max

    The covariance is symmetrised and ridge-regularised before passing to
    CVXPY to guarantee positive definiteness.  CLARABEL is used as the
    solver.  An equal-weight portfolio is returned on any solver failure.

    Parameters
    ----------
    mu        : (n,) expected return vector
    Sigma     : (n, n) covariance matrix
    w_prev    : (n,) previous portfolio weights
    gamma     : risk-aversion coefficient
    tau       : L1 turnover penalty coefficient
    w_max     : maximum weight per asset
    eps_ridge : ridge regularisation added to the diagonal of Sigma

    Returns
    -------
    w : (n,) optimal weights, non-negative, sum to 1, each <= w_max
    """
    n = len(mu)
    Sigma_reg = (Sigma + Sigma.T) / 2.0 + eps_ridge * np.eye(n)

    w = cp.Variable(n)
    objective = cp.Maximize(
        mu @ w
        - gamma * cp.quad_form(w, Sigma_reg)
        - tau * cp.norm1(w - w_prev)
    )
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= w_max,
    ]
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL, warm_start=True)
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            result = np.clip(w.value, 0.0, w_max)
            total = result.sum()
            if total < 1e-12:
                raise ValueError("Solver returned zero-weight portfolio.")
            return result / total
    except Exception as exc:  # noqa: BLE001
        logger.warning("MVO solver failed (%s); returning equal-weight fallback.", exc)

    return np.ones(n) / n


def compute_turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    """
    One-way portfolio turnover.

    TO = 0.5 * ||w_new - w_prev||_1

    A value of 1.0 means the entire portfolio was replaced; 0.0 means no
    change.

    Parameters
    ----------
    w_new  : (n,) new portfolio weights
    w_prev : (n,) previous portfolio weights

    Returns
    -------
    float in [0, 1]
    """
    return 0.5 * float(np.sum(np.abs(w_new - w_prev)))


def compute_n_eff(w: np.ndarray) -> float:
    """
    Effective number of portfolio positions (inverse Herfindahl index).

    N_eff = 1 / sum_i w_i^2

    Ranges from 1 (fully concentrated) to n (equally weighted).

    Parameters
    ----------
    w : (n,) portfolio weights (non-negative, should sum to 1)

    Returns
    -------
    float >= 1
    """
    denom = float(np.sum(w ** 2))
    if denom < 1e-12:
        return float(len(w))
    return 1.0 / denom
