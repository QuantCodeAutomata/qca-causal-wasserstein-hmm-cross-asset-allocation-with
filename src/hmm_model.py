"""
Gaussian HMM fitting with model-order selection via penalised predictive
log-likelihood, plus helpers for one-step-ahead forecasting.

All forward-algorithm computations are carried out in log space (logsumexp)
to prevent numerical underflow on long sequences.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

# Regularisation ridge added to every HMM component covariance to prevent
# numerical singularity (Custom — scipy multivariate_normal requires strict PSD).
_COV_RIDGE = 1e-6

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class HMMResult:
    """Container for a fitted Gaussian HMM and its summary statistics."""

    def __init__(self) -> None:
        self.model: Optional[GaussianHMM] = None
        self.n_components: int = 0
        self.means: np.ndarray = None       # (K, D)
        self.covars: np.ndarray = None      # (K, D, D)
        self.transmat: np.ndarray = None    # (K, K)
        self.startprob: np.ndarray = None   # (K,)
        self.log_likelihood: float = -np.inf


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_hmm(
    X: np.ndarray,
    n_components: int,
    n_restarts: int = 3,
    n_iter: int = 100,
    tol: float = 1e-3,
    random_state: Optional[int] = None,
) -> HMMResult:
    """
    Fit a Gaussian HMM with full covariance using multiple random restarts.

    Parameters
    ----------
    X            : (T, D) observation array
    n_components : number of hidden states K
    n_restarts   : number of random-restart attempts
    n_iter       : EM maximum iterations per attempt
    tol          : EM convergence tolerance
    random_state : base random seed (None → use attempt index as seed)

    Returns
    -------
    HMMResult holding the best-fitting model by total log-likelihood.
    If all restarts fail, the returned HMMResult has log_likelihood = -inf.
    """
    best = HMMResult()

    for i in range(n_restarts):
        seed = (random_state + i) if random_state is not None else i
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=n_iter,
                tol=tol,
                random_state=seed,
            )
            model.fit(X)
            # model.score returns average log-likelihood per observation
            ll = model.score(X) * len(X)
            if ll > best.log_likelihood:
                # Regularise all component covariances before storing so that
                # Wasserstein distance, template update, and logpdf calls are
                # numerically safe throughout the pipeline.
                reg_covars = np.stack(
                    [_regularise_cov(model.covars_[k]) for k in range(n_components)]
                )
                # Write regularised covars back into the model object so that
                # any subsequent call to _log_emissions via model.covars_ also
                # sees PSD matrices.
                model.covars_ = reg_covars
                best.model = model
                best.n_components = n_components
                best.means = model.means_.copy()
                best.covars = reg_covars
                best.transmat = model.transmat_.copy()
                best.startprob = model.startprob_.copy()
                best.log_likelihood = ll
        except Exception as exc:  # noqa: BLE001
            logger.debug("HMM fit restart %d/%d failed: %s", i + 1, n_restarts, exc)
            continue

    return best


# ---------------------------------------------------------------------------
# Forward algorithm (log-space, internal helper)
# ---------------------------------------------------------------------------

def _regularise_cov(cov: np.ndarray, ridge: float = _COV_RIDGE) -> np.ndarray:
    """
    Return a symmetrised, ridge-regularised copy of *cov* that is guaranteed PSD.

    Steps:
      1. Symmetrise: cov = (cov + cov.T) / 2
      2. Clip negative eigenvalues to 0 (projection onto PSD cone)
      3. Add ridge * I
    """
    cov = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    cov = (eigvecs * eigvals) @ eigvecs.T
    cov += ridge * np.eye(cov.shape[0])
    return cov


def _log_emissions(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Compute log-emission probabilities for every (t, k) pair.

    Covariance matrices are regularised via :func:`_regularise_cov` before the
    multivariate Gaussian pdf is evaluated, preventing ``LinAlgError`` from
    near-singular matrices that can appear during HMM-EM convergence.

    Returns
    -------
    log_emiss : (T, K) array
    """
    T = len(X)
    K = model.n_components
    log_emiss = np.empty((T, K))
    for k in range(K):
        cov_k = _regularise_cov(model.covars_[k])
        log_emiss[:, k] = multivariate_normal.logpdf(
            X, mean=model.means_[k], cov=cov_k
        )
    return log_emiss


def _forward_log(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Run the (normalised) forward algorithm in log space on sequence X.

    Returns
    -------
    log_alpha : (T, K) array of normalised log filtered state probabilities
                log_alpha[t, k] = log P(z_t = k | x_{0:t})  (up to normalisation)
    """
    K = model.n_components
    T = len(X)
    log_emiss = _log_emissions(model, X)
    log_start = np.log(np.clip(model.startprob_, 1e-300, None))
    log_trans = np.log(np.clip(model.transmat_, 1e-300, None))

    log_alpha = np.empty((T, K))
    log_alpha[0] = log_start + log_emiss[0]
    log_alpha[0] -= logsumexp(log_alpha[0])   # normalise

    for t in range(1, T):
        # log P(z_t = j | x_{0:t-1}) = logsumexp_i [ log_alpha[t-1,i] + log_trans[i,j] ]
        log_pred = logsumexp(
            log_alpha[t - 1, :, np.newaxis] + log_trans, axis=0
        )                                      # shape (K,)
        log_alpha[t] = log_pred + log_emiss[t]
        log_alpha[t] -= logsumexp(log_alpha[t])

    return log_alpha


# ---------------------------------------------------------------------------
# Predictive log-likelihood (one-step-ahead, causal)
# ---------------------------------------------------------------------------

def compute_predictive_loglik(model: GaussianHMM, X_val: np.ndarray) -> float:
    """
    Compute the sum of one-step-ahead predictive log-likelihoods on X_val.

    For t = 1, ..., T-1:
      - alpha_{t-1} = filtered state distribution given x_{0:t-1}
      - pred_t      = alpha_{t-1} @ transmat               (predictive probs)
      - log p(x_t | x_{0:t-1}) = log sum_k pred_t[k] * p(x_t | z_t=k)

    Uses the log-space forward algorithm (scipy.stats.multivariate_normal.logpdf
    for emissions, scipy.special.logsumexp for numerical stability).

    Parameters
    ----------
    model : fitted GaussianHMM
    X_val : (T, D) validation observations

    Returns
    -------
    float — summed predictive log-likelihood over t = 1 .. T-1
    """
    T = len(X_val)
    if T < 2:
        return 0.0

    K = model.n_components
    log_emiss = _log_emissions(model, X_val)
    log_start = np.log(np.clip(model.startprob_, 1e-300, None))
    log_trans = np.log(np.clip(model.transmat_, 1e-300, None))

    # Initialise with x_0
    log_alpha = log_start + log_emiss[0]
    log_alpha -= logsumexp(log_alpha)

    total = 0.0
    for t in range(1, T):
        # One-step-ahead predictive distribution (in log space)
        log_pred = logsumexp(
            log_alpha[:, np.newaxis] + log_trans, axis=0
        )                                          # (K,)
        # Predictive log p(x_t | x_{0:t-1})
        total += float(logsumexp(log_pred + log_emiss[t]))
        # Update filtered distribution
        log_alpha = log_pred + log_emiss[t]
        log_alpha -= logsumexp(log_alpha)

    return total


# ---------------------------------------------------------------------------
# Model order selection
# ---------------------------------------------------------------------------

def select_hmm_order(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k_min: int,
    k_max: int,
    lambda_k: float = 1.0,
    n_restarts: int = 3,
    n_iter: int = 100,
    tol: float = 1e-3,
) -> Tuple[int, Dict[int, float]]:
    """
    Select HMM order by penalised predictive log-likelihood.

    For each candidate K in [k_min, k_max]:
      1. Fit GaussianHMM(K) on X_train.
      2. Evaluate one-step-ahead predictive LL on X_val.
      3. score(K) = PredLL(K) - lambda_k * complexity(K)
         complexity(K) = K*D + K*D*(D+1)/2 + K*K
                         (means   + covariances       + transition matrix)

    Parameters
    ----------
    X_train   : (N_train, D) training observations
    X_val     : (N_val, D)   validation observations
    k_min     : minimum candidate state count
    k_max     : maximum candidate state count
    lambda_k  : complexity penalty weight
    n_restarts: random restarts per candidate
    n_iter    : EM max iterations
    tol       : EM convergence tolerance

    Returns
    -------
    (best_K, scores) where scores is a dict {K: penalised score}
    """
    D = X_train.shape[1]
    scores: Dict[int, float] = {}

    for K in range(k_min, k_max + 1):
        result = fit_hmm(X_train, K, n_restarts=n_restarts, n_iter=n_iter, tol=tol)
        if result.model is None:
            logger.warning("HMM order %d: all restarts failed; skipping.", K)
            continue

        pred_ll = compute_predictive_loglik(result.model, X_val)
        complexity = K * D + K * D * (D + 1) // 2 + K * K
        score = pred_ll - lambda_k * complexity
        scores[K] = score
        logger.debug(
            "K=%d  PredLL=%.2f  complexity=%d  score=%.2f", K, pred_ll, complexity, score
        )

    if not scores:
        raise RuntimeError("HMM order selection failed for all candidate K values.")

    best_K = max(scores, key=scores.__getitem__)
    return best_K, scores


# ---------------------------------------------------------------------------
# One-step-ahead predictive state probabilities
# ---------------------------------------------------------------------------

def compute_predictive_probs(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Compute the one-step-ahead predictive state probabilities for position T.

    Given observations x_0, ..., x_{T-1} (all of X), returns
    P(z_T = k | x_0, ..., x_{T-1}) for k = 0, ..., K-1.

    Steps:
      1. Run forward algorithm on X[:-1] (observations 0 .. T-2).
      2. Predictive: pred = exp(log_alpha_{T-2}) @ transmat, then normalise.

    If X has only one observation, returns startprob @ transmat (normalised).

    Parameters
    ----------
    model : fitted GaussianHMM
    X     : (T, D) observation sequence

    Returns
    -------
    pred : (K,) array of predictive state probabilities, sums to 1
    """
    K = model.n_components
    log_trans = np.log(np.clip(model.transmat_, 1e-300, None))

    if len(X) == 1:
        # Only x_0 observed; pred from start distribution
        log_start = np.log(np.clip(model.startprob_, 1e-300, None))
        log_pred = logsumexp(log_start[:, np.newaxis] + log_trans, axis=0)
    else:
        log_alpha = _forward_log(model, X[:-1])   # shape (T-1, K)
        log_pred = logsumexp(
            log_alpha[-1, :, np.newaxis] + log_trans, axis=0
        )                                          # (K,)

    pred = np.exp(log_pred)
    pred = np.clip(pred, 0.0, None)
    pred /= pred.sum()
    return pred
