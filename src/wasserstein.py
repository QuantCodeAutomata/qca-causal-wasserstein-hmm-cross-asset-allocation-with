"""
Gaussian 2-Wasserstein distance and persistent regime-template tracking.

W2^2 between two Gaussians is computed via the closed-form Bures metric.
The TemplateBank maintains G persistent regime templates with EMA updates,
matching HMM components to templates via nearest-W2-distance assignment.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSD matrix square root
# ---------------------------------------------------------------------------

def real_psd_sqrt(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute the PSD matrix square root of M via eigendecomposition.

    Symmetrises M, clips negative eigenvalues to *eps*, and returns
    Q @ diag(sqrt(clipped_eigvals)) @ Q^T.

    Parameters
    ----------
    M   : (D, D) symmetric positive semi-definite matrix
    eps : floor for eigenvalues before taking the square root

    Returns
    -------
    (D, D) PSD square-root matrix S such that S @ S ≈ M
    """
    M_sym = (M + M.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------------
# Gaussian 2-Wasserstein distance
# ---------------------------------------------------------------------------

def gaussian_w2_sq(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """
    Squared Gaussian 2-Wasserstein distance (Bures metric).

    W2^2(N(mu1,Sigma1), N(mu2,Sigma2)) =
        ||mu1 - mu2||^2
        + Tr(Sigma1) + Tr(Sigma2)
        - 2 * Tr( (Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{1/2} )

    Uses eigendecomposition-based PSD square roots (real_psd_sqrt) to
    handle near-singular covariances robustly.

    Parameters
    ----------
    mu1, mu2       : (D,) mean vectors
    Sigma1, Sigma2 : (D, D) covariance matrices

    Returns
    -------
    float ≥ 0
    """
    diff = mu1 - mu2
    term1 = float(diff @ diff)

    S2 = real_psd_sqrt(Sigma2)
    M = S2 @ Sigma1 @ S2
    M_sqrt = real_psd_sqrt(M)

    term2 = (
        float(np.trace(Sigma1))
        + float(np.trace(Sigma2))
        - 2.0 * float(np.trace(M_sqrt))
    )

    return max(0.0, term1 + term2)


# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------

class TemplateBank:
    """
    Maintains G persistent regime templates with exponential-moving-average
    (EMA) updates.

    Each template is a Gaussian characterised by a mean vector and a full
    covariance matrix.  Template identity is preserved across HMM re-fits via
    many-to-one nearest-W2 assignment: every HMM component is independently
    matched to its closest template in W2 distance.
    """

    def __init__(
        self,
        G: int,
        means: np.ndarray,
        covars: np.ndarray,
        eta: float = 0.05,
    ) -> None:
        """
        Parameters
        ----------
        G      : number of persistent templates
        means  : (G, D) initial template means
        covars : (G, D, D) initial template covariances
        eta    : EMA smoothing rate (0 < eta <= 1)
        """
        if means.shape[0] != G or covars.shape[0] != G:
            raise ValueError("means and covars first dimension must equal G.")
        self.G = G
        self.means = means.copy()    # (G, D)
        self.covars = covars.copy()  # (G, D, D)
        self.eta = eta

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign_components(
        self,
        comp_means: np.ndarray,
        comp_covars: np.ndarray,
    ) -> np.ndarray:
        """
        Many-to-one nearest-template assignment.

        For each HMM component k, independently find the template g that
        minimises W2^2(template_g, component_k).

        Parameters
        ----------
        comp_means  : (K, D) component means
        comp_covars : (K, D, D) component covariances

        Returns
        -------
        assignments : (K,) int array with values in {0, ..., G-1}
        """
        K = len(comp_means)
        assignments = np.empty(K, dtype=int)
        for k in range(K):
            dists = np.array([
                gaussian_w2_sq(
                    self.means[g], self.covars[g],
                    comp_means[k], comp_covars[k],
                )
                for g in range(self.G)
            ])
            assignments[k] = int(np.argmin(dists))
        return assignments

    # ------------------------------------------------------------------
    # Moment aggregation
    # ------------------------------------------------------------------

    def aggregate_template_moments(
        self,
        comp_means: np.ndarray,
        comp_covars: np.ndarray,
        comp_probs: np.ndarray,
        assignments: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate component moments into per-template moments using
        probability-weighted averaging.

        For template g, let I_g = {k : assignments[k] = g}.
        If I_g is non-empty:
            total_g  = sum_{k in I_g} comp_probs[k]
            w_k      = comp_probs[k] / total_g   (normalised weights)
            bar_mu_g = sum_{k in I_g} w_k * comp_means[k]
            bar_S_g  = sum_{k in I_g} w_k * comp_covars[k]
        If I_g is empty, bar_mu_g = self.means[g] and bar_S_g = self.covars[g]
        (template unchanged this step).

        Parameters
        ----------
        comp_means  : (K, D)
        comp_covars : (K, D, D)
        comp_probs  : (K,) predictive state probabilities (non-negative, sum ≤ 1)
        assignments : (K,) template indices

        Returns
        -------
        template_probs : (G,) summed predictive probability per template
        bar_means      : (G, D) weighted template means
        bar_covars     : (G, D, D) weighted template covariances
        """
        D = self.means.shape[1]
        template_probs = np.zeros(self.G)
        bar_means = self.means.copy()
        bar_covars = self.covars.copy()

        for g in range(self.G):
            mask = assignments == g
            if not np.any(mask):
                continue
            probs_g = comp_probs[mask]
            total = probs_g.sum()
            if total < 1e-300:
                continue
            template_probs[g] = total
            w = probs_g / total                    # normalised weights (sum = 1)
            bar_means[g] = (w[:, np.newaxis] * comp_means[mask]).sum(axis=0)
            bar_covars[g] = (w[:, np.newaxis, np.newaxis] * comp_covars[mask]).sum(axis=0)

        return template_probs, bar_means, bar_covars

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def update(
        self,
        bar_means: np.ndarray,
        bar_covars: np.ndarray,
        assignments: np.ndarray,
    ) -> None:
        """
        EMA update for every template that received at least one assignment.

        mu_g    <- (1 - eta) * mu_g    + eta * bar_mu_g
        Sigma_g <- (1 - eta) * Sigma_g + eta * bar_Sigma_g

        Templates with no assigned components are left unchanged.

        Parameters
        ----------
        bar_means  : (G, D) aggregated means from aggregate_template_moments
        bar_covars : (G, D, D) aggregated covariances
        assignments: (K,) template indices (used to determine which templates
                     received at least one component)
        """
        active = np.unique(assignments)
        for g in active:
            self.means[g] = (1.0 - self.eta) * self.means[g] + self.eta * bar_means[g]
            self.covars[g] = (1.0 - self.eta) * self.covars[g] + self.eta * bar_covars[g]

    # ------------------------------------------------------------------
    # Portfolio moments
    # ------------------------------------------------------------------

    def compute_portfolio_moments(
        self,
        template_probs: np.ndarray,
        n_assets: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute portfolio expected return and covariance as a template-
        probability-weighted sum over the first *n_assets* dimensions.

        mu_t    = sum_g p_g * mu_g[:n_assets]
        Sigma_t = sum_g p_g * Sigma_g[:n_assets, :n_assets]

        This is the simple weighted sum specified by the paper, NOT the full
        GMM covariance (which would add a cross-mean scatter term).

        Parameters
        ----------
        template_probs : (G,) predictive probabilities per template (normalised
                         internally; need not already sum to 1)
        n_assets       : number of assets (first dimensions of mean/cov to use)

        Returns
        -------
        mu    : (n_assets,) expected return vector
        Sigma : (n_assets, n_assets) covariance matrix
        """
        total = template_probs.sum()
        if total < 1e-300:
            probs = np.ones(self.G) / self.G
        else:
            probs = template_probs / total

        mu = np.zeros(n_assets)
        Sigma = np.zeros((n_assets, n_assets))
        for g in range(self.G):
            mu += probs[g] * self.means[g][:n_assets]
            Sigma += probs[g] * self.covars[g][:n_assets, :n_assets]

        return mu, Sigma
