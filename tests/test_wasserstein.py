"""Tests for Wasserstein distance and template tracking."""
import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest
from src.wasserstein import gaussian_w2_sq, real_psd_sqrt, TemplateBank


def test_w2_identical_distributions():
    """W2 distance between identical Gaussians should be 0."""
    mu = np.array([0.01, 0.02, 0.0])
    Sigma = np.eye(3) * 0.01
    dist = gaussian_w2_sq(mu, Sigma, mu, Sigma)
    assert dist < 1e-8, f'Expected ~0, got {dist}'


def test_w2_mean_only_difference():
    """W2^2 between same-covariance Gaussians equals squared mean diff."""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([1.0, 0.0])
    Sigma = np.eye(2)
    dist = gaussian_w2_sq(mu1, Sigma, mu1, Sigma)
    assert dist < 1e-8
    dist12 = gaussian_w2_sq(mu1, Sigma, mu2, Sigma)
    assert abs(dist12 - 1.0) < 1e-6, f'Expected 1.0, got {dist12}'


def test_w2_nonnegative():
    """W2 distance is always non-negative."""
    rng = np.random.RandomState(0)
    for _ in range(10):
        D = 5
        mu1, mu2 = rng.randn(D), rng.randn(D)
        A = rng.randn(D, D); Sigma1 = A @ A.T + np.eye(D) * 0.1
        B = rng.randn(D, D); Sigma2 = B @ B.T + np.eye(D) * 0.1
        assert gaussian_w2_sq(mu1, Sigma1, mu2, Sigma2) >= 0


def test_real_psd_sqrt_positive_definite():
    """Square root of PD matrix satisfies sqrtA @ sqrtA == A."""
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    sqrtA = real_psd_sqrt(A)
    assert np.allclose(sqrtA @ sqrtA, A, atol=1e-6)


def test_template_bank_assignment():
    """Each component is assigned to exactly one template."""
    G, D = 4, 5
    rng = np.random.RandomState(1)
    tmpl_means = rng.randn(G, D)
    tmpl_covars = np.stack([np.eye(D) * 0.1 for _ in range(G)])
    bank = TemplateBank(G, tmpl_means, tmpl_covars, eta=0.05)
    K = 3
    comp_means = rng.randn(K, D)
    comp_covars = np.stack([np.eye(D) * 0.1 for _ in range(K)])
    assignments = bank.assign_components(comp_means, comp_covars)
    assert len(assignments) == K
    assert all(0 <= a < G for a in assignments)


def test_template_probs_sum_to_one():
    """Template probabilities must sum to 1."""
    G, D, K = 4, 5, 3
    rng = np.random.RandomState(2)
    bank = TemplateBank(G, rng.randn(G, D),
                        np.stack([np.eye(D) for _ in range(G)]), eta=0.05)
    comp_means = rng.randn(K, D)
    comp_covars = np.stack([np.eye(D) for _ in range(K)])
    comp_probs = np.array([0.5, 0.3, 0.2])
    assignments = bank.assign_components(comp_means, comp_covars)
    tmpl_probs, _, _ = bank.aggregate_template_moments(
        comp_means, comp_covars, comp_probs, assignments
    )
    assert abs(tmpl_probs.sum() - 1.0) < 1e-6, f'Sum={tmpl_probs.sum()}'
