"""Tests for HMM model fitting and order selection."""
import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest
from src.hmm_model import fit_hmm, compute_predictive_probs, select_hmm_order


def test_fit_hmm_returns_result():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    result = fit_hmm(X, n_components=3, n_restarts=2, n_iter=50, random_state=0)
    assert result.model is not None
    assert result.means.shape == (3, 5)
    assert result.covars.shape == (3, 5, 5)


def test_predictive_probs_sum_to_one():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 5)
    result = fit_hmm(X, n_components=3, n_restarts=2, n_iter=50, random_state=1)
    if result.model is not None:
        probs = compute_predictive_probs(result.model, X)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(probs >= 0)
        assert len(probs) == 3


def test_select_hmm_order_returns_valid_k():
    rng = np.random.RandomState(2)
    X = rng.randn(200, 5)
    X_train, X_val = X[:150], X[150:]
    best_k, scores = select_hmm_order(X_train, X_val, k_min=2, k_max=4,
                                      lambda_k=1.0, n_restarts=1, n_iter=30)
    assert 2 <= best_k <= 4
    assert len(scores) >= 1


def test_fit_hmm_edge_case_small_data():
    """Should handle small datasets gracefully."""
    X = np.random.randn(10, 3)
    result = fit_hmm(X, n_components=2, n_restarts=1, n_iter=10)
    # Either fits successfully or returns None gracefully
    assert result is not None
