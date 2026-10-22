"""Tests for MVO portfolio optimization."""
import sys
sys.path.insert(0, '/workspace/project')
import numpy as np
import pytest
from src.portfolio import solve_mvo, compute_turnover, compute_n_eff


def test_weights_sum_to_one():
    mu = np.array([0.001, 0.0005, 0.0008, 0.0002, 0.0003])
    Sigma = np.eye(5) * 0.001
    w_prev = np.ones(5) / 5
    w = solve_mvo(mu, Sigma, w_prev)
    assert abs(w.sum() - 1.0) < 1e-4


def test_weights_nonnegative():
    mu = np.array([0.001, 0.0005, 0.0008, 0.0002, 0.0003])
    Sigma = np.eye(5) * 0.001
    w_prev = np.ones(5) / 5
    w = solve_mvo(mu, Sigma, w_prev)
    assert np.all(w >= -1e-6)


def test_weights_max_constraint():
    mu = np.array([0.01, 0.0, 0.0, 0.0, 0.0])
    Sigma = np.eye(5) * 1e-4
    w_prev = np.ones(5) / 5
    w = solve_mvo(mu, Sigma, w_prev, gamma=0.01, tau=0.0, w_max=0.6)
    assert np.all(w <= 0.6 + 1e-4)


def test_turnover_zero_for_same_weights():
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert compute_turnover(w, w) == 0.0


def test_turnover_formula():
    w_new = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    w_old = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    expected = 0.5 * np.sum(np.abs(w_new - w_old))
    assert abs(compute_turnover(w_new, w_old) - expected) < 1e-10


def test_n_eff_equal_weight():
    w = np.ones(5) / 5
    assert abs(compute_n_eff(w) - 5.0) < 1e-6


def test_n_eff_concentrated():
    w = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert abs(compute_n_eff(w) - 1.0) < 1e-6
