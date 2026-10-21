"""Finite sums independently exercise the manuscript's probability claims."""
import numpy as np
import pytest

from ai_planner.diagnostics import (
    coarse_histogram_tv,
    discrete_log_ratio,
    effective_sample_size,
    finite_state_audit,
    stationary_distribution,
    total_variation,
    transition_matrix,
)


@pytest.fixture
def finite_problem():
    prior = np.array([0.12, 0.19, 0.31, 0.38])
    cost = np.array([0.1, 0.7, 1.2, 0.35])
    beta = 0.8
    p = prior * np.exp(-cost / beta)
    p /= p.sum()
    q = np.array([[0.51, 0.18, 0.20, 0.11],
                  [0.15, 0.47, 0.26, 0.12],
                  [0.17, 0.23, 0.35, 0.25],
                  [0.21, 0.16, 0.19, 0.44]])
    return p, q, prior, beta


def test_exact_mh_detailed_balance_and_rejections(finite_problem):
    p, q, _, _ = finite_problem
    ell = discrete_log_ratio(p, q)
    kernel = transition_matrix(q, ell)
    np.testing.assert_allclose(kernel.sum(axis=1), 1, atol=2e-15)
    np.testing.assert_allclose(p[:, None] * kernel, p[None, :] * kernel.T, atol=2e-15)
    np.testing.assert_allclose(p @ kernel, p, atol=2e-15)
    np.testing.assert_allclose(stationary_distribution(kernel), p, atol=2e-14)
    assert np.any(np.diag(kernel) > np.diag(q))
    # Removing repeated states changes the invariant law to the jump-chain law.
    jump = kernel.copy()
    np.fill_diagonal(jump, 0)
    jump /= jump.sum(axis=1, keepdims=True)
    assert total_variation(stationary_distribution(jump), p) > 0.02


@pytest.mark.parametrize("error_scale", [0.0, 0.2, 2.0, -3.0])
def test_accepted_flow_refresh_and_finite_step_bounds(finite_problem, error_scale):
    p, q, prior, beta = finite_problem
    potential = error_scale * np.array([-0.7, 0.1, 0.8, -0.2])
    d = discrete_log_ratio(p, q) + potential[None, :] - potential[:, None]
    audit = finite_state_audit(p, q, d, prior, beta=beta, j_max=1.2,
                               rhos=[0, 0.05, 0.3, 1], steps=[0, 1, 4, 25, 80])
    assert audit["accepted_flow_identity_error"] < 2e-14
    assert audit["L_bal"] >= -1e-14
    assert audit["one_step_tv"] <= audit["flow_bound"] + 1e-14
    assert audit["one_step_tv"] <= audit["epsilon_a"] + 1e-14
    assert audit["epsilon_a"] <= audit["ratio_bound"] + 1e-14
    for row in audit["refresh"]:
        if row["rho"] == 0:
            assert row["stationary_bound"] is None
            continue
        assert row["stationary_tv"] <= row["stationary_bound"] + 1e-13
        for finite in row["finite_steps"]:
            assert finite["tv"] <= finite["bound"] + 1e-13
    assert audit["refresh"][-1]["stationary_tv"] < 1e-13


def test_identity_kernel_failure_and_local_movement(finite_problem):
    p, _, _, _ = finite_problem
    wrong = np.array([0.6, 0.2, 0.1, 0.1])
    defects = []
    for jump_probability in [0.1, 0.01, 0.001]:
        kernel = (1 - jump_probability) * np.eye(4) + jump_probability * wrong
        np.testing.assert_allclose(stationary_distribution(kernel), wrong, atol=5e-14)
        defects.append(total_variation(p @ kernel, p))
    np.testing.assert_allclose(np.array(defects[1:]) / defects[:-1], 0.1, atol=1e-13)
    # Exact reversible near-identity proposals can have acceptance 1 while
    # bounded accepted movement tends to zero.
    tiny_q = 0.9999 * np.eye(4) + 0.0001 * p[None, :]
    acceptance = np.exp(np.minimum(0, discrete_log_ratio(p, tiny_q)))
    assert np.sum(p[:, None] * tiny_q * acceptance) > 1 - 1e-13
    movement = 1 - np.eye(4)
    assert np.sum(p[:, None] * tiny_q * acceptance * movement) < 1e-4


def test_rejects_nonantisymmetric_logits_and_invalid_minorization(finite_problem):
    p, q, prior, beta = finite_problem
    with pytest.raises(ValueError, match="antisymmetric"):
        finite_state_audit(p, q, np.ones_like(q), prior, beta=beta, j_max=1.2)
    with pytest.raises(ValueError, match="minorization"):
        finite_state_audit(p, q, discrete_log_ratio(p, q), prior, beta=beta, j_max=0)


def test_ess_and_coarse_tv_interfaces():
    rng = np.random.default_rng(42)
    independent = rng.normal(size=(3, 1000))
    sticky = np.repeat(independent[:, :100], 10, axis=1)
    assert effective_sample_size(independent) > effective_sample_size(sticky) * 3
    assert effective_sample_size(np.ones(100)) == 0
    axis = np.linspace(-0.9, 0.9, 10)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1)
    assert coarse_histogram_tv(grid, grid, np.ones((10, 10)), bins=5) == 0
