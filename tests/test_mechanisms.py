from dataclasses import replace

import jax
import numpy as np

from ai_planner.decoders import IdentityDecoder
from ai_planner.proposals import RWProposal
from ai_planner.mechanisms import reconstruct_rejection_count
from ai_planner.samplers import initialize, rejection_sample, run_chain
from ai_planner.targets import make_analytic_target, quadrature


def test_identity_quadrature_uses_gaussian_prior_and_changed_target():
    target = replace(make_analytic_target("two"), decoder=IdentityDecoder())
    # At beta=beta0 the Gaussian expectation has a closed-form product integral.
    c = target.context
    analytic_normalizer = c.eps_floor
    for center, covariance, weight in zip(c.centers, c.covariances, c.weights):
        s = np.asarray(covariance)
        m = np.asarray(center)
        integral = np.sqrt(np.linalg.det(s) / np.linalg.det(s + np.eye(2))) * np.exp(
            -.5 * m @ np.linalg.solve(s + np.eye(2), m))
        analytic_normalizer += (1 - c.eps_floor) * weight * integral
    grid = quadrature(target, 128)
    np.testing.assert_allclose(grid["normalizer"], analytic_normalizer, atol=2e-6)
    np.testing.assert_allclose(grid["region_masses"], [.5, .5], atol=1e-12)
    assert abs(grid["normalizer"] - quadrature(make_analytic_target("two"), 128)["normalizer"]) > .01


def test_prior_mode_draws_prior_independently_of_supplied_proposal():
    target = make_analytic_target("constant")
    chain = initialize(target, jax.random.PRNGKey(32), z=np.array([80., -70.]))
    _, trace, counters = run_chain(target, RWProposal(.00001), chain, 32, mode="prior")
    assert np.max(np.abs(trace["z"])) < 5
    assert np.all(trace["accepted"])
    assert counters["exact_cost_evaluations"] == 0


def test_reconstruct_rejection_batches_without_reevaluating_costs():
    target = make_analytic_target("two")
    key = jax.random.PRNGKey(71017)
    samples, counters, _ = rejection_sample(target, 80, key)
    assert reconstruct_rejection_count(samples[-1], key) == counters["exact_cost_evaluations"]
