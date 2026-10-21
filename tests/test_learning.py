from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from ai_planner.acceptance import RatioModel, acceptance_probability, exact_log_ratio
from ai_planner.proposals import NeuralProposal, PCNProposal, RWProposal, normal_log_prob
from ai_planner.targets import make_analytic_target
from ai_planner.training import (TrainingConfig, bounded_movement, load_checkpoint,
                                 ratio_labels, score_function_loss, train)


def test_gaussian_proposals_match_scipy_and_initialize_pcn():
    z, zp = jnp.array([0.2, -1.1, 0.7]), jnp.array([-0.7, 0.8, 1.2])
    context = jnp.array([0.3, -0.2])
    p = NeuralProposal.create(jax.random.PRNGKey(1), 3, 2, hidden_dims=(8, 8))
    pcn = PCNProposal(.65)
    np.testing.assert_allclose(p.log_prob(zp, z, context), pcn.log_prob(zp, z), atol=1e-12)
    np.testing.assert_allclose(normal_log_prob(z), norm.logpdf(z).sum(), atol=1e-12)
    np.testing.assert_allclose(RWProposal(.4).log_prob(zp, z),
                               norm.logpdf(zp, loc=z, scale=.4).sum(), atol=1e-12)
    np.testing.assert_allclose(p.log_prob(zp, z, context),
                               norm.logpdf(zp, loc=np.sqrt(1-.65**2)*z,
                                           scale=.65).sum(), atol=1e-12)
    # No clipping: an unconstrained Gaussian must retain full real support.
    draws = p.sample(jax.random.PRNGKey(3), jnp.zeros((4096, 3)), context)
    assert np.max(np.abs(draws)) > 1.8


def test_ratio_antisymmetry_acceptance_and_exact_label_refresh():
    z, zp, context = jnp.array([.4, -.7]), jnp.array([-.8, .1]), jnp.array([1., 2.])
    ratio = RatioModel.create(jax.random.PRNGKey(0), 2, 2, hidden_dims=(8, 8))
    np.testing.assert_allclose(ratio(z, zp, context), -ratio(zp, z, context), atol=1e-14)
    assert ratio(z, z, context) == 0
    a = acceptance_probability(jnp.array([-1000., -1., 0., 1., 1000.]))
    assert np.all((a >= 0) & (a <= 1))
    p = NeuralProposal.create(jax.random.PRNGKey(1), 2, 2, hidden_dims=(8,))
    before = ratio_labels(p, p.params, z, zp, 1.2, .5, .7, context)
    mutated = jax.tree.map(lambda v: v + .06, p.params)
    after = ratio_labels(p, mutated, z, zp, 1.2, .5, .7, context)
    assert abs(before - after) > .001
    np.testing.assert_allclose(after, -exact_log_ratio(
        zp, z, .5, 1.2, .7, p, context, mutated), atol=1e-12)
    np.testing.assert_allclose(before, -(0.5-1.2)/.7, atol=1e-12)


def test_score_function_matches_exact_discrete_sum():
    reward = jnp.array([.1, .4, .9])
    logits = jnp.array([-.6, .3, .1])
    exact = jax.grad(lambda t: jnp.sum(jax.nn.softmax(t) * reward))(logits)
    weights = jax.lax.stop_gradient(jax.nn.softmax(logits))
    baseline = .32
    score = jax.grad(lambda t: jnp.sum(
        weights * (reward - baseline) * jax.nn.log_softmax(t)))(logits)
    np.testing.assert_allclose(score, exact, atol=1e-14)
    eps = 1e-5
    finite = [(np.dot(jax.nn.softmax(logits + eps*jnp.eye(3)[i]), reward)
               - np.dot(jax.nn.softmax(logits - eps*jnp.eye(3)[i]), reward)) / (2*eps)
              for i in range(3)]
    np.testing.assert_allclose(score, finite, atol=1e-10)


def test_gaussian_score_stops_sample_reward_and_baseline_gradients():
    class ScalarMean:
        def sample(self, key, source, context, params):
            return params + .7 * jax.random.normal(key, source.shape, dtype=jnp.float64)
        def log_prob(self, destination, source, context, params):
            return normal_log_prob(destination, params, .7)

    proposal = ScalarMean()
    key = jax.random.PRNGKey(3)
    z = jnp.zeros((65536, 1))
    # The reward and baseline deliberately depend on theta in this test.
    # Both paths must be stopped by the actual production training helper.
    def loss(theta):
        return score_function_loss(proposal, theta, key, z, jnp.zeros(0),
                                   lambda x, y: y[:, 0]**2 + theta,
                                   theta)[0]
    got = -float(jax.grad(loss)(jnp.array(.6)))
    # d/d mu E[N(mu,.7)^2] = 2 mu; the added theta is frozen.
    np.testing.assert_allclose(got, 1.2, atol=.05)
    sample = proposal.sample(key, z, None, .6)[:, 0]
    manual = np.mean(np.asarray(sample**2 * (sample-.6)/.7**2))
    np.testing.assert_allclose(got, manual, atol=1e-12)


def test_movement_ignores_unused_noise_coordinates():
    z = jnp.array([.1, .2, .3, .4, .5, .6])
    zp = z.at[2:].add(9)
    # A valid six-dimensional decoder need not use all its inputs.
    decoder = lambda x: x[:2]
    assert bounded_movement(decoder(z), decoder(zp), .5) == 0
    assert 0 < bounded_movement(decoder(z), decoder(z).at[0].add(.2), .5) < 1


def test_checkpoint_resume_reproduces_next_round_and_frozen_decoder(tmp_path):
    target = make_analytic_target("constant")
    target_id, decoder_version = target.target_id, target.decoder.version
    config = TrainingConfig(rounds=2, ratio_steps=3, proposal_steps=2, batch_size=8,
                            target_samples=16, pairs_per_context=32,
                            calibration_samples=16, hidden_dims=(8, 8))
    complete = train([target], config, tmp_path / "complete")
    first = train([target], replace(config, rounds=1), tmp_path / "resumed")
    loaded = load_checkpoint(first.checkpoint_path)
    np.testing.assert_array_equal(loaded.state.key, first.state.key)
    source = jnp.array([.1, -.1])
    np.testing.assert_array_equal(
        loaded.proposal.sample(loaded.state.key, source, target.context_features),
        first.proposal.sample(first.state.key, source, target.context_features))
    resumed = train([target], config, tmp_path / "resumed", resume=first.checkpoint_path)
    for a, b in zip(jax.tree.leaves(complete.state), jax.tree.leaves(resumed.state)):
        np.testing.assert_array_equal(a, b)
    assert complete.counters == resumed.counters
    assert complete.history == resumed.history
    assert target.target_id == target_id and target.decoder.version == decoder_version
    meta = json.loads((tmp_path / "complete" / "checkpoint.json").read_text())
    assert meta["proposal_version"] == meta["ratio_fitted_to_proposal_version"]
    assert meta["completed_rounds"] == 2
    assert meta["counters"]["training_oracle_evaluations"] > 0
    assert all(p["method"] == "independent_exact_rejection" for p in meta["provenance"])
