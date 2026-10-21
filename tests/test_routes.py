"""Deterministic physical and causal route tests, not large statistical audits."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ai_planner.envs.point_mass import (
    SimulatorConfig,
    Snapshot,
    segment_disk_collision,
    simulate,
)
from ai_planner.routes import (
    RouteTarget,
    align_complete,
    execute_selected,
    make_route_context,
    make_route_targets,
    prefix_then_rollout,
    support_grid,
    warm_started_cem,
)


def test_decode_cost_deterministic_and_snapshot_unchanged():
    target = make_route_targets()["test"][0]
    original = target.context
    z = jnp.asarray([-0.9, 0.2])
    candidate = target.decoder.decode(target.context, z)
    np.testing.assert_array_equal(candidate, target.decoder.decode(target.context, z))
    np.testing.assert_array_equal(target.cost(candidate), target.cost(candidate))
    assert target.context == original
    assert candidate.shape == (20, 2)
    assert np.max(np.abs(candidate)) <= target.simulator.command_bound


def test_swept_collision_catches_disk_crossing_and_tangent():
    start, end, center = map(jnp.asarray, [(-1.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
    assert segment_disk_collision(start, end, center, 0.1)
    assert segment_disk_collision(start + jnp.array([0, 0.1]), end + jnp.array([0, 0.1]), center, 0.1)
    assert not segment_disk_collision(start + jnp.array([0, 0.11]), end + jnp.array([0, 0.11]), center, 0.1)
    # One deliberately large physical step crosses the entire obstacle.
    config = SimulatorConfig(dt=1.0, substeps=1, kp=1.0, kd=0.01, max_speed=10, robot_radius=0)
    path = simulate(Snapshot(position=(-1.0, 0.0)), jnp.asarray([[1.0, 0.0]]),
                    jnp.asarray([[0.0, 0.0, 0.1, 1.0]]), config)
    assert path.final_state.collided
    np.testing.assert_array_equal(path.final_state.position, (-1.0, 0.0))
    np.testing.assert_array_equal(path.final_state.velocity, (0.0, 0.0))


def test_free_space_pd_tracking_stable_and_collision_absorbing():
    config = SimulatorConfig()
    commands = jnp.tile(jnp.asarray([0.75, -0.4]), (20, 1))
    no_obstacles = jnp.asarray([[0.0, 0.0, 0.3, 0.0]])
    rollout = simulate(Snapshot(), commands, no_obstacles, config)
    assert not rollout.final_state.collided
    assert np.linalg.norm(np.asarray(rollout.final_state.position) - commands[-1]) < 1e-4
    assert np.linalg.norm(rollout.final_state.velocity) < 1e-3
    failed = simulate(Snapshot(collided=True), commands, no_obstacles, config)
    np.testing.assert_array_equal(failed.positions, np.tile([-0.95, 0.0], (181, 1)))


def test_simulator_batch_equals_scalar():
    target = make_route_targets()["test"][0]
    zs = jnp.asarray([[-1.1, 0.5], [1.0, -0.6], [0.0, 0.1]])
    chunks = jax.vmap(lambda z: target.decoder.decode(target.context, z))(zs)
    batched = jax.jit(jax.vmap(target.rollout))(chunks)
    for index, chunk in enumerate(chunks):
        scalar = target.rollout(chunk)
        np.testing.assert_allclose(batched.positions[index], scalar.positions, atol=1e-14)
        np.testing.assert_array_equal(batched.collisions[index], scalar.collisions)


def test_causal_initial_snapshot_and_execution_use_selected_plan():
    target = make_route_targets()["test"][0]
    context = target.context
    k = context.executed_prefix
    initial = simulate(context.initial_snapshot, jnp.asarray(context.selected_old_plan[:k]),
                       context.obstacles, target.simulator)
    np.testing.assert_array_equal(context.snapshot.position, initial.final_state.position)
    np.testing.assert_array_equal(context.snapshot.velocity, initial.final_state.velocity)
    selected = target.decoder.decode(context, jnp.asarray([-1.1, 0.2]))
    next_target, execution = execute_selected(target, selected)
    np.testing.assert_array_equal(next_target.context.snapshot.position, execution.final_state.position)
    np.testing.assert_array_equal(next_target.context.retained_suffix, selected[k:])
    np.testing.assert_array_equal(next_target.context.cached_plan,
                                  align_complete(selected, k, context.completion_decay))
    assert next_target.target_id != target.target_id
    assert next_target.context.cached_plan != context.cached_plan


def test_tail_completion_is_damped_selected_suffix():
    chunk = jnp.asarray([[-0.8, 0.0], [-0.4, 0.1], [0.0, 0.2], [0.4, 0.3]])
    completed = align_complete(chunk, 2, decay=0.5)
    np.testing.assert_allclose(completed, [[0, 0.2], [0.4, 0.3], [0.6, 0.35], [0.7, 0.375]])
    with pytest.raises(ValueError):
        align_complete(chunk, 2, decay=1.1)


def test_identical_prefix_different_suffix_continuations():
    target = make_route_targets()["test"][0]
    first = target.decoder.decode(target.context, jnp.asarray([1.1, 0.2]))
    second = target.decoder.decode(target.context, jnp.asarray([-1.1, -0.2]))
    second = second.at[: target.context.executed_prefix].set(first[: target.context.executed_prefix])
    first_result = prefix_then_rollout(target, first)
    second_result = prefix_then_rollout(target, second)
    np.testing.assert_array_equal(first_result.prefix_positions, second_result.prefix_positions)
    np.testing.assert_array_equal(first_result.prefix_state.position, second_result.prefix_state.position)
    np.testing.assert_array_equal(first_result.prefix_cost, second_result.prefix_cost)
    assert not np.allclose(first_result.continuation_positions, second_result.continuation_positions)
    discarded_a = prefix_then_rollout(target, first, preserve_suffix=False)
    discarded_b = prefix_then_rollout(target, second, preserve_suffix=False)
    np.testing.assert_array_equal(discarded_a.continuation_positions, discarded_b.continuation_positions)
    np.testing.assert_array_equal(discarded_a.cost, discarded_b.cost)
    assert first_result.simulator_steps == 2 * 3 * 9


def test_all_context_defining_fields_invalidate_target_hash():
    target = make_route_targets()["test"][0]
    alternatives = [
        replace(target, beta=target.beta * 1.1),
        replace(target, goal_weight=target.goal_weight + 0.01),
        replace(target, cost_version="changed"),
        replace(target, simulator=replace(target.simulator, kd=17.0)),
        replace(target, decoder=replace(target.decoder, cache_weight=0.2)),
        replace(target, context=replace(target.context, completion_decay=0.3)),
        replace(target, context=replace(target.context, continuation_version="changed")),
        replace(target, context=replace(target.context, snapshot=Snapshot(position=(-0.7, 0.1)))),
        replace(target, context=replace(target.context, goal=(0.9, 0.1))),
        replace(target, context=replace(target.context, obstacles=((0, 0, 0.3, 1), (0, 0.7, 0.5, 0)))),
    ]
    assert all(alternative.target_id != target.target_id for alternative in alternatives)
    assert replace(target).target_id == target.target_id


def test_support_inspection_has_good_bad_candidates_and_valid_alternative():
    target = RouteTarget(make_route_context("route_invalidation"))
    grid = support_grid(target, resolution=9)
    assert grid["success"].any()
    assert grid["collision"].any()
    assert not np.any(grid["success"] & (grid["coefficients"][:, 0] > 0))
    assert np.any(grid["success"] & (grid["coefficients"][:, 0] < 0))
    assert np.min(grid["cost"]) >= 0
    assert np.max(grid["cost"]) <= target.j_max
    assert grid["oracle_evaluations"] == 81
    assert grid["simulator_steps"] == 81 * 180
    assert np.isclose(grid["target_mass"].sum(), 1)
    assert target.metrics(jnp.asarray(target.context.cached_plan)).collided


def test_cheap_decoding_features_do_not_call_simulator(monkeypatch):
    import ai_planner.routes as routes
    target = make_route_targets()["test"][0]

    def forbidden(*args, **kwargs):
        raise AssertionError("Cheap features called simulator")

    monkeypatch.setattr(routes, "simulate", forbidden)
    candidate = target.decoder.decode(target.context, jnp.array([0.2, 0.8]))
    assert target.features(candidate).shape == (40,)
    assert target.context_features.shape == (25,)


def test_context_split_and_cem_accounting():
    family = make_route_targets()
    ids = [target.target_id for split in family.values() for target in split]
    assert len(ids) == len(set(ids)) == 9
    result = warm_started_cem(family["test"][0], jax.random.PRNGKey(29), budget=13, population=8)
    assert result["oracle_evaluations"] == 13
    assert result["simulator_steps"] == 13 * 180
    assert result["z"].shape == (13, 2)
    assert result["best_index"] is not None
    assert not result["collision"][result["best_index"]]
