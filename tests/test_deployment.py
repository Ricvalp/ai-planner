import jax
import numpy as np

from ai_planner.deployment import run_episode, save_episode
from ai_planner.envs.point_mass import Snapshot, simulate
from ai_planner.proposals import PCNProposal
from ai_planner.routes import align_complete, make_route_targets


def test_physical_replanning_is_causal_and_has_distinct_target_ids(tmp_path):
    target = make_route_targets()["test"][2]
    result = run_episode(target, PCNProposal(), None, jax.random.PRNGKey(13), budget=4, replans=2)
    summary = result["summary"]
    assert summary["actual_replans"] == 2
    assert summary["counters"]["exact_cost_evaluations"] == 8
    assert summary["physical_execution_steps"] == 2 * 2 * 9
    assert summary["total_simulator_steps"] == 8 * 180 + 2 * 2 * 9
    assert len(set(result["context_ids"])) == 3
    for i in range(2):
        snapshot = Snapshot(tuple(result["snapshot_positions"][i]), tuple(result["snapshot_velocities"][i]))
        actual = simulate(snapshot, result["executed_prefix_commands"][i],
                          target.context.obstacles, target.simulator)
        np.testing.assert_allclose(actual.positions, result["physical_prefixes"][i], atol=1e-14)
        np.testing.assert_allclose(actual.final_state.position, result["snapshot_positions"][i + 1], atol=1e-14)
        np.testing.assert_array_equal(result["retained_suffixes"][i], result["selected_chunks"][i, 2:])
        completed = align_complete(result["selected_chunks"][i], 2)
        np.testing.assert_array_equal(result["completed_tails"][i], completed[-2:])
        if i < 1:
            np.testing.assert_array_equal(result["old_cached_plans"][i + 1], completed)
    saved = save_episode(tmp_path / "episode", result)
    with np.load(saved) as data:
        np.testing.assert_array_equal(data["physical_prefixes"], result["physical_prefixes"])
