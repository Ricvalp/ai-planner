"""Short receding-horizon episodes with actual execution between frozen targets."""

import json
from pathlib import Path
import time

import jax
import numpy as np

from .planning import plan
from .routes import execute_selected
from .samplers import COUNTER_NAMES


def run_episode(target, proposal, ratio, key, budget=16, replans=5, mode="exact", rho=0.0):
    """Plan, validate, execute a prefix, then use its actual successor and suffix.

    Each replan is a new target and a new bounded planner invocation. These
    trajectories must not be combined into a stationary MCMC trace. ``replans``
    is a finite demonstration horizon, not a guarantee of reaching the goal.
    """
    if replans < 1:
        raise ValueError("At least one physical replan is required.")
    original_target = target
    context_ids = [target.target_id]
    snapshot_positions = [target.context.snapshot.position]
    snapshot_velocities = [target.context.snapshot.velocity]
    cached_plans, selected_chunks, prefix_commands = [], [], []
    physical_prefixes, retained_suffixes, completed_tails = [], [], []
    selected_z, lookahead_goal_error, lookahead_success, validated_cost, fallbacks = [], [], [], [], []
    pool_z, pool_cost, pool_candidates = [], [], []
    counts = {name: 0 for name in COUNTER_NAMES}
    compilation_seconds, execution_seconds = 0.0, 0.0
    physical_execution_steps = 0
    physical_execution_seconds = 0.0
    key = jax.numpy.asarray(key)
    for _ in range(replans):
        result = plan(target, proposal, ratio, key, budget, mode=mode, rho=rho)
        k = target.context.executed_prefix
        cached_plans.append(np.asarray(target.context.cached_plan))
        selected_chunks.append(result["candidate"])
        prefix_commands.append(result["candidate"][:k])
        selected_z.append(result["z"])
        lookahead_goal_error.append(result["metrics"]["goal_error"])
        lookahead_success.append(result["metrics"]["success"])
        validated_cost.append(result["cost"])
        fallbacks.append(result["fallback"])
        pool_z.append(result["pool_z"])
        pool_cost.append(result["pool_cost"])
        pool_candidates.append(result["pool_candidates"])
        for name in COUNTER_NAMES:
            counts[name] += result["counters"][name]
        compilation_seconds += result["compilation_seconds"]
        execution_seconds += result["warmed_execution_seconds"]

        started = time.perf_counter()
        target, physical = execute_selected(target, result["candidate"])
        jax.block_until_ready(physical)
        physical_execution_seconds += time.perf_counter() - started
        physical_execution_steps += k * target.simulator.substeps
        physical_prefixes.append(np.asarray(physical.positions))
        retained_suffixes.append(np.asarray(target.context.retained_suffix))
        completed_tails.append(np.asarray(target.context.completed_tail))
        context_ids.append(target.target_id)
        snapshot_positions.append(target.context.snapshot.position)
        snapshot_velocities.append(target.context.snapshot.velocity)
        key = jax.numpy.asarray(result["key"])
        actual_goal_error = float(np.linalg.norm(np.asarray(target.context.snapshot.position) - target.context.goal))
        if target.context.snapshot.collided or actual_goal_error <= target.success_tolerance:
            break

    actual_goal_error = float(np.linalg.norm(np.asarray(target.context.snapshot.position) - target.context.goal))
    steps = len(selected_chunks)
    return {
        "summary": {
            "initial_target_name": original_target.name,
            "initial_target_id": original_target.target_id,
            "scenario": original_target.context.scenario,
            "mode": mode,
            "rho": float(rho),
            "requested_replans": replans,
            "actual_replans": steps,
            "oracle_budget_per_replan": budget,
            "actual_final_goal_error": actual_goal_error,
            "actual_collision": bool(target.context.snapshot.collided),
            "actual_goal_reached": bool(
                not target.context.snapshot.collided and actual_goal_error <= target.success_tolerance
            ),
            "fallback_count": int(sum(fallbacks)),
            "fallback_frequency": float(np.mean(fallbacks)),
            "lookahead_success_frequency": float(np.mean(lookahead_success)),
            "physical_execution_steps": physical_execution_steps,
            "physical_execution_wall_seconds": physical_execution_seconds,
            "total_simulator_steps": counts["simulator_steps"] + physical_execution_steps,
            "compilation_seconds": compilation_seconds,
            "warmed_planning_seconds": execution_seconds,
            "counters": counts,
            "interpretation": "Separate frozen targets at actual physical replans; not a stationary chain.",
        },
        "context_ids": np.asarray(context_ids),
        "snapshot_positions": np.asarray(snapshot_positions),
        "snapshot_velocities": np.asarray(snapshot_velocities),
        "old_cached_plans": np.asarray(cached_plans),
        "selected_chunks": np.asarray(selected_chunks),
        "executed_prefix_commands": np.asarray(prefix_commands),
        "physical_prefixes": np.asarray(physical_prefixes),
        "retained_suffixes": np.asarray(retained_suffixes),
        "completed_tails": np.asarray(completed_tails),
        "selected_z": np.asarray(selected_z),
        "lookahead_goal_error": np.asarray(lookahead_goal_error),
        "lookahead_success": np.asarray(lookahead_success),
        "validated_cost": np.asarray(validated_cost),
        "fallbacks": np.asarray(fallbacks),
        "pool_z": np.asarray(pool_z),
        "pool_cost": np.asarray(pool_cost),
        "pool_candidates": np.asarray(pool_candidates),
        "goal": np.asarray(target.context.goal),
        "obstacles": np.asarray(target.context.obstacles),
        "key": np.asarray(key),
    }


def save_episode(path, result):
    """Save raw replay arrays and an adjacent human-readable measured summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **{key: value for key, value in result.items() if key != "summary"})
    path.with_suffix(".json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    return path.with_suffix(".npz")


def main():
    """A small explicit development job; no training or parameter sweep."""
    import argparse
    from .proposals import PCNProposal
    from .routes import make_route_targets
    from .training import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--replans", type=int, default=4)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--beta", type=float, default=0.25)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    loaded = load_checkpoint(args.checkpoint)
    methods = (
        ("pcn", PCNProposal(), None, "exact", 0.0),
        ("learned_exact", loaded.proposal, loaded.ratio, "exact", 0.0),
        ("refresh_0.1", loaded.proposal, loaded.ratio, "learned", 0.1),
    )
    summaries = []
    family = make_route_targets(args.beta)
    context_creation_steps = sum(len(target.context.initial_positions) - 1
                                 for split in family.values() for target in split)
    for scenario_index, target in enumerate(family["test"][1:]):
        for method_index, (name, proposal, ratio, mode, rho) in enumerate(methods):
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 3 * scenario_index + method_index)
            episode = run_episode(target, proposal, ratio, key, args.budget, args.replans, mode, rho)
            episode["summary"]["method"] = name
            episode["summary"]["seed"] = args.seed
            episode["summary"]["checkpoint"] = str(args.checkpoint)
            save_episode(output / f"{target.name}__{name}", episode)
            summaries.append(episode["summary"])
            print(json.dumps({
                "scenario": target.context.scenario, "method": name,
                "actual_final_goal_error": episode["summary"]["actual_final_goal_error"],
                "actual_collision": episode["summary"]["actual_collision"],
                "actual_goal_reached": episode["summary"]["actual_goal_reached"],
                "fallback_count": episode["summary"]["fallback_count"],
            }), flush=True)
    (output / "summary.json").write_text(json.dumps({
        "configuration": vars(args), "episodes": summaries,
        "context_creation_steps": context_creation_steps,
        "total_simulator_steps_including_context_creation": (
            context_creation_steps + sum(row["total_simulator_steps"] for row in summaries)
        ),
        "limitation": "One development seed, four physical replans; unfinished episodes are inconclusive for eventual goal success.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
