"""Resume completed experiment artifacts after an orchestration interruption.

This recovery does not retrain or rerun saved chains. It reconstructs their
accounting from saved counters, completes missing mechanism/planner jobs, and
renders the finished run. Unsaved timings remain unknown.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import jax
import numpy as np

from .evaluation import _sum_counters, write_json
from .mechanisms import run_mechanisms
from .planning import plan
from .proposals import PCNProposal, RWProposal
from .samplers import COUNTER_NAMES
from .targets import analytic_splits
from .training import load_checkpoint


def _read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _audit_count(n, simulator_steps_per_cost=0):
    result = dict.fromkeys(COUNTER_NAMES, 0)
    result.update(decoder_calls=int(n), exact_cost_evaluations=int(n),
                  audit_evaluations=int(n), simulator_steps=int(n) * simulator_steps_per_cost)
    return result


def finish_run(run_dir):
    """Finish an interrupted analytic pilot from its existing verified artifacts."""
    output = Path(run_dir)
    config = _read(output / "config.json")
    metadata = _read(output / "metadata.json")
    if metadata["kind"] != "analytic":
        raise ValueError("This recovery currently supports interrupted analytic pilots")
    seed = int(config.get("seed", 17))
    started = time.perf_counter()
    previous_recovery = _read(output / "recovery.json")
    training, validation, tests = analytic_splits(
        config.get("beta", 1.0), config.get("one_context", False))
    if config.get("test_contexts"):
        tests = [t for t in tests if t.name in config["test_contexts"]]
    expected_contexts = {t.name for t in tests}
    existing_results = _read(output / "results.json")
    main_rows = [r for r in existing_results["metrics"] if r["context"] in expected_contexts]
    if not main_rows:
        raise ValueError("No completed sampling metrics found")
    result = load_checkpoint(output / "checkpoints")
    ablation_path = output / "ablation_no_movement" / "checkpoint.msgpack"
    ablation = load_checkpoint(ablation_path) if ablation_path.exists() else None
    methods = [
        ("prior", PCNProposal(1.0), None, "prior", 0.0),
        ("rw", RWProposal(.35), None, "exact", 0.0),
        ("pcn", PCNProposal(.65), None, "exact", 0.0),
        ("learned_exact", result.proposal, result.ratio, "exact", 0.0),
        ("learned", result.proposal, result.ratio, "learned", 0.0),
    ]
    methods.extend((f"refresh_{rho:g}", result.proposal, result.ratio, "learned", rho)
                   for rho in config.get("refresh_rates", [.1, .3]))

    # Every displayed/replayed oracle call is recoverable from actual artifacts.
    audit_rows = []
    reference_checks = _read(output / "reference_checks.json", [])
    for reference in reference_checks:
        count = dict(reference["counters"])
        count["audit_evaluations"] = count["exact_cost_evaluations"]
        audit_rows.append(count)
    for target in tests:
        for suffix in ("", "__fine"):
            path = output / "grids" / f"{target.name}{suffix}.npz"
            with np.load(path) as grid:
                audit_rows.append(_audit_count(int(grid["exact_cost_evaluations"])))
    for row in main_rows:
        path = output / "traces" / f"{row['context']}__{row['method']}.npz"
        with np.load(path) as trace:
            actual_sampling = dict(zip(
                COUNTER_NAMES, map(int, trace["counters"][:, -1].sum(axis=0))))
            if any(actual_sampling[k] != int(row[k]) for k in COUNTER_NAMES):
                raise ValueError(f"Stored counter mismatch: {path}")
            replay_costs = (np.isfinite(trace["audit_current_cost"]).sum()
                            + np.isfinite(trace["audit_proposed_cost"]).sum())
            audit_rows.append(_audit_count(replay_costs))
    finite = _read(output / "finite_grid.json")
    for audit in finite.values():
        audit_rows.append(_audit_count(int(audit["grid_size"])**2))

    mechanism = None
    if config.get("mechanisms", True):
        print("[recovery] completing missing mechanism artifacts", flush=True)
        mechanism = run_mechanisms(output, tests, seed=seed, resume_existing=True)
        audit_rows.append(mechanism["counters"]["audit"])
    metrics = main_rows + (mechanism["metrics"] if mechanism else [])
    contexts_by_name = {c["name"]: c for c in metadata["contexts"]}
    if mechanism:
        contexts_by_name.update({c["name"]: c for c in mechanism["contexts"]})
    metadata["contexts"] = list(contexts_by_name.values())
    metadata["recovery"] = (
        "Original process stopped at identity quadrature because it retained an older "
        "module in memory. Saved training, sampling, reference, finite-grid and three "
        "temperature artifacts were reused. Rejection batch counts were reconstructed "
        "from Gaussian PRNG draws only, without new cost evaluations.")
    write_json(output / "metadata.json", metadata)

    planning_rows = _read(output / "planning_results.json", [])
    completed = {(r["context"], r["method"], int(r["budget"]), int(r["replicate"]))
                 for r in planning_rows}
    new_plans = 0
    if config.get("planning", True):
        pool_directory = output / "planning_pools"
        pool_directory.mkdir(exist_ok=True)
        # `planning/` is reserved for route physical-execution replay artifacts.
        # Relocate files from the first recovery implementation without reruns.
        for artifact in (output / "planning").glob("*__b*__r*.npz"):
            artifact.replace(pool_directory / artifact.name)
        for target_index, target in enumerate(tests):
            if target.name == "constant":
                continue
            for budget in config.get("planning_budgets", [16, 48]):
                for replicate in range(int(config.get("planning_replicates", 2))):
                    for name, proposal, ratio, mode, rho in methods:
                        job = (target.name, name, budget, replicate)
                        if job in completed:
                            continue
                        key = jax.random.PRNGKey(
                            seed + 90000 + target_index * 1000 + budget * 10 + replicate)
                        planned = plan(target, proposal, ratio, key, budget, mode=mode, rho=rho)
                        row = {k: v for k, v in planned.items()
                               if k not in ("candidate", "z", "key", "attempt_log")
                               and not k.startswith("pool")}
                        row.update(planned["metrics"])
                        row.update(planned["counters"])
                        row.update(context=target.name, method=name, budget=budget, replicate=replicate)
                        planning_rows.append(row)
                        completed.add(job)
                        new_plans += 1
                        np.savez_compressed(
                            pool_directory / f"{target.name}__{name}__b{budget}__r{replicate}.npz",
                            candidate=planned["candidate"], z=planned["z"], key=planned["key"],
                            pool_z=planned["pool_z"], pool_cost=planned["pool_cost"],
                            pool_candidates=planned["pool_candidates"], pool_source=planned["pool_source"])
                        write_json(output / "planning_results.json", planning_rows)
            print(f"[recovery] planning {target.name} complete", flush=True)
    sampling_rows = list(main_rows)
    if mechanism:
        sampling_rows.append(mechanism["counters"]["sampling"])
    elapsed = time.perf_counter() - started
    prior_elapsed = 0.0
    if previous_recovery:
        prior_elapsed = previous_recovery.get(
            "cumulative_recovery_execution_seconds",
            previous_recovery["recovery_execution_seconds"]
            + previous_recovery.get("prior_recovery_execution_seconds", 0.0))
    recovery_record = dict(
        reason="Original process retained the pre-fix identity quadrature guard",
        completed_main_sampling_jobs_reused=len(main_rows),
        mechanism_contexts_reused=mechanism.get("reused_contexts", []) if mechanism else [],
        new_planning_jobs=new_plans, planning_jobs_total=len(planning_rows),
        oracle_evaluations_repeated_for_recovery=0,
        counter_reconstruction="Saved per-transition counters and oracle arrays; rejection counts from exact Gaussian PRNG matching",
        recovery_execution_seconds=elapsed,
        cumulative_recovery_execution_seconds=prior_elapsed + elapsed,
        recovery_timing_scope="Persisted recovery stages before rendering; original pre-interruption total unavailable",
        original_total_elapsed_seconds=None,
        original_timing_note="Interrupted orchestration did not persist a total timer; no value inferred",
    )
    if previous_recovery:
        recovery_record["prior_recovery_execution_seconds"] = previous_recovery["recovery_execution_seconds"]
    accounting = dict(
        sampling=_sum_counters(sampling_rows), audit=_sum_counters(audit_rows),
        training=result.counters,
        training_calibration=result.metadata.get("calibration_counters", {}),
        ablation_training=ablation.counters if ablation else {},
        ablation_calibration=ablation.metadata.get("calibration_counters", {}) if ablation else {},
        planning=_sum_counters([r.get("counters", r) for r in planning_rows]),
        physical_context_creation_steps=0, physical_execution_steps=0,
        total_elapsed_seconds=None, recovery_execution_seconds=elapsed,
        cumulative_recovery_execution_seconds=prior_elapsed + elapsed,
        recorded_sampling_compilation_seconds=sum(float(r.get("compilation_seconds") or 0) for r in metrics),
        recorded_sampling_warmed_execution_seconds=sum(float(r.get("warmed_execution_seconds") or 0) for r in metrics),
        recorded_planning_compilation_seconds=sum(float(r.get("compilation_seconds") or 0) for r in planning_rows),
        recorded_planning_warmed_execution_seconds=sum(float(r.get("warmed_execution_seconds") or 0) for r in planning_rows),
        training_timing=result.metadata.get("timing", {}),
        note=(
            "All oracle counters reconstructed from artifacts and combined with newly executed stages. "
            "No training or completed chain was rerun. Original overall elapsed time and three mechanism "
            "timings were not persisted and remain unavailable. Recorded timing sums exclude those missing values."))
    write_json(output / "recovery.json", recovery_record)
    write_json(output / "accounting.json", accounting)
    write_json(output / "results.json", dict(metrics=metrics, planning=planning_rows))
    scalar_keys = sorted({k for row in metrics for k, v in row.items()
                          if isinstance(v, (int, float, str))})
    with (output / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    from .visualization import render
    render(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    finish_run(parser.parse_args().run_dir)
