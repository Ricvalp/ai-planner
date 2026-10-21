"""Match no-movement ratio-training effort while preserving exploratory outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import jax
import numpy as np

from .evaluation import _audit_replay, _sum_counters, sampling_summary, write_json
from .samplers import COUNTER_NAMES, run_chains
from .targets import analytic_splits
from .training import TrainingConfig, load_checkpoint, train


def _load(path):
    return json.loads(Path(path).read_text())


def match_ablation(run_dir):
    output = Path(run_dir)
    complete_path = output / "matched_ablation.json"
    if complete_path.exists():
        return _load(complete_path)
    started = time.perf_counter()
    config = _load(output / "config.json")
    metadata = _load(output / "metadata.json")
    seed = int(config["seed"])
    desired = load_checkpoint(output / "checkpoints").metadata["completed_rounds"]
    archive = output / "exploratory_round1"
    if not archive.exists():
        archive.mkdir()
        shutil.copytree(output / "ablation_no_movement", archive / "ablation_no_movement")
        (archive / "traces").mkdir()
        original_results = _load(output / "results.json")
        old_rows = [row for row in original_results["metrics"] if row["method"] == "no_movement"]
        write_json(archive / "metrics.json", old_rows)
        shutil.copy2(output / "accounting.json", archive / "accounting.json")
        for row in old_rows:
            name = f"{row['context']}__no_movement.npz"
            shutil.copy2(output / "traces" / name, archive / "traces" / name)
        write_json(archive / "provenance.json", dict(
            reason="Exploratory ablation had one ratio-training round versus two for the main model",
            retained_for_audit=True, main_training_rounds=desired, ablation_training_rounds=1))
    old_rows = _load(archive / "metrics.json")
    original_accounting = _load(archive / "accounting.json")
    context_creation_steps = 0
    if metadata["kind"] == "analytic":
        training, validation, tests = analytic_splits(
            config.get("beta", 1.0), config.get("one_context", False))
    else:
        from .routes import make_route_targets
        splits = make_route_targets(config.get("beta", .25))
        training, validation, tests = splits["train"], splits["validation"], splits["test"]
        context_creation_steps = sum(t.context.executed_prefix * t.simulator.substeps
                                     for group in splits.values() for t in group)
    if config.get("test_contexts"):
        tests = [t for t in tests if t.name in config["test_contexts"]]
    current = load_checkpoint(output / "ablation_no_movement")
    cfg = dict(current.metadata["config"])
    cfg["rounds"] = desired
    cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
    if int(current.state.completed_rounds) < desired:
        print(f"[matched ablation] {metadata['kind']}: resume ratio round 2", flush=True)
        current = train(training, TrainingConfig(**cfg), output / "ablation_no_movement",
                        validation, resume=current.checkpoint_path)
    elif int(current.state.completed_rounds) != desired:
        raise ValueError("Ablation checkpoint has more rounds than main checkpoint")
    if int(current.state.proposal_updates) != 0:
        raise ValueError("No-movement ablation unexpectedly updated the proposal")
    main = load_checkpoint(output / "checkpoints")
    if int(current.state.ratio_updates) != int(main.state.ratio_updates):
        raise ValueError("Main and ablation ratio-update budgets still differ")
    partial_path = output / "matched_ablation.partial.json"
    partial = _load(partial_path) if partial_path.exists() else {"rows": [], "audit": []}
    new_rows = partial["rows"]
    audit_counts = partial["audit"]
    done = {row["context"] for row in new_rows}
    current_results = _load(output / "results.json")
    old_audit = []
    for target_index, target in enumerate(tests):
        name = f"{target.name}__no_movement.npz"
        with np.load(archive / "traces" / name) as trace:
            n_audit = int(np.isfinite(trace["audit_current_cost"]).sum()
                          + np.isfinite(trace["audit_proposed_cost"]).sum())
            counter = dict.fromkeys(COUNTER_NAMES, 0)
            counter.update(decoder_calls=n_audit, exact_cost_evaluations=n_audit,
                           audit_evaluations=n_audit,
                           simulator_steps=n_audit * target.simulator_steps_per_cost)
            old_audit.append(counter)
        if target.name in done:
            continue
        method_names = [r["method"] for r in current_results["metrics"] if r["context"] == target.name]
        method_index = method_names.index("no_movement")
        key = jax.random.PRNGKey(seed + 1000 + target_index * 100 + method_index)
        trace, counts = run_chains(
            target, current.proposal, key, int(config["steps"]), int(config["chains"]),
            current.ratio, "learned", 0.0)
        with np.load(output / "grids" / f"{target.name}.npz") as grid:
            summary = sampling_summary(target, trace, dict(grid), int(config["burn_in"]))
        summary.update(context=target.name, method="no_movement", beta=target.beta, rho=0.0,
                       ablation_training_rounds=desired,
                       ablation_ratio_updates=int(current.state.ratio_updates),
                       ablation_proposal_updates=0, **counts)
        audit_counts.append(_audit_replay(target, current.proposal, current.ratio, trace))
        np.savez_compressed(output / "traces" / name, **trace)
        new_rows.append(summary)
        write_json(partial_path, dict(rows=new_rows, audit=audit_counts))
        print(f"[matched ablation] {target.name}: completed", flush=True)
    replacements = {r["context"]: r for r in new_rows}
    current_results["metrics"] = [
        replacements[r["context"]] if r["method"] == "no_movement" else r
        for r in current_results["metrics"]]
    old_sampling, new_sampling = _sum_counters(old_rows), _sum_counters(new_rows)
    old_audit_total, new_audit_total = _sum_counters(old_audit), _sum_counters(audit_counts)
    accounting = dict(original_accounting)
    accounting["sampling"] = {
        k: int(original_accounting["sampling"][k]) - old_sampling[k] + new_sampling[k]
        for k in COUNTER_NAMES}
    accounting["audit"] = {
        k: int(original_accounting["audit"][k]) - old_audit_total[k] + new_audit_total[k]
        for k in COUNTER_NAMES}
    accounting["exploratory_ablation_sampling"] = old_sampling
    accounting["exploratory_ablation_audit"] = old_audit_total
    accounting["exploratory_ablation_calibration"] = original_accounting["ablation_calibration"]
    accounting["ablation_training"] = current.counters
    accounting["ablation_calibration"] = current.metadata["calibration_counters"]
    accounting["physical_context_creation_steps"] = (
        original_accounting.get("physical_context_creation_steps", 0) + context_creation_steps)
    accounting["matched_ablation_execution_seconds"] = time.perf_counter() - started
    accounting["matched_ablation_note"] = (
        "Final no_movement has the same number of ratio-training rounds and optimizer "
        "updates as the main model. Original one-round chains, prefix audits and final "
        "calibration are charged separately as exploratory work. Resumed training "
        "counters already include both rounds; no round-one training is counted twice.")
    metadata["no_movement_ablation"] = dict(
        matched_training_rounds=desired, matched_ratio_updates=int(current.state.ratio_updates),
        proposal_updates=0, initialization="same initial pCN proposal",
        exploratory_round1="exploratory_round1/",
        note="Fresh q-dependent pair datasets each fit; equal ratio-update effort, no proposal movement updates")
    write_json(output / "results.json", current_results)
    write_json(output / "accounting.json", accounting)
    write_json(output / "metadata.json", metadata)
    scalar_keys = sorted({k for row in current_results["metrics"] for k, value in row.items()
                          if isinstance(value, (int, float, str))})
    with (output / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(current_results["metrics"])
    report = dict(
        training_rounds=desired, ratio_updates=int(current.state.ratio_updates),
        main_ratio_updates=int(main.state.ratio_updates),
        completed_contexts=[r["context"] for r in new_rows],
        old_sampling=old_sampling, new_sampling=new_sampling,
        old_prefix_audit=old_audit_total, new_prefix_audit=new_audit_total,
        old_calibration=original_accounting["ablation_calibration"],
        new_calibration=current.metadata["calibration_counters"],
        old_training=original_accounting["ablation_training"],
        resumed_training=current.counters,
        context_creation_steps=context_creation_steps,
        elapsed_before_render_seconds=time.perf_counter() - started,
        scientific_note="Ratio optimization effort now matched; original confounded comparison retained as exploratory",
    )
    write_json(complete_path, report)
    from .visualization import render
    render(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    for directory in parser.parse_args().run_dirs:
        match_ablation(directory)
