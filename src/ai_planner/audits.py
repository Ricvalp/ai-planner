"""Post-run sampling summaries and held-out accepted-flow Monte Carlo audits.

Saved decoded candidates suffice for movement/ESS summaries. Only the explicitly
counted accepted-flow audit evaluates costs. Its signed balance estimate is never
silently clamped or presented as a population certificate.
"""

from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .acceptance import exact_log_ratio
from .decoders import IdentityDecoder
from .samplers import COUNTER_NAMES
from .targets import analytic_splits, make_analytic_target


AUDIT_VERSION = "heldout_accepted_flow_mc-v1"


def _mean_se(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None


def accepted_flow_summary(exact, predicted, *, beta, j_max, refresh_rates=(0.1, 0.3)):
    """Signed empirical diagnostics for iid pairs drawn from p(z)q(z'|z).

    The algebraic inequalities concern population expectations. Plug-in Monte
    Carlo estimates below have sampling error, including occasionally negative
    balance estimates; they are not certified upper confidence bounds.
    """
    exact, predicted = np.asarray(exact, dtype=float), np.asarray(predicted, dtype=float)
    if exact.shape != predicted.shape or exact.ndim != 1 or len(exact) < 1:
        raise ValueError("Equal nonempty one-dimensional label vectors required.")
    if beta <= 0 or j_max < 0 or not np.isfinite(exact).all() or not np.isfinite(predicted).all():
        raise ValueError("Finite labels, positive temperature and nonnegative bound required.")
    a_star, a = np.exp(np.minimum(0.0, exact)), np.exp(np.minimum(0.0, predicted))
    residual = predicted - exact
    ratio_mse, ratio_mse_se = _mean_se(residual**2)
    balance, balance_se = _mean_se(a * residual)
    epsilon, epsilon_se = _mean_se(np.abs(a - a_star))
    abar, abar_se = _mean_se(a)
    refresh = []
    for rho in refresh_rates:
        if not 0 < rho <= 1:
            raise ValueError("Positive refresh probability at most one required.")
        log_kappa = float(np.log(rho) - j_max / beta)
        if rho == 1 or epsilon == 0:
            log_bound, bound = None, 0.0
        else:
            log_bound = float(np.log1p(-rho) + np.log(epsilon) - log_kappa)
            bound = float(np.exp(log_bound)) if log_bound < np.log(np.finfo(float).max) else None
        refresh.append({
            "rho": float(rho),
            "log_kappa": log_kappa,
            "kappa": float(np.exp(log_kappa)),
            "stationary_bias_bound_point_estimate": bound,
            "log_stationary_bias_bound_point_estimate": log_bound,
            "vacuous": bool(bound is None or bound >= 1.0),
            "interpretation": "Population formula evaluated at MC epsilon_a; not a certified bound.",
        })
    return {
        "n_pairs": len(exact),
        "L_ratio": ratio_mse,
        "L_ratio_standard_error": ratio_mse_se,
        "L_bal_signed": balance,
        "L_bal_standard_error": balance_se,
        "epsilon_a": epsilon,
        "epsilon_a_standard_error": epsilon_se,
        "abar": abar,
        "abar_standard_error": abar_se,
        "exact_acceptance": float(a_star.mean()),
        "sqrt_L_ratio_point_estimate": float(np.sqrt(ratio_mse)),
        "sqrt_abar_L_bal_over_2_point_estimate": float(np.sqrt(abar * balance / 2)) if balance >= 0 else None,
        "balance_bound_status": "defined plug-in estimate" if balance >= 0 else "undefined: signed MC L_bal is negative",
        "refresh": refresh,
        "interpretation": (
            "Independent exact rejection sources and fresh proposals estimate population expectations. "
            "Finite-sample signed L_bal can be negative; it is preserved. These plug-in values "
            "are Monte Carlo diagnostics, not certified bounds on continuous TV or stationary bias."
        ),
    }


def _route_from_definition(definition):
    """Restore an immutable recorded snapshot without executing its history."""
    from .envs.point_mass import SimulatorConfig, Snapshot
    from .routes import RouteContext, RouteDecoder, RouteTarget

    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value

    definition = dict(definition)
    context = {key: tuples(value) for key, value in definition.pop("context").items()}
    for field in ("snapshot", "initial_snapshot"):
        context[field] = Snapshot(**{key: tuples(value) for key, value in context[field].items()})
    return RouteTarget(
        context=RouteContext(**context),
        decoder=RouteDecoder(**definition.pop("decoder")),
        simulator=SimulatorConfig(**definition.pop("simulator")),
        **definition,
    )


def _run_targets(config, metadata):
    if config["experiment"] == "routes":
        main = [_route_from_definition(context["definition"])
                for context in metadata["contexts"] if context["split"] == "test"]
        targets = list(main)
    else:
        _, _, main = analytic_splits(config.get("beta", 1.0), config.get("one_context", False))
        targets = list(main)
        base = make_analytic_target("two", beta=1.0)
        targets.extend(
            replace(base, beta=beta, context=replace(base.context, name=f"two_beta_{beta:g}"))
            for beta in (0.6, 1.0, 1.6)
        )
        targets.append(replace(base, decoder=IdentityDecoder(),
                               context=replace(base.context, name="two_identity_decoder")))
    if config.get("test_contexts"):
        main = [target for target in main if target.name in config["test_contexts"]]
    return main, {target.name: target for target in targets}


def _rewrite_sampling_summaries(run_dir, config, targets):
    from .evaluation import sampling_summary, write_json
    path = run_dir / "results.json"
    result = json.loads(path.read_text())
    updated = {}
    for row in result["metrics"]:
        name, method = row["context"], row["method"]
        if name not in targets:
            raise ValueError(f"Cannot reconstruct saved target {name!r}.")
        with np.load(run_dir / "traces" / f"{name}__{method}.npz") as artifact:
            trace = dict(artifact)
        with np.load(run_dir / "grids" / f"{name}.npz") as artifact:
            grid = dict(artifact)
        burn = int(row.get("burn_in_per_chain", config.get("burn_in", 256)))
        summary = sampling_summary(targets[name], trace, grid, burn)
        # Context, method, timings and original accounting survive this update.
        row.update(summary)
        updated[(name, method)] = row
    result["summary_rederivation"] = {
        "version": "decoded-feature-movement-and-coordinate-ess-v1",
        "oracle_evaluations": 0,
        "source": "Saved repeated Markov states and decoded candidates; no simulator rerun.",
    }
    write_json(path, result)
    scalar_keys = sorted({key for row in result["metrics"] for key, value in row.items()
                          if isinstance(value, (int, float, str))})
    with (run_dir / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["metrics"])
    mechanisms_path = run_dir / "mechanisms.json"
    if mechanisms_path.exists():
        mechanisms = json.loads(mechanisms_path.read_text())
        for row in mechanisms["metrics"]:
            replacement = updated.get((row["context"], row["method"]))
            if replacement is not None:
                row.update(replacement)
        write_json(mechanisms_path, mechanisms)
    return len(updated)


def augment_run(run_dir, *, n_pairs=512):
    """Augment an existing run without repeating previously saved oracle work."""
    from .evaluation import write_json
    from .training import load_checkpoint
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    main_targets, targets = _run_targets(config, metadata)
    expected_ids = {context["name"]: context["target_id"] for context in metadata["contexts"]}
    for target in targets.values():
        if target.name in expected_ids and target.target_id != expected_ids[target.name]:
            raise ValueError(f"Saved target definition mismatch for {target.name}.")
    rewritten = _rewrite_sampling_summaries(run_dir, config, targets)
    checkpoint_path = run_dir / "checkpoints" / "checkpoint.msgpack"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = load_checkpoint(checkpoint_path)
    artifact_dir = run_dir / "acceptance_audits"
    artifact_dir.mkdir(exist_ok=True)
    counters = dict.fromkeys(COUNTER_NAMES, 0)
    rows = []
    total_compile_seconds = 0.0
    total_execution_seconds = 0.0
    generated_evaluations = 0
    for index, target in enumerate(main_targets):
        if target.beta != checkpoint.metadata["temperature"]:
            raise ValueError("Acceptance audit cannot reuse a checkpoint at another temperature.")
        seed = int(config.get("seed", 17)) + 200000 + index
        signature_data = {
            "version": AUDIT_VERSION, "target_id": target.target_id,
            "checkpoint_sha256": checkpoint_hash, "n_pairs": n_pairs, "seed": seed,
        }
        signature = json.dumps(signature_data, sort_keys=True)
        artifact_path = artifact_dir / f"{target.name}.npz"
        if artifact_path.exists():
            with np.load(artifact_path) as artifact:
                if str(artifact["signature"]) != signature:
                    raise ValueError(f"Existing accepted-flow artifact signature mismatch: {artifact_path}")
                arrays = dict(artifact)
        else:
            with np.load(run_dir / "traces" / f"{target.name}__rejection_reference.npz") as reference:
                available = np.asarray(reference["z"]).reshape(-1, target.noise_dim)
            if len(available) < n_pairs:
                raise ValueError(f"{target.name} has fewer than {n_pairs} saved exact rejection sources.")
            source = jnp.asarray(available[:n_pairs])
            key = jax.random.PRNGKey(seed)

            def evaluate_pairs(draw_key, z):
                zp = checkpoint.proposal.sample(draw_key, z, target.context_features)
                costs = jax.vmap(target.evaluate)(jnp.concatenate((z, zp)))
                old_cost, new_cost = costs[:n_pairs], costs[n_pairs:]
                exact = exact_log_ratio(z, zp, old_cost, new_cost, target.beta,
                                        checkpoint.proposal, target.context_features)
                predicted = checkpoint.ratio(z, zp, target.context_features)
                return zp, old_cost, new_cost, exact, predicted

            started = time.perf_counter()
            compiled = jax.jit(evaluate_pairs).lower(key, source).compile()
            compilation_seconds = time.perf_counter() - started
            started = time.perf_counter()
            proposed, source_cost, proposed_cost, exact, predicted = compiled(key, source)
            jax.block_until_ready((proposed, source_cost, proposed_cost, exact, predicted))
            execution_seconds = time.perf_counter() - started
            arrays = {
                "signature": np.asarray(signature), "source_z": np.asarray(source),
                "proposed_z": np.asarray(proposed),
                "source_cost": np.asarray(source_cost), "proposed_cost": np.asarray(proposed_cost),
                "exact_log_ratio": np.asarray(exact), "learned_log_ratio": np.asarray(predicted),
                "compilation_seconds": np.asarray(compilation_seconds),
                "warmed_execution_seconds": np.asarray(execution_seconds),
            }
            np.savez_compressed(artifact_path, **arrays)
            generated_evaluations += 2 * n_pairs
        row = accepted_flow_summary(arrays["exact_log_ratio"], arrays["learned_log_ratio"],
                                   beta=target.beta, j_max=target.j_max)
        per_context = dict.fromkeys(COUNTER_NAMES, 0)
        per_context.update(exact_cost_evaluations=2 * n_pairs, decoder_calls=2 * n_pairs,
                           simulator_steps=2 * n_pairs * target.simulator_steps_per_cost,
                           audit_evaluations=2 * n_pairs)
        row.update(
            context=target.name, target_id=target.target_id, beta=target.beta,
            j_max=target.j_max, seed=seed, counters=per_context,
            source_provenance="First saved independent exact rejection samples; no best-of-N selection.",
            source_cost_evaluations=n_pairs, proposed_cost_evaluations=n_pairs,
            artifact=str(artifact_path.relative_to(run_dir)),
        )
        rows.append(row)
        for name in COUNTER_NAMES:
            counters[name] += per_context[name]
        total_compile_seconds += float(arrays["compilation_seconds"])
        total_execution_seconds += float(arrays["warmed_execution_seconds"])
    audit = {
        "version": AUDIT_VERSION, "checkpoint_sha256": checkpoint_hash,
        "n_pairs_per_context": n_pairs, "contexts": rows, "counters": counters,
        "compilation_seconds": total_compile_seconds,
        "warmed_execution_seconds": total_execution_seconds,
        "mechanism_audits_excluded": "Temperature and identity-decoder cases use exact acceptance only.",
        "sampling_summaries_rederived": rewritten,
        "sampling_summary_oracle_evaluations": 0,
        "context_creation_steps": 0,
        "context_restoration": "Immutable route snapshots restored from saved metadata; physical history not rerun.",
    }
    write_json(run_dir / "acceptance_audit.json", audit)
    accounting_path = run_dir / "accounting.json"
    accounting = json.loads(accounting_path.read_text())
    accounting["extra_audit"] = {
        **counters,
        "name": AUDIT_VERSION,
        "source_cost_evaluations": n_pairs * len(rows),
        "proposed_cost_evaluations": n_pairs * len(rows),
        "compilation_seconds": total_compile_seconds,
        "warmed_execution_seconds": total_execution_seconds,
        "context_creation_steps": 0,
        "note": "Saved once per signature. Original audit accounting is preserved; summaries use zero oracle calls.",
    }
    write_json(accounting_path, accounting)
    return {
        "run": str(run_dir), "sampling_rows_updated": rewritten,
        "new_oracle_evaluations_this_invocation": generated_evaluations,
        "saved_extra_audit_evaluations": counters["exact_cost_evaluations"],
        "contexts": rows,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    result = augment_run(args.run_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "contexts"}, indent=2))


if __name__ == "__main__":
    main()
