"""Small exact-only mechanism checks, saved with the main replay artifact schema."""
from dataclasses import replace
import json
from pathlib import Path
import time

import jax
import numpy as np

from .contexts import canonical
from .decoders import IdentityDecoder, compact
from .proposals import PCNProposal
from .samplers import COUNTER_NAMES, rejection_sample, run_chains
from .targets import make_analytic_target, quadrature


def reconstruct_rejection_count(last_sample, key, *, batch_size=512, max_evaluations=2_000_000):
    """Recover paid batch count by matching the final accepted Gaussian draw.

    This reproduces only random numbers, not costs or accept/reject decisions.
    It works because rejection_sample returns samples in draw order, evaluates
    complete batches, and preserves every returned draw bit-for-bit.
    """
    last_sample = np.asarray(last_sample)
    for total in range(batch_size, max_evaluations + 1, batch_size):
        key, draw_key, _ = jax.random.split(key, 3)
        candidates = np.asarray(jax.random.normal(
            draw_key, (batch_size, len(last_sample)), dtype=np.float64))
        if np.any(np.all(candidates == last_sample, axis=1)):
            return total
    raise ValueError("Saved final rejection sample not found in the declared PRNG stream")


def run_mechanisms(output, targets, proposal=None, ratio=None, seed=17, *,
                   steps=1024, burn=256, n_chains=4, reference_samples=1024,
                   grid_size=64, resume_existing=False):
    """Compare temperatures and decoder maps without reusing learned acceptance.

    Runs four exact pCN cases: CDF decoder beta=.6,1,1.6 and identity decoder
    beta=1. Every case gets its own reference, independently sampled target
    states, and saved grid. The same energy formula with a different decoder
    defines a different target; this is a sensitivity check, not a dominance
    claim. `proposal`/`ratio` are accepted for orchestration compatibility but
    are deliberately unused: these checks only depend on exact pCN.
    """
    from .evaluation import region_labels, sampling_summary, write_json

    if not 0 <= burn < steps or n_chains < 1 or reference_samples < 1:
        raise ValueError("Invalid bounded mechanism sampling sizes")
    if not all(hasattr(t.context, "centers") for t in targets):
        raise ValueError("This mechanism audit is for analytic target contexts")
    output = Path(output)
    if resume_existing and (output / "mechanisms.json").exists():
        return json.loads((output / "mechanisms.json").read_text())
    for directory in ("grids", "traces"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    # Explicit common geometry makes the symmetric half-plane partition exact
    # for both decoder choices; no inferred latent component labels are used.
    base = make_analytic_target("two", beta=1.0)
    selected = [
        replace(base, beta=beta, context=replace(base.context, name=f"two_beta_{beta:g}"))
        for beta in (.6, 1.0, 1.6)
    ]
    selected.append(replace(base, decoder=IdentityDecoder(),
                            context=replace(base.context, name="two_identity_decoder")))
    rows, contexts, references = [], [], []
    sampling_counts = {name: 0 for name in COUNTER_NAMES}
    audit_counts = {name: 0 for name in COUNTER_NAMES}
    started = time.perf_counter()
    reused_contexts = []
    for index, target in enumerate(selected):
        is_identity = isinstance(target.decoder, IdentityDecoder)
        coordinate_label = ("Gaussian CDF coefficient u (identity-decoded output is z)"
                            if is_identity else "Decoded descriptor y = Gaussian CDF coefficient u")
        grid_path = output / "grids" / f"{target.name}.npz"
        fine_path = output / "grids" / f"{target.name}__fine.npz"
        trace_path = output / "traces" / f"{target.name}__pcn.npz"
        reference_path = output / "traces" / f"{target.name}__rejection_reference.npz"
        if resume_existing and grid_path.exists() and fine_path.exists():
            with np.load(grid_path) as saved:
                grid = dict(saved)
            with np.load(fine_path) as saved:
                fine = dict(saved)
            if int(grid["grid_size"]) != grid_size or int(fine["grid_size"]) != 2 * grid_size:
                raise ValueError("Existing mechanism grids use a different grid size")
        else:
            grid, fine = quadrature(target, grid_size), quadrature(target, 2 * grid_size)
        grid["coordinate_label"] = coordinate_label
        fine["coordinate_label"] = coordinate_label
        for reference_grid in (grid, fine):
            for name in ("decoder_calls", "exact_cost_evaluations", "simulator_steps"):
                audit_counts[name] += int(reference_grid[name])
            audit_counts["audit_evaluations"] += int(reference_grid["exact_cost_evaluations"])
        np.savez_compressed(grid_path, **grid)
        np.savez_compressed(fine_path, **fine)
        key = jax.random.PRNGKey(seed + 70000 + index)
        if resume_existing and trace_path.exists():
            with np.load(trace_path) as saved:
                trace = dict(saved)
            if trace["z"].shape != (n_chains, steps, 2):
                raise ValueError("Existing mechanism chain dimensions differ")
            counts = dict(zip(COUNTER_NAMES, map(int, trace["counters"][:, -1].sum(axis=0))))
            counts.update(compilation_seconds=None, warmed_execution_seconds=None,
                          timing_note="Not persisted before interrupted run; unavailable")
            reused_contexts.append(target.name)
        else:
            trace, counts = run_chains(target, PCNProposal(.65), key, steps, n_chains,
                                       mode="exact")
        for name in COUNTER_NAMES:
            sampling_counts[name] += counts[name]
        trace["audit_exact_log_ratio"] = trace["exact_log_ratio"].copy()
        np.savez_compressed(trace_path, **trace)
        row = sampling_summary(target, trace, grid, burn)
        row.update(
            context=target.name, method="pcn", beta=target.beta, rho=0.0,
            coordinate_label=coordinate_label, decoder=target.decoder.version,
            accepted_movement_space="uniform CDF coefficient u",
            mechanism=("identity_decoder" if is_identity else "temperature"),
            exact_only=True, **counts)
        rows.append(row)
        if resume_existing and reference_path.exists():
            with np.load(reference_path) as saved:
                reference = np.asarray(saved["z"])[0]
            if len(reference) != reference_samples:
                raise ValueError("Existing mechanism reference sample size differs")
            attempted = reconstruct_rejection_count(
                reference[-1], jax.random.PRNGKey(seed + 71000 + index))
            reference_counts = dict.fromkeys(COUNTER_NAMES, 0)
            reference_counts.update(exact_cost_evaluations=attempted, decoder_calls=attempted,
                                    simulator_steps=attempted * target.simulator_steps_per_cost)
        else:
            reference, reference_counts, _ = rejection_sample(
                target, reference_samples, jax.random.PRNGKey(seed + 71000 + index))
        for name in COUNTER_NAMES:
            audit_counts[name] += reference_counts[name]
        audit_counts["audit_evaluations"] += reference_counts["exact_cost_evaluations"]
        yy = np.asarray(compact(reference))
        labels = region_labels(target, yy)
        observed = np.bincount(labels, minlength=len(grid["region_masses"])) / reference_samples
        standard_error = np.sqrt(fine["region_masses"] * (1 - fine["region_masses"])
                                 / reference_samples)
        references.append(dict(
            context=target.name, n=reference_samples, exact_only=True,
            normalizer=grid["normalizer"], refined_normalizer=fine["normalizer"],
            region_mass_grid_difference=float(np.max(np.abs(
                grid["region_masses"] - fine["region_masses"]))),
            reference_region_masses=fine["region_masses"],
            rejection_region_masses=observed, rejection_standard_error=standard_error,
            rejection_max_standardized_error=float(np.max(
                np.abs(observed - fine["region_masses"]) / np.maximum(standard_error, 1e-12))),
            chain_region_masses=row["region_masses"],
            chain_region_standard_error=row["region_mass_standard_error"],
            chain_ess=row["ess"],
            provenance="independent exact rejection; compared to finite exact-kernel chains",
            counters=reference_counts))
        np.savez_compressed(reference_path, z=reference[None], y=yy[None])
        contexts.append(dict(
            name=target.name, beta=target.beta, target_id=target.target_id,
            j_max=target.j_max, decoder=target.decoder.version, split="mechanism_audit",
            coordinate_label=coordinate_label, definition=canonical(target)))
        write_json(output / "mechanisms.partial.json", dict(
            metrics=rows, contexts=contexts, reference_checks=references,
            counters=dict(sampling=sampling_counts, audit=audit_counts),
            completed_contexts=[c["name"] for c in contexts], reused_contexts=reused_contexts))
    result = dict(
        metrics=rows, contexts=contexts, reference_checks=references,
        counters=dict(sampling=sampling_counts, audit=audit_counts),
        total_elapsed_seconds=time.perf_counter() - started,
        reused_contexts=reused_contexts,
        timing_scope="This invocation only; any reused stage's unsaved original timing is unavailable",
        temperature_note="Exact pCN only. No fixed-temperature ratio checkpoint is used out of distribution.",
        decoder_note=(
            "Identity and CDF decoders define different targets through J(G(z)). Both are "
            "integrated/plotted in u=2Phi(z)-1 with a uniform coefficient prior. Only the "
            "CDF decoder has decoded output equal to u. The common symmetric two-center "
            "partition is the sign of the first coordinate in either space."),
        uncertainty_note=(
            "One seed, four independent chain starts per setting. Chain standard errors "
            "measure start-to-start variation and do not eliminate finite-chain bias. "
            "Rejection reference standard errors are binomial under independent exact sampling."),
    )
    write_json(output / "mechanisms.json", result)
    return result
