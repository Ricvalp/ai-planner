"""Small frozen-round training, explicit oracle accounting, and reproducible resume.

Only ratio regression uses exact cost values. Proposal learning uses a frozen
ratio, fresh on-policy draws, and a stopped score-function estimator. Checkpoints
are written at round boundaries, where there is no half-consumed minibatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from flax import serialization, struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .acceptance import RatioModel, acceptance_probability, exact_log_ratio
from .proposals import NeuralProposal


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 17
    rounds: int = 2
    hidden_dims: tuple[int, ...] = (32, 32, 32)
    ratio_steps: int = 100
    proposal_steps: int = 20
    batch_size: int = 64
    target_samples: int = 192
    pairs_per_context: int = 512
    calibration_samples: int = 256
    ratio_lr: float = 0.002
    proposal_lr: float = 0.0005
    initial_eta: float = 0.65
    movement_scale: float = 0.5
    baseline_decay: float = 0.95
    max_rejection_attempts: int = 2_000_000

    def __post_init__(self):
        if min(self.rounds, self.ratio_steps, self.batch_size, self.target_samples,
               self.pairs_per_context, self.calibration_samples) <= 0:
            raise ValueError("Training sizes must be positive")
        if self.proposal_steps < 0 or self.movement_scale <= 0:
            raise ValueError("Invalid proposal_steps or movement_scale")
        if not 0 <= self.baseline_decay < 1:
            raise ValueError("baseline_decay must lie in [0,1)")


@struct.dataclass
class TrainingState:
    proposal_params: Any
    ratio_params: Any
    proposal_optimizer: Any
    ratio_optimizer: Any
    key: Any
    baseline: Any
    completed_rounds: Any
    ratio_updates: Any
    proposal_updates: Any


@dataclass
class TrainingResult:
    proposal: NeuralProposal
    ratio: RatioModel
    history: list[dict]
    counters: dict
    checkpoint_path: str | None
    calibration: dict = field(default_factory=dict)
    state: TrainingState | None = None
    metadata: dict = field(default_factory=dict)


def _counters():
    return dict(transitions=0, decoder_calls=0, exact_cost_evaluations=0,
                simulator_steps=0, training_oracle_evaluations=0,
                refresh_current_evaluations=0, refresh_proposed_evaluations=0,
                final_validation_evaluations=0, audit_evaluations=0)


def _count_oracle(counters, target, n, category="training_oracle_evaluations"):
    counters["decoder_calls"] += int(n)
    counters["exact_cost_evaluations"] += int(n)
    counters[category] += int(n)
    counters["simulator_steps"] += int(n) * int(target.simulator_steps_per_cost)


def parameter_version(params):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(params):
        a = np.asarray(leaf)
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()[:16]


def bounded_movement(features, proposed_features, scale):
    """Movement in decoded behavior space; ignored noise coordinates earn zero."""
    return -jnp.expm1(-jnp.sum((proposed_features - features)**2, axis=-1)
                     / (2 * scale**2))


def score_function_loss(proposal, params, key, source, context, reward_fn, baseline):
    """Negative accepted movement score objective with all required stops.

    `reward_fn` receives detached destinations and must use a frozen ratio. Its
    output is detached as well, so this function cannot backpropagate simulator
    or acceptance gradients even when the oracle happens to be differentiable.
    """
    proposed = jax.lax.stop_gradient(proposal.sample(key, source, context, params))
    reward = jax.lax.stop_gradient(reward_fn(source, proposed))
    advantage = jax.lax.stop_gradient(reward - baseline)
    loss = -jnp.mean(advantage * proposal.log_prob(proposed, source, context, params))
    return loss, jnp.mean(reward)


def independent_target_samples(target, n, key, counters, cost_batch=None,
                               max_attempts=2_000_000,
                               category="training_oracle_evaluations"):
    """Exact independent rejection samples, counting overdrawn candidates too."""
    cost_batch = cost_batch or jax.jit(jax.vmap(target.evaluate))
    samples, costs = [], []
    found = attempted = 0
    size = max(256, 2 * n)
    while found < n:
        if attempted + size > max_attempts:
            raise RuntimeError("Rejection oracle reached the configured attempt cap")
        key, draw_key, uniform_key = jax.random.split(key, 3)
        candidates = jax.random.normal(draw_key, (size, target.noise_dim), dtype=jnp.float64)
        values = np.asarray(cost_batch(candidates))
        _count_oracle(counters, target, size, category)
        if np.any(~np.isfinite(values)) or np.any(values < -1e-10) or np.any(
                values > target.j_max + 1e-8):
            raise ValueError("Rejection oracle requires costs within the declared [0,J_max]")
        log_u = np.log(np.asarray(jax.random.uniform(
            uniform_key, (size,), minval=np.finfo(float).tiny, dtype=jnp.float64)))
        selected = log_u < -values / target.beta
        samples.append(np.asarray(candidates)[selected])
        costs.append(values[selected])
        found += int(selected.sum())
        attempted += size
    return (jnp.asarray(np.concatenate(samples)[:n]),
            jnp.asarray(np.concatenate(costs)[:n]), key)


def ratio_labels(proposal, params, source, proposed, source_cost, proposed_cost,
                 beta, context):
    """Recompute proposal-dependent labels even if cost values are cached."""
    return jax.lax.stop_gradient(exact_log_ratio(
        source, proposed, source_cost, proposed_cost, beta, proposal, context, params))


def _normalization(targets):
    contexts = np.stack([np.asarray(t.context_features, dtype=float) for t in targets])
    return contexts.mean(axis=0), np.maximum(contexts.std(axis=0), 1.0)


def _initialize(config, noise_dim, context_dim, mean, scale):
    key, proposal_key, ratio_key = jax.random.split(jax.random.PRNGKey(config.seed), 3)
    proposal = NeuralProposal.create(
        proposal_key, noise_dim, context_dim, config.hidden_dims, eta=config.initial_eta,
        context_mean=mean, context_scale=scale)
    ratio = RatioModel.create(ratio_key, noise_dim, context_dim, config.hidden_dims,
                              context_mean=mean, context_scale=scale)
    p_optimizer = optax.chain(optax.clip_by_global_norm(2.0), optax.adam(config.proposal_lr))
    r_optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(config.ratio_lr))
    state = TrainingState(proposal.params, ratio.params,
                          p_optimizer.init(proposal.params), r_optimizer.init(ratio.params),
                          key, jnp.array(0.0), jnp.array(0), jnp.array(0), jnp.array(0))
    return proposal, ratio, p_optimizer, r_optimizer, state


def _checkpoint_path(path):
    path = Path(path)
    return path / "checkpoint.msgpack" if path.is_dir() or path.suffix == "" else path


def save_checkpoint(path, state, metadata):
    path = _checkpoint_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement avoids a partly written parameter artifact.
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(serialization.to_bytes(state))
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return str(path)


def load_checkpoint(path):
    """Load frozen inference models and complete optimizer/PRNG resume state."""
    path = _checkpoint_path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    cfg = dict(metadata["config"])
    cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
    config = TrainingConfig(**cfg)
    proposal, ratio, _, _, template = _initialize(
        config, metadata["noise_dim"], metadata["context_dim"],
        metadata["context_mean"], metadata["context_scale"])
    state = serialization.from_bytes(template, path.read_bytes())
    state = jax.tree.map(jnp.asarray, state)
    return TrainingResult(proposal.replace_params(state.proposal_params),
                          ratio.replace_params(state.ratio_params),
                          metadata["history"], metadata["counters"], str(path),
                          state=state, metadata=metadata)


def calibration_summary(exact, predicted):
    exact, predicted = np.asarray(exact), np.asarray(predicted)
    a = np.exp(np.minimum(0, predicted))
    a_star = np.exp(np.minimum(0, exact))
    balance_terms = a * (predicted - exact)
    n = len(exact)
    return {
        "n_pairs": n,
        "ratio_mse": float(np.mean((predicted - exact)**2)),
        "acceptance_error": float(np.mean(np.abs(a - a_star))),
        "acceptance_learned": float(a.mean()),
        "acceptance_exact": float(a_star.mean()),
        "balance_mc_signed": float(balance_terms.mean()),
        "balance_mc_standard_error": float(balance_terms.std(ddof=1) / np.sqrt(n))
        if n > 1 else None,
        "interpretation": "Monte Carlo estimates, not certified population bounds",
    }


def calibrate(targets, proposal, ratio, key, n=256, counters=None,
              max_rejection_attempts=2_000_000):
    """Fresh exact-rejection sources and fresh proposals on declared contexts."""
    counters = _counters() if counters is None else counters
    result = {}
    for target in targets:
        context = jnp.asarray(target.context_features)
        cost_batch = jax.jit(jax.vmap(target.evaluate))
        z, costs, key = independent_target_samples(
            target, n, key, counters, cost_batch, max_rejection_attempts,
            category="audit_evaluations")
        key, subkey = jax.random.split(key)
        zp = proposal.sample(subkey, z, context)
        new_costs = cost_batch(zp)
        _count_oracle(counters, target, n, category="audit_evaluations")
        exact = np.asarray(ratio_labels(proposal, proposal.params, z, zp, costs,
                                       new_costs, target.beta, context))
        prediction = np.asarray(ratio(z, zp, context))
        result[target.target_id] = dict(
            z=np.asarray(z), proposed=np.asarray(zp), exact=exact, predicted=prediction,
            summary=calibration_summary(exact, prediction))
    return result, key


def train(targets, config=None, out_dir=None, validation_targets=(), resume=None):
    """Train a conditional proposal/ratio in short alternating frozen rounds.

    `rounds` is the desired total number of completed rounds, including resumed
    rounds. Other training settings and target IDs must match a resumed run.
    Fresh calibration does not consume the saved training random stream.
    """
    config = TrainingConfig() if config is None else config
    if isinstance(config, dict):
        config = TrainingConfig(**config)
    targets = tuple(targets)
    if not targets:
        raise ValueError("Need at least one training context")
    dims = {(t.noise_dim, len(t.context_features)) for t in targets}
    if len(dims) != 1:
        raise ValueError("All conditional targets must share noise/context dimensions")
    noise_dim, context_dim = next(iter(dims))
    if len({float(t.beta) for t in targets}) != 1:
        raise ValueError("This checkpoint fixes beta; supply a single temperature")
    if any(float(t.beta) != float(targets[0].beta) for t in validation_targets):
        raise ValueError("Validation temperature must match the fixed checkpoint")
    mean, scale = _normalization(targets)
    proposal, ratio, p_optimizer, r_optimizer, state = _initialize(
        config, noise_dim, context_dim, mean, scale)
    counters, history, provenance = _counters(), [], []
    if resume is not None:
        previous = load_checkpoint(resume)
        previous_cfg = dict(previous.metadata["config"])
        requested_cfg = json.loads(json.dumps(asdict(config)))
        previous_cfg.pop("rounds")
        requested_cfg.pop("rounds")
        if previous_cfg != requested_cfg:
            raise ValueError("Only rounds may change when resuming")
        if previous.metadata["target_ids"] != [t.target_id for t in targets]:
            raise ValueError("Checkpoint target versions do not match")
        state, counters, history = previous.state, dict(previous.counters), list(previous.history)
        # Calibration costs are saved separately and never become training costs.
        provenance = list(previous.metadata["provenance"])
    cost_batches = [jax.jit(jax.vmap(t.evaluate)) for t in targets]
    contexts = [jnp.asarray(t.context_features) for t in targets]
    timers = {"total_seconds": 0.0, "compilation_and_first_execution_seconds": 0.0,
              "warmed_execution_seconds": 0.0}
    if resume is not None:
        timers.update(previous.metadata.get("timing", {}))

    @jax.jit
    def ratio_update(params, optimizer_state, z, zp, context, label):
        def objective(p):
            return jnp.mean((ratio.predict(z, zp, context, p) - label)**2)
        loss, grad = jax.value_and_grad(objective)(params)
        updates, optimizer_state = r_optimizer.update(grad, optimizer_state, params)
        return optax.apply_updates(params, updates), optimizer_state, loss

    movement_updates = []
    for target, context in zip(targets, contexts):
        feature_fn = jax.vmap(lambda z, t=target: t.features(t.decoder.decode(t.context, z)))

        def make_update(feature_fn=feature_fn, context=context):
            @jax.jit
            def update(p_params, r_params, optimizer_state, key, source, baseline):
                def reward_fn(z, zp):
                    return (acceptance_probability(ratio.predict(z, zp, context, r_params))
                            * bounded_movement(feature_fn(z), feature_fn(zp),
                                               config.movement_scale))
                def objective(params):
                    return score_function_loss(proposal, params, key, source,
                                               context, reward_fn, baseline)
                (loss, reward), gradient = jax.value_and_grad(
                    objective, has_aux=True)(p_params)
                updates, optimizer_state = p_optimizer.update(gradient, optimizer_state, p_params)
                return optax.apply_updates(p_params, updates), optimizer_state, reward, loss
            return update
        movement_updates.append(make_update())

    def metadata():
        return dict(
            format_version=1, config=asdict(config), noise_dim=noise_dim, context_dim=context_dim,
            context_mean=np.asarray(mean).tolist(), context_scale=np.asarray(scale).tolist(),
            temperature=float(targets[0].beta),
            temperature_policy="fixed per checkpoint; changed beta requires recalibration",
            target_ids=[t.target_id for t in targets],
            validation_target_ids=[t.target_id for t in validation_targets],
            decoder_versions=[t.decoder.version for t in targets],
            proposal_version=parameter_version(state.proposal_params),
            ratio_fitted_to_proposal_version=parameter_version(state.proposal_params),
            completed_rounds=int(state.completed_rounds), history=history,
            counters=counters, provenance=provenance, timing=timers,
            normalization="training-context mean; max(training std,1), no cost/label inputs",
            training_state_provenance="independent exact rejection; no burn-in",
        )

    start = time.perf_counter()
    first_ratio = True
    compiled_movement_contexts = set()
    checkpoint_path = None
    for round_index in range(int(state.completed_rounds), config.rounds):
        pools = []
        for target, cost_batch in zip(targets, cost_batches):
            z, costs, key = independent_target_samples(
                target, config.target_samples, state.key, counters, cost_batch,
                config.max_rejection_attempts)
            state = state.replace(key=key)
            pools.append((z, costs))
            provenance.append(dict(round=round_index, context_id=target.target_id,
                                   decoder_version=target.decoder.version,
                                   proposal_version=parameter_version(state.proposal_params),
                                   method="independent_exact_rejection", burn_in=0,
                                   n_samples=config.target_samples))

        def fit_ratio(phase):
            nonlocal state, first_ratio
            datasets = []
            for target, context, cost_batch, (pool, pool_costs) in zip(
                    targets, contexts, cost_batches, pools):
                key, index_key, proposal_key = jax.random.split(state.key, 3)
                indices = jax.random.randint(index_key, (config.pairs_per_context,),
                                             0, len(pool))
                z, costs = pool[indices], pool_costs[indices]
                zp = proposal.sample(proposal_key, z, context, state.proposal_params)
                new_costs = cost_batch(zp)
                _count_oracle(counters, target, len(zp))
                labels = ratio_labels(proposal, state.proposal_params, z, zp, costs,
                                      new_costs, target.beta, context)
                datasets.append((z, zp, labels))
                state = state.replace(key=key)
            for step in range(config.ratio_steps):
                key, context_key, index_key = jax.random.split(state.key, 3)
                i = int(jax.random.randint(context_key, (), 0, len(targets)))
                z, zp, labels = datasets[i]
                indices = jax.random.randint(index_key, (config.batch_size,), 0, len(z))
                tick = time.perf_counter()
                params, opt_state, loss = ratio_update(
                    state.ratio_params, state.ratio_optimizer, z[indices], zp[indices],
                    contexts[i], labels[indices])
                loss_value = float(loss)
                elapsed = time.perf_counter() - tick
                timing_key = ("compilation_and_first_execution_seconds" if first_ratio
                              else "warmed_execution_seconds")
                timers[timing_key] += elapsed
                first_ratio = False
                state = state.replace(ratio_params=params, ratio_optimizer=opt_state,
                                      key=key, ratio_updates=state.ratio_updates + 1)
                if step % 10 == 0 or step == config.ratio_steps - 1:
                    history.append(dict(round=round_index, phase=phase, step=step,
                                        ratio_updates=int(state.ratio_updates),
                                        ratio_batch_mse=loss_value,
                                        proposal_version=parameter_version(state.proposal_params)))

        fit_ratio("ratio_before_proposal")
        for step in range(config.proposal_steps):
            key, context_key, index_key, draw_key = jax.random.split(state.key, 4)
            i = int(jax.random.randint(context_key, (), 0, len(targets)))
            pool = pools[i][0]
            indices = jax.random.randint(index_key, (config.batch_size,), 0, len(pool))
            tick = time.perf_counter()
            params, opt_state, reward, loss = movement_updates[i](
                state.proposal_params, state.ratio_params, state.proposal_optimizer,
                draw_key, pool[indices], state.baseline)
            reward_value, loss_value = float(reward), float(loss)
            elapsed = time.perf_counter() - tick
            timing_key = ("compilation_and_first_execution_seconds"
                          if i not in compiled_movement_contexts
                          else "warmed_execution_seconds")
            timers[timing_key] += elapsed
            compiled_movement_contexts.add(i)
            counters["decoder_calls"] += 2 * config.batch_size
            baseline_before = float(state.baseline)
            state = state.replace(
                proposal_params=params, proposal_optimizer=opt_state, key=key,
                proposal_updates=state.proposal_updates + 1,
                baseline=config.baseline_decay * state.baseline
                + (1 - config.baseline_decay) * reward)
            history.append(dict(round=round_index, phase="proposal", step=step,
                                proposal_updates=int(state.proposal_updates),
                                learned_accepted_movement=reward_value, score_loss=loss_value,
                                baseline_before_update=baseline_before))
        # This phase regenerates both proposed costs and q-dependent labels.
        fit_ratio("ratio_after_proposal")
        state = state.replace(completed_rounds=jnp.asarray(round_index + 1))
        timers["total_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        if out_dir is not None:
            checkpoint_path = save_checkpoint(out_dir, state, metadata())

    proposal = proposal.replace_params(state.proposal_params)
    ratio = ratio.replace_params(state.ratio_params)
    # Fold-in creates an independent audit stream; resume's next update is unchanged.
    calibration_counters = _counters()
    calibration, _ = calibrate(
        tuple(validation_targets) or targets, proposal, ratio,
        jax.random.fold_in(state.key, 87321), config.calibration_samples,
        calibration_counters, config.max_rejection_attempts)
    meta = metadata()
    meta["calibration_counters"] = calibration_counters
    meta["calibration_summaries"] = {k: v["summary"] for k, v in calibration.items()}
    meta["calibration_split"] = "held-out contexts" if validation_targets else "training contexts"
    if out_dir is not None:
        checkpoint_path = save_checkpoint(out_dir, state, meta)
        output = Path(checkpoint_path).parent
        (output / "training_history.json").write_text(json.dumps(history, indent=2))
        arrays = {}
        for i, (_, values) in enumerate(calibration.items()):
            arrays.update({f"context_{i}_{name}": value for name, value in values.items()
                           if name != "summary"})
        arrays["exact_log_ratio"] = np.concatenate([v["exact"] for v in calibration.values()])
        arrays["learned_log_ratio"] = np.concatenate(
            [v["predicted"] for v in calibration.values()])
        np.savez_compressed(output / "calibration.npz", **arrays)
        (output / "calibration.json").write_text(json.dumps({
            "target_ids": list(calibration), "split": meta["calibration_split"],
            "summaries": meta["calibration_summaries"], "counters": calibration_counters
        }, indent=2))
    return TrainingResult(proposal, ratio, history, counters, checkpoint_path,
                          calibration, state, meta)
