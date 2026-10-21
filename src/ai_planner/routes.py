"""Frozen two-coefficient route family, causal snapshots and route optimizers.

These engineered paths are not learned policies. Nothing in decoding or the
cheap feature map calls the simulator, searches for collisions, or reads success
labels. The same full-chunk cost is used for every sampler and planner.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri

from .envs.point_mass import (
    Rollout,
    SimulatorConfig,
    Snapshot,
    simulate,
    snapshot_from_state,
)


@dataclass(frozen=True)
class RouteContext:
    name: str
    scenario: str
    snapshot: Snapshot
    goal: tuple[float, float]
    obstacles: tuple[tuple[float, float, float, float], ...]
    cached_plan: tuple[tuple[float, float], ...]
    selected_old_plan: tuple[tuple[float, float], ...]
    retained_suffix: tuple[tuple[float, float], ...]
    completed_tail: tuple[tuple[float, float], ...]
    initial_positions: tuple[tuple[float, float], ...]
    initial_snapshot: Snapshot
    executed_prefix: int = 2
    completion_decay: float = 0.5
    continuation_version: str = "damped-tail-v1"


@dataclass(frozen=True)
class RouteDecoder:
    noise_dim: int = 2
    horizon: int = 20
    bend_amplitude: float = 0.95
    asymmetry_amplitude: float = 0.16
    timing_amplitude: float = 0.20
    cache_weight: float = 0.15
    command_bound: float = 1.15
    version: str = "analytic_route_2d-v1"

    def __post_init__(self):
        if self.noise_dim != 2 or self.horizon < 4:
            raise ValueError("This inspectable route family uses two coefficients and H >= 4.")
        if not 0 <= self.cache_weight <= 1 or not 0 <= self.timing_amplitude < 1:
            raise ValueError("Invalid cache/timing weight.")

    def decode(self, context: RouteContext, z):
        coefficients = 2.0 * ndtr(jnp.asarray(z, dtype=jnp.float64)) - 1.0
        # Three final commands hold the endpoint, reducing terminal tracking lag.
        t = jnp.minimum(jnp.arange(1, self.horizon + 1, dtype=jnp.float64) / (self.horizon - 2), 1.0)
        u = t + self.timing_amplitude * coefficients[1] * jnp.sin(jnp.pi * t) / jnp.pi
        start = jnp.asarray(context.snapshot.position)
        goal = jnp.asarray(context.goal)
        straight = start[None, :] + u[:, None] * (goal - start)[None, :]
        cache = jnp.asarray(context.cached_plan)
        cache_straight = (1 - t[:, None]) * cache[0] + t[:, None] * cache[-1]
        remembered_shape = self.cache_weight * (cache - cache_straight)
        lateral = self.bend_amplitude * coefficients[0] * jnp.sin(jnp.pi * u)
        lateral += self.asymmetry_amplitude * coefficients[1] * jnp.sin(2 * jnp.pi * u)
        result = straight + remembered_shape + jnp.stack((jnp.zeros_like(lateral), lateral), axis=-1)
        # Projection belongs to this frozen decoder; it does not clip proposal noise.
        return jnp.clip(result, -self.command_bound, self.command_bound)


class RouteMetrics(NamedTuple):
    cost: jax.Array
    goal_error: jax.Array
    collided: jax.Array
    command_variation: jax.Array
    path_length: jax.Array
    success: jax.Array


@dataclass(frozen=True)
class RouteTarget:
    context: RouteContext
    beta: float = 0.25
    decoder: RouteDecoder = RouteDecoder()
    simulator: SimulatorConfig = SimulatorConfig()
    goal_weight: float = 1.0
    collision_weight: float = 2.0
    variation_weight: float = 0.08
    goal_scale: float = 0.2
    variation_scale: float = 0.12
    success_tolerance: float = 0.15
    cost_version: str = "bounded_full_chunk-v1"

    def __post_init__(self):
        if self.beta <= 0 or self.goal_scale <= 0 or self.variation_scale <= 0:
            raise ValueError("Positive temperature and cost scales required.")
        if min(self.goal_weight, self.collision_weight, self.variation_weight) < 0:
            raise ValueError("Cost weights must be nonnegative.")
        if len(self.context.cached_plan) != self.decoder.horizon:
            raise ValueError("Cached command chunk and decoder horizon must agree.")

    @property
    def name(self):
        return self.context.name

    @property
    def noise_dim(self):
        return self.decoder.noise_dim

    @property
    def j_max(self):
        return self.goal_weight + self.collision_weight + self.variation_weight

    @property
    def target_id(self):
        # All physical, decoder, cost, snapshot and continuation fields participate.
        serialized = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(serialized.encode()).hexdigest()[:24]

    @property
    def simulator_steps_per_cost(self):
        return self.decoder.horizon * self.simulator.substeps

    @property
    def context_features(self):
        cache = jnp.asarray(self.context.cached_plan)
        indices = jnp.linspace(0, self.decoder.horizon - 1, 5).astype(jnp.int32)
        return jnp.concatenate(
            (
                jnp.asarray(self.context.snapshot.position),
                jnp.asarray(self.context.snapshot.velocity),
                jnp.asarray(self.context.goal),
                jnp.asarray(self.context.obstacles).reshape(-1),
                cache[indices].reshape(-1),
                jnp.asarray([jnp.log(self.beta)]),
            )
        )

    def features(self, candidate):
        # Entire command shape, no simulated or future-success information.
        return jnp.asarray(candidate).reshape(-1) / (self.simulator.command_bound * jnp.sqrt(self.decoder.horizon))

    def rollout(self, candidate) -> Rollout:
        return simulate(self.context.snapshot, candidate, self.context.obstacles, self.simulator)

    def metrics(self, candidate, rollout: Rollout | None = None) -> RouteMetrics:
        rollout = self.rollout(candidate) if rollout is None else rollout
        error_sq = jnp.sum(jnp.square(rollout.final_state.position - jnp.asarray(self.context.goal)))
        variation = jnp.mean(jnp.sum(jnp.square(jnp.diff(candidate, axis=0)), axis=-1))
        collision = rollout.final_state.collided
        cost = (
            self.goal_weight * error_sq / (self.goal_scale**2 + error_sq)
            + self.collision_weight * collision
            + self.variation_weight * variation / (self.variation_scale**2 + variation)
        )
        error = jnp.sqrt(error_sq)
        return RouteMetrics(
            cost, error, collision, variation, rollout.path_length,
            (~collision) & (error <= self.success_tolerance),
        )

    def cost(self, candidate):
        return self.metrics(candidate).cost

    def evaluate(self, z):
        return self.cost(self.decoder.decode(self.context, z))

    def evaluate_metrics(self, z):
        return self.metrics(self.decoder.decode(self.context, z))


def _tuples(array):
    return tuple(tuple(float(x) for x in row) for row in np.asarray(array))


def align_complete(selected_chunk, executed_prefix: int, decay: float = 0.5, command_bound: float = 1.15):
    """Retain the *executed selection's* suffix, then damp its last increment."""
    if not 0 <= decay <= 1:
        raise ValueError("Tail decay must lie in [0,1].")
    if not 0 < executed_prefix < len(selected_chunk):
        raise ValueError("Execute a nonempty strict prefix of the chunk.")
    selected_chunk = jnp.asarray(selected_chunk, dtype=jnp.float64)
    suffix = selected_chunk[executed_prefix:]
    delta = selected_chunk[-1] - selected_chunk[-2]

    def extend(carry, _):
        previous, increment = carry
        increment = decay * increment
        command = jnp.clip(previous + increment, -command_bound, command_bound)
        return (command, increment), command

    _, tail = jax.lax.scan(extend, (suffix[-1], delta), None, length=executed_prefix)
    return jnp.concatenate((suffix, tail), axis=0)


def execute_selected(target: RouteTarget, selected_chunk):
    """Execute one selected candidate's prefix and update its actual causal cache.

    The caller must have validated the selection. This function does not silently
    score or screen it; its K * substeps are physical execution, not cost queries.
    """
    context = target.context
    k = context.executed_prefix
    rollout = simulate(context.snapshot, selected_chunk[:k], context.obstacles, target.simulator)
    completed = align_complete(selected_chunk, k, context.completion_decay, target.simulator.command_bound)
    next_context = replace(
        context,
        name=context.name + "/next",
        snapshot=snapshot_from_state(rollout.final_state),
        cached_plan=_tuples(completed),
        selected_old_plan=_tuples(selected_chunk),
        retained_suffix=_tuples(selected_chunk[k:]),
        completed_tail=_tuples(completed[-k:]),
        initial_positions=_tuples(rollout.positions),
        initial_snapshot=context.snapshot,
    )
    return replace(target, context=next_context), rollout


def holding_chunk(target: RouteTarget):
    """A deterministic holding/braking fallback; caller records its use."""
    return jnp.tile(jnp.asarray(target.context.snapshot.position), (target.decoder.horizon, 1))


def make_route_context(scenario="unchanged", *, variant=0.0, name=None):
    """Create a shared pre-disturbance history by actually executing a prefix."""
    if scenario not in {"unchanged", "mild_correction", "route_invalidation"}:
        raise ValueError(f"Unknown scenario {scenario!r}")
    config = SimulatorConfig()
    decoder = RouteDecoder()
    initial = Snapshot(position=(-0.95, -0.025 + 0.04 * variant))
    goal = (0.95, 0.025 * variant)
    t = np.minimum(np.arange(1, decoder.horizon + 1) / (decoder.horizon - 2), 1.0)
    nominal = np.column_stack(
        (
            initial.position[0] + t * (goal[0] - initial.position[0]),
            (1 - t) * initial.position[1] + t * goal[1] + (0.62 + 0.025 * variant) * np.sin(np.pi * t),
        )
    )
    obstacles = ((0.015 * variant, 0.0, 0.25, 1.0), (0.0, 0.68, 0.52, 0.0))
    prefix = 2
    actual = simulate(initial, nominal[:prefix], obstacles, config)
    snapshot = snapshot_from_state(actual.final_state)
    completed = align_complete(nominal, prefix, 0.5, config.command_bound)
    # Disturbance is applied only after the causal prefix execution.
    if scenario == "mild_correction":
        goal = (0.95, -0.12 + 0.025 * variant)
        obstacles = ((0.03 + 0.015 * variant, -0.025, 0.25, 1.0), obstacles[1])
    elif scenario == "route_invalidation":
        obstacles = (obstacles[0], (0.015 * variant, 0.68, 0.52, 1.0))
    return RouteContext(
        name=name or f"{scenario}_{variant:+.2f}",
        scenario=scenario,
        snapshot=snapshot,
        goal=goal,
        obstacles=obstacles,
        cached_plan=_tuples(completed),
        selected_old_plan=_tuples(nominal),
        retained_suffix=_tuples(nominal[prefix:]),
        completed_tail=_tuples(completed[-prefix:]),
        initial_positions=_tuples(actual.positions),
        initial_snapshot=initial,
        executed_prefix=prefix,
    )


def make_route_targets(beta=0.25):
    """Nine distinct contexts, with scenario coverage in every held-out split."""
    scenarios = ("unchanged", "mild_correction", "route_invalidation")
    return {
        split: tuple(
            RouteTarget(make_route_context(scenario, variant=variant, name=f"{split}_{scenario}"), beta=beta)
            for scenario in scenarios
        )
        for split, variant in (("train", -0.6), ("validation", 0.25), ("test", 0.8))
    }


def support_grid(target: RouteTarget, resolution=15):
    """Raw support inspection and quadrature arrays; every grid cost is an oracle.

    The midpoint grid has a uniform coefficient prior. Returned rollout arrays
    permit plotting without any hidden simulator evaluations.
    """
    axis = -1.0 + (np.arange(resolution) + 0.5) * 2.0 / resolution
    first, second = np.meshgrid(axis, axis, indexing="ij")
    coefficients = np.column_stack((first.ravel(), second.ravel()))
    z = ndtri((jnp.asarray(coefficients) + 1.0) / 2.0)

    def evaluate_one(noise):
        chunk = target.decoder.decode(target.context, noise)
        rollout = target.rollout(chunk)
        metrics = target.metrics(chunk, rollout)
        return chunk, rollout.positions, metrics

    chunks, positions, metrics = jax.jit(jax.vmap(evaluate_one))(z)
    cost = np.asarray(metrics.cost)
    log_weights = -cost / target.beta
    weights = np.exp(log_weights - log_weights.max())
    masses = weights / weights.sum()
    return {
        "coefficient_axis": axis,
        "coefficients": coefficients,
        "z": np.asarray(z),
        "commands": np.asarray(chunks),
        "positions": np.asarray(positions),
        "cost": cost,
        "goal_error": np.asarray(metrics.goal_error),
        "collision": np.asarray(metrics.collided),
        "success": np.asarray(metrics.success),
        "target_mass": masses,
        "normalizer": float(np.exp(log_weights.max()) * weights.mean()),
        "oracle_evaluations": resolution**2,
        "simulator_steps": resolution**2 * target.simulator_steps_per_cost,
    }


class ContinuationResult(NamedTuple):
    prefix_positions: jax.Array
    prefix_state: object
    prefix_goal_error: jax.Array
    prefix_cost: jax.Array
    continuation_positions: jax.Array
    final_state: object
    cost: jax.Array
    simulator_steps: int


def prefix_then_rollout(target: RouteTarget, candidate, *, preserve_suffix=True, replans=2):
    """Small optional deterministic suffix continuation, with no recursive search.

    Each continuation executes K commands, then aligns/completes its own cached
    suffix. With discarded suffixes, the base policy linearly targets the goal
    from the actual prefix successor, independent of the original candidate.
    """
    if replans < 1:
        raise ValueError("At least one continuation replan is required.")
    k = target.context.executed_prefix
    prefix = simulate(target.context.snapshot, candidate[:k], target.context.obstacles, target.simulator)
    if preserve_suffix:
        cache = align_complete(candidate, k, target.context.completion_decay, target.simulator.command_bound)
    else:
        t = jnp.linspace(1.0 / target.decoder.horizon, 1.0, target.decoder.horizon)
        cache = prefix.final_state.position + t[:, None] * (
            jnp.asarray(target.context.goal) - prefix.final_state.position
        )

    def continue_once(carry, _):
        state, current_cache = carry
        rollout = simulate(state, current_cache[:k], target.context.obstacles, target.simulator)
        next_cache = align_complete(current_cache, k, target.context.completion_decay, target.simulator.command_bound)
        return (rollout.final_state, next_cache), (rollout.positions[1:], current_cache[:k])

    (final, _), (paths, continuation_commands) = jax.lax.scan(
        continue_once, (prefix.final_state, cache), None, length=replans
    )
    final_error_sq = jnp.sum(jnp.square(final.position - jnp.asarray(target.context.goal)))
    executed_commands = jnp.concatenate((candidate[:k], continuation_commands.reshape(-1, 2)))
    variation = jnp.mean(jnp.sum(jnp.square(jnp.diff(executed_commands, axis=0)), axis=-1))
    score = (
        target.goal_weight * final_error_sq / (target.goal_scale**2 + final_error_sq)
        + target.collision_weight * final.collided
        + target.variation_weight * variation / (target.variation_scale**2 + variation)
    )
    return ContinuationResult(
        prefix.positions,
        prefix.final_state,
        jnp.linalg.norm(prefix.final_state.position - jnp.asarray(target.context.goal)),
        target.metrics(candidate[:k], prefix).cost,
        jnp.concatenate((prefix.final_state.position[None], paths.reshape(-1, 2))),
        final,
        score,
        k * (replans + 1) * target.simulator.substeps,
    )


def warm_started_cem(target: RouteTarget, key, budget: int, *, population=8, elite_fraction=0.25):
    """CEM optimizer over the same two Gaussian-noise coordinates.

    Warm start favors the previously selected positive-bend route. All sampled
    candidates are evaluated, including collisions; no sample is clipped in noise
    space. The returned incumbent is explicitly not a posterior draw.
    """
    if budget < 1 or population < 2 or not 0 < elite_fraction <= 1:
        raise ValueError("Positive budget and a valid population/elite fraction required.")
    mean = jnp.asarray([0.55, 0.0], dtype=jnp.float64)
    scale = jnp.ones(2, dtype=jnp.float64)
    zs, costs, success, collisions = [], [], [], []
    evaluated = 0
    batched = jax.jit(jax.vmap(target.evaluate_metrics))
    compiled = {}
    compilation_seconds = 0.0
    execution_seconds = 0.0
    while evaluated < budget:
        started = time.perf_counter()
        size = min(population, budget - evaluated)
        key, draw = jax.random.split(key)
        noise = mean + scale * jax.random.normal(draw, (size, 2), dtype=jnp.float64)
        execution_seconds += time.perf_counter() - started
        if size not in compiled:
            started = time.perf_counter()
            compiled[size] = batched.lower(noise).compile()
            compilation_seconds += time.perf_counter() - started
        started = time.perf_counter()
        metrics = compiled[size](noise)
        order = np.argsort(np.asarray(metrics.cost))
        elites = noise[order[: max(1, int(np.ceil(size * elite_fraction)))]]
        mean = 0.3 * mean + 0.7 * jnp.mean(elites, axis=0)
        scale = jnp.maximum(0.15, 0.3 * scale + 0.7 * jnp.std(elites, axis=0))
        zs.append(np.asarray(noise))
        costs.append(np.asarray(metrics.cost))
        success.append(np.asarray(metrics.success))
        collisions.append(np.asarray(metrics.collided))
        evaluated += size
        execution_seconds += time.perf_counter() - started
    all_z, all_cost = np.concatenate(zs), np.concatenate(costs)
    all_collision = np.concatenate(collisions)
    eligible = np.where(~all_collision, all_cost, np.inf)
    index = int(np.argmin(eligible)) if np.any(~all_collision) else None
    return {
        "z": all_z,
        "cost": all_cost,
        "success": np.concatenate(success),
        "collision": all_collision,
        "best_index": index,
        "best_z": None if index is None else all_z[index],
        "fallback": index is None,
        "oracle_evaluations": evaluated,
        "decoder_calls": evaluated,
        "simulator_steps": evaluated * target.simulator_steps_per_cost,
        "compilation_seconds": compilation_seconds,
        "warmed_execution_seconds": execution_seconds,
        "key": np.asarray(key),
    }
