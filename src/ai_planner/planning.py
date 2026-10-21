"""Budgeted incumbent planning, deliberately separate from posterior sampling."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from .samplers import COUNTER_NAMES, counter_dict, initialize, make_step


def _is_feasible(target, cost):
    if not hasattr(target, "collision_weight"):
        return True
    if target.collision_weight <= target.goal_weight + target.variation_weight:
        raise ValueError("Collision feasibility inference requires a dominant collision penalty.")
    # Every colliding candidate costs at least collision_weight; every
    # noncolliding one costs at most goal_weight + variation_weight.
    return float(cost) < target.collision_weight


def plan(target, proposal, ratio, key, budget, mode="exact", rho=0.0,
         attempt_cap_factor=8, validate_every=4):
    """Return a best evaluated feasible candidate under an exact oracle budget.

    ``budget`` includes the final validation. Learned-only states are evaluated
    every ``validate_every`` attempts when their cost is unknown. Any remaining
    budget after the attempt cap is spent on independent prior candidates, and
    that count is reported. Refresh is never attempted with fewer than two
    available cost slots when the current cost is unknown.

    Final route validation computes physical metrics in the same single rollout.
    This optimizer's output is not a target-distributed posterior sample.
    """
    if budget < 2:
        raise ValueError("Budget must allow an evaluated candidate and final validation.")
    if attempt_cap_factor < 1 or validate_every < 1:
        raise ValueError("Positive attempt cap factor and validation interval required.")
    if mode not in {"exact", "learned", "prior", "cem"}:
        raise ValueError(f"Unknown planner mode: {mode}")
    if mode == "cem" and not hasattr(target, "metrics"):
        raise ValueError("CEM comparison is implemented for the route task.")
    rho = float(rho)

    compile_seconds = 0.0
    warmed_seconds = 0.0
    started = time.perf_counter()
    chain = initialize(target, key, evaluate=False)
    state = chain.state
    jax.block_until_ready(state)
    warmed_seconds += time.perf_counter() - started
    pool_z, pool_candidates, pool_cost, pool_source = [], [], [], []
    attempt_log = []
    budget_fill_evaluations = 0

    def remember(z, candidate, cost, source):
        pool_z.append(np.asarray(z).copy())
        pool_candidates.append(np.asarray(candidate).copy())
        pool_cost.append(float(cost))
        pool_source.append(source)

    started = time.perf_counter()
    cost_fn = jax.jit(target.cost).lower(state.candidate).compile()
    decoder_fn = jax.jit(lambda z: target.decoder.decode(target.context, z)).lower(state.z).compile()
    validation_fn = jax.jit(target.metrics if hasattr(target, "metrics") else target.cost).lower(state.candidate).compile()
    compile_seconds += time.perf_counter() - started

    def evaluate_current(current, source="periodic_validation"):
        value = cost_fn(current.candidate)
        counts = current.counters.at[2].add(1).at[3].add(target.simulator_steps_per_cost)
        remember(current.z, current.candidate, value, source)
        return current._replace(cost=value, valid=jnp.array(True), counters=counts)

    if mode == "cem":
        from .routes import warm_started_cem
        result = warm_started_cem(target, state.key, budget - 1)
        compile_seconds += result["compilation_seconds"]
        warmed_seconds += result["warmed_execution_seconds"]
        started = time.perf_counter()
        for z, value in zip(result["z"], result["cost"]):
            candidate = decoder_fn(jnp.asarray(z))
            remember(z, candidate, value, "cem")
        # CEM evaluates each decoded candidate once. Re-decoding to save its
        # concrete incumbent/pool here is also charged.
        counts = state.counters.at[0].set(budget - 1).at[1].add(2 * (budget - 1))
        counts = counts.at[2].set(budget - 1).at[3].set((budget - 1) * target.simulator_steps_per_cost)
        state = state._replace(counters=counts, key=jnp.asarray(result["key"]))
        warmed_seconds += time.perf_counter() - started
    else:
        started = time.perf_counter()
        state = evaluate_current(state, "initial")
        warmed_seconds += time.perf_counter() - started
        if mode != "prior":
            started = time.perf_counter()
            step_fn = jax.jit(make_step(target, proposal, ratio, mode, rho)).lower(state).compile()
            compile_seconds += time.perf_counter() - started
        attempts = 0
        cap = attempt_cap_factor * budget
        started = time.perf_counter()
        while int(state.counters[2]) < budget - 1:
            remaining = budget - 1 - int(state.counters[2])
            required = 1 if bool(state.valid) else 2
            if mode == "prior" or attempts >= cap or (mode != "prior" and remaining < required):
                # Independent fill makes the budget view explicit and bounded.
                next_key, draw = jax.random.split(state.key)
                z = jax.random.normal(draw, (target.noise_dim,), dtype=jnp.float64)
                candidate = decoder_fn(z)
                counts = state.counters.at[0].add(1).at[1].add(1)
                state = state._replace(z=z, candidate=candidate, valid=jnp.array(False),
                                       key=next_key, counters=counts)
                source = "prior" if mode == "prior" else "budget_fill_prior"
                state = evaluate_current(state, source)
                if mode != "prior":
                    budget_fill_evaluations += 1
                attempts += 1
                continue

            previous = state
            state, trace = step_fn(state)
            attempts += 1
            attempted_cost = float(trace.proposed_cost)
            if np.isfinite(attempted_cost):
                if not bool(previous.valid):
                    # Missing old costs arise on exact-refresh branches, whose
                    # log ratio is exactly -(J_new-J_old)/beta.
                    if int(trace.branches) != 2:
                        raise RuntimeError("Unexpected uncached non-refresh exact step.")
                    old_cost = attempted_cost + target.beta * float(trace.exact_log_ratio)
                    remember(previous.z, previous.candidate, old_cost, "refresh_current")
                remember(trace.proposed_z, trace.proposed_candidate, attempted_cost, "exact_proposal")
            attempt_log.append(
                (np.asarray(trace.before_z), np.asarray(trace.proposed_z),
                 bool(trace.accepted), int(trace.branches), attempted_cost)
            )
            if (mode == "learned" and attempts % validate_every == 0 and not bool(state.valid)
                    and int(state.counters[2]) < budget - 1):
                state = evaluate_current(state)
        jax.block_until_ready(state)
        warmed_seconds += time.perf_counter() - started

    eligible = np.asarray([_is_feasible(target, cost) for cost in pool_cost])
    best_index = int(np.argmin(np.where(eligible, pool_cost, np.inf))) if eligible.any() else None
    fallback = best_index is None
    started = time.perf_counter()
    if fallback:
        from .routes import holding_chunk
        candidate = holding_chunk(target)
        selected_z = np.full(target.noise_dim, np.nan)
    else:
        candidate = jnp.asarray(pool_candidates[best_index])
        selected_z = pool_z[best_index]
    final = validation_fn(candidate)
    jax.block_until_ready(final)
    counts = state.counters.at[2].add(1).at[3].add(target.simulator_steps_per_cost).at[7].add(1)
    warmed_seconds += time.perf_counter() - started
    if hasattr(final, "cost"):
        final_cost = float(final.cost)
        metrics = {
            field: bool(getattr(final, field)) if field in {"collided", "success"} else float(getattr(final, field))
            for field in final._fields
        }
    else:
        final_cost = float(final)
        metrics = {"cost": final_cost}
    counters = counter_dict(counts)
    if counters["exact_cost_evaluations"] != budget:
        raise RuntimeError("Planner oracle budget accounting is inconsistent.")
    return {
        "z": np.asarray(selected_z),
        "candidate": np.asarray(candidate),
        "cost": final_cost,
        "metrics": metrics,
        "fallback": fallback,
        "best_index": best_index,
        "counters": counters,
        "compile_seconds": compile_seconds,
        "compilation_seconds": compile_seconds,
        "warmed_execution_seconds": warmed_seconds,
        "attempts": counters["transitions"],
        "attempt_cap": attempt_cap_factor * budget,
        "budget_fill_evaluations": budget_fill_evaluations,
        "pool_z": np.asarray(pool_z),
        "pool_candidates": np.asarray(pool_candidates),
        "pool_cost": np.asarray(pool_cost),
        "pool_source": np.asarray(pool_source),
        "attempt_log": attempt_log,
        "key": np.asarray(state.key),
    }
