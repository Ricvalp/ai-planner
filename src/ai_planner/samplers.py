"""Frozen scalar-chain kernels. Scalar lax.cond genuinely skips oracle branches.

Do not vmap a mixed kernel across chains: batching a conditional can turn it into
a select and evaluate both branches. Independent chains are dispatched separately.
"""
from dataclasses import dataclass
from typing import NamedTuple
import time
import numpy as np
import jax
import jax.numpy as jnp


COUNTER_NAMES = (
    "transitions", "decoder_calls", "exact_cost_evaluations", "simulator_steps",
    "training_oracle_evaluations", "refresh_current_evaluations",
    "refresh_proposed_evaluations", "final_validation_evaluations", "audit_evaluations",
)


def counter_dict(values):
    return dict(zip(COUNTER_NAMES, map(int, np.asarray(values))))


def normal_log_prob(z):
    return -.5 * jnp.sum(z*z + jnp.log(2*jnp.pi), axis=-1)


class KernelState(NamedTuple):
    z: object
    candidate: object
    cost: object
    valid: object
    key: object
    counters: object


@dataclass
class Chain:
    state: KernelState
    target_id: str


class StepTrace(NamedTuple):
    before_z: object
    z: object
    proposed_z: object
    candidate: object
    proposed_candidate: object
    accepted: object
    exact_log_ratio: object
    learned_log_ratio: object
    log_uniform: object
    cost: object
    valid: object
    proposed_cost: object
    branches: object
    counters: object


def initialize(target, key, z=None, evaluate=False):
    key, initial_key = jax.random.split(key)
    z = jax.random.normal(initial_key, (target.noise_dim,), dtype=jnp.float64) if z is None else jnp.asarray(z, dtype=jnp.float64)
    candidate = target.decoder.decode(target.context, z)
    cost = target.cost(candidate) if evaluate else jnp.array(jnp.nan)
    counters = jnp.zeros(len(COUNTER_NAMES), dtype=jnp.int64).at[1].set(1)
    if evaluate:
        counters = counters.at[2].set(1).at[3].set(target.simulator_steps_per_cost)
    return Chain(KernelState(z, candidate, cost, jnp.array(evaluate), key, counters), target.target_id)


def rebind(chain, target):
    """New target/decoder/cost/context invalidates both the decoded object and cost."""
    if chain.target_id == target.target_id:
        return chain
    s = chain.state
    candidate = target.decoder.decode(target.context, s.z)
    return Chain(s._replace(candidate=candidate, cost=jnp.array(jnp.nan), valid=jnp.array(False),
                            counters=s.counters.at[1].add(1)), target.target_id)


def make_step(target, proposal, ratio=None, mode="exact", rho=0.):
    rho = float(rho)
    if mode not in ("exact", "learned", "prior"):
        raise ValueError(mode)
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0,1]")
    if mode == "learned" and ratio is None:
        raise ValueError("learned acceptance needs a frozen ratio")
    ctx = target.context_features

    def step(state, unused=None):
        key, branch_key, proposal_key, uniform_key = jax.random.split(state.key, 4)
        refresh = jax.random.bernoulli(branch_key, rho)
        # Sampling is cheap; no simulated/cost quantity enters either network.
        zp = jax.lax.cond(refresh | jnp.asarray(mode == "prior"),
                          lambda _: jax.random.normal(proposal_key, state.z.shape, dtype=jnp.float64),
                          lambda _: proposal.sample(proposal_key, state.z, ctx), None)
        cp = target.decoder.decode(target.context, zp)
        counts = state.counters.at[0].add(1).at[1].add(1)
        nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
        d = ratio(state.z, zp, ctx) if ratio is not None else nan
        u = jax.random.uniform(uniform_key, (), minval=jnp.finfo(jnp.float64).tiny,
                               maxval=1., dtype=jnp.float64)
        logu = jnp.log(u)

        def expensive(_):
            oldcost = jax.lax.cond(state.valid, lambda _: state.cost,
                                   lambda _: target.cost(state.candidate), None)
            newcost = target.cost(cp)
            missing = (~state.valid).astype(jnp.int64)
            ncost = 1 + missing
            c = counts.at[2].add(ncost).at[3].add(ncost*target.simulator_steps_per_cost)
            c = c.at[5].add(refresh.astype(jnp.int64)*missing).at[6].add(refresh.astype(jnp.int64))
            cost_ratio = -(newcost-oldcost) / target.beta
            ell = jax.lax.cond(refresh, lambda _: cost_ratio,
                               lambda _: cost_ratio + normal_log_prob(zp)-normal_log_prob(state.z)
                               + proposal.log_prob(state.z, zp, ctx)-proposal.log_prob(zp, state.z, ctx), None)
            accept = logu < jnp.minimum(0., ell)
            return accept, ell, jnp.where(accept, newcost, oldcost), jnp.array(True), newcost, c

        def cheap(_):
            accept = jnp.array(True) if mode == "prior" else logu < jnp.minimum(0., d)
            return accept, nan, jnp.where(accept, nan, state.cost), state.valid & ~accept, nan, counts

        if mode == "exact":
            accepted, ell, cost, valid, newcost, counts = expensive(None)
        elif rho == 0:
            accepted, ell, cost, valid, newcost, counts = cheap(None)
        else:
            accepted, ell, cost, valid, newcost, counts = jax.lax.cond(refresh, expensive, cheap, None)
        z = jnp.where(accepted, zp, state.z)
        candidate = jnp.where(accepted, cp, state.candidate)
        branch = jnp.where(refresh, 2, {"exact": 0, "learned": 1, "prior": 3}[mode])
        next_state = KernelState(z, candidate, cost, valid, key, counts)
        trace = StepTrace(state.z, z, zp, candidate, cp, accepted, ell, d, logu, cost, valid,
                          newcost, branch, counts)
        return next_state, trace
    return step


def run_chain(target, proposal, chain, steps, ratio=None, mode="exact", rho=0.):
    chain = rebind(chain, target)
    step = make_step(target, proposal, ratio, mode, rho)
    run = jax.jit(lambda s: jax.lax.scan(step, s, None, length=steps))
    start = time.perf_counter()
    executable = run.lower(chain.state).compile()
    compile_seconds = time.perf_counter()-start
    start = time.perf_counter()
    state, trace = executable(chain.state)
    jax.block_until_ready(state)
    seconds = time.perf_counter()-start
    return Chain(state, target.target_id), {k: np.asarray(v) for k, v in trace._asdict().items()}, {
        "compilation_seconds": compile_seconds, "warmed_execution_seconds": seconds,
        **counter_dict(state.counters)}


def run_chains(target, proposal, key, steps, n_chains=4, ratio=None, mode="exact", rho=0.):
    """One compiled scalar scan reused across independent chain keys."""
    keys = jax.random.split(key, n_chains)
    initial = [initialize(target, k) for k in keys]
    run = jax.jit(lambda s: jax.lax.scan(make_step(target, proposal, ratio, mode, rho),
                                       s, None, length=steps))
    start = time.perf_counter()
    executable = run.lower(initial[0].state).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    outputs = [executable(c.state) for c in initial]
    jax.block_until_ready(outputs)
    seconds = time.perf_counter()-start
    traces = {k: np.stack([np.asarray(o[1]._asdict()[k]) for o in outputs])
              for k in StepTrace._fields}
    counts = sum((np.asarray(o[0].counters) for o in outputs), np.zeros(len(COUNTER_NAMES), dtype=int))
    return traces, {"compilation_seconds": compile_seconds, "warmed_execution_seconds": seconds,
                    **counter_dict(counts)}


def rejection_sample(target, n, key, batch_size=512, max_evaluations=2_000_000):
    """Independent exact target samples; discarded batch surplus is charged."""
    samples = []; total = 0; accepted = 0
    evaluate = jax.jit(jax.vmap(target.evaluate))
    while accepted < n:
        if total + batch_size > max_evaluations:
            raise RuntimeError("rejection oracle budget exhausted")
        key, kz, ku = jax.random.split(key, 3)
        z = jax.random.normal(kz, (batch_size, target.noise_dim), dtype=jnp.float64)
        costs = evaluate(z)
        keep = np.asarray(jnp.log(jax.random.uniform(ku, (batch_size,), minval=jnp.finfo(jnp.float64).tiny,
                                                   dtype=jnp.float64)) < -costs / target.beta)
        a = np.asarray(z)[keep]
        samples.append(a); accepted += len(a); total += batch_size
    counts = dict.fromkeys(COUNTER_NAMES, 0)
    counts.update(decoder_calls=total, exact_cost_evaluations=total,
                  simulator_steps=total*target.simulator_steps_per_cost)
    return np.concatenate(samples)[:n], counts, key


def save_chain(path, chain):
    np.savez_compressed(path, **chain.state._asdict(), target_id=chain.target_id)


def load_chain(path):
    with np.load(path) as f:
        state = KernelState(*(jnp.asarray(f[k]) for k in KernelState._fields))
        return Chain(state, str(f["target_id"]))
