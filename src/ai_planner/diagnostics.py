"""Independent, float64 probability audits.

The finite grid is its own discrete model. Its proposal probabilities are
normalized on the grid and its Hastings ratios are recomputed from those
probabilities; it is not a transition matrix for the continuous sampler.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from scipy.special import logsumexp, ndtri


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """TV for two normalized finite probability vectors."""
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def effective_sample_size(values: np.ndarray) -> float:
    """Sum per-chain ESS using the initial-positive paired autocorrelation sum.

    The last axis is time. Constant chains receive ESS zero: this convention
    flags a chain that never explores instead of suggesting independent draws.
    These short-chain estimates are diagnostics, not convergence certificates.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("ESS expects [time] or [chain,time] observations")
    total = 0.0
    for row in values:
        n = len(row)
        if n < 3:
            continue
        centered = row - row.mean()
        if np.dot(centered, centered) < 1e-25:
            continue
        fft = np.fft.rfft(centered, n=1 << (2 * n - 1).bit_length())
        cov = np.fft.irfft(fft * fft.conj())[:n] / n
        corr = cov / cov[0]
        positive_sum = 0.0
        last_pair = np.inf
        # Geyer pairs rho_1+rho_2, rho_3+rho_4, ...; monotone truncation
        # provides a conservative estimate for the positive-correlation cases.
        for k in range(1, n - 1, 2):
            pair = float(corr[k] + corr[k + 1])
            if pair <= 0:
                break
            pair = min(pair, last_pair)
            positive_sum += pair
            last_pair = pair
        total += n / max(1.0, 1.0 + 2.0 * positive_sum)
    return float(total)


def coarse_histogram_tv(
    samples_y: np.ndarray,
    grid_y: np.ndarray,
    grid_density: np.ndarray,
    bins: int = 10,
) -> float:
    """Sampling-dependent histogram TV on a fixed compact-domain partition.

    Reference density is integrated using equally spaced midpoint cells. This
    is deliberately not called exact continuous total variation.
    """
    samples = np.asarray(samples_y).reshape(-1, 2)
    grid = np.asarray(grid_y).reshape(-1, 2)
    weights = np.asarray(grid_density, dtype=np.float64).ravel()
    if len(samples) == 0 or weights.sum() <= 0:
        raise ValueError("Nonempty samples and positive reference mass required")
    edges = np.linspace(-1, 1, bins + 1)
    observed = np.histogram2d(samples[:, 0], samples[:, 1], bins=(edges, edges))[0]
    reference = np.histogram2d(grid[:, 0], grid[:, 1], bins=(edges, edges), weights=weights)[0]
    if observed.sum() != len(samples):
        raise ValueError("Descriptor samples must lie in [-1,1]^2")
    return total_variation(observed / observed.sum(), reference / reference.sum())


def _probability_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("Strictly positive finite probability masses required")
    return values / values.sum()


def _proposal_matrix(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n, n) or np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("A finite, strictly positive square proposal is required")
    return values / values.sum(axis=1, keepdims=True)


def discrete_log_ratio(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Element [i,j] is log[p_j q_ji / (p_i q_ij)]."""
    log_flow = np.log(p)[:, None] + np.log(q)
    return log_flow.T - log_flow


def transition_matrix(q: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Include rejection mass on the diagonal, including proposed self moves."""
    transition = q * np.exp(np.minimum(0.0, logits))
    transition[np.diag_indices_from(transition)] += 1.0 - transition.sum(axis=1)
    return transition


def stationary_distribution(kernel: np.ndarray) -> np.ndarray:
    """Solve the finite stationary system and check numerical residuals."""
    from scipy.linalg import solve

    n = len(kernel)
    equation = kernel.T - np.eye(n)
    equation[-1] = 1.0
    right = np.zeros(n)
    right[-1] = 1.0
    law = solve(equation, right, assume_a="gen")
    if law.min() < -1e-10:
        raise ArithmeticError("Stationary solve returned negative mass")
    law = np.maximum(law, 0.0)  # remove only floating-point solve roundoff
    law /= law.sum()
    if np.max(np.abs(law @ kernel - law)) > 1e-9:
        raise ArithmeticError("Stationary solve failed its residual check")
    return law


def finite_state_audit(
    p: np.ndarray,
    q: np.ndarray,
    logits: np.ndarray,
    refresh_q: np.ndarray,
    *,
    j_max: float,
    beta: float,
    rhos: Iterable[float] = (0.0, 0.1, 0.3, 1.0),
    steps: Iterable[int] = (0, 1, 5, 20, 100),
    initial: np.ndarray | None = None,
) -> dict:
    """Exactly sum accepted-flow, ratio, stationary, and finite-step bounds.

    ``refresh_q`` must be a state-independent prior proposal for which
    p_i / refresh_q_i is proportional to exp(-J_i/beta), 0<=J_i<=j_max.
    The minorization is explicitly checked before its bounds are reported.
    Unlike Monte Carlo estimates, all expectations here are finite sums.
    """
    p = _probability_vector(p)
    n = len(p)
    q = _proposal_matrix(q, n)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != q.shape or not np.all(np.isfinite(logits)):
        raise ValueError("A finite logit for every ordered pair is required")
    if np.max(np.abs(logits + logits.T)) > 1e-9:
        raise ValueError("Accepted-flow identity requires antisymmetric logits")
    if beta <= 0 or j_max < 0:
        raise ValueError("beta must be positive and j_max nonnegative")
    refresh_q = np.asarray(refresh_q, dtype=np.float64)
    if refresh_q.ndim == 1:
        refresh_q = np.tile(_probability_vector(refresh_q), (n, 1))
    refresh_q = _proposal_matrix(refresh_q, n)
    if not np.allclose(refresh_q, refresh_q[:1], atol=1e-12):
        raise ValueError("Exact refresh must use an independence proposal")
    ell = discrete_log_ratio(p, q)
    exact_acceptance = np.exp(np.minimum(0.0, ell))
    acceptance = np.exp(np.minimum(0.0, logits))
    kernel = transition_matrix(q, logits)
    reference = transition_matrix(refresh_q, discrete_log_ratio(p, refresh_q))
    minorization = float(np.exp(-j_max / beta))
    if np.min(reference - minorization * p[None, :]) < -1e-11:
        raise ValueError("The supplied prior, target, and cost bound violate minorization")
    flow = p[:, None] * q
    accepted_flow = flow * acceptance
    abar = float(accepted_flow.sum())
    balance = float(np.sum(accepted_flow * (logits - ell)))
    ratio_mse = float(np.sum(flow * (logits - ell) ** 2))
    epsilon = float(np.sum(flow * np.abs(acceptance - exact_acceptance)))
    normalized_flow = accepted_flow / abar
    # Log-domain expression avoids underflow in the accepted density ratio.
    log_accepted = np.log(flow) + np.minimum(0.0, logits)
    kl = float(np.sum(normalized_flow * (log_accepted - log_accepted.T)))
    balance_bound = float(np.sqrt(max(0.0, abar * balance / 2)))
    one_step_tv = total_variation(p @ kernel, p)
    exact_kernel = transition_matrix(q, ell)
    exact_flow = p[:, None] * exact_kernel
    exact_defect = float(np.max(np.abs(exact_flow - exact_flow.T)))
    chosen_steps = sorted(set(int(step) for step in steps))
    if not chosen_steps or chosen_steps[0] < 0:
        raise ValueError("Finite-step audits require nonnegative steps")
    if initial is None:
        initial = np.zeros(n)
        initial[0] = 1.0
    initial = np.asarray(initial, dtype=np.float64)
    if initial.shape != (n,) or np.any(initial < 0) or initial.sum() <= 0:
        raise ValueError("Invalid starting law")
    initial /= initial.sum()
    initial_tv = total_variation(initial, p)
    refresh_results = []
    for rho in rhos:
        rho = float(rho)
        if not 0 <= rho <= 1:
            raise ValueError("rho must lie in [0,1]")
        mixed = (1 - rho) * kernel + rho * reference
        stationary = stationary_distribution(mixed)
        log_kappa = float(np.log(rho) - j_max / beta) if rho else None
        kappa = float(np.exp(log_kappa)) if rho else 0.0
        numerator = (1 - rho) * epsilon
        if rho and kappa > 0:
            stationary_bound = numerator / kappa
        elif rho and numerator == 0:
            stationary_bound = 0.0
        else:
            stationary_bound = None
        law = initial.copy()
        finite = []
        for step in range(max(chosen_steps) + 1):
            if step in chosen_steps:
                if rho:
                    contraction = float(np.exp(step * np.log1p(-kappa))) if kappa < 1 else float(step == 0)
                    # Finite geometric sum remains stable when kappa is tiny.
                    geometric = -np.expm1(step * np.log1p(-kappa)) / kappa if 0 < kappa < 1 else float(step > 0) if kappa == 1 else float(step)
                    bound = float(contraction * initial_tv + numerator * geometric)
                else:
                    bound = None
                finite.append({"step": step, "tv": total_variation(law, p), "bound": bound})
            law = law @ mixed
        refresh_results.append({
            "rho": rho,
            "kappa": kappa,
            "log_kappa": log_kappa,
            "stationary_tv": total_variation(stationary, p),
            "stationary_bound": stationary_bound,
            "bound_vacuous": stationary_bound is None or stationary_bound >= 1.0,
            "stationary_residual": float(np.max(np.abs(stationary @ mixed - stationary))),
            "one_step_tv": total_variation(p @ mixed, p),
            "finite_steps": finite,
        })
    return {
        "expectation_kind": "exact finite sum; separate discrete model",
        "num_states": n,
        "abar": abar,
        "L_bal": balance,
        "L_ratio": ratio_mse,
        "epsilon_a": epsilon,
        "accepted_flow_KL": kl,
        "accepted_flow_identity_error": abs(kl - balance / abar),
        "one_step_tv": one_step_tv,
        "flow_bound": balance_bound,
        "ratio_bound": float(np.sqrt(ratio_mse)),
        "exact_detailed_balance_error": exact_defect,
        "exact_stationarity_tv": total_variation(p @ exact_kernel, p),
        "refresh": refresh_results,
        "roundoff_note": "Only the nonnegative square-root argument is rounded up from negative floating-point zero; raw L_bal is retained.",
    }


def finite_grid_audit(
    target,
    proposal=None,
    ratio: Callable | None = None,
    grid_size: int = 11,
    rhos: Iterable[float] = (0.0, 0.1, 0.3, 1.0),
) -> dict:
    """Audit exact, controlled-error, and optionally learned logits on a grid.

    Uniform midpoint descriptor masses represent the transformed Gaussian
    prior. Continuous q density is multiplied by dz/dy and row-normalized.
    All discrete Hastings ratios are then recomputed, including row constants.
    """
    import jax
    import jax.numpy as jnp

    if target.decoder.noise_dim != 2:
        raise ValueError("The complete finite-grid visualization is two dimensional")
    if grid_size < 3 or grid_size > 21:
        raise ValueError("Use a small finite audit grid between 3 and 21")
    axis = -1 + (np.arange(grid_size) + 0.5) * 2 / grid_size
    y = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    z = ndtri((y + 1) / 2)
    costs = np.asarray(jax.vmap(target.evaluate)(jnp.asarray(z)), dtype=np.float64)
    if costs.shape != (len(z),):
        raise ValueError("target.evaluate(z) must return its scalar exact cost")
    log_p = -costs / target.beta
    p = np.exp(log_p - logsumexp(log_p))
    context = target.context_features
    if callable(context):
        context = context()
    if proposal is None:
        eta = 0.55
        delta = z[None, :, :] - np.sqrt(1 - eta**2) * z[:, None, :]
        log_q = -0.5 * (delta**2).sum(axis=-1) / eta**2 - 2 * np.log(eta) - np.log(2 * np.pi)
    else:
        log_q = np.asarray(jax.vmap(lambda source: jax.vmap(
            lambda destination: proposal.log_prob(destination, source, context)
        )(jnp.asarray(z)))(jnp.asarray(z)), dtype=np.float64)
    # dz/dy = 1/(4 phi(z)); the constant 1/4 cancels per row.
    log_prior = -0.5 * (z**2).sum(axis=-1) - np.log(2 * np.pi)
    log_q = log_q - log_prior[None, :]
    log_q -= logsumexp(log_q, axis=1, keepdims=True)
    q = np.exp(log_q)
    # Probability-interface values in this small grid should remain representable.
    if np.any(q == 0):
        raise ArithmeticError("Discrete q underflow: use a less local proposal or smaller grid")
    ell = discrete_log_ratio(p, q)
    error = 1.25 * (y[None, :, 0] - y[:, None, 0])
    variants = {"exact": ell, "controlled_error": ell + error}
    if ratio is not None:
        variants["learned"] = np.asarray(jax.vmap(lambda source: jax.vmap(
            lambda destination: ratio(source, destination, context)
        )(jnp.asarray(z)))(jnp.asarray(z)), dtype=np.float64)
    kwargs = {"j_max": float(target.j_max), "beta": float(target.beta), "rhos": tuple(rhos)}
    audits = {name: finite_state_audit(p, q, d, np.ones(len(p)), **kwargs) for name, d in variants.items()}
    wrong = p * np.exp(2 * y[:, 0])
    wrong /= wrong.sum()
    identity = []
    for rate in (0.2, 0.01, 0.0001):
        almost_identity = (1 - rate) * np.eye(len(p)) + rate * wrong[None, :]
        identity.append({
            "jump_probability": rate,
            "one_step_tv": total_variation(p @ almost_identity, p),
            "stationary_tv": total_variation(wrong, p),
            "expected_stationary_tv": total_variation(wrong, p),
        })
    return {
        "context_id": str(getattr(target, "target_id", "target")),
        "grid_size": grid_size,
        "beta": float(target.beta),
        "j_max": float(target.j_max),
        "prior_mass": "uniform midpoint descriptor cells",
        "proposal_definition": "continuous density times dz/dy, row-normalized; discrete Hastings recomputed",
        "states_y": y.tolist(),
        "target_mass": p.tolist(),
        "variants": audits,
        "identity_failure": identity,
        "identity_failure_note": "P=(1-epsilon)I+epsilon*q has stationary q for every positive epsilon; a small one-step defect alone gives no stationary certificate.",
    }
