"""Exact MH labels and algebraically antisymmetric learned log ratios."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

from .proposals import normal_log_prob


def acceptance_probability(logit):
    return jnp.exp(jnp.minimum(0.0, logit))


def exact_log_ratio(z, zp, cost, costp, beta, proposal, context, params=None):
    """Costs are explicit values: no hidden evaluation or decoder Jacobian."""
    return (normal_log_prob(zp) - normal_log_prob(z)
            - (costp - cost) / beta
            + proposal.log_prob(z, zp, context, params)
            - proposal.log_prob(zp, z, context, params))


class PairNetwork(nn.Module):
    hidden_dims: tuple[int, ...] = (32, 32, 32)

    @nn.compact
    def __call__(self, z, zp, context):
        context = jnp.broadcast_to(context, (*z.shape[:-1], context.shape[-1]))
        h = jnp.concatenate((context, z, zp), axis=-1)
        for width in self.hidden_dims:
            h = nn.tanh(nn.Dense(width, dtype=jnp.float64, param_dtype=jnp.float64)(h))
        return nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64)(h)[..., 0]


@dataclass(frozen=True)
class RatioModel:
    model: PairNetwork
    params: Any
    context_mean: Any
    context_scale: Any

    @classmethod
    def create(cls, key, noise_dim, context_dim, hidden_dims=(32, 32, 32),
               context_mean=None, context_scale=None):
        model = PairNetwork(tuple(hidden_dims))
        params = model.init(key, jnp.zeros(noise_dim), jnp.zeros(noise_dim),
                            jnp.zeros(context_dim))["params"]
        return cls(model, params,
                   jnp.zeros(context_dim) if context_mean is None else jnp.asarray(context_mean),
                   jnp.ones(context_dim) if context_scale is None else jnp.asarray(context_scale))

    def predict(self, z, zp, context, params=None):
        context = (jnp.asarray(context) - self.context_mean) / self.context_scale
        variables = {"params": self.params if params is None else params}
        return (self.model.apply(variables, z, zp, context)
                - self.model.apply(variables, zp, z, context))

    def __call__(self, z, zp, context):
        return self.predict(z, zp, context)

    def replace_params(self, params):
        return replace(self, params=params)


@dataclass(frozen=True)
class PerturbedRatio:
    """Controlled antisymmetric perturbation of another frozen ratio callable."""
    base: Any
    amplitude: float = 1.0
    coordinate: int = 0

    def __call__(self, z, zp, context):
        return self.base(z, zp, context) + self.amplitude * (
            jnp.tanh(zp[..., self.coordinate]) - jnp.tanh(z[..., self.coordinate]))
