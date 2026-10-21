"""Explicit Gaussian proposals; the very same density scores both directions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def normal_log_prob(value, mean=0.0, scale=1.0):
    """Diagonal Gaussian log density, summed over the final coordinate axis."""
    value, mean, scale = map(jnp.asarray, (value, mean, scale))
    return jnp.sum(-0.5 * ((value - mean) / scale) ** 2
                   - jnp.log(scale) - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)


@dataclass(frozen=True)
class RWProposal:
    scale: float = 0.35

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")

    def sample(self, key, z, context=None, params=None):
        return z + self.scale * jax.random.normal(key, jnp.shape(z), dtype=jnp.float64)

    def log_prob(self, destination, source, context=None, params=None):
        return normal_log_prob(destination, source, self.scale)


@dataclass(frozen=True)
class PCNProposal:
    eta: float = 0.65

    def __post_init__(self):
        if not 0 < self.eta <= 1:
            raise ValueError("eta must lie in (0, 1]")

    def sample(self, key, z, context=None, params=None):
        return jnp.sqrt(1 - self.eta**2) * z + self.eta * jax.random.normal(
            key, jnp.shape(z), dtype=jnp.float64)

    def log_prob(self, destination, source, context=None, params=None):
        return normal_log_prob(destination, jnp.sqrt(1 - self.eta**2) * source, self.eta)


class ProposalNetwork(nn.Module):
    noise_dim: int
    hidden_dims: tuple[int, ...] = (32, 32, 32)
    eta: float = 0.65
    min_scale: float = 0.08
    max_scale: float = 1.8
    residual_scale: float = 1.0

    @nn.compact
    def __call__(self, z, context):
        context = jnp.broadcast_to(context, (*z.shape[:-1], context.shape[-1]))
        h = jnp.concatenate((z, context), axis=-1)
        for width in self.hidden_dims:
            h = nn.tanh(nn.Dense(width, dtype=jnp.float64, param_dtype=jnp.float64)(h))
        # Zero output weights initialize exactly at an interpretable pCN kernel.
        raw = nn.Dense(2 * self.noise_dim, kernel_init=nn.initializers.zeros_init(),
                       bias_init=nn.initializers.zeros_init(), dtype=jnp.float64,
                       param_dtype=jnp.float64)(h)
        mean = jnp.sqrt(1 - self.eta**2) * z + self.residual_scale * jnp.tanh(
            raw[..., :self.noise_dim])
        lo, hi = np.log(self.min_scale), np.log(self.max_scale)
        p = (np.log(self.eta) - lo) / (hi - lo)
        offset = np.log(p / (1 - p))
        log_scale = lo + (hi - lo) * jax.nn.sigmoid(raw[..., self.noise_dim:] + offset)
        return mean, jnp.exp(log_scale)


@dataclass(frozen=True)
class NeuralProposal:
    model: ProposalNetwork
    params: Any
    context_mean: Any
    context_scale: Any

    @classmethod
    def create(cls, key, noise_dim, context_dim, hidden_dims=(32, 32, 32),
               eta=0.65, min_scale=0.08, max_scale=1.8, residual_scale=1.0,
               context_mean=None, context_scale=None):
        if not 0 < min_scale < eta < max_scale or eta > 1:
            raise ValueError("Require 0 < min_scale < eta <= 1 and eta < max_scale")
        model = ProposalNetwork(noise_dim, tuple(hidden_dims), eta, min_scale,
                                max_scale, residual_scale)
        params = model.init(key, jnp.zeros(noise_dim), jnp.zeros(context_dim))["params"]
        mean = jnp.zeros(context_dim) if context_mean is None else jnp.asarray(context_mean)
        scale = jnp.ones(context_dim) if context_scale is None else jnp.asarray(context_scale)
        if np.any(np.asarray(scale) <= 0):
            raise ValueError("Normalization scales must be positive")
        return cls(model, params, mean, scale)

    def statistics(self, z, context, params=None):
        context = (jnp.asarray(context) - self.context_mean) / self.context_scale
        return self.model.apply({"params": self.params if params is None else params},
                                z, context)

    def sample(self, key, z, context, params=None):
        mean, scale = self.statistics(z, context, params)
        return mean + scale * jax.random.normal(key, mean.shape, dtype=jnp.float64)

    def log_prob(self, destination, source, context, params=None):
        mean, scale = self.statistics(source, context, params)
        return normal_log_prob(destination, mean, scale)

    def replace_params(self, params):
        return replace(self, params=params)
