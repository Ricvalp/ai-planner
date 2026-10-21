"""Frozen decoder interface: the sampler never requires an inverse or Jacobian."""
from dataclasses import dataclass
from typing import Protocol, Any
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri


class Decoder(Protocol):
    noise_dim: int
    version: str

    def decode(self, context: Any, z): ...


def compact(z):
    return 2 * ndtr(jnp.asarray(z, dtype=jnp.float64)) - 1


def inverse_compact(y):
    # Interior quadrature nodes avoid clipping. Endpoints are mathematically infinite.
    return ndtri((jnp.asarray(y, dtype=jnp.float64) + 1) / 2)


@dataclass(frozen=True)
class AnalyticDecoder:
    noise_dim: int = 2
    version: str = "analytic_2d_cdf_v1"

    def decode(self, context, z):
        return compact(z)


@dataclass(frozen=True)
class IdentityDecoder:
    noise_dim: int = 2
    version: str = "identity_v1"

    def decode(self, context, z):
        return jnp.asarray(z, dtype=jnp.float64)
