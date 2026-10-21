"""Bounded analytic targets and an independent quadrature reference."""
from dataclasses import dataclass, field, replace
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import logsumexp
from .contexts import frozen_hash
from .decoders import AnalyticDecoder, inverse_compact


@dataclass(frozen=True)
class AnalyticContext:
    name: str
    centers: tuple
    covariances: tuple
    weights: tuple
    eps_floor: float = .015
    beta0: float = 1.
    constant: bool = False

    def __post_init__(self):
        if not 0 < self.eps_floor < 1 or self.beta0 <= 0:
            raise ValueError("positive energy scale and floor in (0,1) required")
        if len(self.centers) != len(self.weights) or not np.isclose(sum(self.weights), 1):
            raise ValueError("one normalized nonnegative weight per center")
        if min(self.weights) < 0 or np.any(np.linalg.eigvalsh(self.covariances) <= 0):
            raise ValueError("covariances positive definite; weights nonnegative")


@dataclass(frozen=True)
class AnalyticTarget:
    context: AnalyticContext
    beta: float = 1.
    decoder: object = field(default_factory=AnalyticDecoder)
    cost_version: str = "floor_gaussian_energy_v1"
    simulator_steps_per_cost: int = 0

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError("temperature must be positive")

    @property
    def name(self):
        return self.context.name

    @property
    def noise_dim(self):
        return self.decoder.noise_dim

    @property
    def j_max(self):
        return 0. if self.context.constant else -self.context.beta0 * np.log(self.context.eps_floor)

    @property
    def target_id(self):
        return frozen_hash(self)

    @property
    def context_features(self):
        c = self.context
        centers = np.zeros((4, 2)); covs = np.zeros((4, 4)); weights = np.zeros(4)
        n = len(c.centers)
        centers[:n] = c.centers
        covs[:n] = np.asarray(c.covariances).reshape(n, 4)
        weights[:n] = c.weights
        return jnp.asarray(np.r_[centers.ravel(), covs.ravel(), weights,
                                c.eps_floor, c.beta0, self.beta, c.constant])

    def cost(self, candidate):
        c = self.context
        if c.constant:
            return jnp.asarray(0., dtype=jnp.float64)
        delta = candidate - jnp.asarray(c.centers)
        exponent = -.5 * jnp.einsum("mi,mij,mj->m", delta,
                                   jnp.linalg.inv(jnp.asarray(c.covariances)), delta)
        weight = c.eps_floor + (1 - c.eps_floor) * jnp.sum(jnp.asarray(c.weights) * jnp.exp(exponent))
        return -c.beta0 * jnp.log(weight)

    def evaluate(self, z):
        return self.cost(self.decoder.decode(self.context, z))

    def features(self, candidate):
        return candidate


def make_analytic_target(kind="two", *, name=None, shift=0., beta=1., floor=.015):
    if kind in ("two", "rare", "constant"):
        centers = np.array([[-.52, .0], [.52, .0]])
        weights = (.5, .5) if kind != "rare" else (.9, .1)
    else:
        centers = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
        weights = (.25,) * 4
    centers[:, 1] += shift
    covs = []
    for i in range(len(centers)):
        angle = .6 * (-1)**i if kind == "rotated" else .0
        r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        widths = [.22, .105] if kind == "rotated" else [.17, .21]
        covs.append(r @ np.diag(np.square(widths)) @ r.T)
    context = AnalyticContext(name or kind, tuple(map(tuple, centers)),
                              tuple(tuple(map(tuple, v)) for v in covs), weights,
                              eps_floor=floor, constant=kind == "constant")
    return AnalyticTarget(context, beta)


def analytic_splits(beta=1., one_context=False):
    if one_context:
        t = make_analytic_target(beta=beta)
        return [t], [], [t]
    train = [make_analytic_target(k, name=f"train_{k}", shift=-.055, beta=beta)
             for k in ("two", "rare", "four", "rotated")]
    validation = [make_analytic_target("two", name="validation_two", shift=.075, beta=beta)]
    test = [make_analytic_target(k, name=k, beta=beta) for k in ("two", "rare", "four", "rotated", "constant")]
    return train, validation, test


def quadrature(target, n=128):
    """Midpoint quadrature in u=2 Phi(z)-1; this coefficient prior is uniform.

    With the analytic CDF decoder, u is also the decoded output y. For an
    identity decoder, the evaluated candidate is z and u is only a plotting/
    integration coordinate. Its contours must not be labeled decoded output.
    """
    if target.noise_dim != 2:
        raise ValueError("compact quadrature requires two-dimensional Gaussian noise")
    axis = -1 + (np.arange(n) + .5) * 2 / n
    yy = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1)
    zz = inverse_compact(yy.reshape(-1, 2))
    costs = np.asarray(jax.jit(jax.vmap(target.evaluate))(zz)).reshape(n, n)
    logw = -costs / target.beta
    lognorm = logsumexp(logw) - np.log(n*n)
    masses = np.exp(logw - logsumexp(logw))
    if hasattr(target.context, "centers"):
        centers = np.asarray(target.context.centers)
        regions = np.argmin(np.sum((yy[..., None, :] - centers)**2, axis=-1), axis=-1)
    else:
        regions = (yy[..., 0] > 0).astype(int)
    region_masses = np.bincount(regions.ravel(), weights=masses.ravel())
    return dict(y=yy, cost=costs, density=masses / (2/n)**2, masses=masses,
                regions=regions, region_masses=region_masses, log_normalizer=lognorm,
                normalizer=np.exp(lognorm), grid_size=n,
                exact_cost_evaluations=n*n, decoder_calls=n*n,
                simulator_steps=n*n*target.simulator_steps_per_cost)
