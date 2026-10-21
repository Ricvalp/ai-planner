"""A deterministic PD-controlled point mass with swept-disk collision checks.

The simulation is a pure function of its snapshot, commands, obstacles and
configuration. Every candidate is evaluated from the supplied snapshot. Collision
is absorbing; attempted physical substeps still count toward rollout accounting.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SimulatorConfig:
    dt: float = 0.02
    substeps: int = 9
    mass: float = 1.0
    kp: float = 70.0
    kd: float = 16.0
    max_acceleration: float = 35.0
    max_speed: float = 3.0
    command_bound: float = 1.15
    workspace_bound: float = 1.2
    robot_radius: float = 0.035

    def __post_init__(self):
        if self.dt <= 0 or self.substeps < 1 or self.mass <= 0:
            raise ValueError("Positive integration step, substeps and mass required.")
        if min(self.kp, self.kd, self.max_acceleration, self.max_speed) <= 0:
            raise ValueError("PD gains and dynamic limits must be positive.")
        if not 0 < self.command_bound <= self.workspace_bound:
            raise ValueError("Commands must lie inside the workspace.")
        if self.robot_radius < 0:
            raise ValueError("Robot radius cannot be negative.")


@dataclass(frozen=True)
class Snapshot:
    position: tuple[float, float] = (-0.95, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    collided: bool = False


class PhysicalState(NamedTuple):
    position: jax.Array
    velocity: jax.Array
    collided: jax.Array


class Rollout(NamedTuple):
    positions: jax.Array
    velocities: jax.Array
    collisions: jax.Array
    command_positions: jax.Array
    command_velocities: jax.Array
    final_state: PhysicalState
    path_length: jax.Array


def segment_disk_collision(start, end, center, radius):
    """True even when a segment enters and exits a disk between samples."""
    delta = end - start
    length_sq = jnp.dot(delta, delta)
    fraction = jnp.clip(jnp.dot(center - start, delta) / jnp.maximum(length_sq, 1e-30), 0.0, 1.0)
    closest = start + fraction * delta
    return jnp.sum(jnp.square(closest - center)) <= radius**2


def segment_collision(start, end, obstacles, config: SimulatorConfig):
    """Obstacles have rows ``(center_x, center_y, radius, enabled)``."""
    hits = jax.vmap(segment_disk_collision)(
        jnp.broadcast_to(start, (obstacles.shape[0], 2)),
        jnp.broadcast_to(end, (obstacles.shape[0], 2)),
        obstacles[:, :2],
        obstacles[:, 2] + config.robot_radius,
    )
    boundary = jnp.any(jnp.abs(end) > config.workspace_bound - config.robot_radius)
    return jnp.any(hits & (obstacles[:, 3] > 0.5)) | boundary


def _bounded_norm(vector, bound):
    return vector * jnp.minimum(1.0, bound / jnp.maximum(jnp.linalg.norm(vector), 1e-30))


def physical_step(state: PhysicalState, command, obstacles, config: SimulatorConfig):
    """Semi-implicit Euler with bounded acceleration and speed."""
    command = jnp.clip(command, -config.command_bound, config.command_bound)
    acceleration = _bounded_norm(
        (config.kp * (command - state.position) - config.kd * state.velocity) / config.mass,
        config.max_acceleration,
    )
    velocity = _bounded_norm(state.velocity + config.dt * acceleration, config.max_speed)
    proposed_position = state.position + config.dt * velocity
    collision = state.collided | segment_collision(state.position, proposed_position, obstacles, config)
    # Keep the last safe physical position, and brake in the absorbing state.
    position = jnp.where(collision, state.position, proposed_position)
    velocity = jnp.where(collision, jnp.zeros_like(velocity), velocity)
    next_state = PhysicalState(position, velocity, collision)
    return next_state, next_state


def simulate(snapshot: Snapshot | PhysicalState, commands, obstacles, config: SimulatorConfig = SimulatorConfig()):
    """Execute all commands and retain every physical integration substep."""
    state = PhysicalState(
        jnp.asarray(snapshot.position, dtype=jnp.float64),
        jnp.asarray(snapshot.velocity, dtype=jnp.float64),
        jnp.asarray(snapshot.collided),
    )
    commands = jnp.asarray(commands, dtype=jnp.float64)
    obstacles = jnp.asarray(obstacles, dtype=jnp.float64)
    repeated_commands = jnp.repeat(commands, config.substeps, axis=0)
    final, states = jax.lax.scan(
        lambda carry, command: physical_step(carry, command, obstacles, config),
        state,
        repeated_commands,
    )
    positions = jnp.concatenate((state.position[None, :], states.position), axis=0)
    velocities = jnp.concatenate((state.velocity[None, :], states.velocity), axis=0)
    collisions = jnp.concatenate((state.collided[None], states.collided), axis=0)
    return Rollout(
        positions,
        velocities,
        collisions,
        states.position[config.substeps - 1 :: config.substeps],
        states.velocity[config.substeps - 1 :: config.substeps],
        final,
        jnp.sum(jnp.linalg.norm(jnp.diff(positions, axis=0), axis=-1)),
    )


def snapshot_from_state(state: PhysicalState) -> Snapshot:
    """Host-side immutable snapshot creation after actual execution."""
    return Snapshot(
        tuple(float(v) for v in state.position),
        tuple(float(v) for v in state.velocity),
        bool(state.collided),
    )
