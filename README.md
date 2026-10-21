# ai-planner

An executable research pilot for **Ai-Plan: Learning Simulator-Corrected Markov
Kernels for Action-Chunk Refinement**. It trains a proposal and an antisymmetric
acceptance model around a **frozen, engineered decoder**. It includes an
auditable two-dimensional target and a deterministic planar route-repair toy.
The manuscript's illustrative results are not used as data.

## Start on CPU

```bash
uv sync --extra dev --locked
uv run pytest -q
uv run ai-planner demo analytic --config configs/analytic_smoke.yaml
uv run ai-planner demo routes --config configs/route_smoke.yaml
```

The demo commands include small training jobs, posterior comparisons, planning
comparisons, numerical reference checks, and rendering. Pilot configurations
use one development seed and modest networks; they are not large sweeps.
CPU is the default locked installation. The tested interpreter and exact
package versions are recorded in each run's `metadata.json`.

## Inspect the saved runs

The development outputs are included:

- [Analytic replay](outputs/analytic_pilot/figures/replay.html)
- [Route replay](outputs/route_pilot/figures/replay.html)
- [Guided notebook](notebooks/01_ai_plan_intuition.ipynb)
- [Actual run report](docs/RUN_REPORT.md)
- [Discretionary choices](docs/DECISIONS.md)

Open the HTML locally in a browser; it embeds its data and needs no server.
The notebook and rendering load saved arrays. They do not train or simulate on
slider changes. Each run includes PNG/PDF figures, compressed traces and grids,
JSON/CSV measurements, configuration, source/version identifiers, and checkpoints.

```bash
uv run ai-planner render --run outputs/analytic_pilot
uv run ai-planner audit --run outputs/analytic_pilot
uv run ai-planner render --run outputs/route_pilot
uv run ai-planner audit --run outputs/route_pilot
```

`audit` verifies the saved rejection-versus-quadrature checks and finite-grid
exact stationarity. Algebraic invariants are tested by pytest. Short chain
statistics remain estimates, and a passed numerical audit is not a certificate
for a trained neural sampler.

## Train, evaluate, resume

```bash
uv run ai-planner train analytic --config configs/analytic_pilot.yaml
uv run ai-planner train routes --config configs/route_pilot.yaml
uv run ai-planner evaluate \
  --checkpoint outputs/analytic_pilot/checkpoints/checkpoint.msgpack \
  --config configs/analytic_smoke.yaml --output outputs/analytic_evaluation
```

For optimizer resume, copy the original configuration, increase only
`training.rounds`, set a new output directory, and use `train ... --resume
outputs/analytic_pilot/checkpoints/checkpoint.msgpack`. All other training
settings and target IDs must match. Checkpoints preserve optimizer states and
the random stream; tests compare uninterrupted and resumed updates bitwise.
`--exact-only` runs the non-neural baselines and references without training.

An accelerator is optional. Keep the tested CPU lock for reproducibility.
In a **separate environment**, follow the current
[official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html)
for the accelerator and driver versions actually available, then verify the
Flax/Optax combination and rerun the tests. Do not modify system CUDA or drivers
to run this pilot. No measured GPU or wall-clock speed advantage is claimed.

## Interpretation

The latent target is `normal(z) * exp(-cost(decode(context,z))/beta)`.
There is no decoder Jacobian in its MH ratio. Exact correction preserves this
target but can mix slowly. Learned acceptance can reduce cost queries while
introducing bias; state-independent random exact refresh provides a quantitative,
often very loose, bound. Rejections remain in every posterior trace.

Posterior sampling and best-feasible planning are separate outputs. A planner
incumbent is not a posterior sample. The two-dimensional route family is
engineered, and its simulator is a PD-controlled point mass with finite
collision penalties. The repository makes no claim about DSBM pretraining,
robotic contact, passivity, or real-world safety.

See [algorithm](docs/ALGORITHM.md) and [experiments](docs/EXPERIMENTS.md)
for the mathematical and accounting details.
