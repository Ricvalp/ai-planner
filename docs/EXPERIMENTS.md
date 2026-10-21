# Reproducing the bounded experiments

`configs/analytic_pilot.yaml` and `configs/route_pilot.yaml` specify the actual
development runs. Each trains one seed, two alternating rounds, width-32
networks and short movement blocks. No large sweeps are launched.

The analytic pilot has four training contexts, one held-out validation context,
and five principal test contexts. The route pilot has three scenarios in each
split, each with a different causal geometry variant. Split IDs and full frozen
definitions are in `metadata.json`; training provenance is in the checkpoint
JSON.

## Posterior mode

Compare prior, random-walk MH, pCN, a high-locality random walk, learned proposal
with exact acceptance, the same proposal with learned acceptance, refresh rates
`.1,.3`, controlled ratio error, and no movement learning. All posterior
comparisons use a fixed number of transitions, retaining rejected states and
including burn-in in oracle accounting.

Report region masses and their error, per-chain standard errors, region
switches, region-indicator ESS, coarse 8×8 histogram discrepancy, acceptance,
and accepted movement. A constant observable in a stuck chain receives ESS
zero by convention. These are short-run diagnostics rather than a guarantee
of convergence.

Rejection samples provide independent exact target draws. Grid refinement
compares midpoint resolutions `n` and `2n`. The separate 11×11 finite-state
audit checks exact, injected, and neural acceptance and refresh bounds by
finite sums and stationary linear solves.

## Planner mode

Budgets 16 and 48 include final validation. Independent best-of-N, random walk,
pCN, learned exact, learned approximate and refreshed methods share the cost
and decoder. Routes additionally use warm-started CEM. Success is full-chunk
lookahead feasibility and goal tolerance. Saved selected execution prefixes
show the separate physical controller action. No averaging of incompatible
top/bottom routes is used.

The plot against simulator substeps converts actual charged candidate costs
to the deterministic rollout workload, not batch count. Training and audit
work remain visible in `accounting.json`. Lower cost-query count does not
imply lower wall time for this cheap simulator.

## Files and commands

```bash
uv run pytest -q
uv run ai-planner train analytic --config configs/analytic_pilot.yaml
uv run ai-planner train routes --config configs/route_pilot.yaml
uv run ai-planner audit --run outputs/analytic_pilot
uv run ai-planner audit --run outputs/route_pilot
```

Supplementary diagnostics and physical replanning are explicit commands:

```bash
uv run python -m ai_planner.audits outputs/analytic_pilot
uv run python -m ai_planner.audits outputs/route_pilot
uv run python -m ai_planner.deployment \
  --checkpoint outputs/route_pilot/checkpoints/checkpoint.msgpack \
  --output outputs/route_pilot/deployment --budget 16 --replans 4 --seed 71
```

The acceptance audit reuses saved independent source samples and checkpoints
completed pair evaluations; rerunning it does not repeat oracle work.
`uv run python -m ai_planner.recovery outputs/analytic_pilot` can finish an
interrupted analytic evaluation from its saved artifacts. Optimizer resume is
separate and uses the `train --resume` interface documented in the README.

- `traces/*.npz`: pre-move states, proposals, post-move states, decoded
  candidates, accept/reject decisions, branch IDs, log uniforms, known costs,
  validity bits and cumulative counters. Prefix replay audit labels are
  separate from inference logits.
- `grids/*.npz`: midpoint coordinates, costs, normalized mass/density,
  region partition, numerical normalizer and reference costs.
- `reference_checks.json`: independent sample uncertainty and grid sensitivity.
- `finite_grid.json`: exact sums, KL identity, invariant/finite-step errors
  and explicitly vacuous bounds.
- `checkpoints/`: complete train state, provenance, calibration and optimizer
  histories. The no-movement ablation has its own checkpoint.
- `planning_results.json`: optimizer outcomes and budgets, separate from
  posterior metrics. `planning/*.npz` stores selected causal executions.
- `routes/*.npz`: full commands and simulated positions for actual replay
  proposals plus pretraining support checks.
- `figures/`, `replay.html`: figures and server-free interactive replay.

To run a later independent-seed comparison, copy a pilot YAML and change
`seed` and `output`; aggregate complete runs explicitly. No multi-seed evidence
is implied by this first pass.
