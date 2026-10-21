# Actual development run report

**Date:** 20 October 2026, Europe/Rome. **Status:** complete bounded toy pilot.
All numerical results below come from saved runs in this repository.
The manuscript's fictional empirical numbers were not used.

Start with the [analytic replay](../outputs/analytic_pilot/figures/replay.html)
and [route replay](../outputs/route_pilot/figures/replay.html). Both work locally
without a notebook server. The [guided notebook](../notebooks/01_ai_plan_intuition.ipynb)
loads these artifacts and has been executed without retraining.

## What ran

The primary runs use development seed 17, float64, CPU, three-layer width-32
Flax Linen networks, and two alternating training rounds. Analytic training
uses 800 ratio updates and 48 movement updates; route training uses 960 ratio
updates and 40 movement updates. Independent exact rejection supplies training
states. Test contexts are distinct from training/validation contexts.

The analytic main comparison has 5 contexts × 10 methods × 4 independent chains
× 2,304 transitions, discarding 256 transitions per chain for statistics.
Four additional exact-only sensitivity cases cover temperature and the identity
decoder. The route comparison has 3 contexts × 10 methods × 4 chains × 1,792
transitions, also discarding 256. Rejections remain in every trace.

Planning evaluates budgets 16 and 48, including final validation, with three
paired random streams per context. Those repetitions are **not independent
training seeds**. There are 168 analytic and 144 route planning comparisons.
Six additional route episodes execute four actual replans each. No large sweep
or DSBM pretraining project was run.

The locked stack is Python 3.11.15, JAX/JAXlib 0.6.2, Flax Linen 0.10.6,
Optax 0.2.5, NumPy 2.2.6 and SciPy 1.15.3 on Linux x86-64 CPU.
A separate clean installation with system Python 3.12.3 also passed the focused
tests. No global environment, CUDA installation or driver was modified.

## Mathematical and software validation

- **46 focused tests pass.** Coverage includes independent SciPy densities,
  exact and learned antisymmetry, pCN cancellation, finite-state balance and
  refresh bounds, score-function gradients and stopped derivatives, cache
  invalidation, actual runtime oracle skipping, deterministic checkpoint
  resume, six-dimensional sampling, ignored-noise-coordinate rewards,
  swept collisions, causal cache selection, suffix continuation and artifact
  recovery.
- Both saved-run `audit` commands pass. Main analytic independent rejection
  region estimates are within **1.69 binomial standard errors** of refined
  quadrature; route estimates are within **1.11 standard errors**. These are
  independent-sample checks, not claims that every finite MH chain is stationary.
- Doubling the analytic grid from 96 to 192 changes the largest region mass by
  at most **4.02×10⁻⁶** across main cases. Doubling route grids from 80 to 160
  changes region masses by at most **1.90×10⁻⁴**. Collision boundaries are
  discontinuous, so this is a numerical sensitivity estimate.
- Exact finite-state MH preserves its discrete target to roughly **10⁻¹⁶** TV.
  Accepted-flow KL identities and finite-step bounds pass numerical checks.
  This 11×11 model has its own normalized proposal rows and discrete ratios.
- The notebook's 15 cells executed without training or simulator calls.
  Standalone HTML controls were exercised, including recorded planner frames.
  Saved figures were inspected. Final tiny analytic and route CLI integration
  runs completed generation, training, evaluation and rendering from scratch.

## Sampling results

Representative results below use post-burn-in samples. Region TV is a fixed
partition diagnostic; coarse TV uses an 8×8 histogram and includes sampling
error. ESS refers to a region indicator, not an exact convergence measure.

| Context / method | Acceptance | Region TV | Coarse TV | Region ESS |
|---|---:|---:|---:|---:|
| Two modes: pCN | .450 | .009 | .045 | 618 |
| Two modes: learned proposal + exact MH | .478 | .010 | .061 | 820 |
| Two modes: learned acceptance | .542 | .038 | .175 | 963 |
| Two modes: refresh .1 | .518 | .071 | .167 | 905 |
| Two modes: refresh .3 | .525 | .037 | .142 | 1,231 |
| Rare mode: pCN | .453 | .017 | .058 | 778 |
| Rare mode: learned proposal + exact MH | .447 | .016 | .046 | 1,196 |
| Route invalidation: pCN | .633 | <.001 | .031 | 0* |
| Route invalidation: learned proposal + exact MH | .539 | <.001 | .027 | 0* |
| Route invalidation: learned acceptance | .577 | .064 | .159 | 36 |
| Route invalidation: refresh .1 | .558 | .050 | .128 | 402 |
| Route invalidation: refresh .3 | .519 | .037 | .088 | 852 |

`*` The correct invalidation target has negligible upper-route mass, so a
constant region indicator is expected in these short exact chains. The
conservative implementation reports zero indicator ESS; it does **not** mean
every coordinate is frozen. Coordinate ESS and decoded-plan movement are
saved separately.

The learned proposal improved some exploration measures under exact correction:
the rare-mode ESS increased from 778 to 1,196, and four-mode switches increased
from 1,763 to 2,072. This one-seed result does not establish a general efficiency
advantage. Two-mode histogram error was slightly worse for learned exact than
pCN. Exact correction remains the reliable probability reference.

The high-locality two-mode chain accepted **98.4%** of proposals but made
**zero region switches** and had coarse histogram TV **.721**. Its pooled
region TV happened to be zero because separately stuck chains started on
balanced sides. The replay shows individual chains, preventing that pooled
coincidence from masquerading as mixing.

Learned acceptance distorted densities. On the constant-cost test, exact pCN
accepted every proposal, while learned acceptance had region TV **.149** and
coarse TV **.452**. This constant context was not in training and is an explicit
generalization failure. Controlled antisymmetric error also produced large
mode distortions. Low ratio-training loss does not imply calibration on all
contexts.

Refresh generally reduced bias in route invalidation, but the finite two-mode
region estimate worsened at `.1` before improving at `.3`. Neither a monotonic
empirical trend nor a learned-method win is guaranteed. Lower temperature
reduced pCN acceptance and region ESS in the sensitivity cases. The identity
decoder comparison integrates a different target and is not evidence that one
decoder is universally better.

## Acceptance audits and exact-refresh bounds

Fresh test-context audits use 512 independent rejection source states with fresh
frozen proposals, charging both source and proposed costs. Route acceptance
error estimates are:

| Route context | Mean absolute acceptance error | Standard error |
|---|---:|---:|
| Unchanged | .1104 | .0091 |
| Mild correction | .0998 | .0086 |
| Invalidation | .0699 | .0062 |

All signed balance estimates from these runs were positive, but the code and
tests preserve negative finite-sample estimates rather than clamping them.
These are Monte Carlo diagnostics, not certified bounds.

For routes, `Jmax/β=12.32`, so `log(kappa)=-14.6226` at `rho=.1`.
The resulting test-audit plug-in stationary bounds range from approximately
141,000 to 223,000 at `.1`: completely vacuous. The finite-state learned model
on the unchanged route has actual stationary TV `.1453` without refresh,
`.1284` at `.1`, and `.0960` at `.3`; its corresponding conservative bounds are
also vacuous. At `rho=1`, exact refresh preserves the discrete target.

Analytic learned finite-grid stationary TV is `.1763`, `.1619` and `.1342`
at refresh rates `0,.1,.3`; its positive-rate bounds are `79.55` and `20.62`.
These finite-state values do not estimate the exact stationary law of the
continuous neural sampler. The constant analytic target has a nonvacuous
Monte Carlo plug-in bound at `.3`, but finite sampling still does not certify it.

## Planning and physical execution

Most route planning comparisons were already easy for independent prior
sampling and pCN: each achieved **9/9 full-chunk lookahead successes** at both
budgets. Learned exact and learned/refreshed methods also achieved 9/9.
CEM and random-walk MH each had one fallback among nine budget-48 cases;
both achieved 9/9 at budget 16. Different budgets use different paired random
streams, so those outcomes are not nested monotonic trials. This small sample
does not establish that CEM or random walk is inferior.

At budget 48, mean route costs were about `.03509` for independent prior,
`.03515` for pCN, `.03515` for learned exact and `.03524` for learned-only.
Those differences are too small and under-replicated for a superiority claim.
The analytic best-cost results favored learned-only in this development run,
but an incumbent optimizer's cost is not evidence of posterior fidelity.

Each of the six deployment episodes used 64 oracle evaluations and 72 actual
physical execution substeps across four replans. **None collided; none reached
the goal within four replans.** In route invalidation, pCN used two holding
fallbacks while learned exact and refresh used none. Final distances remained
between 1.33 and 1.51 for that scenario. Eventual success is untested, and a
successful simulated full chunk must not be reported as actual goal attainment.
The saved episodes preserve a new target ID, actual successor, selected prefix,
retained suffix and completed tail at each physical replan.

## Work accounting, timing and failures

Every run has `accounting.json`, per-method timing/counters, and exact-cost
validity traces. Training, calibration, numerical grids, independent-rejection
references, replay audits, planning validations and physical execution are
reported separately. Final added acceptance audits cost 5,120 analytic and
3,072 route evaluations. The six deployment episodes cost 69,120 oracle
substeps, 432 actual execution substeps and 162 shared context-creation substeps.

The final named pilot ledgers, including matched-ablation and exploratory
overhead but excluding separately logged integration checks, contain:

| Ledger | Exact cost evaluations | Oracle simulator substeps |
|---|---:|---:|
| Analytic pilot and its recorded supplements | 1,072,362 | 0 |
| Route pilot and its recorded supplements | 286,951 | 51,651,180 |
| Separate six-episode deployment run | 384 | 69,120 |

Most of the analytic work is independent-rejection/reference auditing, not
training or deployment. Route grid and replay simulations are also charged.
These counts must not be reduced to the number of batched function calls.

The primary route pipeline took **77.01 seconds** on this machine, including
its training, sampling, references and planning, before the later supplementary
acceptance/deployment audits and matched-ablation completion. Primary network-training orchestration took
**4.13 seconds analytic** and **4.45 seconds route**. These times include the
recorded compilation and host work appropriate to each phase; individual
sampler compilation and warmed execution times are separate. There is no
claimed wall-clock speedup from avoiding this cheap simulator.

The analytic pipeline initially stopped at the identity-decoder sensitivity
audit because the running process had imported an older quadrature guard.
All main training, chains, grids and three temperature cases had already been
saved. Recovery reused those arrays and reconstructed paid rejection batches
from the recorded random stream, without reevaluating completed oracle work.
The missing identity case and planning were then completed. **The original
analytic total elapsed time is unknown**, and remains null rather than being
invented; phase times and measured recovery times are retained.

Other integration failures were a planner-artifact directory collision during
rendering and a YAML indentation error in a tiny integration configuration.
Both were fixed. The render failure required no oracle rerun; the YAML error
occurred before any experiment. These failures are software debugging outcomes,
not omitted unfavorable scientific trials.

The initial no-movement ablation had fewer ratio-fitting rounds than the main
model. It was resumed to match the main two-round ratio schedule before the
final comparison. Earlier artifacts and their extra sampling/calibration costs
are retained as exploratory overhead. Only the additional round and affected
no-movement sampling rows were run again, taking 5.42 seconds analytic and
4.72 seconds route before rendering. Its unchanged proposal fingerprint and
shared sampling random keys were checked. The final ablation remains mixed:
it is close to learned acceptance on some region partitions and worse on route
invalidation; it does not support a universal improvement from movement training.

## Remaining limitations

The learned ratio is insufficiently calibrated, especially near the route
collision boundary and outside its training context family. Proposal gains
are modest and not replicated across training seeds. Finite-chain bias and
finite-bin histogram noise remain. Conservative refresh bounds are generally
too weak to be practically informative. The deployment horizon is too short to
establish eventual task completion.

This is an inspectable sampler over an engineered two-coordinate plan family.
There is no trained generative decoder, demonstrated DSBM advantage, realistic
contact model, physical passivity proof, or real-robot safety claim.
Commands and discretionary choices are in [EXPERIMENTS.md](EXPERIMENTS.md)
and [DECISIONS.md](DECISIONS.md). The next research comparison should use
independent training seeds and better-calibrated acceptance, while retaining
exact correction as the reference.
