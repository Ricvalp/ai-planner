# Implementation decisions

These choices instantiate the underspecified geometry and numerics in the
concept manuscript. They are not manuscript measurements.

## Numerical stack and architecture

- CPU-first `uv` environment; tested Python 3.11.15, JAX/JAXlib 0.6.2,
  Flax Linen 0.10.6, Optax 0.2.5, NumPy 2.2.6, SciPy 1.15.3.
  The complete transitive stack is in `uv.lock`. System Python 3.12.3 is also
  present; `uv` selected its compatible managed 3.11 interpreter.
- Float64 is enabled before package array construction, including networks.
  Width 32 with three tanh layers is deliberately smaller than the suggested
  width 128 development default. This keeps the first CPU pilot bounded.
- The proposal starts exactly at pCN `eta=.65` using a zero output head.
  Its mean residual is bounded by tanh, scale smoothly lies in `(.08,1.8)`,
  and Gaussian samples themselves remain unbounded.
- Context normalization subtracts the training-context mean and divides by
  `max(training standard deviation,1)`. It uses no target samples or test
  labels. Each checkpoint fixes temperature.
- No DSBM training, model download, web application, remote tracker, optimizer
  distillation, or robotic simulator integration is included.

## Analytic target and training

- Two-mode centers are `(-.52,0),(.52,0)`; four-mode centers are the
  `±.5` corners. Default widths are `.17,.21`; rotated widths `.22,.105`
  at alternating angles `±.6` radians. Rare/common energy weights are `.9,.1`.
  The energy floor is `.015`, `beta0=1`.
- Train contexts shift vertical centers by `-.055`; the validation two-mode
  context shifts by `+.075`; test contexts use zero shift. The constant-cost
  case is an additional out-of-family stress test. One-context debugging is
  supported with `one_context: true` and is not called held-out generalization.
- Midpoint quadrature uses the complete 2D domain and a doubled resolution.
  No mixture-component probability is substituted for a region integral.
- Training uses independent exact rejection samples rather than finite-chain
  bootstrapping. Rejected proposals and unused accepted batch surplus count.
  Finite exact-chain mixing is still measured against the independent oracle.
- Raw ratio MSE is primary. Movement scale `.5` is fixed a priori in the
  declared feature units; it is not selected using held-out performance.
  Baseline decay `.95`; gradient norm clips are 2 (proposal) and 10 (ratio).
  Short proposal blocks are followed by fresh proposal-dependent ratio labels.
- The final no-movement ablation matches the main model's two-round ratio
  update budget. Its initial one-round exploratory checkpoint was resumed,
  with original artifacts and paid evaluation overhead retained separately.
- Temperature sensitivity uses exact pCN at `.6,1,1.6`; no acceptance model is
  reused as calibrated at an untrained temperature. The identity decoder uses
  the same energy formula on `z`, so it defines a different target, explicitly
  plotted in the Gaussian-CDF integration coordinates.
- Controlled continuous error adds `1.5*(z'_1-z_1)` to the learned ratio.
  The finite-grid audit additionally perturbs the exact discrete ratio, which
  isolates algebra from neural fit error.

## Point-mass world and decoder

- Horizon `H=20`, prefix `K=2`. This replaces the suggested `16,4` so the
  actual disturbance snapshot remains before the route fork. Both route
  strategies remain reachable and support is checked before training.
- Semi-implicit PD tracking: mass 1, `kp=70`, `kd=16`, `dt=.02`, nine substeps
  per command, acceleration norm cap 35, speed cap 3. Workspace `±1.2`,
  command bounds `±1.15`, point-mass collision radius `.035`. Commands and
  actual positions are saved separately.
- Central disk radius `.25`; the active upper blocker is near `(0,.68)` with
  radius `.52`. Swept segment checks catch obstacle crossings between samples.
  Collision is absorbing; remaining scheduled substeps are still charged.
- The two compact coefficients are `c=2Φ(z)-1`. With
  `t=min(i/18,1)` and `u=t+.2*c2*sin(pi*t)/pi`, the decoder combines a
  straight current-to-goal path, `.15` times cached path shape, and the
  vertical bend `.95*c1*sin(pi*u)+.16*c2*sin(2*pi*u)`. Final commands hold the
  goal. Projection is part of the fixed decoder, never the noise proposal.
- Cost is bounded goal error `r/(.2²+r)` plus `2*collision` plus
  `.08*v/(.12²+v)` for mean squared command variation. Thus `Jmax=3.08`.
  Full-chunk lookahead success requires no collision and goal error ≤ `.15`.
  It is distinct from what physically happens in the executed prefix.
- Cached plans arise from actually executing a nominal prefix. Disturbances
  happen afterward, before the comparison snapshot. Tail completion uses
  damped last increments with decay `.5` from the selected, executed plan.
- Each scenario has distinct train/validation/test variants `-.6,.25,.8`.
  Context features include position, velocity, goal, obstacle parameters and
  five cached command samples; full snapshot restoration belongs to evaluation.
- Optional suffix continuation preserves each candidate's suffix and executes
  a fixed number of deterministic continuation prefixes. It never recursively
  calls Ai-Plan. Discarding suffixes yields identical continuation from equal
  prefix successor states, tested explicitly.

## Planning and accounting

- Posterior comparisons use fixed transition counts and independent chain
  starts. Planning comparisons use fixed true cost budgets, including one
  final validation. Planning incumbents are never used as target samples.
- A learned-only planner validates unknown current states every four attempts.
  The attempt cap is eight times the oracle budget. Any unspent oracle budget
  after the cap is filled by independent prior candidates and reported.
  Unevaluated learned proposals never enter the incumbent pool.
- Collision feasibility can be inferred from exact scalar cost in this
  specific cost design: every noncollision cost is ≤1.08 and every collision
  cost is ≥2. The planner checks the required weight inequality.
  Final validation supplies full physical metrics. No feasible candidate means
  deterministic holding fallback, with frequency recorded.
- CEM uses the same two noise coordinates, initial mean `(.55,0)`, unit
  standard deviations, population 8, elite fraction `.25`, smoothing `.7`,
  and scale floor `.15`. It is labeled an optimizer.
- Compilation is measured with explicit lowering where available. Training
  reports first execution plus compilation separately from later execution.
  Host-loop dispatch overhead remains in execution timings. No synthetic
  oracle delay or claimed speedup is used.
- Density/reference/replay oracle work is reported separately from deployment.
  Rendering only reads artifacts. Independent chain dispatch prevents a
  batched conditional from executing both oracle branches.
- Three repeated planning random streams within one development seed are
  paired across methods. They are not three independent training seeds.
  Small pilot outcomes are descriptive and cannot establish superiority.
- The final no-movement ablation receives the same two ratio-training rounds
  and number of ratio optimizer updates as the main method, with zero proposal
  updates. An initial one-round comparison confounded movement learning with
  acceptance fitting effort. Its checkpoints, calibration, chains and metrics
  remain under each pilot's `exploratory_round1/`. The second round resumed
  the existing optimizer and PRNG state; only the eight affected posterior
  comparisons were rerun. Exploratory sampling/audit/calibration expenditure
  is charged separately, and training counters include both resumed rounds.
