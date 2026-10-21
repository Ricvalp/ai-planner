# ai-planner: an executable, visual Ai-Plan research pilot

## 0. Mission and scope

Build a new repository named **`ai-planner`** from an empty directory. Use the Python package name **`ai_planner`**.

Implement the sampling-and-planning method in the attached concept manuscript, **Ai-Plan: Learning Simulator-Corrected Markov Kernels for Action-Chunk Refinement**. Start with small, inspectable experiments rather than a full robotics benchmark. The immediate goal is to understand what is being learned, verify the probability calculations, and visualize the trade-off between exploration, target bias, and simulator cost.

There are two required experiments, sharing one sampler implementation:

1. **Analytic conditional 2D targets:** the two-/four-mode diagnostic proposed in manuscript §5.1, instantiated precisely below. Its normalizing constant and region probabilities can be checked by numerical integration. It is the principal correctness and intuition experiment.
2. **Plan repair in a 2D world:** a small deterministic point-mass simulator with two obstacle-avoidance routes. A fixed, two-dimensional noise input generates an entire candidate command chunk. Refinement modifies that input, and the simulator determines the cost. This connects the diagnostic to action-chunk planning without first requiring a trained high-dimensional generator.

**Important scope decision:** the initial decoders are deliberately analytic and frozen. They stand in for the pretrained generator `G_psi` in the manuscript. We train the **proposal network** and the **acceptance/log-ratio network**, not a DSBM generator in this pass. The manuscript's main sampling results apply to any fixed measurable decoder. This is a faithful test of the Ai-Plan sampling layer, but it is **not evidence for the superiority of DSBM, a learned generative prior, or a dissipative physical policy**.

Provide a clean, dimension-independent decoder interface so a genuinely pretrained DSBM/FM/diffusion model can be added later. Do not build an entire DSBM trainer merely to obtain the first visualization, and do not claim the analytic decoder was learned from demonstrations. A DSBM trainer and realistic contact-rich pushing are explicitly outside this first implementation.

Implement the complete required toy pipeline, not only scaffolding. Run unit tests and bounded end-to-end development jobs, save actual outputs, and document negative results honestly. Do not manufacture a win against pCN, independent sampling, or CEM.

## 1. Sources and how to use them

If present, read `references/Ai_Plan_Concept_Manuscript.pdf`, particularly:

- §3 and §4.1: fixed context, frozen decoder, noise-space target, terminal tilt;
- §4.3: explicit stochastic proposal and exact/learned log ratios;
- §4.4: acceptance-error and accepted-flow bounds; the separate movement objective;
- §4.5: exact refresh and approximate stationary-law error;
- §4.6: alternating frozen rounds, deployment, and optional distillation;
- §5.1: analytic conditional targets and ambiguous-route repair;
- Appendices A.2–A.5 and B: proof conventions, proposal gradients, pCN, frozen quantities, and accounting.

**All empirical numbers in the concept manuscript are fictional illustrations.** They must not appear as measured results, target scores, pretrained checkpoints, or evidence that the implementation should win.

The manuscript leaves the energy functions, geometry, and numerical implementation underspecified. The explicit benchmark definitions below are new design choices for this pilot. Record deviations and discretionary choices in `docs/DECISIONS.md` rather than silently changing the algorithm.

The old Action Bridge repository's VAE, teacher-forced action-time dynamics, Push-T losses, and BC-to-RL instructions are not specifications for this new repository. In particular, do not import an old instruction saying that no SB solver should ever be implemented, or substitute a legacy path-control loss for Ai-Plan's sampler objectives.

This file is sufficiently self-contained to implement the required pilot even without the PDF. Preserve the PDF when supplied; do not rewrite it.

## 2. Framework decision: JAX

Use **JAX + Flax Linen + Optax** for the numerical core and networks. Linen's explicit parameter dictionaries are convenient for freezing and versioning the small networks. There is no need for distributed training, NNX state machinery, or two competing neural-network frameworks.

Rationale:

- We repeatedly evaluate small functions over many candidates, contexts, and chains. `jit`, `vmap`, and `lax.scan` are a good fit for that workload.
- Explicit random keys help distinguish generator randomness, proposal randomness, acceptance uniforms, and environment/continuation randomness.
- Functional simulator states make snapshot restoration and batched candidate evaluation straightforward.
- We are not depending on a large existing PyTorch robotics model in this first repository.

Costs to manage: compilation latency, static-shape constraints, hidden recompilation, and more deliberate control-flow/debugging. PyTorch also has `torch.compile`; do not describe JIT as unique to JAX or claim an unmeasured speed advantage.

Use NumPy/SciPy for independent CPU checks and plotting preparation where useful. Use Matplotlib for saved figures and a notebook or lightweight Plotly visualization for interaction. Core algorithm code must live in importable modules, not notebook cells. Do not add a web application, database, distributed experiment system, or compulsory online tracking service.

Use `pyproject.toml` with `uv` and a reproducible lockfile. Default installation must work on CPU. Document optional accelerator installation using current official JAX instructions, and verify compatibility with the available Python/platform rather than blindly selecting the newest unsupported wheel. Do not modify system CUDA, drivers, or the global environment.

Start numerical audits in float64; explicitly enable JAX's x64 mode before array construction. Small neural networks may use float32, with deliberate casts and tests at the probability interface. Never use mixed precision for the reference-density checks without validating it.

## 3. Definitions and the three different iteration axes

At a fixed planning context `x`, let:

- `z in R^d`: random input to the frozen decoder;
- `varphi(z) = Normal(0, I_d)`: its known base density;
- `A = G(x, z)`: the decoded candidate; a 2D descriptor in experiment 1, a full action chunk in experiment 2;
- `J_x(A)`: deterministic, nonnegative, bounded evaluation cost;
- `beta > 0`: planning temperature;
- `Z_x`: unknown target normalizer.

The target is

```math
p_x(z) = Z_x^{-1} varphi(z) exp[-J_x(G(x,z))/beta].
```

The desired action distribution is the pushforward of this law through `G`. We do not need an inverse decoder, its Jacobian determinant, or an evaluable action-chunk density.

Keep these axes distinct in code, figures, and documentation:

1. **Physical time:** the simulator executes commands.
2. **Revision time:** a future DSBM decoder could internally revise an entire chunk. The analytic pilot decoder has no learned SB revision trajectory.
3. **MCMC refinement step:** Ai-Plan changes `z`, thus proposing another complete decoded candidate from the same frozen context.

A Markov-chain refinement proposal is **not** a physical action and is not a call to the SB reverse drift. Do not feed a newly proposed chunk back as the decoder's source within a fixed-context MCMC chain. The source inside `x` stays fixed until an actual new robot observation creates a new planning problem.

## 4. Implement the actual Ai-Plan sampler

### 4.1 Explicit, evaluable proposals

Start with the manuscript's simple conditional diagonal-Gaussian proposal:

```math
q_theta(z' | z, x) = Normal(mu_theta(z,x), diag(s_theta(z,x)^2)).
```

A small MLP predicts a residual mean and bounded log-standard-deviation. Three hidden layers of width 128 are a reasonable development default; width 256 is a larger configuration. Codex can choose the exact residual parameterization, normalization, and activation.

Initialize it near a useful pCN/local proposal rather than with enormous random jumps. Enforce strictly positive lower and upper scale limits. Do not clip Gaussian samples after drawing them: that changes the proposal law. Bounded decoder outputs are a separate matter.

Implement both `sample(...)` and `log_prob(destination, source, context, params)`. The reverse quantity is evaluated by calling the SAME proposal network at `z'` and scoring `z`.

### 4.2 Exact MH ratio and acceptance

For the frozen context and decoder, compute

```math
ell_theta(z,z';x)
 = log varphi(z') - log varphi(z)
 - [J_x(G(x,z')) - J_x(G(x,z))]/beta
 + log q_theta(z | z',x) - log q_theta(z' | z,x).
```

Then

```math
a_star(z,z';x) = min(1, exp(ell_theta(z,z';x))).
```

Implement in log space: accept when `log(u) < min(0, ell)`, with an independent uniform `u` in `(0,1)`. Sum log densities over all noise coordinates. Evaluate forward and reverse proposal terms with identical frozen parameters and context.

An exact MH transition may accept a higher-cost plan because of the prior/proposal factors. Never replace MH with greedy cost descent while retaining claims of target invariance.

**Rejecting means the state remains unchanged.** Store repeated states in chain traces and target-sample buffers. Keeping only accepted states samples the jump chain and generally biases the distribution.

### 4.3 Learned log-ratio and approximate acceptance

Use

```math
d_phi(z,z';x) = g_phi(x,z,z') - g_phi(x,z',z),
a_phi(z,z';x) = min(1, exp(d_phi(z,z';x))).
```

A three-layer MLP is enough. Enforce antisymmetry algebraically, not with a penalty. The context must be the same in both evaluations. Any feature normalization or pair feature must preserve this construction.

The network must not receive the true proposed simulator cost, exact `ell`, target grid density, or future route-success label at fast inference. Otherwise it is not avoiding the costly evaluation. It may receive normal observable scene/context information and cheaply decoded features, provided their computation is included in runtime accounting.

Training labels come from the exact ratio above. The default supervised objective is

```math
L_ratio(phi; theta) = E_{z ~ p_x, z' ~ q_theta(.|z,x)}
                     [(d_phi(z,z';x) - ell_theta(z,z';x))^2].
```

Use exact simulator/cost VALUES as labels, not their gradients. Raw squared error is the primary loss because it corresponds to the manuscript's stated bound. A robust auxiliary loss can be an ablation, but is not the same certificate.

The exact ratio depends on `theta`. After proposal parameters change, cached costs can be reused, but proposal log-density terms and ratio labels must be recomputed. Refresh the acceptance fit before reporting the new approximate sampler.

### 4.4 Accepted-flow audit

For a fixed context, write

```math
F(z,z') = p_x(z) q_theta(z'|z,x),
abar = E_F[a_phi],
L_bal = E_F[a_phi (d_phi - ell_theta)],
epsilon_a = E_F[abs(a_phi - a_star)].
```

Implement the manuscript's population diagnostics:

```math
TV(p_x T_{theta,phi}, p_x) <= sqrt(abar * L_bal / 2),
TV(p_x T_{theta,phi}, p_x) <= epsilon_a <= sqrt(L_ratio).
```

`L_bal` is nonnegative at the population level under these definitions, but individual terms and finite-sample estimates can be negative. Do not silently clamp a Monte Carlo loss and present it as a certified bound. Verify the identity on a finite-state diagnostic, and label Monte Carlo estimates with their sampling limitations.

Use ratio regression as the main acceptance-learning objective initially. Implement `L_bal` as an audit; an optional balance-loss optimization experiment can be added only after the supervised version works.

Do not minimize balance error as the proposal-learning objective: an almost-identity proposal can have an excellent balance score and terrible exploration.

### 4.5 Separate proposal-learning objective

Use the paper's bounded accepted-movement objective:

```math
M(theta) = E_{p_x(z) q_theta(z'|z,x)}
           [a_phi(z,z';x) * m_x(z,z')],  0 <= m_x <= 1.
```

Use a bounded distance between **decoded plans or behavior features**, not only distance between latent vectors. For example,

```math
m_x(z,z') = 1 - exp(-||Phi_x(G(x,z')) - Phi_x(G(x,z))||^2/(2*l^2)).
```

`Phi_x` is a fixed feature map and `l` a fixed scale selected from training contexts. In the analytic toy, the decoded 2D coordinates suffice. In the route toy, use normalized path coordinates or a few fixed path-shape features. Simulated features are possible, but their evaluations count. Never reward changes in unused random-input coordinates.

With `phi`, the target, decoder, and feature map frozen, implement the score-function gradient from Appendix A.4:

```math
grad_theta M = E[(a_phi*m_x - baseline(x,z))
                 * grad_theta log q_theta(z'|z,x)].
```

The baseline must not depend on the current sampled `z'`. A running scalar baseline or a small prediction from `(x,z)` is sufficient.

**Autodiff rule:** draw `z'` from the current proposal, stop its gradient before evaluating the score-function log probability, and stop the reward/baseline gradients in this update. Do not accidentally combine a reparameterization derivative with the score-function estimator. Do not backpropagate through accept/reject booleans or simulator outcomes.

Generate fresh proposals for score-function updates. Reusing stale proposal samples after updating `theta` is not an on-policy score estimator unless explicitly importance corrected.

When `a_phi` is replaced by exact MH acceptance, it depends explicitly on `theta` through both proposal densities. The displayed gradient would then be incomplete. Do not make that substitution silently.

Surrogate exploitation is a real risk: use short proposal-update blocks, refit the ratio model, and audit with exact acceptance. High learned acceptance alone is not evidence of progress.

### 4.6 Exact refresh

Implement the actual mixture kernel

```math
P = (1-rho) * T_{theta,phi} + rho * P_ref,   0 < rho <= 1,
```

where `P_ref` is exact independence MH with `z' ~ varphi`. Its acceptance is

```math
min(1, exp(-(J_new - J_current)/beta)).
```

Choose the refresh branch with a state-independent Bernoulli draw. A fixed every-k-th-step refresh schedule is a different, periodically varying construction; do not label it the homogeneous mixture from the theorem.

For a known bound `0 <= J <= J_max`, log

```math
kappa = rho * exp(-J_max/beta).
```

The manuscript bounds frozen-kernel bias by `(1-rho)*epsilon_a/kappa` and gives a geometric finite-step term. Evaluate these as mathematical diagnostics where target expectations can be computed; do not claim that finite neural training certifies them. Show when the bound is vacuous. Calculate very small constants in log space.

### 4.7 Cost-cache correctness

Maintain whether the current state has a valid exact cost and the context/decoder version to which that cost belongs.

- An exact proposal can reuse a valid current-state cost.
- A learned-acceptance move can enter a state whose true cost has never been evaluated. Mark that cost unknown.
- A later exact refresh from such a state must evaluate BOTH the current and proposed costs if neither is cached. Count both.
- Rejection preserves the current state and its cost-validity metadata.
- A new scene, cost configuration, decoder, or continuation policy invalidates affected cost caches.
- Logging and plotting must not secretly invoke an uncounted simulator.

Never use a surrogate cost as the current cost in an allegedly exact MH ratio.

## 5. Alternating training and the bootstrap distribution

For a fixed training round:

1. Freeze decoder, target/cost, temperature, context definitions, and a proposal snapshot.
2. Obtain target-distributed or approximately target-distributed `z` states using exact-correction sampling.
3. Draw proposal pairs and compute exact log-ratio labels.
4. Fit the antisymmetric ratio network with the proposal frozen.
5. Freeze the ratio network; take a short block of accepted-movement proposal updates.
6. Recompute proposal-dependent labels, refresh coverage as necessary, and refit the ratio network.
7. Freeze both networks for evaluation.

Finite MH chains are NOT automatically stationary. Use multiple initial states, at least one global-refresh mechanism, trace/region diagnostics, and documented burn-in. For the analytic toy, compare against independent exact rejection samples as described below. Do not use best-of-N candidates as though they were samples from the target for `E_F`.

Track sample provenance: context ID, decoder/cost/proposal versions, method used to obtain target states, burn-in, and whether the samples are exact independent, correlated exact-kernel, or approximate-kernel outputs.

Parameter updates happen BETWEEN sampling rounds. Do not claim a continually adapting chain already has the invariant law of a frozen kernel.

Checkpoint proposal parameters, ratio parameters, optimizer states, random keys, feature normalization, target versions, and training counters. Resuming a checkpoint must be reproducible on the same platform.

## 6. Experiment A: genuinely two-dimensional, auditable targets

### 6.1 Why this experiment

The manuscript explicitly proposes a compact 2D analytic diagnostic with two and four modes. It is ideal for visualizing:

- true target versus sampled density;
- high acceptance with slow exploration;
- errors introduced by learned acceptance;
- the effect of exact refresh;
- what changes when proposal learning is turned off.

We specify the following convenient instantiation. These formulas are implementation choices, not verbatim formulas from the manuscript.

### 6.2 Gaussian noise and a compact descriptor domain

Use `d=2`, `z ~ N(0,I_2)`. Define a fixed analytic decoder

```math
y = G(z) = 2 Phi_std(z) - 1,  y in (-1,1)^2,
```

applied coordinatewise, where `Phi_std` is the scalar standard-normal CDF. Under the Gaussian prior, `y` is uniform on the square. This makes compact-domain quadrature particularly transparent while keeping the exact Gaussian noise-space target and pCN baseline.

There are no trained decoder parameters in this experiment. Label it `analytic_2d`, not `pretrained_dsbm`. Use a numerically stable CDF/inverse-CDF implementation for grid conversions. Do not add a decoder Jacobian term to the noise-space MH target.

### 6.3 Bounded two-/four-mode energies

Let the context specify centers `c_m`, positive-definite 2D matrices `S_m`, mixture weights `pi_m >= 0` summing to one, and a fixed floor `0 < eps_floor < 1`. Define

```math
w_x(y) = eps_floor + (1-eps_floor) * sum_m pi_m
         * exp[-0.5 (y-c_m)^T S_m^{-1}(y-c_m)],
J_x(y) = -beta_0 * log w_x(y).
```

`beta_0 > 0` is a fixed energy-scale constant; `beta` is the planning temperature. Because `eps_floor <= w_x <= 1`,

```math
0 <= J_x <= -beta_0 log eps_floor = J_max.
```

The descriptor-space target is proportional to

```math
w_x(y)^(beta_0/beta)
```

on the square. The sampler itself still runs in `z` space using the exact Gaussian prior.

Use at least:

- two separated approximately symmetric modes;
- two modes with rare-versus-common weighting;
- four modes, including a rotated/anisotropic version;
- a constant-cost case, where the target is exactly the Gaussian prior in `z` space.

A useful starting geometry has centers near `(+/-0.5,0)` or the four `(+/-0.5,+/-0.5)` corners. Choose widths and floors that produce visible but not numerically singular modes. A fixed floor adds background mass, so normalized region masses are NOT identical to the input `pi_m`. Compute them.

Train over a small family of contexts; split contexts, not repeated samples from identical contexts, into training/validation/test sets. Include a one-context mode for debugging. Temperature must either be an explicit conditioning input trained over a declared range, or be fixed for that acceptance-model checkpoint. A temperature slider must not silently reuse a supposedly calibrated ratio model at an untrained temperature; label that case as an out-of-distribution audit or use exact acceptance.

### 6.4 Numerical reference and independent sample oracle

Compute the normalizer and region probabilities by quadrature over the compact descriptor square. Use stable log weights, cell areas, and a grid-resolution comparison. Save the grid and mass estimates for reproducing every figure.

Define a fixed, documented region partition, for example a Voronoi partition by mode centers. Report **region masses** rather than claiming exact latent mixture-component labels for samples from a floor-plus-mixture energy.

Provide an independent rejection-sampling oracle:

```text
sample z from the Gaussian prior;
accept with probability exp(-J_x(G(z))/beta).
```

This is valid because the cost is nonnegative and the acceptance weight is at most one. It is for small diagnostic audits and optional clean target minibatches, not a free baseline: count its cost evaluations. Compare exact-MH bootstrap samples with it.

Histogram-vs-grid TV is a **coarse, sampling-dependent diagnostic**, not exact continuous TV. Include sample count, grid size, and repeat estimates where meaningful.

### 6.5 A separate finite-state mathematical audit

Add a small finite grid, around 11x11 or 15x15 states. Define normalized target masses and proposal rows on that grid and recompute the discrete Hastings ratios. Keep rejection probability on the diagonal.

The finite-grid kernel is a separate discretized problem, not the exact transition matrix of the continuous sampler. Do not use a huge dense matrix for a fine plotting grid.

For this small problem compute exactly:

- `p T - p` and its TV norm;
- stationary distribution of the learned-acceptance/refresh kernel;
- `abar`, `L_bal`, `L_ratio`, and `epsilon_a`;
- both one-step bounds and the refresh bound;
- finite-step law error for a chosen starting distribution;
- the identity-like-kernel failure mode.

Use exact acceptance, a deliberately perturbed antisymmetric log-ratio, and the learned pair network. This is where algebraic claims should be checked without Monte Carlo ambiguity. The exact discrete ratio must use the discrete proposal probabilities, not the continuous proposal density by accident.

## 7. Experiment B: visualize how noise refinement repairs a whole plan

### 7.1 Required environment

Build a small **deterministic 2D point-mass simulator**, not a full physics-engine integration. It should have position and velocity, a common PD target-tracking controller, one central obstacle, and an optional route-blocking obstacle.

A suitable model is

```math
mass * dv/dt = k_p * (a - r) - k_d * v,
dr/dt = v,
```

where `r` is physical position and `a` is a commanded target. Use stable substeps, bounded commands, deterministic snapshots, and collision checks on swept segments or sufficiently fine substeps. A collision can terminate the evaluation in an absorbing failure state; no rigid-contact solver is needed for this route-repair toy.

Log commanded targets separately from realized positions. This is a planning/geometry test, not a contact-passivity experiment. Codex may choose units, time step, gains, and obstacle shapes, but must verify stable free-space tracking and ensure the same simulator is used for every method.

Suggested scale: a workspace around `[-1.2,1.2]^2`, start on the left, goal on the right, and a central disk leaving two corridors. Begin with `H=16` commands, `K=4` executed per replan, and an action duration allowing the nominal full chunk to traverse the workspace. These are starting values, not constraints if another small setting is clearer.

### 7.2 Transparent two-dimensional frozen decoder

Keep the random input dimension at TWO for this first route demo, so we can plot the complete planning target rather than a misleading projection of a high-dimensional noise law.

Use a fixed smooth path family. For example, decode two bounded coefficients into a broad lateral bend and a timing/asymmetry adjustment of the cached nominal command sequence. Smooth spline or basis-function parameterizations are acceptable. Endpoint adjustment toward the current goal can be an explicitly documented deterministic part of the decoder.

The decoder must:

- accept the real current context and cached plan;
- be deterministic given `(x,z)`;
- produce an entire bounded `[H,2]` target chunk;
- represent both routes with non-negligible prior probability;
- not perform a hidden collision search or call the task-cost oracle;
- not use the successful-route label or exact density grid as input;
- change both immediately executed and unexecuted parts when appropriate.

State clearly that this is an **engineered low-dimensional plan family**, not a trained generative policy. Its purpose is to isolate Ai-Plan and let us inspect exactly where it searches. A favorable result establishes sampler behavior within this family, not general robotics performance.

Before training, plot a grid of decoded plans and simulate them. Ensure there really are good and bad candidates and that a valid alternative is within the decoder's support. If the decoder cannot express a repair, diagnose that as a support limitation rather than trying to fix it by training acceptance.

Prefer the same Gaussian-to-compact-coefficient map as experiment A, allowing quadrature in coefficient coordinates. If a different map is chosen, quadrature must include its actual induced coefficient prior rather than assuming a uniform distribution. The noise-space MH target remains Gaussian-prior times the cost weight, without a decoder Jacobian. Generic sampler code must not assume a two-dimensional input even though this benchmark uses one.

### 7.3 Context construction and disturbances

Provide three scenarios:

1. **Unchanged:** a cached route remains valid.
2. **Mild correction:** goal or obstacle shifts modestly; the strategy can remain valid after editing.
3. **Route invalidation:** a blocker closes the previously selected corridor while the other remains reachable.

Generate a causal initial snapshot by creating a nominal chunk, actually executing its prefix, and retaining its suffix. Then introduce the chosen disturbance before the next replan. Do not pair an unexecuted model prediction with a state advanced along some separate recording.

Changes must occur before the snapshot used to compare candidate evaluations. During a fixed-context chain, the scene and all disturbance information remain fixed. Apply route blocking early enough that both the nominal and alternative strategies are not already physically irrecoverable.

### 7.4 Cost and continuation

The initial scoring mode should simulate the entire candidate chunk from the same saved state. The physical robot still executes only its first `K` actions. This simple full-chunk lookahead is a deliberate implementation choice to avoid recursive planning in the first visual demo.

Use a bounded cost comprising goal error, collision/constraint violation, and a modest command-variation term. Normalize continuous terms using fixed bounded maps such as `r/(s^2+r)` for nonnegative squared error `r` and fixed scale `s`; sum nonnegative weighted terms and record the resulting known `J_max`. Report unnormalized physical metrics as well.

Use finite collision penalties in the sampling target; hard execution screening is a separate controller decision. Do not introduce infinite penalties and continue using the bounded-cost minorization unchanged.

Add a small **optional** `prefix_then_rollout` scorer, after full-chunk scoring works:

- execute the candidate's first `K` commands;
- preserve/complete its candidate-specific suffix;
- continue for a fixed small number of replans using a frozen deterministic base policy, or a fixed bank of continuation random inputs;
- never call Ai-Plan recursively inside this cost;
- count all continuation simulation steps.

This scorer should include a regression test: identical executed prefixes imply identical first-prefix physical states and costs, but different cached suffixes may cause later continuations to differ. If the suffix is discarded, deterministic continuations from the same physical/history state must agree.

### 7.5 Execution and memory

At deployment-style evaluation, generate and refine at the actual current snapshot, execute the selected prefix, read the actual successor, and align/complete the selected chunk. Completion can use fixed damped extrapolation of the last command increments, with an explicit decay in `[0,1]`.

Tail completion is not MCMC and is not SB revision. It may change the source proposal at the next physical replan, which defines a new target distribution. Do not concatenate chains across different contexts and report them as one stationary MCMC run.

Use a simple safe fallback such as braking/holding when no validated candidate is available. Report fallback frequency; screening or fallback changes the execution policy from the raw posterior sampler.

## 8. Sampling quality and planning quality are separate experiments

Implement two explicit output modes:

### Posterior-sampling mode

Return final/post-burn-in Markov states. Keep rejections as repetitions. Assess target-region probabilities, distribution discrepancy, autocorrelation, and effective sample size (ESS) for relevant observables. A fixed number of samples or transitions is used for density audits; varying the stopping time according to random cost expenditure can introduce extra sampling bias and must not be silently treated as the fixed-n theorem.

### Planner mode

Maintain the best feasible candidate among candidates whose costs were actually evaluated, or sample a terminal candidate and validate it. Execute one complete candidate's prefix. Do not average incompatible top/bottom plans.

The best-feasible incumbent is **not** a sample from the target. Never use best-of-N planning success as evidence of correct mode probabilities or stationary sampling.

For the fast learned-acceptance sampler, unevaluated proposed plans do not magically enter the exact-cost incumbent pool. Account for any final ranking or validation cost. A final check may improve safety but does not correct all distributional bias.

## 9. Required comparisons

Keep the comparison compact and mechanistic.

For both experiments, implement:

1. **Unmodified generator/prior sampling.** In planner mode, independent best-of-N under an equal cost budget.
2. **Gaussian random-walk MH.** Useful local baseline and high-acceptance/low-movement demonstration.
3. **pCN-MH.** `z' = sqrt(1-eta^2)*z + eta*epsilon`, with independent standard Gaussian `epsilon`, `0 < eta <= 1`; exact acceptance depends only on the cost difference.
4. **Learned proposal + exact MH.** Main test of proposal efficiency, with no acceptance approximation confound.
5. **Same learned proposal + learned acceptance.** Deliberately approximate, `rho=0`; report its bias.
6. **Same learned proposal + learned acceptance + exact refresh.** Small configurable set of refresh rates.

For the route toy add **warm-started CEM over the same two coefficient/noise coordinates**, using the same cost and evaluation budget. It is an optimizer, not a posterior-sampling baseline. Do not label arbitrary weighted averaging as MPPI; full MPPI is not required in this pilot.

Add cheap mechanism ablations rather than a large hyperparameter sweep:

- no proposal learning / no movement objective;
- ratio regression versus a controlled antisymmetric error injection;
- high-locality proposals with high acceptance but low mode switching;
- identity decoder versus the fixed plan decoder;
- no exact refresh versus several nonzero rates.

There is no scientific acceptance criterion requiring the learned method to win. On a tiny problem, pCN, rejection sampling, or CEM may be excellent. Characterize what happens.

## 10. Fair cost accounting

Every run needs independent counters for:

- proposal attempts / total MCMC transitions;
- decoder calls;
- exact cost evaluations;
- physical simulator substeps or transitions;
- training-time oracle evaluations;
- exact-refresh current-state and proposed-state evaluations;
- final validation evaluations;
- audit-only evaluations;
- compilation time and warmed execution time.

Count rejected proposals, burn-in, all lookahead completions, and cache misses. Batch size is not a discount on the number of simulated candidates. Caching may save real work, but it must use immutable context/decoder/cost IDs.

Run two budget views:

- fixed transition/sample count for stationary diagnostics;
- fixed true oracle/simulator budget for planning performance, with a wall-clock or attempt cap so learned-only steps are not unlimited free work.

The analytic energy and small JAX simulator may be cheaper than the acceptance network. Do not assert a wall-clock speedup because the number of cost queries fell. Do not insert artificial sleeps and call them actual simulator speedups. An explicit oracle-delay sensitivity study may be optional and clearly labeled synthetic.

**JAX trap:** masking an expensive computation does not necessarily avoid executing it. `where` and some vmapped conditional constructions can evaluate both branches. Compile approximate and exact transitions separately or use control flow that actually skips oracle work; verify via counters and profiling. If batched refresh scheduling shares randomness across chains, document the induced dependence and use independent replicate batches for uncertainty estimates.

## 11. Visualizations: central deliverable, not an afterthought

Produce a guided notebook `notebooks/01_ai_plan_intuition.ipynb` that imports the package and can be run after a short command-line demo. Keep training optional when reopening saved artifacts.

It should explain and display:

1. The Gaussian prior in noise space and the compact descriptor mapping.
2. Cost landscape versus normalized target density; these are NOT the same object.
3. Current Markov state, proposed state, exact acceptance probability, learned acceptance probability, and accept/reject result.
4. Chain trajectories and density estimates over the true target contours.
5. High acceptance with negligible mode exploration.
6. Changes in target mass error and exploration as refresh rate changes.
7. For route repair, the noise/descriptor point beside its **entire decoded command chunk** and realized physical rollout.
8. Cached old plan, retained suffix, completed tail, proposed repaired plan, and actually executed prefix as visually distinct objects.
9. The effects of temperature, proposal locality, and a fixed surrogate error on sampled distributions.

Use sliders or saved-frame controls for context, method, temperature, refresh rate, and chain step. Do not make network training happen unintentionally on every slider move. Recompute or load the target appropriate to changed parameters; never show samples from one target over another target's contours.

Save static PNG/PDF figures and a standalone HTML visualization or GIF so the user can inspect results without a notebook server. Choose a minimal interaction library; an HTML replay of precomputed data is sufficient. Record all displayed values from actual runs.

Recommended named outputs:

- `target_and_prior`
- `proposal_acceptance_replay`
- `chain_and_region_occupancy`
- `ratio_calibration`
- `refresh_bias_tradeoff`
- `finite_grid_bound_audit`
- `route_repair_replay`
- `success_vs_simulator_steps`
- `sampler_vs_planner_summary`

Do not call a projection of a higher-dimensional distribution the exact two-dimensional target. The initial experiments intentionally avoid this problem by keeping `d=2`.

## 12. Tests and mathematical invariants

Keep deterministic unit tests fast. Put statistically demanding checks in a separate audit command with explicit sample sizes and tolerances.

Required tests:

- `decode(x,z)` and `cost(x,A)` are deterministic, and repeated evaluation restores the same snapshot.
- Standard-normal log density, proposal log density, and exact log ratios agree with independent NumPy/SciPy calculations.
- `ell(z,z') = -ell(z',z)` and `d_phi(z,z') = -d_phi(z',z)` for fixed parameters/context.
- `d_phi(z,z)=0` and learned acceptance lies in `[0,1]`.
- Constant cost reduces the target to the prior; pCN acceptance is identically one in this case.
- pCN's general Hastings ratio matches the cost-only expression.
- Exact finite-state MH satisfies detailed balance and preserves the discrete target.
- Finite-state accepted-flow/ratio bounds and refresh stationary/finite-step bounds hold numerically.
- Tiny proposals may accept often but have small accepted movement; accepted-only trace filtering is never used for stationary estimates.
- Proposal gradient: compare the score-function formula with an exactly summed small discrete example and/or finite differences with controlled Monte Carlo error. Test stop-gradient placement.
- Updating proposal parameters recomputes ratio labels; decoder parameters remain unchanged throughout sampler training.
- Learned-only transitions make zero true-cost calls unless explicitly auditing/validating.
- Refresh after an unevaluated learned move evaluates the current cost correctly.
- Rejection leaves the current latent, decoder output, and cost-validity state consistent.
- Simulator batching agrees with scalar execution.
- Collision detection catches crossing an obstacle between command samples.
- Causal cache updates use the plan actually selected and executed.
- Identical-prefix / different-suffix continuation test from §7.4.
- Frozen target hash changes when any target-defining parameter changes; stale costs cannot be reused.
- Checkpoint reload and PRNG-state restoration reproduce the next transitions on the same machine.
- High-dimensional interface smoke test (for example `d=6`) does not break sampler mathematics; no plotting code assumes all decoders are 2D.
- An auxiliary test decoder that ignores some noise coordinates does not get movement reward solely from moving in those unused coordinates.

For independent rejection samples, verify region masses within statistical uncertainty of quadrature. Avoid flaky fixed thresholds on tiny samples. State when an integration or finite-chain approximation is being tested rather than exact arithmetic.

## 13. Software organization and minimum public interface

A suggested layout (Codex may simplify it):

```text
ai-planner/
  PROMPT.md
  pyproject.toml
  uv.lock
  README.md
  src/ai_planner/
    targets.py
    decoders.py
    contexts.py
    proposals.py
    acceptance.py
    samplers.py
    training.py
    envs/point_mass.py
    evaluation.py
    diagnostics.py
    visualization.py
    cli.py
  configs/
    analytic_smoke.yaml
    analytic_pilot.yaml
    route_smoke.yaml
    route_pilot.yaml
  notebooks/01_ai_plan_intuition.ipynb
  tests/
  docs/
    ALGORITHM.md
    EXPERIMENTS.md
    DECISIONS.md
    RUN_REPORT.md
  references/Ai_Plan_Concept_Manuscript.pdf
```

Avoid elaborate plugin registries or duplicate parallel implementations. Simple dataclasses/PyTrees and pure functions are sufficient.

Essential concepts, not mandated signatures:

- decoder: explicit `noise_dim`, `decode(context,z)`, frozen artifact/version ID;
- evaluator: deterministic `cost(context,A)` plus physical metrics and cost counters;
- proposal: `sample` and `log_prob`;
- acceptance: exact labels and antisymmetric learned logits;
- chain state: current `z`, optional decoded candidate, exact-cost value/validity, PRNG state;
- trace: proposals, accepted/rejected states, scores, branch type, costs, and counters;
- context: immutable snapshot and all information that defines the target.

Use `lax.scan` for fixed-length chain/simulator loops and `vmap` for independent candidates or contexts where useful. Keep plotting, checkpoint I/O, and orchestration outside JIT. Do not put simulator gradients into the main proposal update merely because the toy happens to be differentiable.

Suggested CLI capabilities:

```bash
uv sync --extra dev
uv run pytest -q
uv run ai-planner demo analytic --config configs/analytic_smoke.yaml
uv run ai-planner train analytic --config configs/analytic_pilot.yaml
uv run ai-planner audit --run outputs/<run_id>
uv run ai-planner demo routes --config configs/route_smoke.yaml
uv run ai-planner train routes --config configs/route_pilot.yaml
uv run ai-planner evaluate --checkpoint <path> --config <path>
uv run ai-planner render --run outputs/<run_id>
```

Exact argument names may differ. Document the commands that actually work, including a CPU-only path. Do not leave example paths to nonexistent checkpoints as the only runnable quickstart.

Each run saves resolved configuration, versions, seeds, checkpoints, JSON/CSV metrics, chains, target grids where applicable, and figures. Use an artifact format that permits rendering without re-running the simulator. Do not require W&B or a remote service.

## 14. Implementation order and compute budget

Complete in this order:

### Milestone 1 — exact sampler and visual truth

Implement analytic targets, frozen decoder, quadrature/rejection oracle, Gaussian random-walk and pCN exact MH, finite-grid tests, and a first visualization. No neural training is needed yet. This catches probability/sign/indexing bugs before learning can conceal them.

### Milestone 2 — learned Ai-Plan components

Implement ratio regression, accepted-movement proposal updates, frozen alternating rounds, learned acceptance, exact refresh, calibration and bias audits. Compare the SAME learned proposal under exact versus approximate acceptance.

### Milestone 3 — planar plan repair

Use the same sampler with the fixed 2D path decoder, point-mass simulator, causal cache, disturbances, CEM/independent comparisons, and synchronized latent-to-physical replay. Start with full-chunk scoring; add the optional continuation scorer only after that is validated.

### Milestone 4 — usable handoff

Finish the notebook/HTML replay, documentation, checkpoint resume, budget plots, and a compact actual-run report. The report must state what improved, what did not, and what remains untested.

Use one development seed and small networks first. A handful of frozen contexts and hundreds to a few thousand optimizer updates are reasonable smoke/pilot starting points; choose sizes based on measured runtime. Supply configurations for later multi-seed evaluation, but do not launch large sweeps or long jobs by default. Do not spend most of the implementation effort on repeated preliminary training gates.

Spend the necessary coding and debugging effort for correctness. If resources prevent a long experiment, still implement it fully, run a bounded end-to-end check, and provide the exact resumption command. Do not report a small smoke run as a conclusive research comparison.

## 15. Acceptance criteria

The pilot is complete when:

- It installs from a clean environment and all focused unit tests pass.
- Both required experiments run through generation, training, sampling, evaluation, and visualization.
- The frozen decoder is visibly distinct from the learned proposal and learned acceptance model.
- Exact correction matches the numerical reference within documented numerical/statistical error on the small analytic cases.
- The code demonstrates and explains at least one high-acceptance/poor-exploration case.
- Learned acceptance can be compared with exact acceptance and exact refresh without hidden oracle calls.
- A user can watch a proposal in two-dimensional noise/descriptor space change an entire route and its actual simulated outcome.
- Sampling fidelity and best-found planning performance are reported separately.
- All compute/cost claims include the relevant evaluations and wall-clock measurements.
- No fictional numbers from the manuscript are reused as results.
- The report does not claim a pretrained neural decoder, a DSBM advantage, real-robot safety, or physical passivity that was not tested.

Learning may fail to outperform pCN or CEM in the first pilot. That is a valid outcome. Correctly diagnosing whether the limitation is proposal expressivity, acceptance calibration, poor mixing, target temperature, or decoder support is more valuable than forcing a favorable plot.

## 16. Explicitly deferred work

Do not implement these in the first pass:

- full DSBM/FM/diffusion pretraining or a pretrained robotics model download;
- Push-T, MuJoCo, Isaac, or a robotic-arm integration;
- the separate one-step occupancy-matching pseudo-paper;
- actor–critic RL, policy fine-tuning, or distillation of planner outputs;
- adaptive/noisy simulator objectives without rederiving the sampling target;
- mean-field particle repulsion;
- general nonlinear diffusion bridges;
- distributed/multi-GPU training;
- a large web dashboard.

The future extension is to replace the analytic `G` with a genuinely pretrained conditional generator, freeze its weights, collect actual simulator contexts, and rerun exactly the same outer sampling code. Its noise dimension will generally be larger (all stochastic revision innovations must be explicit); the analytic 2D contour guarantees will no longer apply. The new decoder changes the target, so learned ratio models and proposal checkpoints must be retrained or explicitly recalibrated rather than reused without checking.

## 17. References for implementation decisions

The local concept manuscript is the authority for the proposed Ai-Plan method, but its experiments are hypothetical. External references supply background, not evidence that this pilot will work:

- Ai-Sampler: `https://arxiv.org/abs/2406.02490`
- pCN / MCMC Methods for Functions: `https://arxiv.org/abs/1202.0709`
- DSBM (for the later decoder extension): `https://arxiv.org/abs/2303.16852`
- JAX JIT: `https://docs.jax.dev/en/latest/jit-compilation.html`
- JAX vectorization: `https://docs.jax.dev/en/latest/automatic-vectorization.html`
- JAX scan: `https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html`
- JAX PRNG: `https://docs.jax.dev/en/latest/random-numbers.html`
- JAX benchmarking: `https://docs.jax.dev/en/latest/benchmarking.html`
- JAX installation: `https://docs.jax.dev/en/latest/installation.html`
- JAX float64: `https://docs.jax.dev/en/latest/default_dtypes.html`
- Flax Linen: `https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html`
- Optax: `https://optax.readthedocs.io/en/latest/getting_started.html`
- PyTorch compilation (considered alternative): `https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html`

Do not copy a library API from memory when the installed version differs. Pin and test one compatible stack, and state that stack in the final run report.
