# ADDENDUM: Ai-Plan second experimental round

**Read this file after `PROMPT.md` and use both as the implementation specification.**

Repository: `https://github.com/Ricvalp/ai-planner`  
Review baseline: `a5510734277a963fd40b051f951f0189a78b6595`  
Prepared: 28 October 2026

## 0. Assignment, precedence, and completion

Extend the existing repository and actually run the bounded experiments below. The central question is:

> Does a learned state-dependent proposal, evaluated with exact MH acceptance, discover successful repairs with fewer simulator evaluations than tuned simple methods on held-out contexts?

The second question is whether any improvement survives complete physical execution.

This addendum replaces the original instructions to start from an empty directory and to run only one development training seed. Preserve the working implementation, original prompt, manuscript, existing reports, checkpoints, and first-round outputs. Inspect the actual checkout and accommodate later changes; the review baseline is a reference, not a request to reset the repository.

All original mathematical invariants remain in force unless explicitly refined here. Continue with JAX, Flax Linen, Optax, the deterministic simulator, and frozen analytic decoders. The main inference method in this round is the learned proposal with **exact MH**. The learned ratio network remains part of the original alternating training procedure.

Implement and execute the required core in Sections 1–12, then produce the artifacts and report in Sections 14–16. Section 13 distinguishes bounded conditional work from deferred research. Negative or inconclusive learning results are valid outcomes and do not authorize dropping required comparisons.

Make routine implementation choices independently. Keep reasonable freedom over geometry, basis functions, module layout, training schedule, and plotting details. Record choices before test evaluation. Do not keep enlarging the experiment until Ai-Plan wins.

## 1. Inspect and preserve the first-round implementation

Read the original prompt, available manuscript, and:

- `docs/ALGORITHM.md`, `docs/DECISIONS.md`, `docs/EXPERIMENTS.md`, and `docs/RUN_REPORT.md`.
- The sampler, proposal, ratio model, training, planner, route decoder, simulator, deployment, and evaluation implementations.
- Current configuration files and relevant tests.

Run the existing focused tests before substantive changes. Record the source revision, dirty-tree status, environment, and initial test outcome. Diagnose failures rather than treating an inherited report as a fresh test result.

The existing evidence suggests questions to investigate, not numerical targets to reproduce: modest proposal gains, approximate-acceptance bias, easy full-chunk planning, and a short deployment horizon. Prior conversational support estimates and replanning counts are not new repository measurements. Reproduce any numbers you want to cite.

Use a new output namespace such as `outputs/round2/<run_id>/` and a new report, `docs/RUN_REPORT_ROUND2.md`. Preserve historical evidence. If changing a decoder, evaluator, feature map, or context schema, assign a new version and retrain affected models.

## 2. Required scope and order

| Stage | Required work | Proceed when |
| --- | --- | --- |
| A | Preserve exact mathematics, refactor context handling and budget records, run physical execution controls | Tests establish correct evaluation and causal execution |
| B | Calibrate a varied 2D repair family and create immutable data splits | Development scenes contain inspectable successful and unsuccessful candidates |
| C | One-seed development study, baseline tuning, proposal-objective audits | The pipeline runs and the final protocol can be frozen |
| D | Three-seed analytic and fixed-context planning confirmation | Settings are frozen; learned superiority is not a prerequisite |
| E | Complete closed-loop episodes on a prespecified small subset | The physical execution controls pass |
| F | Render saved data, aggregate results, document limitations and reproduction commands | Every required experiment has an explicit completed or resource-blocked status |

Correctness failures are engineering gates: repair them before making performance claims. A learned method losing to pCN, Sobol, or CEM is not a gate failure. Finish the bounded main comparison.

Do not introduce SB/DSBM training, a robotics engine, a different framework, distributed training, policy distillation, or a dashboard in this assignment.

## 3. Preserve the algorithm and distinguish the evaluation modes

For an immutable context and frozen decoder, retain:

```math
p_x(z) = Z_x^{-1} varphi(z) exp[-J_x(G(x,z))/beta],
varphi = Normal(0,I).

ell_theta(z,z';x)
  = log varphi(z') - log varphi(z)
    - (J_x(G(x,z')) - J_x(G(x,z)))/beta
    + log q_theta(z|z',x) - log q_theta(z'|z,x).
```

Use the same proposal parameters in both directions. Do not add a decoder Jacobian, clip Gaussian proposal draws, replace MH with greedy descent, or omit repeated states after rejection.

Keep contexts, the source cache, model parameters, cost, temperature, simulator, and continuation definition fixed within each chain. Physical execution creates a new context and target. Re-evaluate candidates as needed under that new target.

There are two different evaluation protocols:

1. **Sampling:** a deterministic number of transitions, with a declared initialization and burn-in policy. Use these traces for density, expectation, mode, ESS, and stationary-bias diagnostics.
2. **Planning:** a bounded number of actual evaluator calls, with a documented stopping rule. Use the evaluated candidate pool to select an incumbent.

A random oracle-budget stopping time does not automatically preserve the target law of an exact transition kernel. Do not use budget-stopped endpoints, best-found plans, or concatenated replanning traces as target samples.

An exact invariant distribution also does not imply that a short chain started from a bad plan has equilibrated.

## 4. Evaluation infrastructure and fair accounting

### 4.1 Compile numerical contexts dynamically

Refactor the hot evaluation and transition paths so numerical context values are JAX arrays or suitable PyTree leaves. Keep only shapes and configuration that changes the program structure static. Avoid rebuilding a closure and executable for every new physical snapshot.

Use fixed-capacity obstacle arrays and masks if needed. Keep context IDs and serialization metadata outside hot numerical computations.

Verify that same-shaped contexts with different positions, velocities, goals, obstacles, and caches reuse the compiled computation while producing correct distinct results. Preserve exact branch behavior: do not use batching or masking that executes an expensive oracle branch on nominally cheap moves.

Do not evaluate the ratio network inside the timed learned-proposal/exact-MH path. Compute its predictions separately for audits and replay. Preserve measurements of compilation, warmed execution, and host orchestration; synchronize asynchronous work before stopping timers.

### 4.2 Return physical metrics with an exact evaluation

An exact route rollout should return cost and physical metrics together, including collision and goal-success flags. Cache the record with the candidate and immutable evaluator/context identity. This avoids running the same rollout again merely to discover whether a candidate succeeded.

**New primary validation convention:** an already exactly evaluated candidate in the same unchanged deterministic context can reuse that record for final validation. Every rollout actually performed counts; a valid cached read does not. If a new check really is necessary, count it before reporting a result within budget.

This explicitly refines the first-round convention that reserved an additional rollout for final validation. Save a `validation_policy` field. Do not mix old and new budget conventions in one comparison without translating and labeling them.

Changes in state, obstacles, goal, decoder, simulator, evaluator, or continuation invalidate affected records. Temperature belongs to the target identity: invalidate target-dependent state conservatively. A raw physical rollout may be reused across a temperature-only change only if its evaluator identity is separately verified and target quantities are recomputed.

If cost weights change, do not infer feasibility from an old scalar-cost threshold unless its required inequality still holds. Prefer the physical flags returned by the rollout.

### 4.3 Count and retain the actual work

Keep separate ledgers for:

- Candidate proposals, transitions, decoder calls, exact evaluations, and simulator substeps.
- Training, validation/tuning, quadrature, rejection-reference generation, and audits.
- Planning inference, extra validation, fallback validation, and physical execution.
- Compilation, warmed numerical work, end-to-end planning time, and experiment orchestration.
- Shared artifact reuse versus new physical computation.

Count rejected proposals, rejected reference draws, discarded batch surplus, burn-in, and cache misses. A batched call is not one candidate evaluation.

Every exactly evaluated proposal may enter the planner pool, including MH-rejected proposals. Unevaluated proposals may not. Audit-only results must never influence online selection.

Do not silently fill an exhausted or stalled learned search with prior draws. If a fallback search policy is retained, name it, log its proposals separately, and include its cost. Do not resample an unaffordable refresh branch until a cheaper branch appears.

## 5. Physical execution controls and decoder progress

Perform these controls before interpreting closed-loop learned results, initially on the existing three scenarios.

### 5.1 Full-chunk execution

Find a feasible chunk by an independently charged grid search. Execute all its commands from the same snapshot and compare predicted versus realized positions, velocities, collision status, and final goal error.

This is a deterministic consistency check, not a learned-planner trial.

### 5.2 Prefix execution with cache continuation

Execute that same chunk through repeated prefixes, retaining its actual suffix and appending the documented tail. Generate no replacement plan during this control.

Through the original chunk's duration, the command sequence and physical trajectory must agree with full execution, within a justified numerical tolerance. Preserve the existing causal rule: the next cache comes from the chunk actually selected and executed.

Test a nonzero-tail-increment case as well as endpoint holds, so tail completion is exercised.

### 5.3 Grid-search replanning

At each replan, evaluate a coefficient grid from the actual successor state, select an incumbent using the same selection rule as other planners, execute its prefix, and continue. Start with a `41 x 41` grid; refine a few development controls if necessary.

Use a default cap of 64 replans, with early stopping on actual goal success or collision. Log the corresponding physical duration from the simulator settings. At the review baseline, `K=2`, nine substeps per command, and `dt=0.02` imply 23.04 seconds for 64 replans; derive this from the running configuration rather than hard-coding it.

Label this a grid reference over the decoder family. It is not a budget-matched competitor or a proof of global physical optimality.

### 5.4 A minimal progress-preserving decoder comparison

If grid replanning repeatedly postpones progress, implement one alternative that continues the aligned cache's remaining timing. Compare it with the current decoder before changing the cost as well.

A suitable starting family is:

```math
G(x,z)_k = project_bounds(
    C_x[k] + s1*u1*b1(k)*n_x + s2*u2*b2(k)*n_x
),
u = 2*Phi_std(z)-1.
```

Here `C_x` is the actual aligned/completed cache; the shapes, amplitudes, and normal direction are fixed documented functions of observable context. This is illustrative, not a mandated final formula.

Prefer `G(x,0)=C_x` when there is no documented deterministic goal adjustment. Verify the identity explicitly if claimed. Preserve meaningful arrival/hold timing and allow edits early enough to influence the executed prefix.

The decoder must remain a deterministic two-coordinate family, with no collision search, evaluator call, grid lookup, winning-route label, or optimized test-specific parameter. Check that both small corrections and route changes remain representable on calibration scenes.

Cache identity alone does not guarantee closed-loop progress; retain the grid-completion diagnostic. If cost shaping or a deadline is subsequently needed, compare one change at a time, preserve bounded nonnegative costs, update `J_max`, and version the resulting target.

Freeze the chosen decoder and cost before the main training and confirmation runs. Historical checkpoints are not automatically valid for the new target.

## 6. A controlled family of repair problems

### 6.1 Separate calibration, training/validation, and testing

Use three distinct stages:

1. **Calibration scenes:** choose geometry ranges, decoder edit scales, cost weights, and one global temperature for the route family.
2. **Training and validation scenes:** fit models and select baseline/model settings.
3. **Test scenes:** evaluate the frozen protocol once on independently generated scene/history seeds.

Split entire histories or episodes, not adjacent replans. Record reproducible generation seeds and immutable context manifests. Check for duplicate contexts across splits.

Generate cached plans in the old scene, execute the causal prefix, then apply the disturbance. Do not construct the old cache using the post-disturbance successful route. Reject physically invalid generated inputs only under prespecified rules, logging rejection counts.

Include:

- Unchanged controls where keeping the old route is appropriate.
- Mild corrections where local repair is useful.
- Route invalidations where the old corridor is blocked but an alternative may remain reachable.

Vary geometry and causal history, including start/goal offsets, obstacle locations or widths, old route side, and reachable velocity/cache variation. Training on a single geometry with one scalar perturbation is insufficient for the intended context-generalization test.

### 6.2 Calibrate difficulty using two probabilities

For full-chunk success set `S_x`, estimate:

```math
p_prior(x) = E_varphi[1{z in S_x}],
Z_x = E_varphi[exp(-J_x(G(x,z))/beta)],
p_target(x) =
    E_varphi[1{z in S_x} exp(-J_x(G(x,z))/beta)] / Z_x.
```

A successful repair means the declared full-chunk collision/constraint and goal-tolerance criteria hold. Keep this distinct from collision-free feasibility and actual episode completion.

Aim on calibration scenes for approximate prior-success levels of 20%, 5%, and 1%, while retaining substantial target success mass, preferably at least about 0.5 on the intended solvable calibration cases. These are benchmark-design goals, not pass/fail conditions imposed on individual test scenes.

For intuition, if successes cost zero and failures cost `c`:

```math
p_target = p_prior / (p_prior + (1-p_prior)*exp(-c/beta)).
```

At `p_prior=0.01`, target success mass 0.8 requires `c/beta` of approximately 5.98. In the actual continuous-cost problem, compute the probabilities rather than applying this simplified formula as an estimate.

Create difficulty through transparent geometry: narrower corridors, offset obstacles requiring correlated coefficient changes, and invalidated route choices. Report whether the successful set has one component, separated alternatives, or thin/curved geometry. Equal success mass does not imply equal search difficulty.

Do not feed success probabilities, grids, reference optima, or successful coefficients into the proposal, ratio model, decoder, or online planner. Use ordinary observable scene features.

### 6.3 Freeze the generator of test scenes

After calibration, freeze the parameter ranges and global cost/temperature. Generate test seeds once.

Do not redraw a test scene because its prior mass, target mass, or algorithm performance is inconvenient. Report all generated valid scenes, including target-misaligned or apparently unsupported cases. A prespecified supplementary subset with numerically observed feasible support is acceptable if overall results and all exclusions remain visible.

Stratify plots using measured prior-success probability after generation. Intended difficulty settings and measured probability bins may differ. Do not silently force equal measured bins by selecting test scenes.

A temperature change defines a new target/checkpoint condition unless temperature was explicitly trained as a conditioning variable. The primary experiment uses one fixed temperature per family.

## 7. Numerical references without excessive memory or leakage

Use uniform midpoint quadrature in the current bounded coefficient coordinates. Under `u=2*Phi_std(z)-1`, the coefficient prior is uniform. If this map changes, use its actual induced prior.

For route audits, begin with a `101 x 101` grid and refine to `201 x 201` on a prespecified subset spanning difficulty and scenario, plus any unresolved boundary cases. The refinement decision must depend on numerical uncertainty, not which method wins.

Compute and save cost, collision, goal error, success, normalized target mass, `Z_x`, prior/target success mass, and the best audited grid candidate. Report resolution sensitivity.

Implement summary-only batched grids. Do not retain every command and physical substep trajectory for every fine-grid point in every scene. Save full trajectories only for a small representative selection needed for replay.

Use independent exact rejection samples for small reference checks and target-distributed audit sources. Count all draws. Stop at explicit attempt caps; a small `Z_x` makes the current training/reference sampler expensive. Do not replace exact samples with grid minima or selected best plans.

A grid finding no success means “no successful candidate found at this resolution.” A finite-grid minimum is a reference value; a continuous method may improve on it. Preserve negative gaps to that reference rather than clamping them.

Keep audit artifacts separate from runtime contexts. Verify that removing audit files does not change a planner's output for fixed runtime inputs and random keys.

## 8. Training and diagnostics of what the proposal learns

### 8.1 Preserve the original alternating objective

Keep the original ratio-regression and bounded decoded-movement objectives. During a proposal block:

- Freeze target, decoder, features, ratio parameters, and the baseline rule.
- Draw fresh destinations from the current proposal.
- Stop sampled-destination gradients before the score log-density.
- Stop reward and baseline gradients.
- Recompute proposal-dependent exact ratio labels and refit the ratio after proposal changes.

Do not replace learned acceptance with detached exact acceptance in the existing gradient estimator. The exact acceptance depends on proposal parameters and requires a different derivative treatment.

The existing reward already uses decoded features. Preserve it; do not replace it with raw latent displacement.

Keep target-state source provenance explicit. Training from independent target samples does not by itself train recovery from invalidated initial plans. Do not use best-of-N samples as target-distributed sources.

### 8.2 Audit surrogate and exact movement at checkpoints

At initialization and after each completed alternating block, estimate on validation contexts:

```math
M_surrogate = mean_i[a_phi(z_i,z'_i) * m_x(z_i,z'_i)],
M_exact     = mean_i[a_MH(z_i,z'_i)  * m_x(z_i,z'_i)].
```

Use the same pairs for the two estimates within a checkpoint. Draw sources from an independent target-reference pool and fresh destinations from that checkpoint's current proposal.

Do not reuse destinations drawn from an earlier proposal and call them an expectation under the new proposal without correction. Reusing a fixed independent source pool for paired comparisons is acceptable if documented.

Charge audit evaluations, use a separate audit random stream, and save uncertainty and the reward gap. Plot the two rewards alongside acceptance error, mode exploration, and validation repair performance. Evaluate final test diagnostics only after model selection is frozen.

Bound frequent checkpoint audits: prespecify a small subset, initially four validation contexts and 256 pairs per context, with a reusable independent source pool. Use fresh current-proposal destinations at every checkpoint. Broader validation audits occur only at prespecified model-selection checkpoints. Record these sample sizes and all reference-generation costs.

If the learned reward rises while exact movement deteriorates, test shorter proposal blocks or improved ratio fitting on development/validation data. Record this as surrogate exploitation evidence. Do not respond by silently changing the objective.

### 8.3 Bound model selection and preserve checkpoint comparability

Start with the existing small network. Explore at most two documented development training schedules per family before freezing one main configuration. Short blocks of roughly 5–20 proposal updates are reasonable starting points; use measured curves to select the total number of blocks.

Do not repeat a large architecture/temperature/seed sweep. If one modest capacity change is justified by clear fitting diagnostics, charge it as development work and explain the choice.

Save the initialized proposal and selected checkpoints. For the no-movement exact baseline, use the corresponding initialized proposal with no movement updates. If comparing learned acceptance in an ablation, match the ratio-fitting schedule too.

Context normalization and feature scales must use training data only. Persist model, optimizer, RNG, baseline, dataset hashes, and target versions for faithful resumption.

## 9. Analytic sampling and first-passage study

Retain the original two-mode, four-mode/anisotropic, rare-mode, and constant-cost cases. Expand the nonconstant families to multiple independent contexts with variation in centers, orientations, widths, and weights. Keep constants as a clearly labeled stress test if absent from training.

### 9.1 Exact-kernel methods

Required:

- Prior independence MH.
- pCN with validation-selected scale.
- Random-walk MH with validation-selected scale.
- Initialized neural proposal with exact MH.
- Trained proposal with exact MH.
- Trained proposal plus state-independent exact prior refresh, with one declared mixture rate, initially `rho=0.1`.

For the final item, both the learned-proposal branch and the prior-refresh branch use exact acceptance. Branch selection is an independent state-independent Bernoulli draw. Preserve the original high-locality example as a small regression/intuition control.

Tune pCN over `eta in {0.1,0.3,0.6,0.9,1.0}`. For random walk, a small fixed scale set such as `{0.1,0.3,0.7,1.5}` is sufficient. Select using a declared validation metric. Do not choose a different setting from each test context's outcome.

### 9.2 Two initialization protocols

- **Target-start exploration:** independent target-distributed starts for within-target mixing and estimation diagnostics. These are paid diagnostic starts, not free deployment initialization.
- **Poor-start recovery:** a common documented unfavorable initial state for mode discovery and first passage. Choose it through a fixed geometric or causal rule, without test-grid optimization.

Report target-start and poor-start results separately. For poor starts, first-passage statistics include the initial transient; do not burn away the repair problem.

For density/expectation audits, use fixed transition counts, save both initial and post-burn-in results where relevant, and retain rejected states. Exact target starts do not require burn-in mathematically; any discarded prefix must be declared consistently.

### 9.3 Primary measurements

Report mode/region-mass error against quadrature, errors in a few fixed expectations, per-chain switching/occupancy, relevant observable ESS, accepted decoded movement, and first-passage curves. Show oracle-normalized and time-normalized efficiency.

Do not let pooled region counts hide independently trapped chains. A constant observable should be marked zero/undefined under the existing conservative convention, with the target's region mass explaining whether constancy is expected.

Use one fixed histogram partition for comparable coarse-TV diagnostics and show independent-reference sampling noise. Do not call histogram discrepancy continuous TV.

## 10. Fixed-context planning study

### 10.1 Required planning methods

Use the same frozen decoder, evaluator, context, candidate eligibility rule, and total evaluation budget:

1. Independent prior candidate search.
2. Scrambled Sobol candidate search in coefficient coordinates.
3. Validation-tuned pCN with exact MH.
4. Initialized neural proposal with exact MH.
5. Learned proposal with exact MH.
6. Existing warm-started CEM.
7. Dense grid in a separate reference panel, with its full cost disclosed.

The matched exact-refresh method from the analytic study may be added as a small named route diagnostic; it need not multiply every route trial.

For Sobol, map uniform points to the actual coefficient prior and then to noise coordinates if required by the shared interface. Use reproducible independent scrambles. Handle inverse-CDF endpoint numerics explicitly without clipping arbitrary Gaussian proposal samples. Sobol best-candidate search is not a posterior sampler.

Tune CEM on validation data using at most four prespecified population/elite/smoothing configurations. Retain covariance/scale floors and include small-budget behavior. Avoid an unconditional preferred-side warm start when scene sides vary; use the same declared causal incumbent information available to other planners.

pCN settings may be selected separately for sampling and planning using declared validation objectives. Freeze each setting across its test set.

### 10.2 Common starts and candidate selection

Use the same causally available incumbent for all methods and charge any evaluation performed under the new context. If the incumbent has no known noise representation, retain it separately and initialize stateful chains with the same documented representable candidate. Charge that extra initialization consistently for all methods if needed for the primary comparison.

Do not assume `z=0` represents the cache for the historical decoder. Verify the property for any new decoder that claims it.

Use a fixed selection rule, initially minimum exact cost among collision-free candidates. Make fallback evaluation causal and budget-independent: after evaluating the common initial incumbent, if it is ineligible, evaluate the same deterministic holding/braking fallback early for every method and charge that call. Retain a collision-free fallback as an eligible incumbent. This early check is part of every subsequent budget prefix, not an unpaid check added after the budget expires.

If neither the fallback nor any later paid candidate is collision-free, return `no_validated_action`; in a deployment episode, terminate and record this failure rather than executing an unchecked action. A different fallback convention is acceptable only if it is fixed in the protocol, never exceeds the budget, and works consistently for all prefix checkpoints. Do not call holding “safe” without checking it in the actual state.

Record both:

- **Any-success-by-budget:** whether a successful repair was found anywhere in the paid candidate pool.
- **Selected-plan success:** whether the candidate actually returned by the fixed selection rule succeeds.

These need not coincide when continuous cost trades off goal accuracy and command variation. Do not choose a successful candidate retrospectively using audit data to improve a deployment score.

### 10.3 Honest budget-progress curves

Use checkpoints `B in {4,8,16,32,64}`. A budget includes every exact evaluation actually performed, including initialization and necessary fallback/final checks. Reused same-context metric records are free reads under Section 4.2.

For a prefix-consistent search, run once to the largest budget and derive smaller-budget incumbents from that stream. Keep the seed and algorithm independent of the requested terminal budget.

CEM requires an explicit choice:

- A budget-independent anytime population schedule, with honest per-evaluation prefixes; or
- Separate budget-aware runs, charged separately and labeled as such.

Do not use results from later members of an unfinished batch to select an earlier-budget incumbent. Time curves for batched methods must reflect when results become available, not pretend every partial budget enjoys full-batch timing.

Do not exceed a budget or perform hidden final evaluation. If a required branch cannot fit, use the declared budget-exhaustion rule and return the best paid candidate; this is planner behavior, not a stationarity statement.

### 10.4 Planning metrics

Required curves:

- Any-success and selected-plan success versus exact evaluations.
- Best eligible exact cost versus evaluations; show the fraction with no eligible incumbent.
- Calls to first successful repair, treating non-discovery by the cap as censored.
- The same main outcomes versus measured elapsed time.
- Performance versus measured prior-success mass, accompanied by target-success mass.
- Gaps to the best audited grid candidate, allowing negative values.

For independent prior candidates, show the theoretical reference `1-(1-p_prior)^N` with `N` equal to actual fresh independent draws. If a common incumbent is present, account for its known success separately. Do not substitute the nominal budget for `N` without subtracting non-draw work.

Keep frozen-context full-chunk outcomes distinct from physical completion.

## 11. Bounded development and confirmation protocol

Use these defaults unless a documented numerical or resource reason requires an adjustment before test evaluation.

### 11.1 Datasets and runs

| Component | Default scope |
| --- | --- |
| Calibration/development route controls | Approximately 12 representative contexts spanning all three scenarios |
| Main training contexts | Approximately 32 per family, with varied observable geometry/context |
| Validation contexts | Approximately 16 per family, disjoint from training/calibration |
| Analytic confirmation | 12 nonconstant test contexts, approximately four per main family, plus the constant-cost control |
| Route confirmation | 60 repair contexts: intended three difficulty settings x two repair types x ten scene seeds |
| Unchanged route controls | Six additional held-out contexts, reported separately from repair discovery |
| Independent training seeds | `17, 29, 43` for the frozen main configuration |
| Search repetitions | Five per context in development; ten per context in route confirmation |
| Planning budget | Development up to 32; confirmation up to 64, with the checkpoints in Section 10 |
| Analytic chains | Four independent chains per context/initialization protocol, initially 4096 transitions |
| Analytic burn-in | Declare explicitly; 512 is a reasonable poor-start density-audit default, separate from first-passage analysis |

In this table, “per family” means per experiment family: analytic targets versus route repair. It does not mean 32 training contexts for each analytic subfamily. State the actual totals and allocation across analytic subfamilies or route scenarios in the manifests.

Keep dataset-generation, model-training, proposal, acceptance, audit, and online-search RNG streams separate and reproducible. A search repetition is not an independent training seed.

Use the same held-out context IDs across methods and training seeds. Pair exogenous conditions and common initial candidates. Common random numbers may be used where appropriate, but do not force different proposal laws to produce identical destinations.

Prior, Sobol, pCN, random-walk, and CEM baselines need only run once per context/search seed/configuration. Reuse those results in paired comparisons without pretending they are fresh independent trials for each trained model. The initialized neural control must use each training seed's own saved initialization when those proposal laws differ; do not replace distinct initializations with one shared neural baseline.

Generate the route test set from the calibrated fixed ranges; the intended cells do not require exact measured success-probability bins. Keep every valid generated test scene.

### 11.2 Freeze model selection before confirmation

Before running main test comparisons, save a protocol manifest containing:

- Dataset seeds, IDs, and hashes.
- Decoder, cost, temperature, simulator, and context-feature versions.
- Training schedule, selected checkpoint rule, and three training seeds.
- Baseline settings and the validation metric used to choose them.
- Budgets, initialization, validation/selection rules, and stopping conditions.
- Fixed metrics, representative replay IDs, closed-loop subset, and planned exclusions.
- Resource caps and what counts as a completed run.

Use validation for checkpoint selection. Do not select a different checkpoint by its test results or choose one successful training seed as the main method.

If a genuine implementation bug is found after freezing, version the protocol and rerun affected comparisons fairly. Retain and explain the earlier exploratory results.

### 11.3 Resource discipline and resumption

Start with the one-seed development run and measure runtime, memory, rejection efficiency, and artifact size. Use these measurements to plan the complete fixed main run.

Use one local CPU environment or one already available accelerator. Avoid distributed jobs or infrastructure changes. Set explicit finite rejection-attempt and rollout-loop limits. A suggested active-computation cap is four hours and a generated-artifact cap is 5 GB for the default second-round command; make these configurable and record actual values.

Prioritize the complete required main comparison over optional ablations. Use batched summary grids and checkpoint after each context block and training seed. Resume without repeating completed paid evaluations, while preserving ledgers.

Do not silently shrink the confirmation set after seeing outcomes. If a real resource cap prevents completion, save the exact completed trial IDs, counters, checkpoints, and an executable resumption command. Mark the confirmation partial; do not present the development subset as completed confirmation.

Do not declare a learned advantage from a single favorable seed or use endlessly extended training as a completion criterion.

### 11.4 Statistical reporting and practical effect

Show each training seed separately and an aggregate across the prespecified scene set. Report uncertainty across independent scenes and online searches. If using a hierarchical bootstrap, describe its levels and keep paired methods together. With three training seeds, label the result an initial reproducibility check.

Do not treat all repeated searches from one trained network as independent model replications. Do not duplicate shared baseline trials to narrow uncertainty.

Before testing, declare a useful effect to inspect, for example calls required to reach 80% repair success. Report when a method never reaches that level. A twofold reduction would be interesting, but is not a mandatory outcome and must not determine which scenes are retained.

Report both online cost and offline cost. At matched performance, estimate a call-based amortization scale only when the learned method saves a positive number of calls:

```math
N_break_even = additional_offline_evaluations / calls_saved_per_decision.
```

Separate this from wall-clock amortization. Charge reference/rejection work used for training even if the corresponding arrays are reused by multiple ablations; distinguish marginal new computation from standalone method data requirements.

## 12. Complete closed-loop evaluation

Prespecify 12 initial scenes: four unchanged, four mild correction, and four invalidation, selected by seed/intended setting rather than comparative planner outcome. Use held-out histories and common exogenous disturbance schedules.

Run the main practical methods: independent prior, Sobol, tuned pCN, learned-exact, and CEM. The initialized neural baseline is optional here because it is already tested in fixed contexts.

Use:

- Budgets 8 and 16 exact evaluations per replan.
- Three online search repetitions per initial scene.
- All three learned training seeds.
- Up to 64 replans, with actual success/collision termination.
- No early success label based solely on a predicted full chunk.

Reuse deterministic nonlearned trials rather than rerunning them for each learned seed. After methods choose different actions, their states and caches should diverge naturally; do not reset them to a shared reference trajectory.

Run grid replanning on the small execution-control set from Section 5, not on every cell of this factorial comparison.

Report actual completion, collision, timeout, `no_validated_action`, time to goal, fallback frequency, number of replans, total exact evaluations, simulator substeps, and total planning time per episode. Include remaining goal distance for timed-out or terminated episodes.

Record visited contexts and compare their feature ranges with training. If mismatch appears important, report it. A later visited-context training ablation must use separate training episodes and a new held-out evaluation; do not train on test episodes and relabel the results as generalization.

A decoder or cost change after this analysis defines a new experiment, not an invisible repair to the old table.

## 13. Conditional and deferred follow-ups

### 13.1 Bounded context-only proposal control

If development/validation shows a plausible learned-proposal benefit and the required core fits the resource budget, run one context-only control:

```math
q_context(z'|x)
```

A small conditional Gaussian mixture trained on the same allowed offline target pools is acceptable. Disclose capacity, training objective, and standalone data cost. Use exact independence MH for a small sampling comparison and independent draws for planning.

Decide whether to include this control using development/validation evidence before looking at the confirmation results. If not included, provide a concrete follow-up protocol and mark it unrun; do not create nonfunctional CLI placeholders.

Interpret a match with Ai-Plan as evidence that context amortization may explain the benefit. A state-dependent advantage would support iterative refinement. Neither outcome is required.

### 13.2 Acceptance efficiency: next study, not the default main sweep

Preserve the existing learned-acceptance and refresh code and regression tests, but do not multiply the primary confirmation across many acceptance settings.

Document the next matched study using one frozen proposal:

- Exact MH.
- Learned acceptance.
- Learned acceptance with prior refresh.
- Exact MH with the identical prior-refresh mixture.
- Delayed acceptance as an exact screening comparator.

For the last method, a frozen antisymmetric predicted log ratio `d` and exact log ratio `ell` give:

```math
a1 = min(1, exp(d)),
a2 = min(1, exp(ell-d)).
```

Use independent uniform tests; evaluate the expensive exact ratio only after the first test passes. Antisymmetry makes the forward/reverse ratio of the product equal to `exp(ell)`. This preserves detailed balance for the frozen kernel but can reject more than ordinary MH. Its value must be judged by computation-normalized accuracy/exploration.

If implemented as a small optional extension, verify the factorization on the existing finite-state audit and verify actual oracle skipping. Do not let it delay the required exact-proposal experiments.

For future refresh diagnostics, the independent-prior kernel with nonnegative cost permits the sharper minorization `kappa=rho*Z_x`. Quadrature-based `Z_x` and estimated acceptance error produce plug-in diagnostics, not certified bounds. Exact refresh does not make the remaining approximate transitions exact.

### 13.3 Dimensionality and SB: explicitly deferred

Document a later progression to 2, 8, and 32 influential analytic path coefficients, followed by one frozen SB generator. Added coordinates must affect decoded commands; appended ignored Gaussian coordinates test only interface behavior.

Keep prior trajectory scale and problem difficulty interpretable as dimension grows. For an SB, expose all stochastic sampling innovations, keep the source chunk fixed within each chain, and count both generator and simulator work.

Do not implement or train that generator in this assignment. The report should explain what the current evidence suggests testing next.

## 14. Focused verification

Reuse the existing mathematical tests. Add focused tests only for new behavior or concrete risks:

- Dynamic numerical contexts reuse compiled kernels without retaining old context values.
- Exact inference skips the unused ratio network; cheap branches skip the oracle at runtime.
- Full-chunk execution and prefix/cache replay agree.
- Claimed cache-zero decoder identity holds; both coefficients influence intended commands.
- Context/evaluator changes invalidate affected cached costs and metrics. Ordinary new contexts within a supported conditional model reuse its frozen weights; they do not require retraining. Decoder, evaluator, feature-schema, or trained-family changes require an explicit checkpoint compatibility decision and appropriate retraining or recalibration.
- Initial and fallback validation accounting is correct; budgets cannot be exceeded.
- MH-rejected but evaluated candidates remain eligible for planning.
- Prefix budget curves match separately stopped prefix-compatible searches.
- CEM's declared budget/population semantics are implemented.
- Feasibility flags agree with actual physics after cost changes.
- Movement audit pairs are current-proposal pairs and use a separate RNG stream.
- Dataset splits and runtime inputs contain no audit-only information.
- Checkpointed resumption reproduces an uninterrupted small run.
- Quadrature and independent reference samples agree within stated numerical/statistical limitations.

Keep algebraic/deterministic unit tests fast. Put costly statistical checks in explicit audit commands, with seeds and tolerances justified. Run tiny end-to-end analytic and route checks, then the bounded scientific protocol. Do not repeatedly rerun unrelated expensive pilots.

## 15. Required deliverables

Choose a simple extension of the existing CLI. Provide actual working commands for development, frozen confirmation, deployment, audit, rendering, and resume. Commands must point to real configurations and artifacts.

Suggested additions:

- `configs/round2_development.yaml` and `configs/round2_confirmation.yaml`.
- `docs/ROUND2_PROTOCOL.md`, `docs/RUN_REPORT_ROUND2.md`, and updated usage documentation.
- An extension of the existing notebook or `notebooks/02_ai_plan_repair_benchmark.ipynb`.
- Resolved protocol/data manifests, checkpoints, per-trial JSON/CSV, summary grids, accounting ledgers, and replay data under the new run directory.

Save enough data to reproduce figures without training or simulation. Avoid saving every full trajectory from large grids. Choose replay scenes by prespecified IDs or a declared representative-selection rule; include failures and record the rule.

Required figures:

1. `execution_and_cache_controls`: prediction, execution, cache continuation, and grid-replanning progress.
2. `repair_support_and_target`: successful set, prior mass, target mass, and representative paths.
3. `proposal_learning_diagnostics`: surrogate versus exact movement, calibration, and exploration across checkpoints.
4. `repair_success_vs_evaluations`: the main figure, stratified by measured prior success and annotated with target success.
5. `repair_success_vs_time` and `best_cost_vs_evaluations`.
6. `analytic_exploration_and_first_passage`: per-chain behavior and target errors.
7. `closed_loop_outcomes_and_progress`: actual goal distance, completion/collision/timeout, and episode computation.
8. A standalone replay linking noise/coefficients, candidate commands, and actual physical execution.

Save static PNG/PDF figures and a lightweight HTML replay using the existing rendering approach. Reopening the notebook or moving a replay slider must not train or simulate implicitly.

## 16. Report structure and final completion check

The report must state:

1. What changed from the first round and why.
2. Which protocol version and target versions were used.
3. The exact experiments completed, with scene/search/training-seed counts.
4. What the physical controls establish and which decoder was selected.
5. Difficulty distributions, target success mass, support limitations, and grid sensitivity.
6. Baseline tuning and all model-selection work.
7. Per-training-seed and aggregate results, with uncertainty and failure breakdowns.
8. Whether learned and exact movement objectives agree during training.
9. Whether gains concern target exploration, first repair, selected plans, or physical completion.
10. Online and offline accounting, timing, memory/artifact footprint, and any partial runs.
11. What did not improve, what remains unresolved, and the next experiment justified by the evidence.
12. Exact reproduction/resumption commands.

Use an explicit experiment manifest with statuses such as `complete`, `failed_correctness`, `partial_resource_limit`, `conditional_not_triggered`, and `deferred`. An implemented configuration is not an executed result.

Completion requires working code, focused verification, real saved outputs for the bounded runs that fit the declared resources, and an honest report. Do not fabricate favorable results, import manuscript numbers, or hide failed trials. A resource-blocked confirmation remains partial even if its code is complete.

In the final handoff, lead with the main empirical answer, distinguish full confirmation from partial work, list the most useful figures/replays, and give exact commands. Avoid a long implementation chronology.

## References

The original prompt and local concept manuscript remain the method sources. Relevant background for the specified follow-ups:

- pCN: [Cotter et al., MCMC Methods for Functions](https://arxiv.org/abs/1202.0709).
- Delayed acceptance: [Banterle et al., Accelerating Metropolis-Hastings Algorithms by Delayed Acceptance](https://arxiv.org/abs/1503.00996).
- [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).

Use documentation matching the installed dependency versions. These references motivate the methods; they are not evidence of an Ai-Plan performance advantage.

