# Algorithm and invariants

The supplied concept manuscript, §§3–4 and Appendices A–B, is the method source.
Its empirical section is explicitly hypothetical. `PROMPT.md` supplies the
pilot's bounded analytic energies and implementation requirements.

For an immutable context and decoder, the noise-space target is

`p(z) ∝ φ(z) exp(-J(G(x,z))/β)`.

The decoder is a deterministic measurable map of all explicit Gaussian noise.
Neither an inverse decoder nor an action-density Jacobian is needed. A future
pretrained generator can implement `noise_dim`, `version`, and
`decode(context,z)`; the sampler and Gaussian density calculations sum over any
noise dimension. A six-dimensional smoke test exercises this interface.

## Frozen transitions

The Gaussian proposal exposes sampling and log density. Exact acceptance uses

`ell = logφ(z') - logφ(z) - (J' - J)/β + logq(z|z') - logq(z'|z)`.

Both proposal terms use the same parameters and context. The implementation
compares an independent `log(u)` with `min(0,ell)`. Proposal samples are never
clipped. Rejection repeats the current latent, candidate, and appropriate cache
metadata. pCN preserves the Gaussian prior, so its ratio reduces to the cost
difference; this cancellation is tested independently against SciPy.

The learned ratio is `d=g(x,z,z')-g(x,z',z)`. It receives no exact cost, label,
grid density, or route-success indicator. With `rho>0`, a state-independent
Bernoulli chooses exact independence MH. This is a homogeneous mixture, not
an every-k-th-step schedule.

The chain stores an exact cost and validity bit. Accepted learned moves
invalidate it. Exact refresh computes both current and proposed costs when the
current cost is unknown; a rejection after this evaluation can retain the newly
known current cost. Rebinding to a different target hash invalidates cost and
redecodes the candidate. Target hashes include context, decoder, cost, temperature,
simulation settings and continuation definitions.

Scalar `lax.scan` and scalar `lax.cond` keep oracle branches conditional at
runtime. Independent chains reuse the same compiled scalar kernel; the mixed
kernel is deliberately not vmapped across chains. A callback-based test counts
actual runtime oracle executions, in addition to the functional counters.

## Alternating training

Each round freezes target and decoder. Independent rejection draws from the
prior, accepted with probability `exp(-J/β)`, supply exact independent target
states. All rejected draws and batch surplus are charged. This replaces the
potentially biased finite-chain bootstrap in the training loop; exact MH chains
are compared separately with those independent samples.

The ratio model minimizes raw squared exact-log-ratio error. Source costs are
cached; proposal costs and proposal-dependent labels are refreshed around every
short proposal-update block. The proposal reward is frozen learned acceptance
times `1-exp(-||features'-features||²/(2l²))`. Features are decoded coordinates
or normalized full command chunks, never unused latent coordinates.

The proposal gradient is a score-function estimator. Destinations are stopped
before `logq`; rewards and the past-batch scalar baseline are stopped as well.
Fresh destinations are drawn for every update. There is no derivative through
the simulator or an accept/reject Boolean. Replacing the frozen learned
acceptance with exact MH would introduce additional proposal derivatives and
is not done.

Parameters are frozen again for evaluation. Checkpoints include both networks,
both optimizers, random key, baseline, counters, context normalization, versions
and update counts. A separate folded-in calibration random stream leaves
optimizer resume reproducible.

## Bounds and numerical truth

For `F=pq`, compute `abar=E_F[a]`, `L_bal=E_F[a(d-ell)]`,
`L_ratio=E_F[(d-ell)²]`, and `epsilon_a=E_F[|a-a*|]`.
The one-step bounds are

`TV(pT,p) ≤ sqrt(abar L_bal / 2)` and
`TV(pT,p) ≤ epsilon_a ≤ sqrt(L_ratio)`.

The finite-state audit explicitly normalizes proposal rows and recomputes
**discrete** Hastings ratios. It checks accepted-flow KL, detailed balance,
one-step error, stationary distributions, and finite-step errors. It is a
separate discrete model, not the exact transition matrix of a continuous
sampler. Raw signed Monte Carlo balance estimates and their standard errors
are retained; they are not silently converted into certificates.

For exact refresh, `log(kappa)=log(rho)-Jmax/β`. The frozen-kernel bound is
`(1-rho) epsilon_a/kappa`; finite-step error also includes the geometric initial
law term. Bounds above one are explicitly vacuous. The route cost's collision
penalty makes the conservative minorization particularly weak.

Compact midpoint grids integrate in `u=2Φ(z)-1`. For the analytic CDF decoder,
this is also the decoded descriptor. For the identity-decoder sensitivity
experiment it is only an integration coordinate; the decoded output is `z`.
Two grid resolutions quantify quadrature sensitivity. Region masses use a
fixed Voronoi partition (or coefficient sign for routes), not mixture labels.
Histogram TV is a finite-bin, sample-dependent discrepancy.

## Three clocks

Physical substeps advance position and velocity. MCMC steps revise the random
input for an entire new candidate from the same context. There is no learned SB
revision-time trajectory in this analytic pilot. Executing a selected prefix
creates a new snapshot and a new target; chains from successive replans are not
combined as samples from a stationary distribution.
