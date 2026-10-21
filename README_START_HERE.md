# Ai-Plan implementation handoff

This package is an implementation brief for Codex, **not an implemented repository** and not a set of experimental results.

## Start

1. Put these files in a new directory named `ai-planner`.
2. Open that directory with Codex.
3. Give Codex the following instruction:

> Read PROMPT.md completely and consult references/Ai_Plan_Concept_Manuscript.pdf. Implement the required ai-planner toy repository from scratch. Finish the analytic 2D sampler, learned proposal and acceptance training, exact-refresh diagnostics, and planar route-repair visualization. Run the focused tests and bounded end-to-end development experiments; save the real outputs and a run report. Prioritize mathematical correctness and usable visualizations, and take the coding/debugging time needed. Do not launch large sweeps or introduce a DSBM pretraining project in this first pass. Document all discretionary choices and any failed or inconclusive comparisons.

## What will be trained?

The first version trains two small JAX networks: a conditional proposal in the frozen decoder's random-input space, and an antisymmetric approximation to the Metropolis–Hastings log ratio. An exact-acceptance mode remains available throughout.

The initial decoder is analytic, fixed, and fully specified. It substitutes for the pretrained generator only to make the sampling experiment transparent and runnable without another training project. It is not called a pretrained DSBM model. A dimension-independent decoder interface is required so a real pretrained generator can replace it later.

## Why two small experiments?

The analytic two-/four-mode problem gives numerical reference densities and a separate finite-state audit of the equations. The route-repair problem gives the same random-input coordinates an interpretation as edits to a whole action chunk, whose physical consequences are evaluated by a simple deterministic simulator.

The important comparison is not acceptance rate alone. Inspect target-region masses, mode traversal, accepted movement, finite-budget correction quality, and actual cost/runtime accounting together.

## Source versus design

The method and the broad two-dimensional diagnostic are from the supplied Ai-Plan concept draft, especially §§4–5 and Appendix A. The compact Gaussian-CDF decoder, explicit bounded energies, JAX stack, and first-pass two-coordinate route decoder are implementation choices in PROMPT.md. They are not previously measured results or hidden implementation details from the draft.

All numerical results in the manuscript's experimental section are marked hypothetical. They are not targets to reproduce.

## Package contents

- `PROMPT.md`: self-contained implementation specification and experiment plan.
- `references/Ai_Plan_Concept_Manuscript.pdf`: the original eight-page concept draft.
- `README_START_HERE.md`: this handoff guide.
