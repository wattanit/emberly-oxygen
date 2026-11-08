# Phase 12 — Sampler and the S-9 gate: To-do

Working checklist for Phase 12 of the [implementation plan](../emberly-oxygen-implementation-plan-0.2.0.md) (the second part of milestone M9). Tick items as they land; one task is one commit, tests first.

Proves: the controls and generator are correct, S-9 judges the engine's distribution, and it can fail. Satisfies FR-26, FR-27, FR-30, FR-5, S-9, HC-1, HC-4.

Layout: `crates/oxygen/src/rng.rs`, `sampling.rs`, `fault.rs` (two new faults), harness additions under `crates/oxygen/tests/common/`. Behavior: Specification §8 and §11.

Dependencies: none new. Out of scope: wiring into `generate`, the CLI (Phase 13); any change to the forward pass.

## Decision points for the owner

1. **The rule for S-9's tolerance, written before the numbers (applies to 12.8).** *Recommendation:* the rule used for S-1: the tolerance is the smallest round decimal that is at least twice the worst measured error, so the worst case uses at most half of it.
2. **What counts as a crossable cutoff (applies to 12.5, 12.8).** *Recommendation:* a setting whose cutoff margin is smaller than twice the engine's worst measured logit error (3.5e-4 in 0.1.2) is reported and not judged, as a near-tie is for S-3. If any setting is exempted, say so in the release note.

## 12.1 Generator (HC-1, FR-28)

- [x] Compile the authors' C reference in the scratchpad, record outputs for known seeds. Tests first, then `rng.rs`.

## 12.2 Settings and validation (FR-30)

- [x] Tests first: each invalid value rejected naming the option; boundary values accepted. Then `Sampling`.

## 12.3 The controls (FR-26, FR-27)

- [x] Tests first on hand-computed cases (penalty sign rule, tie at k, top-p boundary, one token kept). Then the controls.

## 12.4 Distribution function (FR-26)

- [x] Tests first, including that swapping the order changes a combined result. Then the function.

## 12.5 S-9 in the harness (FR-5, S-9, S-5)

- [x] Tests first with a hand-made stored distribution. Then the check, the table column and the margin reporting.

## 12.6 Negative controls (HC-4, S-9)

- [x] Add the two faults; a test asserts S-9 fails on each on the combined setting.

## 12.7 Measure (S-9)

- [x] Per category and setting, the worst error and margins; record in the plan's section 5.

## 12.8 Stop: owner decides

- [ ] The owner decides the tolerance and the boundary rule from the table and the decision points above.

**Exit check:** as in the plan's section 3.
