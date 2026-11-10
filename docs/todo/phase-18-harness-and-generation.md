# Phase 18 — The harness judges the batched pass; generation starts from it: To-do

Working checklist for Phase 18 of the [implementation plan](../emberly-oxygen-implementation-plan-0.4.0.md) (second part of milestone M13). Tick items as they land; one task is one commit, tests first.

Proves: S-1 to S-4 and S-11 pass with the batched pass in every category, the four faults fail it, and generation, sampling and chat start from `prefill`. Satisfies FR-29, FR-40, FR-41, S-1 to S-5, S-8, S-11, HC-4.

Layout: `crates/oxygen/tests/`, `generate.rs`. Behavior: Specification §8, §11, §14, §17.

Dependencies: none new. Out of scope: any threshold change, any oracle or fixture change.

## 18.1 Record the guard ids (FR-29)

- [x] Run the greedy, seeded and chat guard tests before any change and record the ids.

## 18.2 The harness through `prefill` (FR-41)

- [x] Logits, intermediates and top-1 from `prefill` with `Logits::All` and capture; thresholds and references untouched; read the table.

## 18.3 Generation starts from `prefill` (FR-40, FR-29)

- [x] Tests first: first token and a short greedy run equal the `step` route on the tiny model.
- [x] Then `generate` and `generate_with`; rerun the guard ids; record each difference with its near-tie gap.

## 18.4 Batched against one-token (FR-41)

- [x] Both paths over the prompt set; report the largest differences; fail above the bound of 17.5.

## 18.5 Negative controls (HC-4)

(The new control was written in 18.2; see plan §5.)

- [x] Tests first: the harness fails on each of the four faults; no fault, all pass.

## 18.6 Logging (S-8, FR-23)

(Done in the same commit as 18.3; see plan §5.)

- [x] One `prefill` event, then steps; per layer once at trace; change the debug event count test; output identical at every verbosity in generation and in chat.

## 18.7 Full gate (HC-4)

- [x] `cargo test --no-fail-fast`, whole summary read; record the table, worst errors per category and the chat reported-only figures in §5.
