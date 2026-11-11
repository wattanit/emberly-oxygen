# Phase 24 — `prefill` uses it; the gate and the bounds: To-do

Working checklist for Phase 24 of the [implementation plan](../emberly-oxygen-implementation-plan-0.4.2.md) (second part of milestone M17). Tick items as they land; one task is one commit, tests first.

Proves: the harness passes unchanged in its thresholds through the blocked pass, the drift from 0.4.1 and from the one-token path is measured and bounded, and `UnfinishedTile` fails the harness. Satisfies FR-46, FR-47, FR-41, FR-43, S-1 to S-6, HC-4.

Layout: `crates/oxygen/src/forward.rs`, `fault.rs`, `tests/`. Behavior: Specification §4, §7, §11, §17.

Dependencies: none new. Out of scope: the decode step, the attention, `unsafe`.

## 24.1 Record the 0.4.1 ids (FR-47, FR-29)

- [ ] The guard tests and the 30-run id listing on the 0.4.1 build, kept in the scratchpad.

## 24.2 `Fault::UnfinishedTile` (HC-4, FR-46)

- [ ] Negative control first: the harness fails on a prompt whose length is not a multiple of the tile's input width; with no fault nothing changes.
- [ ] Then the fault.

## 24.3 `prefill` uses the blocked product (FR-40, FR-41, FR-46)

- [ ] Tests re-stated for the new arithmetic (tolerances for `prefill` against `step`; causal and one-token tests).
- [ ] Then the projections call `matmul_blocked_pooled`; the unembedding of `Logits::Last` and the decode step keep `matvec`.

## 24.4 Measure the drift and set the bounds (FR-47, FR-41)

- [ ] Blocked against unblocked, and blocked against one-token, per prompt; record in §5; set the two bounds with the owner; move the open item to §6.

## 24.5 The tests of Specification §11 and the gate (FR-47, FR-43, S-1 to S-6)

- [ ] Blocked against unblocked, batched against one-token, thread-count independence exactly 0; a faulted run over the bound.
- [ ] Correctness test: thresholds unchanged, all five faults fail; the guard of 24.1 rerun, each id difference recorded with its near-tie gap.

## 24.6 The full gate

- [ ] `cargo test --no-fail-fast`, whole summary read; harness table recorded in §5.
