# Emberly Oxygen — Implementation Plan for Release 0.1.1

Nov 7, 2026 · @Wattanit

**Target release:** 0.1.1\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.2 (upstream, approved), Emberly Oxygen — Technical Specification 0.2 (upstream, draft when this plan was written)

This is a living working document, not part of the SFD suite: it has no document version or status and is edited in place as work proceeds. It expands the Technical Specification's milestone M7 into two phases with ordered tasks and exit checks. It owns sequencing and task breakdown only; WHAT stays in the Requirements and HOW stays in the Specification, and every task cites the IDs it satisfies. A discovery that changes HOW is recorded in §5 and absorbed by a Specification version bump, never decided here. The foundation versions above are the as-built pin for this release: if either bumps while the release is in progress, the pin is updated and the change recorded in §5. The previous release's plan is [0.1.0](emberly-oxygen-implementation-plan-0.1.0.md); its §8 rows are the figures this release is compared with.

Phases are numbered on from 0.1.0's (Phases 1 to 6), so their to-do files in `docs/todo/` keep unique names.

## 1. Working rules

All of 0.1.0's rules hold. The ones this release leans on, and what it adds:

- **Phase order is fixed.** Phase 8 does not start until Phase 7's exit check passes, because a speed number from an engine that has not passed the gate is not a result (HC-4).
- **Tests before engine code.** The kernel's tests are written first and the order-pinning test (task 7.2) is seen to fail against the 0.1.0 kernel, so the new tests are known to be able to fail.
- **One task, one commit.** Each task ends with a passing `cargo test --no-fail-fast` whose whole summary is read, and a commit.
- **Every test names the HC, FR or S item it verifies** (Specification §17).
- **One technique only.** The change is the matrix-vector product's partial sums (FR-25). No threads, no `unsafe`, no intrinsics, no tiling, no change to any other kernel (Specification §4). If safe code does not give the expected speed-up, the task stops and the issue goes to the owner; it is not worked around.
- **No dependency outside Specification §15.** This release adds none.
- **Work happens on `release/0.1.1`.** Nothing is pushed, tagged or merged to `develop` or `main` without the owner's explicit order at that moment (standing rule, 2026-11-06).
- **The 0.1.0 loop is kept as a reference,** under the `bench-kernels` feature (task 7.1), so tests and the benchmark can always compare the new kernel with the engine's own history.

## 2. Phase 7 — Kernel (first half of M7)

**Proves:** the partial-sum kernel is correct: it passes the unchanged harness in every category and leaves the negative controls failing. **Satisfies:** FR-25, HC-1, HC-4, S-1 to S-6, S-8.

| # | Task | Cites |
| --- | --- | --- |
| 7.1 | Keep 0.1.0's index-order loop as `kernels::matvec_index_order`, available only with the `bench-kernels` feature. The engine's own `matvec` is unchanged in this task. Existing matrix-vector tests keep passing against it. | FR-25, HC-2, Spec §12, §17 |
| 7.2 | Kernel tests first, against a not-yet-written kernel (`kernels::matvec_lanes`, generic over the number of sums so Phase 8 can sweep 4, 8 and 16). They cover: agreement with the index-order reference and with `matrixmultiply` on seeded random inputs at the model's five shapes, within an absolute tolerance far below S-1; every length from 0 to 40, so lengths below the number of sums (the tail) and not divisible by it are covered; exact equality on small-integer inputs, where every sum is exact; two runs bit-identical (S-6); and an **order-pinning test**: an input built so that index-order and partial-sum addition give different last bits (large values cancelling small ones), checked against the value hand-computed from Spec §4's combining order. The pinning test fails when run against the 0.1.0 loop. | FR-25, S-6, Spec §4, §17 |
| 7.3 | Implement `matvec_lanes` in safe Rust per Spec §4: independent partial sums over `chunks_exact`, one fixed combining order, the tail added in index order. The engine's `matvec` becomes `matvec_lanes` at 8 sums (initial, Spec §4). Callers and the forward pass are untouched. The combining order chosen is written in the function's doc comment and matches the pinning test. | FR-25, HC-1, Spec §4 |
| 7.4 | Check that the compiler vectorized it: emit the release assembly of the kernel and look for vector multiply-add instructions in its inner loop; record what was found. **If it did not vectorize, stop and raise it with the owner** (Spec §4: no `unsafe` as a workaround). Also confirm by `grep` that the engine contains no `unsafe` and no new dependency. A quick `--stats` run on the short prompt gives a first speed estimate, not recorded as a result. | FR-25, HC-1, Spec §4, §15 |
| 7.5 | The full gate: `cargo test --no-fail-fast`, whole summary read. Record per category the worst logit and intermediate errors next to 0.1.0's (§4 below). Confirm the three negative controls still fail the harness, and that run-to-run reproducibility (S-6) and output at every verbosity level (S-8) still hold. A test that asserted a bit-exact 0.1.0 value and now fails for that reason is not loosened silently: the failure and the reason go to the owner. | HC-4, S-1 to S-6, S-8, Spec §11, §17 |

**Exit check:** the harness passes every prompt in every category with the worst errors recorded and, if they moved noticeably from 0.1.0's, explained; all three negative controls fail as before; the kernel tests (including the order pin) pass; the vectorization evidence is recorded; there is no `unsafe` and no new dependency in the engine; the owner has reviewed the tests and the evidence.

## 3. Phase 8 — Measurement and release (second half of M7)

**Proves:** the speed-up is real, measured against 0.1.0, and costs nothing in correctness. **Satisfies:** FR-8, FR-9, FR-25, S-7, HC-2, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 8.1 | Extend the benchmark's kernel group (Spec §12): at each of the five shapes it times `matrixmultiply`, the 0.1.0 index-order loop and partial-sum widths 4, 8 and 16, each row naming its variant; the engine's own kernel keeps its row name so the series stays comparable. It first checks that all variants agree. Workloads and procedure stay as in 0.1.0. | FR-25, HC-2, FR-9, Spec §12 |
| 8.2 | Width sweep: run the kernel group as a dry run (nothing recorded) and choose the width by the rule fixed in decision point 1 of the to-do list, written before the numbers are seen. Set the engine's width to the winner and rerun the full gate (task 7.5). A width other than 8 is a change to Spec §4's initial value, recorded in §5 for a bump. | FR-25, S-1 to S-5, Spec §4 |
| 8.3 | Prepare the measurement: bump the workspace version to 0.1.1 so the record carries it, and confirm a clean tree. | FR-9, Spec §16 |
| 8.4 | Run `cargo test --no-fail-fast`, then `cargo bench` straight after on the benchmark machine at the default log level, as in 0.1.0 (Spec §16; the warm-machine bias is the same on both sides of the comparison). A failing test run stops here. | HC-4, S-7, S-8, Spec §16 |
| 8.5 | Record the comparison with 0.1.0 in §4 below: workloads (time to first token, prompt and generation tokens/s, peak memory) and kernel shapes. Include what the speed-up is worth for a step and how much of a step the matrix-vector product is now (the share of everything else has grown by the same factor). | FR-8, FR-25, Requirements §12 |
| 8.6 | Release note `docs/releases/0.1.1.md`: what changed, why it was expected to help, what was measured against 0.1.0, what was learned, and negative results (a width that did not help, a gap between expectation and result, anything that did not vectorize or stopped scaling). README status line and benchmark section updated to match. | Requirements §12 |
| 8.7 | Owner's review of the rows and the note. Then **stop and ask** before merging or tagging `v0.1.1`: neither happens without the owner's explicit order. | Spec §16 |

**Exit check:** `results/benchmarks.jsonl` holds the 0.1.1 rows from a clean tree at the measured commit, every row carrying its conditions; the comparison with 0.1.0 is recorded; the release note exists and is accurate against the record; the owner has reviewed; the merge and tag are done only on the owner's order.

## 4. Measurements to record

Filled in during the phases. Nothing here is tuned; thresholds change only by a foundation bump.

- Worst logit and intermediate error per category (task 7.5), next to 0.1.0's: english 1.81e-4 and 4.78e-6, thai 2.36e-4 and 5.38e-6, code 1.97e-4 and 5.91e-6, utf8-split 2.16e-4 and 4.59e-6, long 1.81e-4 and 5.21e-6 (plan 0.1.0 §8, Apple M5; the M4 Max reproduced the worst values).
- Vectorization evidence (task 7.4).
- Run-to-run logit difference (task 7.5) for S-6, expected bit-identical.
- Width sweep (task 8.2): GFLOP/s per shape for widths 4, 8 and 16, the index-order loop and `matrixmultiply`.
- Benchmark comparison (task 8.5) against these 0.1.0 rows (clean tree, commit `0b4a5f3`, M4 Max, median of 5): short 1.682 s to first token, 19.03 prompt and 18.97 generation tokens/s; long 26.493 s, 19.33 and 18.48; peak memory 811.6 MB.

## 5. Upstream feedback

Recorded here and absorbed by the upstream owner at the next bump; the upstream documents are not edited from this one. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Before Phase 7 | Specification 0.2 status | The Specification was `draft` when this plan was written and pinned; the owner approved its content and the status is the owner's to set. The pin stands, and is updated here if the status or version changes. |

## 6. Open Questions

- **Which partial-sum width the engine uses (Specification §19).** Resolver: task 8.2 sweep, by the rule set before measuring.
- **Whether the engine's non-matrix-vector operations matter after the speed-up.** In 0.1.0 they were 1.4% of a step; a several-fold faster kernel makes them a larger share. Resolver: task 8.5's figure; any change is a later release's decision.
