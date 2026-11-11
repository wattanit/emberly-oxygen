# Emberly Oxygen — Implementation Plan for Release 0.4.2

Nov 12, 2026 · @Wattanit

**Target release:** 0.4.2\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.10 (upstream; approved on 2026-11-12), Emberly Oxygen — Technical Specification 0.8 (upstream; approved on 2026-11-12)

This is a living working document, not part of the SFD suite: it has no document version or status and is edited in place as work proceeds. It expands the Technical Specification's milestones M17 and M18 into three phases with ordered tasks and exit checks. It owns sequencing and task breakdown only; WHAT stays in the Requirements and HOW stays in the Specification, and every task cites the IDs it satisfies.

Phases are numbered on from 0.4.1's (Phases 20 to 22), so their to-do files in `docs/todo/` keep unique names.

**What this release is.** One technique: register blocking of the batched matrix-matrix product (FR-46 to FR-48). A tile of output elements is held in registers, each product added by a fused multiply-add, in safe code. It is the first release that gives up bit-identity with the previous release on purpose (FR-47): the oracle judges it, a bounded drift check against the retained 0.4.1 product guards it, and the output stays independent of the thread count. The decode step, the attention, sampling, chat and the tokenizer do not change.

## 1. Working rules

All of 0.4.1's rules hold. The ones this release leans on, and what it adds:

- **Phase order is fixed.** Phase 24 does not start until Phase 23's exit check passes, and Phase 25 does not start until Phase 24's does.
- **What is not touched.** `matvec`, `dot_lanes`, the decode step, the attention, the pool's design, the other `ops.rs` functions, the tokenizer, `sampling.rs`, `chat.rs` and the thresholds of S-1 to S-4, S-9 and S-11 stay as they are. `matmul` (the 0.4.1 loop) stays in the library unchanged as the baseline. If a task seems to need a change to one of them, it stops and the issue goes to the owner.
- **One technique only.** Register blocking with fused multiply-add. No `unsafe`, no SIMD intrinsics, no packed copy of the weights, no lower-precision weights, no parallelism crate (HC-1, FR-46; owner, 2026-11-12: `unsafe` only as a last resort and only by the owner's choice).
- **The kernel's speed is fragile.** In the profile only some tile shapes vectorize (§5). Every change to the kernel is followed by its rate at the three shapes against the numbers in §5, and the rate row in the benchmark record is what shows a regression later.
- **Tests before code.** Each new unit (blocked kernel, its pooled form, the fault, the benchmark support code) gets its tests first, seen to fail before the code exists. `UnfinishedTile` gets its negative control before the blocked pass is trusted.
- **Equality is exact where the sum order is fixed, and bounded where it moved.** The blocked kernel is compared with the chunked fused sum written out in the test, exactly (0 difference), at every shape and thread count. Against `matmul` and against the one-token path the difference is measured, and the bounds are set from it with the owner (Specification §19).
- **Guard the ids before changing anything.** Before `prefill` changes (task 24.1), the greedy and seeded ids of the 30 runs 0.4.0 recorded are recorded again for 0.4.1. After the change every difference is recorded with its near-tie gap; a difference at a position with no near-tie is a defect, not a finding.
- **0.4.2 is compared with 0.4.1 directly** (FR-48), at the same lengths and thread counts.
- **One task, one commit**, with a body paragraph and **no attribution line** (the owner's standing rule); check `git log -1 --format=%B` afterwards. Docs-only commits carry the `[docs]` prefix, and docs are committed once, when approved, never as drafts.
- **Every test names the HC, FR or S item it verifies.** No new Rust or Python dependency (Specification §15).
- **Throwaway scripts stay out of the repository**; their results go in §5 with how they were produced.
- **Approved documents are not edited without the owner's permission.** What the work finds goes in §6.
- **The SFD is settled before any merge** (task 25.6).
- **Work happens on `release/0.4.2`.** The owner pushes; nothing is pushed, tagged or merged without the owner's explicit order at that moment.
- **Benchmarks run under `caffeinate`, lid open.**

## 2. Phase 23 — The blocked kernel (first part of M17)

**Proves:** a safe blocked kernel computes exactly the chunked fused sum at every shape, edge and thread count, and it is as fast in the engine as in the profile. **Satisfies:** FR-46, HC-1, HC-2.

| # | Task | Cites |
| --- | --- | --- |
| 23.1 | Tests first for `matmul_blocked(W, X, Y)` and the input packing: equal, exactly, to the chunked fused sum written out in the test, at the model's shapes; at row and input counts that do not divide the tile, are smaller than it, or are 1; at lengths below and above the vector width; against `matrixmultiply` within a tolerance far below S-1; a sum with 16 partial sums instead must differ in some bit (the test can see a changed order); a tile-edge test marks every output element and checks none is left unwritten. | FR-46, HC-2, Spec §4, §17 |
| 23.2 | `ops.rs`: `matmul_blocked` and the packing of the input into `[k][NR]` panels, initial tile 4 × 16, `f32::mul_add`, leftover rows and inputs summed one element at a time in the same order. Safe code, weights read in place. | FR-46, Spec §4 |
| 23.3 | Measure in the library (a scratchpad program through the `bench-kernels` feature, not committed): the rate of `matmul`, `matmul_blocked` and sgemm at the three shapes with 128 inputs on one thread, and a sweep of tile shapes (4 × 16, 4 × 8, 8 × 8, 8 × 4 and the neighbours that the profile found cliffs at) in the real code. Record in §5. If 4 × 16 is not clearly the fastest, or the rate is far under the profile's 118 to 120, stop and take it to the owner (Spec §19). | FR-46, FR-48, Spec §4, §19 |
| 23.4 | `matmul_blocked_pooled`: the pool divides the weight rows into ranges of whole tile multiples where the size allows, the packed input shared as the other inputs are, small products staying on the calling thread (the existing threshold). Tests first: exactly equal to the single-threaded blocked product at counts 1, 2, 3, 7 and 16, at the model's shapes and at odd sizes, so neither the tile nor the row division changes a bit. | FR-46, FR-43, Spec §4 |

**Exit check:** the blocked kernel equals the chunked fused sum exactly at every tested shape and count; its rate in the library is recorded and the tile is decided with the owner; `cargo test` passes with only the tests this phase adds.

## 3. Phase 24 — `prefill` uses it; the gate and the bounds (second part of M17)

**Proves:** the harness passes unchanged in its thresholds through the blocked pass, the drift from 0.4.1 and from the one-token path is measured and bounded, and `UnfinishedTile` fails the harness. **Satisfies:** FR-46, FR-47, FR-41, FR-43, S-1 to S-6, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 24.1 | Before any change, run the guard tests that pin ids and the 30-run id listing (greedy and seeded generation on the 12 prompts, greedy and seeded chat over three turns) on the 0.4.1 build and keep the listing in the scratchpad; it is the 0.4.1 reference for task 24.5. | FR-47, FR-29 |
| 24.2 | The negative control first: `Fault::UnfinishedTile` leaves the last, partial tile of every blocked product unwritten. Tests first: with it the harness fails on a prompt whose length is not a multiple of the tile's input width; with no fault nothing changes. Then the fault. | HC-4, FR-46, Spec §17 |
| 24.3 | `prefill` calls the blocked product for its projections (the final unembedding of `Logits::Last` and the decode step keep `matvec`). Tests first, adjusted where the arithmetic now differs: the tiny-model tests that compared `prefill` with `step` at 1e-5 keep a tolerance and say so; the causal test (a changed later token leaves earlier logits bit-for-bit equal) and the one-token-prompt test are re-stated for the new arithmetic. | FR-40, FR-41, FR-46, Spec §7, §17 |
| 24.4 | Measure on the real model and prompts (§5): the largest logit and intermediate difference between the blocked and the unblocked `prefill` per prompt, and between the blocked `prefill` and the one-token path. Set the two bounds of Specification §11 from the worst by the rule used for S-1, with the owner, and move the open item of Specification §19 to the feedback record (§6). | FR-47, FR-41, Spec §11, §19 |
| 24.5 | The tests of Specification §11: blocked against unblocked (retained `matmul`) with the bound of 24.4, the batched-against-one-token bound reset, and the thread-count independence test through the blocked pass at 1, 2, 3, 7 and the default, exact. A faulted blocked product is far over the bound. Run the correctness test: S-1 to S-4 and S-11 pass with the same thresholds, and the four faults of 0.4.1 and `UnfinishedTile` fail the harness. Rerun the guard of 24.1: every id equal, or each difference recorded in §5 with its near-tie gap. | FR-47, FR-43, S-1 to S-6, HC-4 |
| 24.6 | The full gate: `cargo test --no-fail-fast`, whole summary read; the harness table recorded in §5 against 0.4.1's. | S-1 to S-6, S-11 |

**Exit check:** S-1 to S-4 pass through the blocked pass; every difference from the previous kernel is measured and bounded; thread-count independence is exactly 0; all five faults (four of 0.4.1's and `UnfinishedTile`) fail the harness; no existing test changed except where §6 says the arithmetic moved.

## 4. Phase 25 — The kernel rows, the sweeps and the release (M18)

**Proves:** what the blocked product is worth against 0.4.1 at every length and count, and that decode did not change. **Satisfies:** FR-48, FR-8, FR-9, S-7, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 25.1 | The batched kernel rows in the benchmark's kernel group: `matmul`, `matmul_blocked` and `matrixmultiply`'s sgemm at the three shapes with 128 inputs on one thread, GFLOP/s, with `"inputs": 128` and `"threads": 1`; the blocked kernel checked against the chunked fused sum before it is timed. Tests first for the support code. | FR-48, HC-2, FR-9, Spec §12 |
| 25.2 | Bump the workspace version to 0.4.2 and confirm a clean tree. The default thread count is re-chosen by the sweep's rule after the run of 25.3 (Specification §12), and changed only if the rule picks another count. | FR-44, Spec §12, §16 |
| 25.3 | Run `cargo test --no-fail-fast`, then `cargo bench` straight after, at the default log level, under `caffeinate`. A failing test run stops here. | FR-8, FR-9, S-7 |
| 25.4 | Record in §5: the kernel rows; the prefill sweep and the two workloads against 0.4.1's rows (the figures are in plan 0.4.1 §5); the thread sweep against 0.4.1's at every count; decode shown unchanged; peak memory; where the speed-up stops and why, against the 3× the kernel showed alone (attention is about 20% of a long prefill and is untouched; the unthreaded operations). Say what it is worth, and a negative result as one. | FR-48, FR-9, S-7 |
| 25.5 | Release note `docs/releases/0.4.2.md` and the README (status, version note, the equality rule). | Requirements §12 |
| 25.6 | Settle the SFD before the merge question: §6's rows written into the documents as the owner permits, the sfd-standard compliance review over both, every pin checked here and in the release note, and the owner's approval of the statuses. | G-14, G-16, G-24, G-25 |
| 25.7 | Owner's review of the rows and the note. Then **stop and ask** before merging or tagging `v0.4.2`. | Spec §16 |

**Exit check:** the 0.4.2 rows, including the kernel rows and both sweeps, are in `results/benchmarks.jsonl` from a clean tree; the release note exists; both foundation documents are approved and pinned.

## 5. Measurements to record

Machine: the benchmark machine (Apple M4 Max). Baseline: 452 tests at 0.4.1.

**The profile of 2026-11-12** (a scratchpad program, `blocking`, random data, one core, 128 input rows, release build; not in the repository). GFLOP/s at the model's three prompt shapes (1536 × 576, 576 × 1536, 576 × 576); the shapes agree to within 3%, so one row is given:

| Kernel | GFLOP/s | Output against the 0.4.1 loop |
| --- | --- | --- |
| Current loop (weight row outer, 16 partial sums) | 38.9 to 39.9 | identical |
| Tile 2 × 2 keeping 16 sums per element | 47.7 to 50.7 | identical (checked) |
| Tiles 1 × 2, 1 × 4, 2 × 4, 4 × 2 keeping 16 sums | 20.6 to 34.6 | identical |
| k-outer tile, multiply then add: 4 × 8 or 4 × 16 (inputs packed) | 65 to 69 | max difference 1.0e-5 to 2.1e-5 |
| the same with 8 × 8, 8 × 4, 12 × 4 | 54 to 65 | same |
| the same with 6 × 8 | 14 to 15 | same |
| **k-outer tile with `mul_add`: 4 × 16 (inputs packed)** | **118.3 to 120.2** | max difference 8.6e-6 to 2.2e-5 |
| the same with 4 × 8, 8 × 8, 6 × 8, 12 × 4 | 16 to 18 | same |
| the same with 8 × 4, 8 × 16 | 27 to 33 | same |
| `mul_add`, weights also packed: 4 × 16, 8 × 8 | 114 to 119 | same |
| `mul_add`, weights also packed: 4 × 8 | 70 to 71 | same |
| `matrixmultiply` sgemm | 108.0 to 115 | (reference) |

The numbers it rests on: one core's multiply-add peak is 111 GFLOP/s (plan 0.4.1 §5), so the fused 4 × 16 kernel is at the peak; the unfused kernel runs two instructions per product and tops out near 65. Rust does not fuse a multiply and an add by itself, so `mul_add` is what makes the difference. Vectorization is fragile: most other tile shapes with `mul_add` run at 16 to 33 GFLOP/s, so the tile is chosen by measurement and pinned. The numbers are one core with the data in cache: at 8 threads, with attention and the unthreaded operations untouched, a prompt will gain less than 3×.

Filled in as the phases land.

- **Tasks 23.1, 23.2 and 23.4 (done, one commit).** `matmul_blocked` and `matmul_blocked_pooled` are in (4 × 16 tile, `mul_add`, inputs packed `[k][16]`, leftover rows one at a time, leftover inputs singly), not yet used by `prefill`. `cargo test --no-fail-fast`: **456 passed, 0 failed** (452 at 0.4.1). The blocked product equals an index-order fused sum written in the test exactly at 12 shapes (the model's four, and odd ones from 1 × 37 to 100 × 33, 8 × 1 and 4 × 16) by 11 input counts (1 to 67, around the tile's width of 16); every output element is written; it agrees with `matrixmultiply` within 1e-4 and differs from the 16-partial-sum product in some bit, by under 1e-4; and the pooled form equals it exactly at 1, 2, 3, 7 and 16 threads. The tests were seen failing against stubs before the code.
- **Task 23.3 (measured; no stop needed).** A scratchpad program (`blockrate`, release build, not in the repository) timed `matmul`, `matmul_blocked` and `matrixmultiply`'s sgemm through the `bench-kernels` module at the three shapes with 128 inputs on one thread (best of nine, three calls each), the tile constants edited and the library rebuilt for each shape. The rate includes the input packing, the transposed output copy and the allocation, which the profile's kernel did not. GFLOP/s at 1536 × 576, 576 × 1536, 576 × 576:

  | Tile (rows × inputs) | 1536 × 576 | 576 × 1536 | 576 × 576 |
  | --- | --- | --- | --- |
  | `matmul` (0.4.1) | 36.2 | 38.1 | 36.4 |
  | **4 × 16** | **93.5** | **110.9** | **107.4** |
  | 6 × 8 | 77.5 | 89.4 | 87.0 |
  | 8 × 16 | 74.6 | 82.7 | 83.7 |
  | 8 × 8 | 70.4 | 80.0 | 79.6 |
  | 4 × 32 | 65.8 | 73.3 | 72.2 |
  | 4 × 8 | 58.3 | 64.2 | 64.2 |
  | 8 × 4 | 57.9 | 63.9 | 63.5 |
  | 12 × 4 | 55.5 | 63.6 | 59.2 |
  | sgemm | 106.5 | 106.6 | 106.3 |

  4 × 16 is the fastest at every shape: 2.6× to 2.9× the 0.4.1 loop, and 88% to 104% of sgemm. It is 79% to 93% of the profile's 118 to 120; the difference is the packing, the transposed copy and the allocation (at 1536 × 576 the output is the largest). Those are candidates for a follow-up inside this release if prefill shows them (not measured). The ordering of the tiles agrees with the profile, and the cliffs of the profile are in the real code too, so the tile stays pinned at 4 × 16. Decision for the tile, by the open item of Specification §19: 4 × 16, as the measurement is not in doubt; the owner can overrule.
- **Task 24.1 (done, no commit).** The `chatguard` scratch program recorded the ids of 30 runs on the 0.4.1 build (greedy and seeded generation on the 12 prompts, greedy and seeded chat over three turns): identical to the listing recorded for 0.4.0 (`diff` empty).
- **Tasks 24.2 and 24.3 (done).** `Fault::UnfinishedTile` (a flag on the pool, read by the pooled blocked product, which zeroes the outputs of the inputs outside the whole 16-input panels; only with `negative-controls`) was added with its control first: the harness test failed while `prefill` still used the old product (the fault could not act), and passes now that `prefill` calls `matmul_blocked_pooled` for every projection and for the unembedding of `Logits::All` (the unembedding of `Logits::Last` and the decode step keep `matvec`). `Model::use_unblocked_prefill` (only with `bench-kernels`) runs `prefill` with the retained 0.4.1 product, for the drift check; `matmul` and `matmul_pooled` exist only for tests and that feature, so a normal build has no dead code. The tiny-model test that `Last` equals the last row of `All` bit for bit now allows 1e-5 (the two use different products), the one test of the 0.4.1 arithmetic that changed.
- **Task 24.4, the first measurement and the decision on accuracy.** With the single index-order fused sum per element, the harness passed but the worst error against the 64-bit reference rose from 3.50e-4 to **6.51e-4** in logits (65% of S-1; utf8-split) and from 9.97e-6 to **4.53e-5** in relative intermediates (45% of S-2); the blocked-against-unblocked logit difference reached **1.00e-3**. The cause is one long sequential sum (up to 1536 products) accumulating more rounding than 16 partial sums. A scratchpad variant that sums chunks of 64 products and adds the chunk sums in order cost about 3% of the kernel's speed (114 against 118 GFLOP/s at 4 × 16) and cut its difference from the 16-sum reference by 3.6×. **Owner's decision (2026-11-12): chunk the sums, KC = 64.**
- **Task 24.4, with chunking (measured).** Kernel rate in the library (best of nine; the same program as 23.3): 96.5, 113.8 and 108.3 GFLOP/s at 1536 × 576, 576 × 1536 and 576 × 576, against `matmul` 38.2, 40.0, 39.7 and sgemm 116.4, 109.3, 109.3: no loss from chunking in the engine's kernel. Error against the 64-bit reference, worst by category (logit, relative intermediate): english 2.07e-4, 4.34e-6; thai 1.32e-4, 5.62e-6; code 2.72e-4, 6.77e-6; utf8-split 1.98e-4, 4.96e-6; long 2.72e-4, 7.98e-6. **Overall worst logit error 2.72e-4 (27% of S-1) and worst relative intermediate error 7.98e-6 (8% of S-2), both below 0.4.1's 3.50e-4 and 9.97e-6.** Differences between runs of the engine (scratchpad `drift`, then the tests): blocked against unblocked `prefill`, largest logit difference per prompt 7.2e-5 to **3.53e-4** (english-medium); blocked `prefill` against one token at a time, the same figures (the one-token path is exactly the unblocked one: 0 difference between those, as before); the largest intermediate differences are 2e-3 to 2.9e-2 absolute (the residual stream carries large values; they are reported and not bounded, as before).
- **Bounds (Specification §19, set with the owner).** Both tests of Specification §11 use **1e-3** in logits, S-1's tolerance and 2.8 times the worst measured: two computations each within S-1 of the reference may differ by that much, and a defect in either is far larger. Faulted runs (`UnmaskedAttention` against one token; `UnfinishedTile` against the unblocked product) are over 10 times the bound. These are initial values, set by me from the measurement under the rule "S-1's tolerance, if the worst measured is under half of it"; the owner can change them.
- **Tasks 24.5 and 24.6 (done).** `cargo test --no-fail-fast`: **459 passed, 0 failed** (452 at 0.4.1). S-1 to S-4, S-9, S-10 and S-11 pass in every category through the blocked pass; all five faults (`InterleavedRope`, `PerturbedLogit`, `NoFinalNorm`, `UnmaskedAttention`, and `UnfinishedTile`, with `SkippedRows` from 0.4.1) fail the harness; the thread-count independence test (1, 2, 3, 7 and the default, exact) passes through the blocked pass. The 30 runs of the guard on the 0.4.2 build are **identical to 0.4.1's** (`diff` empty): no id differs, so no near-tie needs recording. That is a measurement, not a promise (FR-47).

## 6. Upstream feedback

Recorded here for the upstream owner; the approved documents are not edited from this plan without the owner's permission. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Task 24.4 (owner's decision, 2026-11-12) | Requirements FR-46, FR-47; Specification §4, §11, §17, §18 | The blocked kernel sums each element in chunks of 64 products, each chunk with fused multiply-adds and the chunk sums added in order, not as one sequential sum: the single sum raised the worst error against the reference to 65% of S-1 and 45% of S-2 (§5). The texts say so, with the measurements. **Absorbed in place into Requirements 0.10 and Specification 0.8 (both `approved`) with the owner's permission of 2026-11-12, no version bump.** |
| Task 24.4 | Specification §11, §19 | The two bounds are measured and set (1e-3 in logits, from a worst of 3.53e-4), the tile is 4 × 16, and the open item moves to the resolved ledger. **Absorbed in place, as above.** |
| Task 24.3 | Specification §17 (batched pass tests) | The test that `Logits::Last` equals the last row of `Logits::All` bit for bit becomes "within 1e-5": the unembedding of `Last` is a matrix-vector product and of `All` a blocked product. The sentence in §17's batched pass tests that says so should be changed at the next absorption. |

## 7. Open Questions

None open. The tile shape and the two bounds were decided at tasks 23.3 and 24.4 (§5, §6); the bounds are mine, under a stated rule, for the owner to change.
