# Emberly Oxygen — Implementation Plan for Release 0.4.1

Nov 11, 2026 · @Wattanit

**Target release:** 0.4.1\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.9 (upstream; approved on 2026-11-11), Emberly Oxygen — Technical Specification 0.7 (upstream; approved on 2026-11-11)

This is a living working document, not part of the SFD suite: it has no document version or status and is edited in place as work proceeds. It expands the Technical Specification's milestones M15 and M16 into three phases with ordered tasks and exit checks. It owns sequencing and task breakdown only; WHAT stays in the Requirements and HOW stays in the Specification, and every task cites the IDs it satisfies.

Phases are numbered on from 0.4.0's (Phases 17 to 19), so their to-do files in `docs/todo/` keep unique names.

**What this release is.** One technique: threads (FR-43 to FR-45). The matrix-vector and matrix-matrix products and the attention are divided among a pool of worker threads by output element, never inside a sum, so the output is bit-identical at every thread count. The arithmetic, the harness thresholds, sampling, chat and the tokenizer do not change. Because the output is unchanged, the release is judged against 0.4.0's bits as well as against the reference.

## 1. Working rules

All of 0.4.0's rules hold. The ones this release leans on, and what it adds:

- **Phase order is fixed.** Phase 21 does not start until Phase 20's exit check passes, and Phase 22 does not start until Phase 21's does.
- **What is not touched.** `dot_lanes` and the arithmetic of every element, the other `ops.rs` functions, the tokenizer, `sampling.rs`, `chat.rs` and the thresholds of S-1 to S-4, S-9 and S-11 stay as they are. If a task seems to need a change to one of them, it stops and the issue goes to the owner.
- **One technique only.** Threads. No SIMD intrinsics, no register blocking or tiling, no lower-precision weights, no parallelism crate and **no `unsafe`** (HC-1, FR-43; owner, 2026-11-11: `std::thread` is fine, `unsafe` only as a last resort and only by the owner's choice). If safe code cannot hand the workers what they read at an acceptable cost, the task stops and raises it with the owner (Specification §19).
- **Tests before code.** Each new unit (pool, threaded kernels, threaded attention, the fault, the sweep's support code) gets its tests first, seen to fail before the code exists. `SkippedRows` gets its negative control before the threaded engine is trusted.
- **Bit-identical, so test for exactly 0.** Every threaded result is compared with the single-threaded one with an exact-equality bound at counts 1, 2, 3, 7 and the default (Specification §11). Counts that do not divide the work (3, 7) are in the set on purpose.
- **Measure before building.** Task 20.1 measures the per-call cost of threads and the safe ways to share the key-value cache; the pool's design follows its result, with the owner.
- **0.4.1 is compared with 0.4.0 directly**, no intermediate engine (FR-45).
- **One task, one commit.** Each task ends with a passing `cargo test --no-fail-fast` whose whole summary is read, and a commit with a body paragraph and **no attribution line**: the owner's standing rule, in every project; check `git log -1 --format=%B` afterwards. Docs-only commits carry the `[docs]` prefix.
- **Every test names the HC, FR or S item it verifies.**
- **No new Rust or Python dependency** (Specification §15).
- **Throwaway scripts stay out of the repository**; their results go in §5 with how they were produced.
- **Approved documents are not edited without the owner's permission.** What the work finds goes in §6.
- **The SFD is settled before any merge** (task 22.6).
- **Work happens on `release/0.4.1`.** The owner pushes; nothing is pushed, tagged or merged without the owner's explicit order at that moment.
- **Benchmarks run under `caffeinate`, lid open**, as 0.4.0's did after the sleeping laptop spoiled a run.

## 2. Phase 20 — The pool and the threaded products (first part of M15)

**Proves:** threads are worth building at these shapes, the pool hands workers what they read in safe code, and a threaded `matvec` and `matmul` equal the single-threaded ones exactly. **Satisfies:** FR-43, FR-44, HC-1, HC-2.

| # | Task | Cites |
| --- | --- | --- |
| 20.1 | Measure, in scratchpad programs: (a) the cost of one hand-over to a persistent worker and back, against creating scoped threads per call, at 1, 2, 4, 8 and 16 workers; (b) the two safe ways of letting workers read the key-value cache (shared ownership per layer with a hand-over per call; scoped borrowing per call), at the real shapes; (c) the product time at each of the eight shapes of a step under 1 to 16 threads, to set the small-product threshold. Record in §5, and take the choice for the open item of Specification §19 to the owner. | FR-43, Spec §4, §19 |
| 20.2 | `pool.rs`: the worker pool as the owner's decision of 20.1 leaves it. Tests first: every range of work runs exactly once; a count of 1 runs on the calling thread and spawns nothing; 0 is rejected naming `threads`; workers are joined when the pool drops; a panic in a worker reaches the caller, not a hang. | FR-43, FR-44, Spec §2 |
| 20.3 | `matvec` and `matmul` take the pool: contiguous row ranges differing by at most one, the calling thread taking one, products under the threshold staying on it. Tests first: exactly equal to the single-threaded result at counts 1, 2, 3, 7 and the default, at the model's five shapes and at odd sizes (one row, fewer rows than workers). | FR-43, HC-2, Spec §4 |
| 20.4 | `Model::load_with_threads(dir, n)` and the pool's place in the session; `Model::load` means the default count. Tests first: the count is stored; 0 is rejected naming `threads`; the model still loads and runs at 1. | FR-44, Spec §2 |

**Exit check:** the open item of Specification §19 has the owner's decision and §5 the measurements behind it; the threaded products equal the single-threaded ones exactly at every tested count; `cargo test` passes with only the tests this phase adds.

## 3. Phase 21 — Threaded attention, the harness and the command line (second part of M15)

**Proves:** the whole engine is bit-identical at every thread count, the harness passes unchanged in its thresholds, and `SkippedRows` fails it. **Satisfies:** FR-43, FR-44, S-1 to S-4, S-6, S-8, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 21.1 | Threaded attention in `step`: query heads divided among workers, the per-head arithmetic untouched. Tests first on the tiny model and then the real one: logits, capture and cache contents exactly equal at counts 1, 2, 3, 7 and the default over a decode run. If heads alone leave bandwidth unused (measured in 22.x), the finer splits of Specification §4 are tried as a recorded follow-up, not here. | FR-43, Spec §4 |
| 21.2 | Threaded attention in `prefill`: rows divided among workers, each row running the existing single-row function. Tests first: `prefill` exactly equal at every tested count, and still equal to `step`; the causal test holds. | FR-43, FR-41, Spec §4, §7 |
| 21.3 | Before the fault, the negative control: `Fault::SkippedRows` (a worker's range skipped), run with an explicit 4 workers, changes the output on the tiny model; with no fault nothing changes. Then the fault. | HC-4, Spec §17 |
| 21.4 | The thread-count independence test over the prompt set (Specification §11): logits and intermediates compared exactly across counts 1, 2, 3, 7 and the default, both paths; a faulted run is over the bound. | FR-43, S-6, Spec §11 |
| 21.5 | The harness through the threaded engine at the default count: run the correctness test and read its table; S-1 to S-4 and S-11 pass with the same thresholds; `SkippedRows` fails it. The existing guard tests that pin ids are run before and after; every id must be equal. | S-1 to S-4, S-6, HC-4 |
| 21.6 | `--threads` on `generate` and `chat`; the count in the `-v` summary and the run summary; 0 rejected naming the option and value. Tests first. Output identical at every count and every verbosity. | FR-44, S-8, Spec §9, §14 |
| 21.7 | The full gate: `cargo test --no-fail-fast`, whole summary read; the harness table recorded in §5 against 0.4.0's. | S-1 to S-6, S-11 |

**Exit check:** the engine is exactly equal at every tested count; S-1 to S-4 pass through the threaded engine; all five faults fail the harness; `--threads` works and rejects 0; no existing test changed except where the summary now reports the count.

## 4. Phase 22 — The thread sweep and the release (M16)

**Proves:** what threads are worth against 0.4.0 at every count, and the default count. **Satisfies:** FR-45, FR-44, FR-8, FR-9, S-7, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 22.1 | The thread sweep group in the benchmark harness (Specification §12): 1, 2, 4, 8, 12 and 16 threads; prompt at 128 and 644 tokens; decode after 32 and 512 token prompts; each result checked equal to the one-thread result before timing; a `threads` field in every row, `"group": "threads"`. Tests first for the support code. | FR-45, FR-9, S-7, Spec §12 |
| 22.2 | Choose the default count from the sweep by the rule of Specification §12 (highest decode speed at 512, ties within 2% to the smaller), and set it. Bump the workspace version to 0.4.1 and confirm a clean tree. | FR-44, Spec §12, §16 |
| 22.3 | Run `cargo test --no-fail-fast`, then `cargo bench` straight after, at the default log level, under `caffeinate`. A failing test run stops here. | FR-8, FR-9, S-7 |
| 22.4 | Record in §5: the sweep per count (medians, spread); both workloads against 0.4.0's rows; peak memory; the one-thread row against 0.4.0's to show the pool costs nothing at 1; where the speed-up stops and why, against the profile (read bandwidth 69 GB/s at one thread, 284 at 12). A same-session A/B against the `v0.4.0` build in a scratchpad worktree if drift matters. Say what it is worth, and a negative result as one. | FR-45, FR-9, S-7 |
| 22.5 | Release note `docs/releases/0.4.1.md` and the README (status line, version note). | Requirements §12 |
| 22.6 | Settle the SFD before the merge question: §6's rows written into the documents as the owner permits, the sfd-standard compliance review over both, every pin checked here and in the release note, and the owner's approval of the statuses. | G-14, G-16, G-24, G-25 |
| 22.7 | Owner's review of the rows and the note. Then **stop and ask** before merging or tagging `v0.4.1`. | Spec §16 |

**Exit check:** the 0.4.1 rows, including the sweep, are in `results/benchmarks.jsonl` from a clean tree; the release note exists; both foundation documents are approved and pinned.

## 5. Measurements to record

Machine: the benchmark machine (Apple M4 Max). Baseline: 420 tests at 0.4.0.

**The profile of 2026-11-11** (a scratchpad program with random weights at the real shapes, not in the repository; the figures the documents cite):

- Read bandwidth by thread count: 1 thread 69 GB/s, 2 135, 4 157, 8 211, 12 284, 16 272.
- One-core f32 multiply-add rate: 111 GFLOP/s.
- Matrix-vector products per decode token: 7.22 ms in all, at 74.5 GB/s over 538 MB, the single core's limit. Shares: q 7.5%, k 2.5%, v 2.5%, o 7.4%, gate 19.9%, up 19.8%, down 19.1%, unembedding 21.1%. A whole step is 7.56 ms, 96% of it products.
- Batched `matmul` at 128 tokens for one layer: 37.9 GFLOP/s, against `matrixmultiply`'s sgemm at 109.7 (2.9×), which is why register blocking, not threads, is the prefill lever for a later release.
- Decode step time by context: 7.69 ms at 8 tokens, 8.67 at 128, 11.32 at 512, 15.49 at 1024, 24.05 at 2000. Attention is about a third of a step at 512 and two thirds at 2000.
- Reading: decode is bandwidth-bound and sits at one core's limit, so threads help it only as far as the memory system's rate rises (about 4× at 12 threads); prefill is compute-bound, so threads help it near linearly.

0.4.0's recorded figures to compare with (plan 0.4.0 §5): short prompt 169.2 tok/s and 189.1 ms to first token; long prompt 137.2 tok/s and 3730.7 ms; generation 115.2 tok/s short and 82.0 long; peak memory 811.6 MB; sweep 188.4, 424.8, 806.8, 3727.9 and 4928.1 ms at 32, 67, 128, 512 and 644 tokens.

Filled in as the phases land.

## 6. Upstream feedback

Recorded here for the upstream owner; the approved documents are not edited from this plan without the owner's permission. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| | | None yet. |

## 7. Open Questions

- **How the workers read the weights and the key-value cache in safe code** (Specification §19, Open). Resolver: task 20.1 measures, the owner decides.
