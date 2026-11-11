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

- **Tasks 20.2 to 20.4 (done).** `pool.rs`, the pooled `matvec` and `matmul` (initial threshold 500,000 multiply-adds), `Tensor` data shared by `Arc` (clones share the numbers), `Model::load_with_threads` and `Model::threads`, and the pool wired into `step` and `prefill`, so the products of the whole engine are threaded (attention is not yet, Phase 21). `cargo test --no-fail-fast`: **436 passed, 0 failed** (420 at 0.4.0). The real model's logits, through `step` and `prefill` at 2, 3 and 7 threads, equal the one-thread logits exactly. A deliberate +1e-3 in the pooled product made the kernel test and the real-model test fail, so both can fail. The tests of 20.3 and 20.4 were written in the same step as the code, not seen failing before it (20.2's and 20.3's kernel tests were); the mutation check stands in for that. `Model::load` now uses the available cores until the sweep sets the default.

- **Task 20.1 (measured; the owner's decision on the open item is pending).** A scratchpad program (`threadcost`, release build, not in the repository; random weights at the real shapes, 9 rounds of repetitions, medians) compared two ways of threading in safe code. **Scoped:** `std::thread::scope` creating the threads at every call, borrowing weights, input and cache. **Persistent:** workers created once, weights shared as `Arc`, the input copied into an `Arc` per call, each worker's output returned over a channel, and the cache held as `Arc<RwLock<Vec<f32>>>` with a read lock per call. Every threaded result was checked exactly equal to the serial one (`assert_eq!` on the whole output: bit-identical at every count measured). Microseconds per call:

  | Product | 1 thread | 2 | 4 | 8 | 12 | 16 |
  | --- | --- | --- | --- | --- | --- | --- |
  | k/v 192×576, scoped / persistent | 9.9 / 7.3 | 22.6 / 10.4 | 27.8 / 12.0 | 62.4 / 19.1 | 93.4 / 26.6 | 121.7 / 32.4 |
  | q/o 576×576 | 16.8 / 17.0 | 28.7 / 16.4 | 32.4 / 16.4 | 64.7 / 20.9 | 95.5 / 28.9 | 124.7 / 35.3 |
  | gate/up 1536×576 | 45.3 / 45.7 | 44.1 / 33.3 | 40.7 / 25.7 | 69.0 / 26.5 | 99.4 / 30.3 | 128.0 / 38.4 |
  | down 576×1536 | 44.0 / 44.5 | 44.6 / 32.3 | 39.1 / 25.3 | 66.1 / 26.5 | 99.0 / 30.9 | 127.7 / 38.5 |
  | unembedding 49152×576 | 1583 / 1592 | 845 / 830 | 700 / 677 | 518 / 493 | 432 / 397 | 438 / 395 |

  Attention (9 heads, head dimension 64, 3 key-value heads), serial / scoped / persistent at 2, 3, 5 and 9 workers:

  | Positions | Serial | 2 | 3 | 5 | 9 |
  | --- | --- | --- | --- | --- | --- |
  | 8 | 1.3 | 17.5 / 6.9 | 21.8 / 8.7 | 35.1 / 11.0 | 67.1 / 17.9 |
  | 128 | 16.5 | 27.7 / 16.4 | 28.4 / 16.4 | 40.5 / 17.1 | 71.1 / 21.8 |
  | 512 | 70.5 | 55.8 / 45.1 | 51.8 / 42.4 | 51.6 / 31.0 | 75.9 / 31.7 |
  | 1024 | 141.4 | 90.3 / 83.6 | 79.0 / 71.2 | 73.1 / 52.1 | 88.4 / 47.1 |
  | 2000 | 279.6 | 167.7 / 162.1 | 127.7 / 119.0 | 101.1 / 83.9 | 104.4 / 71.5 |

  Reading: (1) a scoped thread costs about 8 µs per extra thread at every call (122 µs at 16), which is more than the whole product for everything except the unembedding, so scoped threading would slow most of a step; (2) the persistent pool's hand-over costs about 3 µs per extra worker (about 25 µs at 16 including the input copy), so it pays from the 1536-row products up, and is break-even or worse for the key and value projections (7 µs serial) and the 576-row products at 2 to 4 workers; those stay on the calling thread under the threshold; (3) the unembedding scales 4× at 12 workers, which matches the bandwidth profile (284 GB/s at 12 against 69 at one); (4) attention with a persistent pool gains little below 128 positions (it is lost in hand-over), 1.6× at 512, 2.9× at 1024 and 3.9× at 2000 with 9 workers; and (5) the safe cache sharing worked and `RwLock` read locks cost little, but a real cache grows by appends from the calling thread, so the write must wait until all workers have returned, which the call structure guarantees (each call waits for every worker's output). A rough projection of a short decode step from these rows (7 ms serial): about 4.2 ms with 4 workers, one dispatch per product that is over the threshold, so about 1.7×; the 211 hand-overs per token are the cost, and fusing products that share an input (q, k and v; gate and up) into one dispatch each would cut them, which stays inside the rule (still one output element per worker). That projection is an estimate, not a result.
- **Tasks 21.1 and 21.2 (done, one commit: they share the refactor).** The key-value cache's per-layer vectors are `Arc`ed, and a call hands workers clones that are dropped before it returns (the worker drops its job before answering), so the next append extends the vector in place; a test checks the handle count is 1 after every call. `attention` is now a thin wrapper over a head-range core. Decode attention divides the heads (threshold 256 positions, initial); prompt attention divides the rows, taken in a zigzag order (first, last, second, ...) so that every range mixes cheap and dear rows; `Pool::run` wakes no more workers than there are elements. With the trace level on, the serial path runs (it records the most-attended positions; the output is the same). Exactly-equal tests at 1, 2, 3, 7 and 16 threads; 439 tests pass. The tests were seen failing against stubs before the code.
- **Task 21.3 (done).** `Fault::SkippedRows` makes the pool replace its last worker's answer with zeros (a flag on the pool that exists only with `negative-controls`); `Model::inject_fault` sets it. At an explicit 4 threads with no fault the logits, top-1 and intermediates checks pass; with the fault the logits and the intermediates fail. The control was seen failing before the fault acted.
- **Tasks 21.4 and 21.5 (done).** `tests/threads.rs`: every prompt of the set through `prefill`, at 2, 3, 7 and the default (16) threads against one thread, logits and every intermediate **exactly equal**; one token at a time on the shortest and the longest prompt (over the 256-position attention threshold, so heads are divided too) at 1, 2, 3, 7 and the default, exactly equal; and a `SkippedRows` run at 4 threads over the bound. The correctness test already loads with the default count, so it judged the threaded engine: every category passes with worst logit error **3.50e-4** and worst relative intermediate error **9.97e-6**, the same figures as 0.4.0's. The guard tests that pin ids pass unchanged. The suite is 443 tests; the thread test adds about 60 s to a full run.
- **Tasks 21.6 and 21.7 (done); Phase 21's exit check passes.** `--threads N` on `generate` and `chat` (a count of 0 is rejected by the library with exit code 1, naming `threads` and the value; a non-number is a usage error, exit 2); the count is a field of the `-v` model-loaded event. Tests: zero rejected for both commands; the oracle's continuation identical at 1, 3 and 16 threads with a quiet default level; `-v` shows `threads=3` and, by default, the core count; a greedy chat reply equal at 1 and 5 threads. `cargo test --no-fail-fast`: **447 passed, 0 failed** (420 at 0.4.0). A first look at speed, not a recorded result (one run each, short prompt, 64 tokens): generation 115.6 tokens/s at 1 thread, 172.1 at 4, 170.9 at 8, 154.7 at 16; time to first token 48 ms, 17, 15, 17. That is about 1.5× at 4 threads against the projected 1.7×; the sweep (Phase 22) is the measurement.

## 6. Upstream feedback

Recorded here for the upstream owner; the approved documents are not edited from this plan without the owner's permission. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Task 20.1 (owner's decision, 2026-11-11) | Specification §19 (Open) | Resolved: persistent workers; weights shared as `Arc`; the input copied per call; each worker's output returned over a channel; the cache shared as `Arc<RwLock<..>>` with a read lock per call; no `unsafe` ("unless totally necessary"). Scoped threads per call are rejected: about 8 µs per extra thread, more than most products (plan §5). The Specification's open item moves to its resolved ledger at the next absorption. |
| Drafting | Requirements §2.1 and Specification §4 | Both cite the profile as "plan 0.4.1 §4"; in the plan it is §5 (the figures are the same). A reference fix for the next absorption; no figure is affected. |

## 7. Open Questions

None open. The one the release started with (how the workers read the weights and the key-value cache in safe code) was decided by the owner on 2026-11-11 after task 20.1 (§6).
