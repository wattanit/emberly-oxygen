# Emberly Oxygen — Implementation Plan for Release 0.4.0

Nov 10, 2026 · @Wattanit

**Target release:** 0.4.0\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.8 (upstream; approved on 2026-11-10), Emberly Oxygen — Technical Specification 0.6 (upstream; approved on 2026-11-10)

This is a living working document, not part of the SFD suite: it has no document version or status and is edited in place as work proceeds. It expands the Technical Specification's milestones M13 and M14 into three phases with ordered tasks and exit checks. It owns sequencing and task breakdown only; WHAT stays in the Requirements and HOW stays in the Specification, and every task cites the IDs it satisfies.

Phases are numbered on from 0.3.0's (Phases 14 to 16), so their to-do files in `docs/todo/` keep unique names.

**What this release is.** One technique: batched prefill (FR-40 to FR-42). The prompt's tokens go through one pass over all positions, with a causal mask and a matrix-matrix product that reuses each weight row across the prompt; the unembedding runs only for the last prompt position. Decoding, sampling, chat and the tokenizer do not change. The arithmetic of a logit may change in its last bits (FR-29, FR-41), so the release is judged against the reference and not against 0.3.0's bits.

## 1. Working rules

All of 0.3.0's rules hold. The ones this release leans on, and what it adds:

- **Phase order is fixed.** Phase 18 does not start until Phase 17's exit check passes (the harness cannot judge a pass that does not exist); Phase 19 does not start until Phase 18's does.
- **What is not touched.** `step`, `attention`, `matvec`, the other `ops.rs` functions, the tokenizer, `sampling.rs`, `chat.rs` and the thresholds of S-1 to S-4, S-9 and S-11 stay as they are. This release adds `matmul`, `prefill`, `Logits` and a fault, changes the generation entry (`generate.rs`) to start from `prefill`, and changes the harness's way of reaching the engine, not what it compares. If a task seems to need a change to `attention`, `matvec` or a threshold, it stops and the issue goes to the owner.
- **One technique only.** No tiling, register blocking, threads, hand-written SIMD, `unsafe` or intrinsics in `matmul` (Specification §4). If the loop order alone gives little, that is a recorded negative result (FR-42) raised with the owner, not a reason to add one.
- **Tests before code.** Each new unit (`matmul`, `prefill`, the fault, the sweep's support code) gets its tests first, seen to fail before the code exists. The fault gets its negative control before the batched pass is trusted (Specification §17).
- **Output may change, so say what changed.** FR-29 and FR-41 allow the last bits of the logits, and greedy ids at near-ties, to differ from 0.3.0's. The guard tests that pin ids must be run before and after; any id that differs is recorded in §5 with its near-tie gap, and a difference at a position with no near-tie is a defect, not a finding.
- **0.4.0 is compared with 0.3.0 directly.** The release is one change from the one-token prompt to the batched one, and the benchmark shows that change as a whole: no intermediate arm is built or timed (owner's decision, 2026-11-10).
- **One task, one commit.** Each task ends with a passing `cargo test --no-fail-fast` whose whole summary is read, and a commit with a body paragraph and **no attribution line**: the owner's standing rule, in every project. Docs-only commits carry the `[docs]` prefix.
- **Every test names the HC, FR or S item it verifies** (Specification §17).
- **No new Rust or Python dependency** (Specification §15). The oracle and the fixtures are not touched: the stored expected outputs judge the batched pass as they judged the one-token path.
- **Throwaway scripts stay out of the repository.** A script written to check something once lives in the scratchpad; its result goes in §5 with how it was produced.
- **Approved documents are not edited without the owner's permission.** What the work finds goes in §6. A small fix is written in place only when the owner says so; a large miss means rolling back to before the work, fixing the document and re-approving. No version bump absorbs a mistake: a bump comes only with new scope.
- **The SFD is settled before any merge.** Task 19.7 comes before the question of merging.
- **Work happens on `release/0.4.0`.** The owner pushes; nothing is pushed, tagged or merged to `develop` or `main` without the owner's explicit order at that moment (standing rule, 2026-11-06).

## 2. Phase 17 — The matrix-matrix product and the batched pass (first part of M13)

**Proves:** `matmul` equals the matrix-vector product row by row, `prefill` gives what `step` gives at every position within the bound, the mask is causal, and the loop order is worth trying. **Satisfies:** FR-40, FR-41, HC-1, HC-2.

| # | Task | Cites |
| --- | --- | --- |
| 17.1 | `ops.rs`: `matmul(W, X, Y)`, weight row outermost, calling the existing dot product. Tests first: equal to `matvec` row by row (exactly), equal to `matrixmultiply` within a tolerance far below S-1 at the model's five shapes and at row counts and lengths that are not a multiple of anything, including one row and lengths below the partial-sum width. | FR-40, HC-1, HC-2, Spec §4, §17 |
| 17.2 | `Logits` (`Last`, `All`) and `prefill`'s checks: an unknown token, an empty prompt and a prompt longer than the context are rejected before anything is touched, leaving the cache and the capture as they were. Tests first on the tiny model. | FR-40, FR-18, Spec §7 |
| 17.3 | `prefill`'s body (Specification §7): embedding rows, per layer norm, projections by `matmul`, RoPE per row, all keys and values appended, `attention` per row with `t + 1` positions, output projection, MLP; final norm and unembedding for the last row or for all; capture as `[n, hidden]`. Tests first on the tiny model: equal to `step` at every position within 1e-5 (logits, capture, cache contents); a token changed after position `t` leaves logits up to `t` bit-for-bit equal (causal); a one-token prompt; a prompt that fills the context exactly; `Last` equals the last row of `All`. | FR-40, FR-41, Spec §7, §17 |
| 17.4 | `fault.rs`: `UnmaskedAttention`, and the existing three faults applied in `prefill`. Tests first: each fault changes `prefill`'s output on the tiny model, and with no fault the feature changes nothing. | FR-40, HC-4, Spec §17 |
| 17.5 | Measure on the real model and prompts (§5): the largest logit and intermediate difference between `prefill` and `step` per prompt, and whether the two are bit-identical. Set the bound of Specification §11 from it by the rule used for S-1, with the owner. | FR-41, Spec §11 |
| 17.6 | Dry run, not recorded: time `step` over a 512-token prompt, `step` with the unembedding skipped, and `prefill`, three times each, to see whether the loop order alone gives anything. If the batched arm is not clearly faster than the second, stop and raise it with the owner (Spec §4). | FR-42, Spec §4 |

**Exit check:** `matmul` and `prefill` pass their tests and agree with `step` within the bound on every prompt of the set; the faults are in and change the output; the dry run is recorded in §5 and the owner has seen it; `cargo test` still passes, with only the tests this phase adds.

## 3. Phase 18 — The harness judges the batched pass; generation starts from it (second part of M13)

**Proves:** S-1 to S-4 and S-11 pass with the batched pass in every category, the four faults fail it, and generation, sampling and chat start from `prefill`. **Satisfies:** FR-29, FR-40, FR-41, S-1 to S-5, S-8, S-11, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 18.1 | Before any change, run the guard tests that pin ids (greedy, seeded, chat) and record the ids; they are the 0.3.0 reference for task 18.3. | FR-29 |
| 18.2 | The harness gets its logits, intermediates and top-1 from `prefill` with `Logits::All` and capture, in one pass per prompt. Only the engine's way of producing the values changes: thresholds, references and the near-tie rule do not. Tests first: the comparison functions' own tests are untouched; run the correctness test and read its table. | FR-41, S-1 to S-3, S-5 |
| 18.3 | `generate` and `generate_with` start from `prefill` with `Logits::Last`, then `step` (Specification §8). Tests first: the first token and a short greedy run equal the ones through `step` on the tiny model. Then rerun the guard tests of 18.1: every id equal, or each difference recorded in §5 with its gap (Working rules). Chat follows with no change of its own. | FR-40, FR-29, FR-34 |
| 18.4 | The batched-against-one-token test (Specification §11): both paths over the prompt set, the largest logit and intermediate difference reported, failing above the bound set in 17.5. | FR-41 |
| 18.5 | Negative controls through the batched pass: the harness fails on each of the four faults, including `UnmaskedAttention`; with no fault every check passes. Tests first. | HC-4, S-1, Spec §17 |
| 18.6 | Logging (Specification §14): one `prefill` span and event, then `step` events; per layer once at trace. The existing debug-event-count test changes to one prefill plus generated tokens minus one steps (the change is the Specification's, FR-23); a test shows identical output at every verbosity in generation and in chat. | S-8, FR-23 |
| 18.7 | The full gate: `cargo test --no-fail-fast`, whole summary read. Record the harness table, worst logit and intermediate errors per category against 0.3.0's (35% of S-1 and 10% of S-2 at 16 sums), and the chat reported-only figures. | HC-4, S-1 to S-5, S-11 |

**Exit check:** S-1 to S-4 pass in every category through `prefill`; S-9, S-10 and S-11 pass untouched; all four faults fail the harness; generation and chat run through `prefill`; every greedy or seeded id that differs from 0.3.0's has a near-tie recorded; no existing test changed except the debug event count of 18.6.

## 4. Phase 19 — The prefill sweep and the release (M14)

**Proves:** what the batched prompt is worth against 0.3.0's, and that decode did not change. **Satisfies:** FR-42, FR-8, FR-9, S-7, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 19.1 | The prefill sweep group in the benchmark harness: lengths 32, 67, 128, 512 and 644 as the first `n` ids of one fixed prompt, two paths (one `step` per token, which is how 0.3.0 processed a prompt, and one `prefill`), the paths checked to agree on the last logits before timing, rows with `"group": "prefill"`. Tests first for the support code: the fixed prompts, the lengths, the agreement check, the row. | FR-42, FR-9, S-7 |
| 19.2 | Bump the workspace version to 0.4.0 and confirm a clean tree. | FR-9, Spec §16 |
| 19.3 | Run `cargo test --no-fail-fast`, then `cargo bench` straight after, at the default log level (Specification §16). A failing test run stops here. | FR-8, FR-9, S-7 |
| 19.4 | Record in §5: the sweep per length and path (medians, spread); chat time to first token at 67 and 644 tokens against 0.3.0's 0.55 s and 6.5 s; the two workloads against 0.3.0's rows; decode throughput, expected unchanged, with a same-session A/B against the `v0.3.0` build in a scratchpad worktree if the session's drift matters as in 0.3.0 (plan 0.3.0 §5); peak memory. Compare 0.4.0 with 0.3.0 directly; say what it is worth, and a negative result as one. | FR-42, FR-9, S-7 |
| 19.5 | Release note `docs/releases/0.4.0.md` and the README (the status line, the version note, and the stale `release/0.3.0` wording). | Requirements §12 |
| 19.6 | Settle the SFD before the merge question: §6's rows written into the documents as the owner permits, the sfd-standard compliance review over both, every pin checked here and in the release note, and the owner's approval of the statuses. | G-14, G-16, G-24, G-25 |
| 19.7 | Owner's review of the rows and the note. Then **stop and ask** before merging or tagging `v0.4.0`: neither happens without the owner's explicit order. | Spec §16 |

**Exit check:** the 0.4.0 rows, including the sweep, are in `results/benchmarks.jsonl` from a clean tree; the release note exists; both foundation documents are approved and pinned.

## 5. Measurements to record

Filled in as the phases land. Machine: the benchmark machine (Apple M4 Max), as before. Baseline: 388 tests at 0.3.0.

0.3.0's recorded figures to compare with (plan 0.3.0 §5): short prompt 121.2 tok/s and 264.1 ms to first token; long prompt 100.3 tok/s and 5104.9 ms to first token; generation 116.2 tok/s short and 79.7 long; peak memory 811.6 MB; a chat turn 0.55 s at 67 prompt tokens and 6.5 s at 644. The session's own drift was 3% to 4%.

- **Tasks 17.1 to 17.4 (done).** `matmul`, `Logits`, `Session::prefill` and the `UnmaskedAttention` fault are in; `cargo test --no-fail-fast` passes (409 tests, up from 388 at 0.3.0). The dot product was pulled out of `matvec_lanes` into one `#[inline(always)]` function that both kernels call; the existing order pins in `tests/kernel.rs` still pass. The interleaved-RoPE fault cannot show on the forward tests' tiny model (head dimension 2: the two pairings are the same pairing), so its check uses a model with head dimension 4.
- **Task 17.5 (measured; the bound is not yet set).** A scratchpad program (`measure_prefill`, release build, not in the repository) ran every prompt of the set through `step` one token at a time and through `prefill` with `Logits::All` and capture, and compared the logits at every position and the embedding, every layer's output and the final-norm output. **All 12 prompts, 2,508 positions in all: the two paths are bit-identical** (largest logit difference 0, largest relative intermediate difference 0, argmax equal at every position). So there is no worst error to apply S-1's rule to. The 1e-5 bound of Specification §11 is therefore a guard against a defect and not a measured margin; the owner decides what it is.
- **Task 17.6 (dry run, not recorded in the benchmark record).** Scratchpad programs `dryrun` and `kernelcheck`, release build, Apple M4 Max. The 506-token prompt `long-prose`, three runs each (medians; the runs agree within 0.2%): one-token **4901 ms**; one-token with the unembedding skipped except for the last token **4087 ms** (estimated: the one-token time minus 505 timings of a 49152 × 576 matrix-vector product, 1.612 ms each); batched **3689 ms**. So the skipped logits are worth about **0.81 s (17%)** and batching about **0.40 s (a further 11%)**; together 1.33×. The unembedding is 17% of a token's work, not the "about a fifth" the Specification (§7) says.
  The 0.40 s was not traced to a source. `matmul` against 506 separate `matvec` calls at the four per-layer shapes, each matrix kept warm in the cache by the repetition: q 1.05×, k and v 0.98×, gate and up 0.99×, down 0.99×; **1.00× over the four together** (60.8 ms against 61.0 ms), at about 37 GFLOP/s, compute-bound. That isolated test cannot show weight reuse across tokens in the real pass, where all 540 MB of weights are read between two uses of one matrix and the one-token path loses the cache each time; so part of the 0.40 s may be reuse, and the rest removed per-step allocations and call overhead. The two cannot be told apart without a different experiment (plan §6). The four projections plus the output projection are about 2.95 s of the 3.69 s batched time (80%); the rest, mostly attention, is about 0.74 s.

- **Phase 17 decision (owner, 2026-11-10).** Option A: the release goes on as written, with `matmul` as the loop order only. The finding that the loop order alone gains nothing on this machine (matmul 1.00× against repeated matvec in isolation, with each matrix kept warm in the cache) is a recorded negative result for the release note (FR-42), and further speed-ups (register blocking, SIMD, threads) are left for later releases. The bound of the batched-against-one-token test stays at the Specification's initial 1e-5, a guard against a defect since the measured difference is 0; the owner has not said otherwise.
- **Task 18.1 (done, no commit).** Before `generate` changed, a scratchpad program (`chatguard`, release build, not in the repository) recorded the ids of 30 runs: greedy and seeded (temperature 0.8, seed 1) generation of 32 tokens on each of the 12 prompts, and greedy and seeded (seed 3) chat over three turns of 24 tokens with the instruct model. The existing guard test (`fr28_fr29_greedy_and_seeded_output_is_unchanged...`) pins four more runs; chat had no pinned ids.
- **Tasks 18.2 to 18.6 (done).** The commits are 18.2 (the harness through `prefill`), 18.3 with 18.6 (generation from `prefill`, and its logging; one commit because the logging tests could not stay green between them), and 18.4 (batched against one-token). The control for 18.5 was written first, in 18.2: a negative control for `UnmaskedAttention`, which the harness could not see while it went through `step` (the fault acts only in `prefill`), fails the logits at position 0 and names layer 0; the three older controls pass through `prefill` unchanged, so 18.5 needed no commit of its own. The debug-event test changed as the Specification says (one `prefill` event, then steps at their real positions), and so did the trace test, the two command-line logging tests and the log-file test; output is identical at every level in generation, sampling and chat (tests unchanged).
- **Task 18.3 guard result.** After the change the existing guard test passes and the 30 recorded runs are **identical** (`diff` of the before and after listings is empty): no id differs from 0.3.0's, so no near-tie needs recording. This is the expected consequence of the batched pass being bit-identical to `step` (task 17.5), not a promise (FR-41).
- **Task 18.4 (done).** On all 12 prompts the batched pass and the one-token pass differ by **exactly 0** in every logit and every intermediate (`tests/batched.rs`, bound 1e-5); a faulted batched pass is far over the bound, so the test can fail.
- **Task 18.7, the gate (done).** `cargo test --no-fail-fast`: **416 passed, 0 failed** (388 at 0.3.0). Through `prefill`, every category passes S-1 to S-4, S-9, S-10 and S-11. Worst errors against the 64-bit reference, by category (logit, relative intermediate): english 2.45e-4, 6.13e-6; thai 1.22e-4, 9.97e-6; code 1.23e-4, 5.24e-6; utf8-split 3.50e-4, 7.80e-6; long 2.45e-4, 5.25e-6. Overall **3.50e-4** (35% of S-1) and **9.97e-6** (10% of S-2), the same figures as 0.3.0's. The instruct model's reported-only table is unchanged: largest logit error 8.8e-5 and every greedy reply agreeing (8/8, 64/64, 7/7, 64/64). The four faults, including `UnmaskedAttention`, fail the harness.

- **Tasks 19.1 to 19.3 (done).** The prefill sweep and the version bump are in (420 tests). The first benchmark run was abandoned: the laptop slept during it, so its timings could not be trusted. Nothing from it was recorded (the record was still 47 lines, and the run stopped before its write), and it was rerun under `caffeinate` with the lid open. The rerun is the record: **23 rows appended** (`results/benchmarks.jsonl` now has 70 lines), version 0.4.0, commit `7d4cde5`, clean tree, `bench` profile, no debug assertions, straight after 420 passing tests; groups `workloads` (2), `kernels` (5), `sampling` (6) and the new `prefill` (10). A dry run of the sweep before the official one gave the same figures to within 1% at every length.
- **Task 19.4, the sweep.** Prompt time in milliseconds, up to the logits of the last position, in one process (medians of 5; the spread is under 0.5% except one 438 ms maximum at 67 tokens batched, whose median is 424.8):

  | Prompt tokens | One `step` per token (0.3.0's way) | One `prefill` | Speed-up | Time saved |
  | --- | --- | --- | --- | --- |
  | 32 | 254.6 | 188.4 | 1.35× | 26.0% |
  | 67 | 541.5 | 424.8 | 1.27× | 21.6% |
  | 128 | 1060.1 | 806.8 | 1.31× | 23.9% |
  | 512 | 4950.7 | 3727.9 | 1.33× | 24.7% |
  | 644 | 6550.2 | 4928.1 | 1.33× | 24.8% |

  The saving is a steady 22% to 26% at every length, so it does not grow with the prompt. That fits the dry run of task 17.6: about 17% of the one-token time is the unembedding computed once instead of once per prompt token, and the other 8% or so is not attributed: it may be weight reuse across tokens or removed per-step overhead, and the isolated kernel test (1.00× against repeated `matvec` with the matrix kept warm) cannot tell which.
- **Task 19.4, a chat turn.** Chat's time to first token is the prompt time plus the first token's choice, so the sweep's rows at 67 and 644 tokens are the first and last turn of 0.3.0's ten-turn chat: **0.55 s to 0.42 s at 67 tokens** (541.5 ms to 424.8 ms in the sweep) and **6.5 s to 4.9 s at 644** (6550 ms to 4928 ms). A long chat is still slow, because every turn processes the whole history (Requirements §2.2: reusing the cache across turns is a technique of its own).
- **Task 19.4, the two workloads against 0.3.0's rows (medians).** Short (32 prompt tokens): prompt throughput **121.2 → 169.2 tok/s (+39.6%)**, time to first token **264.1 → 189.1 ms (-28.4%)**. Long (512): prompt throughput **100.3 → 137.2 tok/s (+36.8%)**, time to first token **5104.9 → 3730.7 ms (-26.9%)**. Generation throughput, the part of the engine this release did not change: short 116.2 → 115.2 tok/s (-0.9%), long 79.7 → 82.0 tok/s (+2.9%), both within the 3% to 4% that 0.3.0's session drift showed, so no change is claimed. Peak memory 811.6 MB, unchanged to 0.002%; model load 61.9 → 62.2 ms and 63.6 → 61.7 ms. The kernel group, whose code is unchanged, reads 0.2% to 4.6% higher in this session (in-house GFLOP/s at the five shapes 36.5, 35.8, 35.2, 37.8, 35.2 → 37.5, 37.0, 36.8, 38.4, 35.4), the same session effect in the other direction. The sweep needs no same-session A/B against the `v0.3.0` build: both of its paths run in one process, and the workload figures differ from 0.3.0's by 27% to 40%, far above the 3% to 4% drift. Sampling step costs are unchanged (neutral 162.1 µs, top-p 967 µs, combined 562 µs; 0.3.0: 167.6, 1001, 570).
- **What the release found.** In isolation, with each matrix kept warm in the cache, `matmul` is 1.00× against repeated `matvec`: the kernel is compute-bound at about 37 GFLOP/s, and the projections are 80% of a long prefill. The 1.3× end to end is about 17% from not computing the 49152-way unembedding for every prompt token and about 8% unattributed (weight reuse across tokens, which the isolated test cannot show, or removed per-step overhead). More from batching needs a faster kernel (register blocking, SIMD, threads), which is v4's work (Requirements §2.2).


## 6. Upstream feedback

Recorded here for the upstream owner; the approved documents are not edited from this plan without the owner's permission. This section is a permanent record. **Every row below was absorbed in place into Requirements 0.8 and Specification 0.6 at task 19.6, with the owner's permission of 2026-11-10 and no version bump (the §4 row of the Specification was rejected, with the reason in its ledger).**

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Task 17.6 | Specification §7 (step 4) | §7 says the unembedding is "about a fifth of a token's arithmetic". Measured at 1.612 ms of a 9.7 ms one-token step, it is 17%. A wording fix. |
| Task 17.6 | Requirements §2.2 and FR-40; Specification §4 | §2.2 says the matrix-matrix product is untiled "so the speed-up it gives is the speed-up of batching and no other technique", and FR-40 says "so that this release measures batching alone". Measured: in isolation, with one matrix kept warm in the cache, the matrix-matrix loop order is 1.00× against repeated matrix-vector products; the release's end-to-end speed-up was not traced to its sources (about 17% is the output projection computed once, the rest is unattributed). The sentences should say that no other kernel technique is mixed in, not that the speed-up is the loop order's. §4's "the loop order is the whole technique" describes the code and should stay. **Absorbed in place into Requirements 0.8 and Specification 0.6 with the owner's permission of 2026-11-10 (all rows of this section), no version bump.** |
| Task 17.5 | Specification §11 | §11 says the bound of the batched-against-one-token test (1e-5) is "set after the first measurement". The measurement is 0 on all 12 prompts, so there is no error to set it from; it stays at 1e-5 as a guard against a defect (decision of 2026-11-10). §11 should say so. |
| Task 19.1 (owner's decision of 2026-11-10) | Specification §12 (prefill sweep), §7 (step 4), §18 (M14) | The sweep compares 0.4.0's way of processing a prompt with 0.3.0's directly, with two paths (one step per token, one prefill), not three; the intermediate arm that separated batching from the skipped logits is not built or timed. §12's sweep paragraph, the last sentence of §7 step 4 ("so §12 measures the two apart") and M14's "apart from the unused logits" no longer describe the release. |
| Task 19.1 | Specification §12 (prefill sweep) | As built: the prompts are the first `n` ids of the longest workload's prompt (512 tokens), repeated from its start for the 644-token length, not "one fixed long text"; the metric is `prompt_ms`; a row's `path` is `one_token` or `batched`; the group runs in a full `cargo bench` and alone with `--prefill-only`; nothing is recorded if the two paths' last logits differ by more than 1e-5. |

## 7. Open Questions

None at the start of the release. The bound for the batched-against-one-token test is set at task 17.5, with the owner.
