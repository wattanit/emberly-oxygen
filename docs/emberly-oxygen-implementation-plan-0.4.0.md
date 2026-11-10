# Emberly Oxygen — Implementation Plan for Release 0.4.0

Nov 10, 2026 · @Wattanit

**Target release:** 0.4.0\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.8 (upstream; approved on 2026-11-10), Emberly Oxygen — Technical Specification 0.6 (upstream; approved on 2026-11-10)

This is a living working document, not part of the SFD suite: it has no document version or status and is edited in place as work proceeds. It expands the Technical Specification's milestones M13 and M14 into three phases with ordered tasks and exit checks. It owns sequencing and task breakdown only; WHAT stays in the Requirements and HOW stays in the Specification, and every task cites the IDs it satisfies.

Phases are numbered on from 0.3.0's (Phases 14 to 16), so their to-do files in `docs/todo/` keep unique names.

**What this release is.** One technique: batched prefill (FR-40 to FR-42). The prompt's tokens go through one pass over all positions, with a causal mask and a matrix-matrix product that reuses each weight row across the prompt; the unembedding runs only for the last prompt position. Decoding, sampling, chat and the tokenizer do not change. The release has two effects to measure apart, batching and the skipped logits (Specification §7, §12), and the arithmetic of a logit may change in its last bits (FR-29, FR-41), so the release is judged against the reference and not against 0.3.0's bits.

## 1. Working rules

All of 0.3.0's rules hold. The ones this release leans on, and what it adds:

- **Phase order is fixed.** Phase 18 does not start until Phase 17's exit check passes (the harness cannot judge a pass that does not exist); Phase 19 does not start until Phase 18's does.
- **What is not touched.** `step`, `attention`, `matvec`, the other `ops.rs` functions, the tokenizer, `sampling.rs`, `chat.rs` and the thresholds of S-1 to S-4, S-9 and S-11 stay as they are. This release adds `matmul`, `prefill`, `Logits` and a fault, changes the generation entry (`generate.rs`) to start from `prefill`, and changes the harness's way of reaching the engine, not what it compares. If a task seems to need a change to `attention`, `matvec` or a threshold, it stops and the issue goes to the owner.
- **One technique only.** No tiling, register blocking, threads, hand-written SIMD, `unsafe` or intrinsics in `matmul` (Specification §4). If the loop order alone gives little, that is a recorded negative result (FR-42) raised with the owner, not a reason to add one.
- **Tests before code.** Each new unit (`matmul`, `prefill`, the fault, the sweep's support code) gets its tests first, seen to fail before the code exists. The fault gets its negative control before the batched pass is trusted (Specification §17).
- **Output may change, so say what changed.** FR-29 and FR-41 allow the last bits of the logits, and greedy ids at near-ties, to differ from 0.3.0's. The guard tests that pin ids must be run before and after; any id that differs is recorded in §5 with its near-tie gap, and a difference at a position with no near-tie is a defect, not a finding.
- **Do not credit batching with the skipped logits.** Every speed figure in §5 and in the release note names the arm it comes from (Specification §12). The release note states each effect separately.
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

**Proves:** what batching is worth, apart from the skipped logits, and that decode did not change. **Satisfies:** FR-42, FR-8, FR-9, S-7, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 19.1 | The prefill sweep group in the benchmark harness (Specification §12): lengths 32, 67, 128, 512 and 644 as the first `n` ids of one fixed long text, three arms (one-token, one-token with last logits, batched), arms checked to agree on the last logits before timing, rows with `"group": "prefill"`. Tests first for the support code: the fixed prompts, the agreement check, the row. | FR-42, FR-9, S-7 |
| 19.2 | Bump the workspace version to 0.4.0 and confirm a clean tree. | FR-9, Spec §16 |
| 19.3 | Run `cargo test --no-fail-fast`, then `cargo bench` straight after, at the default log level (Specification §16). A failing test run stops here. | FR-8, FR-9, S-7 |
| 19.4 | Record in §5: the sweep per length and arm (medians, spread); chat time to first token at 67 and 644 tokens against 0.3.0's 0.55 s and 6.5 s; the two workloads against 0.3.0's rows; decode throughput, expected unchanged, with a same-session A/B against the `v0.3.0` build in a scratchpad worktree if the session's drift matters as in 0.3.0 (plan 0.3.0 §5); peak memory. Say what each effect is worth, and a negative result as one. | FR-42, FR-9, S-7 |
| 19.5 | Release note `docs/releases/0.4.0.md` and the README (the status line, the version note, and the stale `release/0.3.0` wording). | Requirements §12 |
| 19.6 | Settle the SFD before the merge question: §6's rows written into the documents as the owner permits, the sfd-standard compliance review over both, every pin checked here and in the release note, and the owner's approval of the statuses. | G-14, G-16, G-24, G-25 |
| 19.7 | Owner's review of the rows and the note. Then **stop and ask** before merging or tagging `v0.4.0`: neither happens without the owner's explicit order. | Spec §16 |

**Exit check:** the 0.4.0 rows, including the sweep, are in `results/benchmarks.jsonl` from a clean tree; every figure names its arm; the release note exists; both foundation documents are approved and pinned.

## 5. Measurements to record

Filled in as the phases land. Machine: the benchmark machine (Apple M4 Max), as before. Baseline: 388 tests at 0.3.0.

0.3.0's recorded figures to compare with (plan 0.3.0 §5): short prompt 121.2 tok/s and 264.1 ms to first token; long prompt 100.3 tok/s and 5104.9 ms to first token; generation 116.2 tok/s short and 79.7 long; peak memory 811.6 MB; a chat turn 0.55 s at 67 prompt tokens and 6.5 s at 644. The session's own drift was 3% to 4%.

- **Tasks 17.1 to 17.4 (done).** `matmul`, `Logits`, `Session::prefill` and the `UnmaskedAttention` fault are in; `cargo test --no-fail-fast` passes (409 tests, up from 388 at 0.3.0). The dot product was pulled out of `matvec_lanes` into one `#[inline(always)]` function that both kernels call; the existing order pins in `tests/kernel.rs` still pass. The interleaved-RoPE fault cannot show on the forward tests' tiny model (head dimension 2: the two pairings are the same pairing), so its check uses a model with head dimension 4.
- **Task 17.5 (measured; the bound is not yet set).** A scratchpad program (`measure_prefill`, release build, not in the repository) ran every prompt of the set through `step` one token at a time and through `prefill` with `Logits::All` and capture, and compared the logits at every position and the embedding, every layer's output and the final-norm output. **All 12 prompts, 2,508 positions in all: the two paths are bit-identical** (largest logit difference 0, largest relative intermediate difference 0, argmax equal at every position). So there is no worst error to apply S-1's rule to. The 1e-5 bound of Specification §11 is therefore a guard against a defect and not a measured margin; the owner decides what it is.
- **Task 17.6 (dry run, not recorded in the benchmark record).** Scratchpad programs `dryrun` and `kernelcheck`, release build, Apple M4 Max. The 506-token prompt `long-prose`, three runs each (medians; the runs agree within 0.2%): one-token **4901 ms**; one-token with the unembedding skipped except for the last token **4087 ms** (estimated: the one-token time minus 505 timings of a 49152 × 576 matrix-vector product, 1.612 ms each); batched **3689 ms**. So the skipped logits are worth about **0.81 s (17%)** and batching about **0.40 s (a further 11%)**; together 1.33×. The unembedding is 17% of a token's work, not the "about a fifth" the Specification (§7) says.
  The 0.40 s is not weight reuse. `matmul` against 506 separate `matvec` calls at the four per-layer shapes: q 1.05×, k and v 0.98×, gate and up 0.99×, down 0.99×; **1.00× over the four together** (60.8 ms against 61.0 ms). The weight matrices are 0.4 to 3.5 MB and fit in the cache, so the weights are not being re-read from memory, and the kernel runs at about 37 GFLOP/s, compute-bound. The four projections plus the output projection are about 2.95 s of the 3.69 s batched time (80%); the rest, mostly attention, is about 0.74 s. The residual 0.40 s is probably the removed per-step allocations and call overhead, not measured.

- **Phase 17 decision (owner, 2026-11-10).** Option A: the release goes on as written, with `matmul` as the loop order only. The finding that the loop order alone gains nothing on this machine (matmul 1.00× against repeated matvec; the weights fit in cache and the kernel is compute-bound) is a recorded negative result for the release note (FR-42), and further speed-ups (register blocking, SIMD, threads) are left for later releases. The bound of the batched-against-one-token test stays at the Specification's initial 1e-5, a guard against a defect since the measured difference is 0; the owner has not said otherwise.

## 6. Upstream feedback

Recorded here for the upstream owner; the approved documents are not edited from this plan without the owner's permission. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Task 17.6 | Specification §7 | §7 says the unembedding is "about a fifth of a token's arithmetic". Measured at 1.612 ms of a 9.7 ms one-token step, it is 17%. A wording fix, to be written in place only with the owner's permission. |

## 7. Open Questions

None at the start of the release. The bound for the batched-against-one-token test is set at task 17.5, with the owner.
