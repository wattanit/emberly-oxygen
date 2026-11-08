# Emberly Oxygen — Implementation Plan for Release 0.1.2

Nov 8, 2026 · @Wattanit

**Target release:** 0.1.2\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.3 (upstream, approved), Emberly Oxygen — Technical Specification 0.3 (upstream, draft when this plan was written)

This is a living working document, not part of the SFD suite: it has no document version or status and is edited in place as work proceeds. It expands the Technical Specification's milestone M8 into two phases with ordered tasks and exit checks. It owns sequencing and task breakdown only; WHAT stays in the Requirements and HOW stays in the Specification, and every task cites the IDs it satisfies. A discovery that changes HOW is recorded in §5 and absorbed by a Specification version bump, never decided here. The foundation versions above are the as-built pin for this release: if either bumps while the release is in progress, the pin is updated and the change recorded in §5. The previous release's plan is [0.1.1](emberly-oxygen-implementation-plan-0.1.1.md); its §4 rows are the figures this release is compared with.

Phases are numbered on from 0.1.1's (Phases 7 and 8), so their to-do files in `docs/todo/` keep unique names.

**What this release is.** One change: the correctness gate judges the engine against a float64 reference instead of PyTorch's float32 output (FR-1, FR-2, S-1 to S-4). The engine does not change. The release exists because 0.1.1 found that S-1 measured distance to PyTorch's rounding, not to the correct answer, and a correct engine reached 99% of its limit.

## 1. Working rules

All of 0.1.1's rules hold. The ones this release leans on, and what it adds:

- **Phase order is fixed.** Phase 10 does not start until Phase 9's exit check passes: the harness cannot be judged against fixtures that do not exist or have not been checked.
- **The engine is not touched.** No change to `crates/oxygen/src` other than what a task names, and no kernel change. The one engine-side edit allowed is the version number (task 10.8). If a task seems to need an engine change, it stops and the issue goes to the owner.
- **The thresholds do not move until the owner decides.** `LOGIT_TOLERANCE`, `INTERMEDIATE_TOLERANCE` and `NEAR_TIE_GAP` keep their values through Phase 10's measurement (task 10.6). Changing one is the owner's decision at task 10.7, and lands as a Requirements bump first (§5).
- **Tests before harness code.** The reader's and the comparison's new tests are written first and seen to fail against the old code (tasks 10.1 and 10.3), so they are known to be able to fail.
- **One task, one commit.** Each task ends with a passing `cargo test --no-fail-fast` whose whole summary is read (Phase 10), or with the oracle command it names (Phase 9), and a commit with a body paragraph and **no attribution line**: the owner's standing rule, in every project.
- **Every test names the HC, FR or S item it verifies** (Specification §17).
- **No new Rust dependency, and no new Python dependency.** The oracle already has `torch`, `transformers` and `safetensors`; the f64 run needs nothing else (Specification §15).
- **Throwaway scripts stay out of the repository.** A script written to measure something once (the reference's noise, the error tables for other kernel widths) lives in the scratchpad; its numbers go in §4 with how they were produced.
- **Work happens on `release/0.1.2`.** The owner pushes; nothing is pushed, tagged or merged to `develop` or `main` without the owner's explicit order at that moment (standing rule, 2026-11-06).

## 2. Phase 9 — Oracle (first half of M8)

**Proves:** the f64 reference exists, is checked, and its own noise is known. **Satisfies:** FR-1, FR-2, FR-4, HC-3.

| # | Task | Cites |
| --- | --- | --- |
| 9.1 | Before touching the oracle, save what the current fixtures hold outside the repository: the manifest and a checksum of every f32 file, in the scratchpad. They are the 0.1.1 fixtures and the baseline for task 9.3's reproducibility check. | FR-4 |
| 9.2 | Extend `oracle/generate.py` (Specification §10): the forward pass, the greedy loop and the file writing become one routine run twice, once per precision; the f64 model is loaded from the same files with `dtype=torch.float64`, its parameter dtype and device asserted as the f32 model's are; hooks, `final_norm_output` and top-two gaps work as before. The f32 outputs and manifest fields stay as they are; the f64 run adds `<id>.f64.safetensors` and the `f64_`-prefixed manifest fields. | FR-1, FR-2, Spec §10 |
| 9.3 | Run it. Record the time it takes and the size of the fixtures folder. Check the f32 files against task 9.1's checksums: the same machine and library versions should reproduce them bit for bit; if not, record how they differ (this is itself a measure of the reference's noise, §4). | FR-2, FR-4 |
| 9.4 | Measure the f64 reference's own noise: run the f64 forward pass twice more with different thread counts and compare to the stored result (largest absolute logit difference, largest relative intermediate difference, per category). A scratchpad script, not committed. The Specification expects it far below S-1; this task shows it. If it is not, the task stops and the owner is asked: a noisy reference defeats the release. | FR-2, S-1, Spec §10 |
| 9.5 | Report how the two runs differ, for the owner: per category, the worst logit and intermediate difference between the f32 and f64 fixtures (the 7.1e-4 of 0.1.1 should reappear) and the number of prompts whose f32 and f64 greedy continuations differ, and where. Recorded in §4. | FR-2, S-3, S-4 |

**Exit check:** both fixture sets exist for every prompt and the f32 set is unchanged or its change is explained; the f64 reference's noise is recorded and is far below S-1; the two runs' difference is recorded; the owner has reviewed the oracle's diff and the numbers.

## 3. Phase 10 — Harness and release (second half of M8)

**Proves:** the gate judges against the f64 reference, can fail, and the release is measured. **Satisfies:** FR-5, FR-6, FR-8, FR-9, S-1 to S-8, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 10.1 | The fixture reader accepts F64 as well as F32 and carries the dtype in its type; tests first: an F64 file parses with exact values, a dtype that does not match the manifest, a missing f64 file and a malformed one each fail with a message naming the command that regenerates them, and the f64 fixtures have the manifest's shapes. | FR-4, FR-7, Spec §11, §17 |
| 10.2 | The manifest's `f64_` fields are read; `discover` fails naming the oracle command if they are absent. The tensors of a prompt load in both precisions, and the harness's existing checks still run on the f32 data while the next tasks move them. | FR-4, FR-7 |
| 10.3 | The comparison functions judge against an f64 reference with the engine's f32 widened before the difference; tests first, with a reference value f32 cannot represent, so a check that narrowed the reference is seen to give a different error. Near-ties come from the f64 gaps; the greedy comparison reads the f64 continuation. Constants unchanged. | S-1 to S-4, FR-2, Spec §11, §17 |
| 10.4 | Wire the checks to the f64 fixtures. Beside each logit and intermediate result the harness measures and reports the distance to the f32 fixtures, per prompt and per category, in the same table; a test shows that figure never fails a prompt. | FR-5, FR-6, S-5, Spec §11 |
| 10.5 | Negative controls against the new reference: all three fail the harness; a test checks the `PerturbedLogit` size exceeds the S-1 limit in force, so a later change of limit cannot leave a control that passes. | HC-4, S-1, S-2, Spec §17 |
| 10.6 | The full gate: `cargo test --no-fail-fast`, whole summary read. Record per category the worst logit and intermediate error against the f64 reference and against the f32 one, for the 16-sum engine (§4), the time the suite takes, and the margin to each limit. A failure is not worked around: it goes to the owner. | HC-4, S-1 to S-6, S-8 |
| 10.7 | **Stop: the owner decides the thresholds** (Requirements §13, Specification §19), from §4's table and the rule in the to-do list's decision point 1, written before the numbers are seen. Keep, tighten or loosen S-1 to S-4. A change is a Requirements bump first (0.4) and the constants follow it; if the limits are kept, the Open Question is resolved in the next foundation bump. | S-1 to S-4, Requirements §13 |
| 10.8 | Prepare the measurement: bump the workspace version to 0.1.2 so the record carries it, and confirm a clean tree. | FR-9, Spec §16 |
| 10.9 | Run `cargo test --no-fail-fast`, then `cargo bench` straight after on the benchmark machine at the default log level, as in 0.1.1 (Spec §16; both sides carry the same warm-machine bias). A failing test run stops here. The engine is unchanged, so these rows also show how repeatable 0.1.1's were. | HC-4, S-7, S-8, Spec §16 |
| 10.10 | Record the comparison with 0.1.1 in §4: workloads and kernel shapes, and the difference between two runs of the same code. | FR-8, FR-9 |
| 10.11 | Release note `docs/releases/0.1.2.md` (what changed, why it was expected to help, what was measured, what was learned, negative results) and the README: status, the oracle's longer run and larger fixtures, the test suite's time. The note must not claim a speed change. | Requirements §12 |
| 10.12 | Owner's review of the rows and the note. Then **stop and ask** before merging or tagging `v0.1.2`: neither happens without the owner's explicit order. | Spec §16 |

**Exit check:** the harness passes every prompt in every category against the f64 reference with the thresholds the owner decided; the f32 distance is reported; all three negative controls fail; `results/benchmarks.jsonl` holds the 0.1.2 rows from a clean tree, every row carrying its conditions; the release note exists and is accurate against the record; the owner has reviewed; the merge and tag are done only on the owner's order.

## 4. Measurements to record

Filled in during the phases. Nothing here is tuned; thresholds change only by the owner's decision at task 10.7 and a foundation bump.

- The oracle's run (task 9.3, Apple M4 Max, 12 torch threads): 99 s wall (183 s user) for both precisions on all 12 prompts; the fixtures folder went from about 0.7 GB to 2.0 GB. All 12 f32 files reproduced the 0.1.1 files bit for bit (SHA-256), and the manifest's existing fields are unchanged apart from `generated_at`.
- The stock f64 reference's own noise (task 9.4; scratchpad script, f64 forward pass of all 12 prompts rerun and compared with the stored f64 file; worst logit absolute difference and worst intermediate relative difference):

  | Category | 12 threads (as generated) | 1 thread |
  | --- | --- | --- |
  | english | 0 and 0 | 0 and 0 |
  | code | 0 and 0 | 0 and 2.0e-17 |
  | utf8-split | 0 and 0 | 0 and 2.0e-17 |
  | thai | 0 and 0 | **1.74e-5** and 4.3e-7 |
  | long | 0 and 0 | **1.65e-5** and 5.9e-7 |

  Same thread count reproduces exactly; one thread differs by up to 1.74e-5 in logits on the thai and long prompts, 1.7% of S-1 and over the 1e-5 (1%) bar of the to-do's decision point 3, so **task 9.4 stopped and went to the owner**. Cause found: the pinned transformers computes RMSNorm (`modeling_llama.py` line 63), the attention softmax (line 197) and the rotary tables (line 111) in float32 even in a float64 model, so the "f64" run still contains f32 rounding that depends on thread count. A scratchpad variant with RMSNorm and softmax forced to float64 (RoPE tables left as the library makes them) differs from the stock f64 run by up to 7.3e-5 in logits (7% of S-1), and from itself at 1 and 12 threads by 2.2e-6 in logits and 4.0e-8 relative in intermediates.
- The difference between the f32 and stock f64 fixtures (task 9.5; worst logit absolute, worst intermediate relative): english 2.88e-4 and 3.45e-5, thai 3.34e-4 and 1.61e-5, code 3.54e-4 and 4.27e-5, utf8-split **7.13e-4** and 2.81e-5, long 2.92e-4 and 2.96e-5. 0.1.1's diagnostic gave 7.12e-4 for the same prompt, so it reappears. Against the patched variant (RMSNorm and softmax in f64) the f32 fixtures differ by 3.08e-4, 3.28e-4, 3.28e-4, 3.93e-4 and 6.39e-4 in logits (code, english, long, thai, utf8-split). The f32 and stock f64 greedy continuations are identical for all 12 prompts, so no near-tie arises in them.
- Worst logit and intermediate error of the 16-sum engine per category, against the f64 and f32 references, with the margin to each limit (task 10.6). 0.1.1's figures against f32: english 3.14e-4 and 3.66e-5, thai 2.79e-4 and 1.51e-5, code 3.18e-4 and 4.02e-5, utf8-split 9.90e-4 and 2.70e-5, long 3.18e-4 and 2.99e-5. *(to fill)*
- Test suite time with the larger fixtures, against 28 s (task 10.6). *(to fill)*
- Workloads and kernel shapes against 0.1.1's rows (task 10.10). *(to fill)*

## 5. Upstream feedback

Recorded here and absorbed by the upstream owner at the next bump; the upstream documents are not edited from this one. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Before Phase 9 | Specification 0.3 status | The Specification is `draft` and this plan pins it so; the status is the owner's to set. The pin is updated here if the status or version changes. |
| Before Phase 9 | Requirements FR-25 | FR-25 says the matrix-vector product was "about 98.6% of a decode step in 0.1.0". Release 0.1.1's like-for-like figure from the recorded rows is 94%; 98.6% came from a dry run on a cooler machine (release note 0.1.1). Suggest the next Requirements bump correct the figure. The Specification §4 was corrected in 0.3. |

| Task 9.4 | Requirements FR-2; Specification §10 | **FR-2 says the 64-bit run is "exact in arithmetic", and §10 says the library keeps only its RoPE tables in f32. Both are wrong:** the pinned library also computes RMSNorm and the attention softmax in float32 in a float64 model (task 9.4, §4). Whatever the owner decides below, FR-2's "Not guaranteed" sentence and §10's reference-limits bullet need correcting, which needs a Requirements bump because 0.3 is approved. |

## 6. Open Questions

- **Which float64 reference S-1 to S-4 are judged against, now that the library's own f64 run is only partly f64 (task 9.4).** Options and a recommendation are in the to-do's decision point 4. Resolver: owner. Blocks task 9.4's exit and Phase 10.
- **Whether S-1 to S-4's thresholds hold against the f64 reference (Requirements §13, Specification §19).** Resolver: owner, at task 10.7, from §4's measurements.
