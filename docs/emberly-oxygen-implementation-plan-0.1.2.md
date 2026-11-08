# Emberly Oxygen — Implementation Plan for Release 0.1.2

Nov 8, 2026 · @Wattanit

**Target release:** 0.1.2\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.4 (upstream, approved; 0.3 was approved and then corrected, §5), Emberly Oxygen — Technical Specification 0.3 (upstream, draft when this plan was written)

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

- The oracle's run (task 9.3, Apple M4 Max, 12 torch threads): about 98 s wall (the first run 99 s, the run after option B 97.5 s) for both precisions on all 12 prompts; the fixtures folder went from about 0.7 GB to 2.0 GB. All 12 f32 files reproduced the 0.1.1 files bit for bit (SHA-256), in both runs, and the manifest's existing fields are unchanged apart from `generated_at`.
- The stock library f64 run's noise (task 9.4, first attempt; kept as the record that led to the owner's decision for option B; scratchpad script, f64 forward pass of all 12 prompts rerun and compared with the stored f64 file; worst logit absolute difference and worst intermediate relative difference):

  | Category | 12 threads (as generated) | 1 thread |
  | --- | --- | --- |
  | english | 0 and 0 | 0 and 0 |
  | code | 0 and 0 | 0 and 2.0e-17 |
  | utf8-split | 0 and 0 | 0 and 2.0e-17 |
  | thai | 0 and 0 | **1.74e-5** and 4.3e-7 |
  | long | 0 and 0 | **1.65e-5** and 5.9e-7 |

  Same thread count reproduces exactly; one thread differs by up to 1.74e-5 in logits on the thai and long prompts, 1.7% of S-1 and over the 1e-5 (1%) bar of the to-do's decision point 3, so **task 9.4 stopped and went to the owner**. Cause found: the pinned transformers computes RMSNorm (`modeling_llama.py` line 63), the attention softmax (line 197) and the rotary tables (line 111) in float32 even in a float64 model, so the "f64" run still contains f32 rounding that depends on thread count. A scratchpad variant with RMSNorm and softmax forced to float64 (RoPE tables left as the library makes them) differs from the stock f64 run by up to 7.3e-5 in logits (7% of S-1), and from itself at 1 and 12 threads by 2.2e-6 in logits and 4.0e-8 relative in intermediates.
- **The reference as released (option B: RMSNorm and softmax forced to float64, rotary tables as published).** Noise, a rerun against the stored f64 files (worst logit absolute difference and worst intermediate relative difference; scratchpad script `noise_and_diff.py noise <threads>`, run from `oracle/`, with the same replacement and its start-up check):

  | Category | 12 threads (as generated) | 1 thread |
  | --- | --- | --- |
  | english | 0 and 0 | 0 and 0 |
  | code | 0 and 0 | 2.1e-13 and 5.6e-15 |
  | utf8-split | 0 and 0 | 1.5e-13 and 2.6e-15 |
  | thai | 0 and 0 | 1.66e-6 and 3.7e-8 |
  | long | 0 and 0 | **2.18e-6** and 4.0e-8 |

  The worst is 2.18e-6, 0.22% of S-1 and under the 1e-5 bar; it is 8 times smaller than the stock run's. Same thread count reproduces exactly. What remains on the thai and long prompts, the only ones with more than about 100 positions where it appears, is not explained; the rotary tables, which stay in float32, are one candidate, and nothing was done to find out. The start-up check was shown to fail without the replacement.
- The difference between the f32 fixtures and the reference as released (task 9.5; worst logit absolute, worst intermediate relative; scratchpad `noise_and_diff.py diff`): english 3.28e-4 and 3.44e-5, thai 3.93e-4 and 1.64e-5, code 3.08e-4 and 4.28e-5, utf8-split **6.39e-4** and 2.83e-5, long 3.28e-4 and 2.99e-5. The f32 and f64 greedy continuations are identical for all 12 prompts. Against the stock library f64 run the same figures were as in the next entry.
- For the record, the difference between the f32 and the stock library f64 fixtures (task 9.5, first attempt; worst logit absolute, worst intermediate relative): english 2.88e-4 and 3.45e-5, thai 3.34e-4 and 1.61e-5, code 3.54e-4 and 4.27e-5, utf8-split **7.13e-4** and 2.81e-5, long 2.92e-4 and 2.96e-5. 0.1.1's diagnostic gave 7.12e-4 for the same prompt, so it reappears. Against the patched variant (RMSNorm and softmax in f64) the f32 fixtures differ by 3.08e-4, 3.28e-4, 3.28e-4, 3.93e-4 and 6.39e-4 in logits (code, english, long, thai, utf8-split). The f32 and stock f64 greedy continuations are identical for all 12 prompts, so no near-tie arises in them.
- **The 16-sum engine against the f64 reference (task 10.6; the gate, every prompt in every category passes, thresholds unchanged).** Worst logit error (S-1, limit 1e-3) and worst relative intermediate error (S-2, limit 1e-4), the share of the limit in brackets, with the distance to the f32 reference beside them (reported, not judged):

  | Category | Logit vs f64 | Intermediate vs f64 | Logit vs f32 | Intermediate vs f32 |
  | --- | --- | --- | --- | --- |
  | english | 2.45e-4 (25%) | 6.13e-6 (6%) | 3.14e-4 | 3.66e-5 |
  | thai | 1.22e-4 (12%) | 9.97e-6 (10%) | 2.79e-4 | 1.51e-5 |
  | code | 1.23e-4 (12%) | 5.24e-6 (5%) | 3.18e-4 | 4.02e-5 |
  | utf8-split | **3.50e-4 (35%)** | 7.80e-6 (8%) | 9.90e-4 | 2.70e-5 |
  | long | 2.45e-4 (25%) | 5.25e-6 (5%) | 3.18e-4 | 2.99e-5 |

  Worst logit error 3.50e-4, 35% of S-1; worst intermediate error 9.97e-6, 10% of S-2. The "vs f32" columns reproduce 0.1.1's table digit for digit (9.90e-4 on `utf8-split`, 99% of S-1 against the f32 reference).
- **The same table for three kernels against the f64 reference** (task 10.6, scratchpad only: `LANES` changed to 1 and to 8, run, restored, `git status` clean; 1 sum is 0.1.0's order, and its and the 8-sum run's "vs f32" columns reproduced 0.1.0's and 0.1.1's recorded figures). Worst logit and worst relative intermediate error per category:

  | Category | 1 sum (0.1.0's order) | 8 sums | 16 sums (released) |
  | --- | --- | --- | --- |
  | english | 4.58e-4, 3.40e-5 | 2.56e-4, 9.19e-6 | 2.45e-4, 6.13e-6 |
  | thai | 3.14e-4, 1.45e-5 | 2.67e-4, 6.97e-6 | 1.22e-4, 9.97e-6 |
  | code | 3.10e-4, 4.37e-5 | 1.50e-4, 4.64e-6 | 1.23e-4, 5.24e-6 |
  | utf8-split | 4.50e-4, 2.37e-5 | 3.20e-4, 1.32e-5 | 3.50e-4, 7.80e-6 |
  | long | 4.58e-4, 2.46e-5 | 2.56e-4, 1.04e-5 | 2.45e-4, 5.25e-6 |
  | worst | **4.58e-4**, **4.37e-5** | 3.20e-4, 1.32e-5 | 3.50e-4, 9.97e-6 |

  Against the f64 reference the partial sums are closer to the truth than 0.1.0's index order, not further: the worst intermediate error falls 4.4× (4.37e-5, 44% of S-2, to 9.97e-6, 10%) and the worst logit error falls from 4.58e-4 to 3.50e-4. The 8-sum kernel is slightly better than 16 on the worst logit (3.20e-4) and on `utf8-split`; the differences between 8 and 16 are within what the sweep did not try to resolve. This is the finding of 0.1.1 shown directly: against the f32 reference the same engines looked 1.4× to 8× worse.
- **Decision of the owner (task 10.7, 2026-11-08): S-1 to S-4 and the near-tie gap are kept as they are** (1e-3, 1e-4, 1e-3). By the rule agreed before the numbers were seen (keep when the worst error is at most half the limit), S-1 is at 35% and S-2 at 10%. The margin that 0.1.1 lost against the f32 reference (99% of S-1) is 65%. The limits are not tightened, because a real bug smaller than a tighter limit could hide and the negative controls are sized for the present values.
- Test suite time with the larger fixtures (task 10.6): `time cargo test --no-fail-fast` 23.8 s wall with the build already compiled (100 s user, 230 tests; the correctness test 13.4 s against 12.7 s a task earlier, so reading about 2 GB of fixtures instead of 0.7 GB cost under a second). 0.1.1 recorded 28 s for 214 tests; how that figure was timed is not recorded here, so no speed-up is claimed.
- **0.1.2's rows against 0.1.1's** (task 10.10; the engine's code is the same, so every difference here is variation between two measurements, not an effect of the release). 0.1.1 at commit `b1a3214`, 0.1.2 at `e132ab0`, both from a clean tree straight after a passing `cargo test`; median of 5 with the range in brackets:

  | Workload | Measure | 0.1.1 | 0.1.2 | Change |
  | --- | --- | --- | --- | --- |
  | short | first token | 0.257 s (0.257 to 0.257) | 0.257 s (0.256 to 0.257) | 0.0% |
  | short | prompt | 124.60 tok/s (124.43 to 124.65) | 124.61 (124.37 to 125.03) | 0.0% |
  | short | generation | 117.48 tok/s (117.18 to 117.55) | 117.82 (117.73 to 117.97) | +0.3% |
  | long | first token | 5.178 s (4.954 to 5.198) | 4.927 s (4.923 to 4.933) | -4.9% |
  | long | prompt | 98.87 tok/s (98.50 to 103.36) | 103.91 (103.78 to 104.01) | +5.1% |
  | long | generation | 79.76 tok/s (79.66 to 83.12) | 83.81 (83.68 to 83.92) | +5.1% |
  | both | peak memory | 811.57 MB and 811.55 MB | 811.57 MB and 811.58 MB | 0.0% |
  | both | model load | 60.5 ms | 60.5 ms and 60.9 ms | under 1% |

  Kernel, the engine's GFLOP/s per shape: 576×576 37.34 to 37.39, 192×576 36.76 to 36.74, 1536×576 36.73 to 36.77, 576×1536 38.30 to 38.68, 49152×576 35.35 to 35.38; within 0.1% except 576×1536 (+1.0%). The other widths and `matrixmultiply` also reproduce (16 sums 36.5, 36.5, 36.3, 37.9 and 35.1 GFLOP/s; 8 sums 30.0, 29.8, 30.0, 26.8 and 29.8; `matrixmultiply` 7.2, 7.0, 7.2, 7.2 and 4.6). The whole bench took 3 minutes 51 seconds (0.1.1: 233 s).

  **The long workload is 5% faster in 0.1.2 and nothing in the code explains it.** Its 0.1.1 range was wide (generation 79.66 to 83.12 tokens/s, first token 4.954 to 5.198 s, a 4% spread) with the median near the bottom; the 0.1.2 range is tight (83.68 to 83.92, 0.3%) and sits above 0.1.1's whole range. Candidates are the machine's state during the earlier run (thermal or background load), code placement after the rebuild, and chance; none was investigated or ruled out. Two consequences: a single long-workload figure carries at least 5% of variation between sessions on this machine, so 0.1.1's 4.3× long-generation gain over 0.1.0 could equally be stated as 4.5×, and the short workload and the kernel are far steadier (0.3% and 0.1%).

## 5. Upstream feedback

Recorded here and absorbed by the upstream owner at the next bump; the upstream documents are not edited from this one. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Before Phase 9 | Specification 0.3 status | The Specification is `draft` and this plan pins it so; the status is the owner's to set. The pin is updated here if the status or version changes. |
| Before Phase 9 | Requirements FR-25 | FR-25 says the matrix-vector product was "about 98.6% of a decode step in 0.1.0". Release 0.1.1's like-for-like figure from the recorded rows is 94%; 98.6% came from a dry run on a cooler machine (release note 0.1.1). Suggest the next Requirements bump correct the figure. The Specification §4 was corrected in 0.3. |

| Task 9.4 | Pin | **Decision of the owner (2026-11-08): option B.** The oracle forces RMSNorm and softmax to float64 for the f64 run and leaves the rotary tables as published. Requirements bumped to 0.4 (FR-2 corrected, FR-25's 94% absorbed) and the Specification's §10 and §17 edited in place (it is `draft`); this plan's pin follows. Requirements 0.4 was approved by the owner on 2026-11-08, after Phase 9's numbers. |
| Task 9.4 | Requirements FR-2; Specification §10 | **FR-2 says the 64-bit run is "exact in arithmetic", and §10 says the library keeps only its RoPE tables in f32. Both are wrong:** the pinned library also computes RMSNorm and the attention softmax in float32 in a float64 model (task 9.4, §4). Whatever the owner decides below, FR-2's "Not guaranteed" sentence and §10's reference-limits bullet need correcting, which needs a Requirements bump because 0.3 is approved. |

| Task 10.7 | Requirements §13 | The Open Question "whether S-1 to S-4's initial thresholds hold against the 64-bit reference" is resolved by the owner's decision of 2026-11-08: they hold and are kept, with the measured errors of §4 (worst logit 3.50e-4, worst intermediate 9.97e-6). Suggest the next Requirements bump move it to "Resolved since 0.4" with those figures, as 0.2's ledger did for 0.1.0. The Specification's matching Open Item was resolved in place (it is `draft`). |

## 6. Open Questions

- ~~**Whether S-1 to S-4's thresholds hold against the f64 reference.**~~ **Resolved by the owner at task 10.7 (2026-11-08):** kept; §4 and §5.
- ~~**Which float64 reference S-1 to S-4 are judged against, now that the library's own f64 run is only partly f64 (task 9.4).** Options and a recommendation are in the to-do's decision point 4. Resolver: owner. Blocks task 9.4's exit and Phase 10.~~ **Resolved by the owner (2026-11-08): option B**; §4 and §5.
