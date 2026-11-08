# Emberly Oxygen — Implementation Plan for Release 0.2.0

Nov 8, 2026 · @Wattanit

**Target release:** 0.2.0\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.6 (upstream, approved on 2026-11-08), Emberly Oxygen — Technical Specification 0.4 (upstream, approved on 2026-11-08)

This is a living working document, not part of the SFD suite: it has no document version or status and is edited in place as work proceeds. It expands the Technical Specification's milestones M9 and M10 into three phases with ordered tasks and exit checks. It owns sequencing and task breakdown only; WHAT stays in the Requirements and HOW stays in the Specification, and every task cites the IDs it satisfies.

Phases are numbered on from 0.1.2's (Phases 9 and 10), so their to-do files in `docs/todo/` keep unique names.

**What this release is.** One feature: sampled generation with temperature, top-k, top-p and a repetition penalty, drawn from a seeded generator (FR-26 to FR-31, S-9, S-10). The greedy path does not change (FR-29). Chat, the instruct model and any change to the oracle's role as a gate are for the next release (Requirements §2.3, §13).

## 1. Working rules

All of 0.1.2's rules hold. The ones this release leans on, and what it adds:

- **Phase order is fixed.** Phase 12 does not start until Phase 11's exit check passes (the harness cannot be judged against distributions that do not exist); Phase 13 does not start until the owner has decided S-9's tolerance (task 12.8).
- **The greedy path is not touched.** No change to `forward.rs`, `ops.rs`, the tokenizer or the existing harness checks. `generate` gains a parameter, and the greedy branch it takes must produce the same IDs as before (FR-29). If a task seems to need a change to the forward pass, it stops and the issue goes to the owner.
- **The tolerance does not move until the owner decides.** S-9's tolerance has no value until task 12.8; the harness reports the measured errors until then. The values for S-10 (200,000 draws, df + 4·sqrt(2·df)) are initial (Specification §11) and change only by the owner's decision.
- **Tests before code.** Each new unit (generator, validation, controls, distribution function, the S-9 check) gets its tests first, seen to fail before the code exists, so each is known to be able to fail.
- **One task, one commit.** Each task ends with a passing `cargo test --no-fail-fast` whose whole summary is read (Phases 12 and 13), or with the oracle command it names (Phase 11), and a commit with a body paragraph and **no attribution line**: the owner's standing rule, in every project. Docs-only commits carry the `[docs]` prefix.
- **Every test names the HC, FR or S item it verifies** (Specification §17).
- **No new Rust dependency, and no new Python dependency.** The generator is written in this project (Specification §8, §15); the oracle already has `torch`, `transformers` and `safetensors`.
- **Throwaway scripts stay out of the repository.** A script written to check something once (the textbook recomputation of task 11.3, the published generator vectors of task 12.1) lives in the scratchpad; its result goes in §4 with how it was produced.
- **The SFD is settled before any merge.** Task 13.11 comes before the question of merging: feedback absorbed by version bumps, a compliance review, the owner's approval of the statuses, pins current.
- **Work happens on `release/0.2.0`.** The owner pushes; nothing is pushed, tagged or merged to `develop` or `main` without the owner's explicit order at that moment (standing rule, 2026-11-06).

## 2. Phase 11 — Sampling reference (first part of M9)

**Proves:** the oracle writes the adjusted distributions and their cutoff margins, they were checked by an independent recomputation, and the harness can read them. **Satisfies:** FR-31, FR-4, HC-3.

| # | Task | Cites |
| --- | --- | --- |
| 11.1 | `oracle/sampling_settings.json`: named settings, each control alone and one combination of all four (Specification §10; initial, tune with use). | FR-26, FR-31 |
| 11.2 | `oracle/sampling_reference.py`: reads the f64 fixtures and manifest, applies the library's processors in the order of Specification §8, writes `<id>.sampling.safetensors` and the manifest's `sampling` entry with the cutoff margins. A start-up check that the processors keep f64, shown to fail if one is fed f32. | FR-31, FR-4, HC-3 |
| 11.3 | Recompute one setting per control from the textbook definition in numpy f64 (a scratchpad script) and compare with the library's output. The library's definition is the reference's (FR-31's honesty clause); any difference, at a tie or at the top-p boundary, is recorded in §4. | FR-31 |
| 11.4 | Run the script. Record the time, the size of the new files, that every distribution sums to 1 within 1e-12, and the size of each setting's support. | FR-31, FR-4 |
| 11.5 | The harness reads the sampling files and manifest entry; tests first: a missing file, a wrong shape and a missing manifest entry each fail with a message naming the command that generates them. No engine change. | FR-4, FR-7 |

**Exit check:** every prompt has a sampling file for every setting; each distribution sums to 1; the textbook recomputation agrees with the library or the difference is recorded; the harness reads the files; `cargo test` still passes untouched.

## 3. Phase 12 — Sampler and the S-9 gate (second part of M9)

**Proves:** the controls and generator are correct, the engine's adjusted distribution is judged against the stored one, and the check can fail. **Satisfies:** FR-26, FR-27, FR-30, FR-5, S-9, HC-1, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 12.1 | `rng.rs`: xoshiro256** seeded through SplitMix64, uniform f64 from the top 53 bits. Tests first: outputs for known seeds equal the authors' published C implementation, compiled in the scratchpad. | HC-1, FR-28 |
| 12.2 | `Sampling` with a greedy default and validation naming the option and value; tests first: each invalid value is rejected, each boundary value accepted. | FR-30 |
| 12.3 | The controls in `sampling.rs`: penalty, temperature, top-k, top-p. Tests first against hand-computed distributions: the penalty's sign rule, a tie at the k-th logit, top-p at its boundary, p small enough to keep one token. | FR-26, FR-27 |
| 12.4 | The public distribution function (controls in f32, softmax in f64) in the order of Specification §8. Tests first: the order matters (a swapped-order result differs on a combined setting). | FR-26 |
| 12.5 | The harness's S-9 check: per prompt and setting, the largest probability difference from the stored distribution, in the category table, with cutoff margins reported. Tests first with a hand-made stored distribution. | FR-5, S-9, S-5 |
| 12.6 | Sampling negative controls: `Fault::SwappedSamplingOrder` and `Fault::PenaltyWrongSign`. A test asserts S-9 fails on each, on the combined setting. | HC-4, S-9, Spec §17 |
| 12.7 | Measure: per category, the worst S-9 error for each setting, the cutoff margins against the engine's own rounding, and any setting whose cutoff could be crossed. Recorded in §4. | S-9 |
| 12.8 | **Stop: the owner decides S-9's tolerance and the cutoff-boundary rule** (Requirements §13, Specification §19) from §4's table and the rule in the to-do's decision point 1. A change to the exemption is a Requirements bump. | S-9 |

**Exit check:** the harness reports S-9 for every prompt and setting; the tolerance is the owner's; both new controls fail it; the greedy path's tests pass untouched.

## 4. Phase 13 — Sampling and release (M10)

**Proves:** seeded generation and the CLI work, greedy is unchanged, the draw test passes, and the release is measured and documented. **Satisfies:** FR-26 to FR-30, FR-8, FR-9, S-8, S-10, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 13.1 | `generate` takes `Sampling`, draws one token per sampled step, returns the seed. Tests first: same seed, same tokens; different seed, different tokens; temperature 0 with every other control set gives the default's IDs; the greedy harness passes untouched. | FR-26, FR-28, FR-29 |
| 13.2 | The S-10 draw test (Specification §11); shown to fail on draws from a different distribution. Record the statistic against its limit. | S-10 |
| 13.3 | CLI options and the `seed: <n>` line on standard error. CLI tests run the binary: an invalid value exits 1 naming the option; standard output stays text only; a sampling run without `--seed` prints one and replaying it reproduces the output. | FR-19, FR-20, FR-30 |
| 13.4 | `-vv` step events when sampling (probability, tokens kept), computed only at that level. A test shows output identical at every verbosity with sampling on. | S-8, FR-23 |
| 13.5 | The full gate: `cargo test --no-fail-fast`, whole summary read; record the time. | HC-4 |
| 13.6 | The sampling cost group in `benches/` (Specification §12); the workloads stay greedy. | FR-8 |
| 13.7 | Bump the workspace version to 0.2.0 and confirm a clean tree. | FR-9, Spec §16 |
| 13.8 | Run `cargo test --no-fail-fast`, then `cargo bench` straight after, at the default log level (Specification §16). A failing test run stops here. | FR-8, FR-9, S-7 |
| 13.9 | Record the comparison with 0.1.2's rows in §5 (the greedy workloads should be unchanged; say so as 0.1.2 did, and do not claim a speed change) and what a sampled step costs next to a forward step. | FR-8, FR-9 |
| 13.10 | Release note `docs/releases/0.2.0.md` and the README (what changed, why it was expected to help, what was measured, what was learned, negative results). | Requirements §12 |
| 13.11 | **Settle the SFD before the merge question:** absorb §6's rows into Requirements 0.7 and Specification 0.5 (bumps), run the sfd-standard compliance review over both, update every pin here and in the release note, and ask the owner to approve the statuses. | G-14, G-16, G-24, G-25 |
| 13.12 | Owner's review of the rows and the note. Then **stop and ask** before merging or tagging `v0.2.0`: neither happens without the owner's explicit order. | Spec §16 |

**Exit check:** greedy output identical to 0.1.2's; S-9 and S-10 pass with the owner's tolerance; the new controls fail; `results/benchmarks.jsonl` holds the 0.2.0 rows from a clean tree; the release note exists; both foundation documents are approved and pinned.

## 5. Measurements to record

Filled in as the phases land. Machine: the benchmark machine (Apple M4 Max), as before.

- **Phase 11 (done).**
  - **Oracle script.** `sampling_reference.py` runs in 1 s (it reads the stored f64 logits and does not run the model) and writes 12 files, 23 MB in all, for 60 distributions (12 prompts × 5 settings). Every distribution sums to 1 within 1e-12 (checked in the script and again by a test).
  - **Support sizes.** Temperature 0.7 and penalty 1.3 keep the whole vocabulary (49152); top-k 40 keeps 40 on every prompt, with no tie at the 40th logit. Top-p 0.9 keeps 1 to 102 tokens (1 on `long-go`, 2 on `code-rust`, 102 on `english-short`). **The combined setting keeps 1 to 7 tokens, and exactly 1 on `code-rust` and `long-go`:** on those two prompts the combined distribution is a point mass, so a swapped order cannot show there. The sampling negative controls (task 12.6) must be judged on a prompt where the combined support is larger (for example `thai-medium`, 7).
  - **Textbook recomputation (task 11.3).** A numpy f64 script written from the textbook definitions, using no library code, agrees with the library's distributions to at most 1.10e-13 over all 60, so the library's definition and the textbook one are the same on this prompt set, including top-p's boundary rule. It found no tie at the k-th logit. (Scratchpad: `textbook_sampling.py`.)
  - **Cutoff margins (first look, for task 12.8).** Smallest top-k margin: 2.18e-4 logits (`thai-medium`), then 9.76e-4 and 1.37e-3. Smallest top-p margin: 6.79e-5 in probability (`utf8-split-emoji`), then 1.16e-4 and 2.09e-4. The engine's worst logit error in 0.1.2 was 3.50e-4, so `thai-medium`'s top-k cutoff is within the engine's rounding and is the first candidate for the crossable-cutoff rule.
- **Phase 12:** per category and setting, worst S-9 error (engine f32 against the stored f64 distribution); cutoff margins; settings whose cutoff the engine's rounding could cross.
- **Phase 12 (measured, awaiting task 12.8).**
  - **S-9 error.** Worst probability error between the engine's distribution and the stored one, over all 60 distributions: **1.98e-6** (`temperature-0.7` on `utf8-split-emoji`). Per category and setting, from 3.2e-7 (`penalty-1.3` on english) to 2.0e-6; `long-go`, whose distributions are nearly point masses, is 7e-11 or 0. The engine's logits are f32 and the reference's are f64, so this is rounding, not a defect.
  - **Cutoffs.** No setting's cutoff was crossed on any prompt: every top-k keeps exactly 40 and every top-p set matches the reference. The engine's own logit error at the last prompt position, the position S-9 uses, is at most **5.05e-5** (`long-prose`); the 3.50e-4 worst case of 0.1.2's tables occurs at other positions of other prompts. Smallest top-k margin: 2.18e-4 logits (`thai-medium`), 4.3 times the worst last-position error (5.05e-5) and 12 times that prompt's own (1.84e-5); next 9.76e-4, 1.37e-3. Smallest top-p margin: 6.79e-5 in probability (`utf8-split-emoji`), against that prompt's S-9 error of 1.4e-6 (49 times); next 1.16e-4, 2.09e-4.
  - **Negative controls.** On `thai-medium`: the swapped order (top-p before temperature) moves the combined setting by 0.050 and nothing else; the inverted penalty moves `penalty-1.3` by 0.85 and `combined` by 0.97 and nothing else. On `english-short` the inverted penalty moves the distribution by only 7.7e-3 (its repeated tokens carry little probability), so a tolerance of 1e-2 would miss it there; the controls therefore run on `thai-medium`. Every setting passes with no fault.
  - **Tolerance now in force.** `SAMPLING_TOLERANCE` is provisional at 1e-2 until task 12.8. The rule of decision point 1 (the smallest round decimal at least twice the worst error) gives 1e-5 from 1.98e-6.
- **Phase 13:** the S-10 statistic against its limit; test suite time; the 0.2.0 rows against 0.1.2's; sampling-step cost per control, against a forward step.

## 6. Upstream feedback

Recorded here and absorbed by the upstream owner at the next bump; the upstream documents are not edited from this one. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Before Phase 11 | Requirements 0.6 companion pin | The header pinned Specification 0.3 when it was approved, before Specification 0.4 existed; the pin went stale (G-16). At the owner's order to approve the Specification, the pin was changed to 0.4 in place, a header-only edit with no text change. The next Requirements bump records it. |
| Before Phase 11 | Requirements FR-26 | Top-k keeps every token tied with the k-th highest logit, as the reference library does (Specification §8), so more than k tokens can survive. FR-26 says "keeps the k highest-scoring tokens". Suggest a clause on ties. |
| Before Phase 11 | Requirements FR-26, FR-29 | Temperature 0 takes the argmax of the raw logits and ignores the other controls, with a warning, so FR-29's "identical to the previous release's greedy output" stays literally true (Specification §8). FR-26 says only that temperature 0 means greedy. Suggest it say the other controls are ignored. |
| Before Phase 11 | Requirements S-9 | Whether a setting whose top-k or top-p cutoff the engine's rounding could cross is judged or only reported (Specification §19). S-9 says "every prompt and setting". Waits for task 12.8. |
| Task 12.4 | Specification §17 | The negative control `SwappedSamplingOrder` is described as "temperature applied after top-k". Temperature divides by a positive number, which does not change the ranking, so it commutes with top-k and that swap gives the same distribution: S-9 could never fail on it. The implemented fault applies top-p before temperature, which does change the result (0.050 on `thai-medium`). §17 needs the description corrected. |
| Task 12.7 | Specification §11, §19 | The cutoff-margin exemption (S-9 reports and does not judge a setting whose cutoff the engine's rounding could cross) was not needed: no cutoff was crossed, and the smallest margins are 4.3 times the worst last-position logit error and 49 times the matching probability error. If the owner decides at 12.8 not to keep the exemption, §11's sentence and the §19 item are removed at the next bump. |

## 7. Open Questions

- **S-9's tolerance, and how a crossable cutoff is treated.** Resolver: owner, at task 12.8. Blocks task 12.8's exit and Phase 13.
