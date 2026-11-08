# Phase 9 — Oracle: To-do

Working checklist for Phase 9 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.2.md) (the first half of milestone M8: the f64 reference exists, is checked, and its own noise is known). Tick items as they land; one task is one commit with a body paragraph and no attribution line, each ending with the oracle command it names run and its output read.

Proves: every prompt has an f32 and an f64 fixture, the f32 set is unchanged or its change is explained, and the f64 reference's run-to-run noise is far below S-1. Satisfies FR-1, FR-2, FR-4, HC-3.

Layout: the oracle is `oracle/generate.py` (run as `cd oracle && uv run python generate.py`), its prompts `oracle/prompts.json`, its output `fixtures/smollm2-135m/` (git-ignored). The fixture format is Specification §10. Python is only in the oracle (HC-3); the engine and both harnesses are not touched in this phase.

Dependencies: none new (`torch`, `transformers`, `safetensors`, `numpy` are already locked).

Out of scope: any Rust change. Any change to the f32 outputs or to the prompt set. A different model.

## Decision points for the owner

Each has a recommendation; **the owner agreed to all of them on 2026-11-08**.

1. **Which prompts get an f64 run. (applies to 9.2.)** *Recommendation:* all twelve, because S-5 judges every category and a subset would leave a category judged against a different reference than the rest. The cost is a few minutes more and about 1.3 GB of disk (Specification §10).
2. **How the reference's noise is measured. (applies to 9.4.)** *Recommendation:* run the f64 forward pass of every prompt once with one thread and once with the default thread count, compare each to the stored f64 result, and report the worst logit difference and worst relative intermediate difference per category. Thread count is the only source of reordering inside one machine; a second machine is not available for this release, and the note says so.
3. **What counts as "far below S-1". (applies to 9.4, written before the numbers.)** *Recommendation:* the reference's noise is acceptable if its worst logit difference is below 1% of the S-1 limit (1e-5) in every category. If not, stop and ask.

## 9.1 Save the current fixtures' fingerprint (FR-4)

- [x] Write, in the scratchpad (not the repository), a copy of `fixtures/smollm2-135m/manifest.json` and the SHA-256 of every `*.safetensors` file there, with the date and the machine. *Done: 12 files, in `scratchpad/fixtures-0.1.1/`; the fixtures were generated 2026-11-05 on the Apple M4 Max with torch 2.14.1, transformers 5.21.0, 12 torch threads.*
- [x] Confirm `git status` is clean and the branch is `release/0.1.2`.

## 9.2 The f64 run (FR-1, FR-2, Spec §10)

- [x] Refactor `generate.py` so one routine runs the forward pass, the greedy loop and the file writing for a given model and dtype; the f32 outputs and every existing manifest field stay as they are (check by diffing the generated manifest's existing fields against task 9.1's copy after 9.3).
- [x] Load the f64 model from the same files with `dtype=torch.float64`, eager attention, `.eval()`, no `device_map`; assert its parameter dtype is `float64` and its device `cpu`, as the f32 model's are.
- [x] Hooks, `final_norm_output` and the hook-versus-library check work as before for the f64 model. The top-two gaps and the greedy continuation (own argmax loop, 32 tokens, EOS stop) are computed from the f64 logits.
- [x] Write `<id>.f64.safetensors` per prompt (same tensor names and shapes, dtype F64, not narrowed) and add to each manifest record `f64_tensors_file`, `f64_prompt_top2_gaps`, `f64_continuation_ids`, `f64_continuation_text`, `f64_continuation_top2_gaps`, `f64_stopped_at_eos`; the manifest's `tensors` note says which file is which precision.
- [x] The script's docstring and `--help` say what it writes and that it takes minutes; the README is updated in task 10.11, not here.

## 9.3 Run it and check the f32 set (FR-2, FR-4)

- [ ] Run `cd oracle && uv run python generate.py`; record the wall time and `du -sh fixtures` in plan §4.
- [ ] Compare each f32 file's SHA-256 with task 9.1's. Same machine and library versions should reproduce them bit for bit. If any differ, record which and by how much (the largest absolute difference), and tell the owner before going on: it means the f32 reference itself is not reproducible, which is the finding of 0.1.1 again.
- [ ] Check the f32 manifest fields are unchanged apart from the new `f64_` ones.

## 9.4 The reference's own noise (FR-2, S-1)

- [ ] Apply decision points 2 and 3. In a scratchpad script (not committed), rerun the f64 forward pass of every prompt with `torch.set_num_threads(1)` and with the default, and compare each result with the stored `<id>.f64.safetensors`: the largest absolute logit difference, and the largest intermediate difference relative to its tensor's largest magnitude (S-2's definition), per category.
- [ ] Record the numbers and the script's command line in plan §4. If the worst logit difference exceeds 1e-5 in any category, stop and ask the owner.

## 9.5 How the two runs differ (FR-2, S-3, S-4)

- [ ] Per category, the worst logit and intermediate difference between the f32 and f64 fixtures, computed in the scratchpad script. 0.1.1's diagnostic had the f32 reference up to 7.12e-4 from the truth; record whether it reappears.
- [ ] How many prompts have an f32 and an f64 greedy continuation that differ, and at which step, and the f64 top-two gap there. A difference at a gap below 1e-3 is a near-tie (S-3), expected; one above it is a surprise and goes to the owner.

## Exit check

- [ ] Both fixture sets exist for every prompt, shapes as in the manifest, and the f64 files are F64.
- [ ] The f32 set is unchanged, or its change is recorded and explained.
- [ ] The f64 reference's noise is recorded and below 1e-5 in logits; the difference between the runs is recorded.
- [ ] The owner has reviewed the oracle's diff and the numbers; nothing is merged, tagged or pushed.
