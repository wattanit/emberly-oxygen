# Phase 11 — Sampling reference: To-do

Working checklist for Phase 11 of the [implementation plan](../emberly-oxygen-implementation-plan-0.2.0.md) (the first part of milestone M9: the oracle writes the sampling distributions and the harness can read them). Tick items as they land; one task is one commit.

Proves: every prompt has a stored adjusted distribution for every setting, checked by an independent recomputation. Satisfies FR-31, FR-4, HC-3.

Layout: `oracle/sampling_settings.json` and `oracle/sampling_reference.py` (run as `cd oracle && uv run python sampling_reference.py`); output beside the f64 fixtures in `fixtures/smollm2-135m/` (git-ignored). Format: Specification §10.

Dependencies: none new.

Out of scope: any engine change, any rerun of the model, chat, the instruct model.

## Decision points for the owner

1. **Which settings. (applies to 11.1.)** *Recommendation:* each control alone plus one combination of all four: temperature 0.7; top-k 40; top-p 0.9; penalty 1.3; all four together. Five settings across twelve prompts is 60 distributions, about 30 MB. Initial; more are added when a bug shows a gap.

## 11.1 Settings (FR-26, FR-31)

- [x] Write `oracle/sampling_settings.json` with the five settings above, each named.

## 11.2 The script (FR-31, FR-4, HC-3)

- [x] `sampling_reference.py` reads the manifest and each prompt's f64 file, takes the last position's logits, applies `RepetitionPenaltyLogitsProcessor`, `TemperatureLogitsWarper`, `TopKLogitsWarper`, `TopPLogitsWarper` in that order with the prompt's IDs as the context, and writes the f64 softmax.
- [x] Start-up check that the processors return f64; shown to fail when fed f32.
- [x] Writes `<id>.sampling.safetensors` and the manifest's `sampling` entry with each setting's cutoff margin.

## 11.3 Independent recomputation (FR-31)

- [x] A scratchpad numpy f64 script recomputes one setting per control from the textbook definition and compares. Record any difference (ties at k, the top-p boundary) in the plan's section 5.

## 11.4 Run and record (FR-31, FR-4)

- [x] Run; record the time, file sizes, that each distribution sums to 1 within 1e-12, and each setting's support size.

## 11.5 Harness reader (FR-4, FR-7)

- [x] Tests first: a missing sampling file, a wrong shape and a missing manifest entry each fail naming the command that generates them. Then the reader.

**Exit check:** as in the plan's section 2.
