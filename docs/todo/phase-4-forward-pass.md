# Phase 4 — Forward pass: To-do

Working checklist for Phase 4 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.0.md) (milestone M4: the math). Tick items as they land; one task is one commit, each ending with a passing `cargo test --no-fail-fast` whose **whole summary is read** (not filtered to a few lines), because Phase 3 showed a red test binary can hide behind a filter. IDs cite the Requirements and Technical Specification.

Proves: the engine's logits, intermediates and top-1 tokens match the oracle in every category (S-1 to S-3, S-5), and the harness is seen to fail when the engine is deliberately broken. Satisfies S-1 to S-3, HC-1, HC-4.

Layout: kernels in `crates/oxygen/src/ops.rs`, the key-value cache in `kv_cache.rs`, the forward pass in `forward.rs`; `Session::step` in `model.rs` calls it. Kernels are plain functions over slices, with no `unsafe`, SIMD or threading (Spec §4). Unit tests sit beside each kernel and need no model files, except the ones that say so.

Dependencies: `matrixmultiply` as a **dev-dependency only** (Spec §15 lists it for tests and benches, so no Specification bump; it is never linked into the engine, HC-2). Nothing else.

Out of scope: generation and sampling (Phase 5), so the **greedy column keeps failing** at this phase's exit; speed (Phase 6), except for the one measurement in 4.1 below.

## Decision point to resolve at 4.1: how long a full run takes

The 12 prompts hold 2,596 positions, and every position computes all layers and the 49,152-row output projection: about 134 M multiply-adds per token, **about 349 G in a full harness run**. Spec §4 asks for "f32 accumulation in index order", and a single running sum is a chain of dependent adds, which the compiler may not reorder or vectorize. If that runs near 1 G multiply-add per second, one `cargo test` takes **6 to 8 minutes**, against Spec §11's "seconds". Task 4.1 measures the real rate. The options, to be decided by the owner and not by this to-do:

- Accept the minutes; the long prompts dominate, and the loop is run per prompt, not in total, while debugging.
- Compute each dot product with a few independent partial sums (still no `unsafe` or SIMD, still plain loops, but a different summation order). The reference already sums in a different order and S-1 and S-2 bound the difference (Spec §4, last line), so it fits the tolerances. It would change the wording "in index order" in Spec §4, so it needs a Specification bump first.
- Leave `long` prompts to a separate, slower test target and run the other categories by default. This weakens S-5 for the default run and is the least preferred.

## 4.1 Matrix-vector product and its speed (HC-1, HC-2, Spec §4, §17)

- [x] `ops::matvec(weight, input, output)`: output element `i` is the dot product of weight row `i` with the input, matching the published `[out, in]` layout; f32 accumulation in index order
- [x] Shape errors are impossible by construction for the engine's own calls; the function asserts input and output lengths (a mismatch is an internal bug, Spec §14)
- [x] Add `matrixmultiply` to `[dev-dependencies]`
- [x] Unit tests against `matrixmultiply::sgemm` (as an `[out, in] × [in, 1]` product) on seeded random inputs, at every shape the model uses: `[576, 576]`, `[192, 576]`, `[1536, 576]`, `[576, 1536]` and the tied output projection `[49152, 576]`; the error bound is stated in the test and is far below S-1
- [x] Hand-computed small cases (identity matrix, a 2×3 example) and zero-sized edge cases
- [x] **Measure:** a test that prints (not asserts) the time and multiply-adds per second of `matvec` at the five shapes, and the estimated time of a full harness run (349 G multiply-adds divided by the measured rate); record the figures in the commit message and below
- [x] **Stop and report the figures to the owner before starting 4.2.** Reported; **decision: option 1, accept the index-order loop** and the roughly 2-minute full run. It keeps this version's logic pure and by-the-book; speed techniques are added and measured one at a time in later versions, with the three options above, if the estimate exceeds about 60 seconds
- [x] Check: every test names HC-1, HC-2 or Spec §4

Measured (Apple M5, the test profile, `cargo test -p emberly-oxygen --lib -- --ignored --nocapture matvec_speed`), matrix-vector part only, so attention, norms and RoPE come on top (up to about 13% more on the 500-token prompts):

| Dot product | multiply-adds per second | per token | full harness run (2,596 positions) |
| --- | --- | --- | --- |
| Index order (Spec §4 as written) | 2.3 to 3.0 G | 47.5 ms | about 123 s |
| 4 partial sums | 5.7 to 5.8 G | 23.3 ms | about 60 s |
| 8 partial sums | 14 to 16 G | 8.8 ms | about 23 s |

The index-order loop is about three times faster than the 1 G per second guessed above, so a full run is about 2 minutes, not 6 to 8. It still exceeds the 60-second threshold set above, hence the stop.

## 4.2 RMSNorm, softmax, SiLU, add, multiply (Spec §4, §17)

- [x] `rms_norm(x, weight, eps, out)`: `x · w / sqrt(mean(x²) + eps)`, with `eps` from the configuration; the reference computes the mean of squares and the reciprocal square root in f32 and then multiplies by the weight, so use that order
- [x] `softmax(x)` in place: subtract the maximum first, then exponentiate and divide by the sum
- [x] `silu(x)` in place: `x / (1 + exp(-x))`
- [x] `add_assign(a, b)` and `mul_assign(a, b)`: elementwise, lengths asserted
- [x] Hand-computed small cases for each; RMSNorm of a known vector with unit weights, with a weight and with `eps` visible (an all-zero input must not produce NaN)
- [x] Softmax: sums to 1 within 1e-6, equal inputs give equal outputs, a large-score case (for example `[1000, 1000, 999]`) does not overflow and a very negative score underflows to 0 rather than NaN, one-element input gives `[1.0]`
- [x] SiLU at 0, at a large positive value (≈ x) and at a large negative value (≈ 0, no NaN from `exp` overflow)
- [x] Check: every test names Spec §4 or §17

## 4.3 RoPE (Spec §4, §17)

- [ ] `rope(x, position, heads, head_dim, theta)`: half-split pairing, dimension `i` rotates with `i + head_dim/2`, applied per head; frequencies `theta^(-2i/head_dim)`, angles `position × frequency`, sines and cosines all computed in **f32** (a more precise angle is a different angle, Spec §4)
- [ ] Applied in place to the query (9 heads) and key (3 heads) of one token
- [ ] Unit tests: identity at position 0; a hand-computed rotation at position 1 for a small `head_dim` (for example 4, with `theta = 10000`); a rotation preserves each pair's length; the first pair at position `p` rotates by exactly `p` radians
- [ ] Compare against values from the reference for SmolLM2's real shape (`head_dim = 64`, `theta = 100000`) at a few positions: the oracle gets a small addition if the harness cannot cover it (see 4.6), recorded in the plan's §9 and not decided here
- [ ] Check: every test names Spec §4 or §17

## 4.4 Key-value cache (Spec §3)

- [x] `KvCache`: per layer, one growing `Vec<f32>` for keys and one for values, laid out `[position][kv_heads × head_dim]`; appending a position is one extend; reading position `t` is a contiguous slice
- [x] Appending and reading are checked by length: appending a slice of the wrong size is an internal bug and asserts
- [x] The cache knows its length in positions and refuses to grow past the model's context length with a `Rejected` error (Spec §14, FR-18), never a panic or silent wrap
- [x] Check: unit tests for append, read-back of each position, per-layer separation, the context limit

## 4.5 Forward pass for one token (FR-1, Spec §7)

- [x] `Session` holds a `KvCache` and the next position; `Session::step(token)` runs Spec §7 steps 1 to 3 and returns logits
- [x] Step 1: embedding row of the token; an ID outside the vocabulary is `Rejected` naming it
- [x] Step 2, per layer: RMSNorm with the attention norm; query, key, value projections; RoPE on query and key; append key and value; each query head attends over cached positions `0..=p` of key-value head `head / (heads / kv_heads)`, scores scaled by `1/sqrt(head_dim)`, softmax, weighted sum of values; output projection added to `x`; RMSNorm with the MLP norm; `x += down(silu(gate(h)) × up(h))`
- [x] Step 3: final RMSNorm, then the tied embedding table as the output projection, giving `vocab` logits
- [x] The residual stream `x` is never normalized in place; only the copies fed to attention, the MLP and the output are
- [x] Capture (off by default): embedding, raw output of every layer, and the final-norm output, exactly as the stub's `Capture` documents them; capture off allocates nothing extra
- [x] A step after the context is full is `Rejected`, and the failed step leaves the session unchanged
- [x] Unit tests on a **tiny synthetic model** (a few dimensions, weights chosen by hand): a one-layer, one-head case small enough to compute by hand; grouped-query indexing with 4 query heads over 2 key-value heads (heads in the same group read the same key-value head, heads in different groups do not); causality (a later token never changes an earlier position's logits); capture shapes and that the capture's last layer plus the final norm reproduces the logits
- [x] Check: no `unsafe`, no threads, no dependency added; every test names FR-1 or Spec §7
- [x] The harness's first run against the real engine: logits, top-1 and intermediates pass for all 12 prompts; largest logit error 2.36e-4 and largest relative intermediate error 5.9e-6; the full run takes about 125 s (the figure estimated in 4.1 was about 123 s for the matrix-vector part)
- [x] `stub_smoke.rs` updated: the model step is real now, so it checks finite logits of the vocabulary's length and one capture per layer; only `generate` is still a stub

## 4.6 Connect to the harness (S-1 to S-3, FR-6)

- [ ] The harness already feeds the engine one token at a time with capture on; nothing in `tests/common/` should need to change except for the checks below
- [ ] Run the shortest prompts first (`english-short`, `utf8-split-emoji`, `english-medium`) and read the first-diverging-layer report (FR-6) on every failure; fix the engine, never the tolerance
- [ ] Work through each failure by the layer report: embedding first, then layer 0, and so on; a bug is found in the first layer that is wrong, and a unit test in `ops` or `forward` is added for it so it cannot come back
- [ ] Then the remaining categories, `long` last
- [ ] Record the **measured maximum logit error and maximum relative intermediate error per category** (plan §8, Phase 4 exit); if they suggest changing 1e-3 or 1e-4, record that in plan §9 and do not change the thresholds here
- [ ] Record how long the full run takes, and compare it with the 4.1 estimate
- [ ] Check: with a real forward pass, the harness table shows encode, decode, streaming, logits, top-1 and intermediates passing in every category; greedy still fails (Phase 5)

## 4.7 Negative controls (Spec §17, S-1, S-2)

The harness must be seen to fail. Three deliberate faults, each behind a test-only switch, and a test asserts the harness fails on each.

- [ ] **Interleaved RoPE pairing:** dimension `2i` pairs with `2i + 1` instead of `i` with `i + d/2`; the intermediates (and so S-2) must fail
- [ ] **A 2e-3 perturbation of one logit:** just over S-1's 1e-3; the logits check must fail and name the position
- [ ] **A dropped final norm:** the logits are computed from the un-normalized residual stream; logits and final-norm intermediate must fail
- [ ] How the switch is compiled in is an implementation choice recorded in plan §9 for the Specification, because it touches the engine's public surface. Candidates: a Cargo feature that only tests enable (for example through a self-referencing dev-dependency), or a hidden constructor. The switch must be impossible to turn on in a normal build and must not change any result when off
- [ ] Each test runs a short prompt through `check_prompt` with the fault on and asserts a specific check fails (and that with the fault off it passes), so a fault that does nothing is itself caught
- [ ] Check: every test names S-1, S-2 or Spec §17

## Phase exit check

- [ ] Logits, intermediates and top-1 checks pass for every prompt in every category (S-5); encode, decode and streaming still pass; greedy still fails
- [ ] All three negative controls make the harness fail as intended, and each passes with the fault off
- [ ] Measured maximum logit error and intermediate error per category are written into plan §8
- [ ] The how-the-switch-is-compiled decision is recorded in plan §9
- [ ] Every test added in this phase names the HC, FR, S or Spec item it verifies
- [ ] `cargo test --no-fail-fast` time recorded, `cargo build` has no warnings, and the summary of every test binary was read
- [ ] README status line updated (Phase 4 done; Phase 3 and 4 summaries)
- [ ] Owner reviews the table output, the measured errors and the negative-control tests before Phase 5 starts

## Notes to carry forward

- Partial sums are a ready-measured candidate technique for a later version: the 4.1 table shows 2x (4 sums) and 5x (8 sums) on the matrix-vector product alone. Adopting it needs a Specification change to §4's "in index order" wording and a version note with its own measurement.
- Phase 5 replaces `generate`: greedy decoding over `Session::step`, with the streaming decoder, and the CLI.
- The 4.1 speed measurement is not a benchmark and is not recorded in `results/`; benchmarking starts in Phase 6.
