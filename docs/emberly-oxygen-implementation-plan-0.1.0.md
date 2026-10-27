# Emberly Oxygen — Implementation Plan for Release 0.1.0

Oct 27, 2026 · @Wattanit

**Target release:** 0.1.0\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.1 (upstream), Emberly Oxygen — Technical Specification 0.1 (upstream)

This is a living working document, not part of the SFD suite: it has no document version or status and is edited in place as work proceeds. This document expands the Technical Specification's milestones M1 to M6 into phases with ordered tasks and exit checks. It owns sequencing and task breakdown only; WHAT stays in the Requirements and HOW stays in the Specification, and every task cites the IDs it satisfies. A discovery that changes HOW is recorded in §9 and absorbed by a Specification version bump, never decided here. The foundation versions above are the as-built pin for this release: if either bumps while the release is in progress, the pin is updated and the change recorded in §9.

## 1. Working rules

- **Phase order is fixed.** Each phase ends with its milestone's proof (Specification §18). The next phase does not start until the exit check passes, because later phases build on what earlier ones prove.
- **Tests before engine code.** The oracle and the harness exist before any engine code (M1, M2), so every engine task is written against a gate that has already been seen to fail. This is the order the Specification's milestones already impose.
- **One task, one commit.** Each task ends with a passing `cargo test` (or the oracle's own check for Phase 1) and a commit, so a regression is traceable to a single task.
- **Every test names the HC, FR or S item it verifies** (Specification §17), so coverage can be listed mechanically. Each task below lists the IDs its tests carry.
- **No dependency outside Specification §15.** Adding one requires a Specification bump first (Specification §15 acceptance policy).
- **Agents write code; the owner reviews at phase exits.** Rewrites are cheap; the tests are the durable asset, so a phase exit reviews the tests and results, not the implementation's elegance.

## 2. Phase 1 — Oracle (M1)

**Proves:** the reference exists. **Satisfies:** FR-1 to FR-4, FR-11, HC-3.

| # | Task | Cites |
| --- | --- | --- |
| 1.1 | Workspace skeleton: root `Cargo.toml` (edition 2024; test profile at full optimization with debug assertions kept), empty `emberly-oxygen` library and `emberly-oxygen-cli` crates, `.gitignore` for `fixtures/`, `models/` and build output, empty `results/`. | Spec §1, §16 |
| 1.2 | `oracle/` as a `uv` project with a lockfile pinning `torch`, `transformers` and `huggingface_hub`. | HC-3, FR-4, Spec §10 |
| 1.3 | Download script: fetches SmolLM2-135M at a pinned commit into `models/<model-id>/` (`config.json`, `model.safetensors`, `tokenizer.json`). It is the only step needing network access. | FR-11, FR-4, Spec §10 |
| 1.4 | `oracle/prompts.json`: at least two prompts per category (`english`, `thai`, `code`, `utf8-split`, `long`), each with an ID and category. The `long` prompts are about 512 tokens; the `utf8-split` prompts are chosen by checking that their tokens split a multi-byte character. | FR-3, Spec §10 |
| 1.5 | Fixture generator: f32 on CPU, eager attention, no gradients; per prompt saves token IDs, logits `[positions, vocab]`, embeddings `[positions, hidden]`, layer outputs `[layers, positions, hidden]`, and a 32-token greedy continuation from the oracle's own argmax loop recording the top-two gap at every step. | FR-1, FR-2, S-3, S-4, Spec §10 |
| 1.6 | Last layer-output entry labeled as post-final-norm in the stored data, as the library returns it. | FR-1, Spec §10 honesty note |
| 1.7 | `fixtures/<model-id>/manifest.json`: model ID and commit, library versions, generating machine, per-prompt records. | FR-1, FR-4, Spec §10 |

**Exit check:** running the generator produces a manifest and one safetensors file per prompt for all five categories; a second run on the same machine reproduces the token IDs and continuations exactly. The `long` fixtures are confirmed to be about 512 tokens. Fixtures are not committed; the prompt set, lockfile and pinned commit are.

## 3. Phase 2 — Harness (M2)

**Proves:** the gate can fail. **Satisfies:** FR-5 to FR-7, S-1 to S-5, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 2.1 | Define the engine surface the harness calls (encode, decode, forward over a token returning logits with optional capture, greedy continuation) as a stub in the library that returns wrong or empty values. The surface is the minimum the harness needs; it is refined by Phases 3 to 5, not designed ahead. | Spec §2, §7, §8 |
| 2.2 | Fixture reader: parses the manifest and the per-prompt safetensors files for tests. The harness has its own reading code because the engine loader does not exist yet and the oracle's output must not be read by the code under test. | FR-4, FR-7, Spec §11 |
| 2.3 | Comparison checks as pure functions with their own unit tests: maximum absolute logit error (S-1), intermediate error relative to the tensor's largest magnitude (S-2), top-1 with the near-tie exception (S-3), greedy match stopping at a near-tie (S-4). Each returns the first diverging position. | S-1 to S-4, FR-6 |
| 2.4 | Per-prompt integration tests over every `fixtures/*/manifest.json`, driven by the manifest so a new model needs no code change. | FR-5, Spec §11, §2.2 of Requirements |
| 2.5 | First-diverging-layer reporting for intermediates. | FR-6 |
| 2.6 | Pass/fail table by prompt and by category; any failing category fails the run. | S-5, FR-5 |
| 2.7 | Missing model files or fixtures fail the run naming the path or the oracle command; the harness never downloads and never runs Python. | FR-7, HC-3, Spec §11 |

**Exit check:** `cargo test` against the stub engine runs offline, prints the table with every category failing, and exits non-zero. Unit tests of the comparison functions pass, including cases at each threshold boundary (just under and just over) and a near-tie case.

## 4. Phase 3 — Loading and tokenization (M3)

**Proves:** the engine reads the model as published. **Satisfies:** FR-11 to FR-15, FR-12, HC-1.

| # | Task | Cites |
| --- | --- | --- |
| 3.1 | Error type with the variants listed in Specification §14 (I/O, malformed file, unsupported component, tensor mismatch, request rejected). Later tasks add no variant outside this list without a Specification bump. | FR-12, Spec §14 |
| 3.2 | `config.json` parsing and the supported-set check; unsupported values rejected naming the field and value. Tests use altered copies of the real configuration. | FR-12, Spec §3 |
| 3.3 | Tensor type: owned `Vec<f32>` with a row-major shape. | HC-1, Spec §3 |
| 3.4 | In-house safetensors reader: header length, JSON header, memory map, F32 copy, BF16 widening, rejection of other dtypes and of sharded weights. Unit tests build tiny safetensors files in memory, including a BF16 widening exactness case. | FR-11, FR-12, HC-1, Spec §5 |
| 3.5 | Weight binding by published names, with missing, mis-shaped and unexpected tensors rejected naming the tensor. | FR-12, Spec §5 |
| 3.6 | Read SmolLM2's `tokenizer.json` and **record the normalizer, pre-tokenizer, post-processor and decoder types it uses.** This resolves Specification §19's first open item and is recorded in §9 of this document as feedback for the Specification. | Spec §6, §19 |
| 3.7 | Byte-to-unicode map and its inverse, with round-trip unit tests over all 256 bytes. | FR-13, FR-14, Spec §6, §17 |
| 3.8 | Hand-written pre-tokenizer splitting. Unit tests on constructed strings covering contractions, digits, punctuation runs, whitespace runs and Thai. | FR-13, Spec §6 |
| 3.9 | BPE merge application by lowest rank, special-token matching first, and the post-processor's BOS behavior. | FR-13, Spec §6 |
| 3.10 | Decoding through the inverse byte map. | FR-14, Spec §6 |
| 3.11 | Streaming decoder holding back up to 3 trailing bytes, emitting U+FFFD for impossible sequences, and flushing at the end. Unit tests split a three-byte Thai character across every possible chunk boundary. | FR-15, Spec §6, §17 |
| 3.12 | Wire the real loader and tokenizer into the harness for the checks that depend only on them: encoding, decoding and the streaming check for `utf8-split` prompts. | FR-13 to FR-15, Spec §11 |

**Exit check:** for every prompt in every category, the harness's encoding and decoding checks pass and the streaming check passes on `utf8-split` prompts; the logit, intermediate and continuation checks still fail because the forward pass is still the stub. An unsupported-configuration test shows the error names the field.

## 5. Phase 4 — Forward pass (M4)

**Proves:** the math. **Satisfies:** S-1 to S-3, HC-1, HC-4.

Kernels come first and one at a time, because a kernel bug found through the full model costs a layer-by-layer search that a unit test would have settled in seconds.

| # | Task | Cites |
| --- | --- | --- |
| 4.1 | Matrix-vector product over the `[out, in]` layout. Unit test against `matrixmultiply` on random inputs at each of the model's matrix shapes (`matrixmultiply` is a dev-dependency only). | HC-1, HC-2, Spec §4, §17 |
| 4.2 | RMSNorm, softmax (max-subtraction), SiLU, elementwise add and multiply, each with hand-computed small cases (softmax sums to 1, and does not overflow on large scores). | Spec §4, §17 |
| 4.3 | RoPE with half-split pairing and f32 frequency and angle computation. Unit tests: identity at position 0, and a hand-computed rotation at position 1. | Spec §4, §17 |
| 4.4 | Key-value cache with the `[position][kv_heads × head_dim]` layout. | Spec §3 |
| 4.5 | Forward pass for one token through steps 1 to 3 of Specification §7, with grouped-query attention indexing, tied output projection and the optional residual-stream capture (off by default). | Spec §7, FR-1 |
| 4.6 | Connect to the harness: logits, intermediates, top-1. Work through any failure by the first-diverging-layer report (FR-6), shortest prompts first. | S-1 to S-3, FR-6 |
| 4.7 | Negative controls behind a test-only switch (interleaved RoPE pairing, a 2e-3 perturbation of one logit, a dropped final norm); a test asserts the harness fails on each. How the switch is compiled (a Cargo feature enabled only by those tests) is an implementation choice recorded in §9 for the Specification, because it touches the engine's public surface. | Spec §17, S-1, S-2 |

**Exit check:** logits, intermediates and top-1 checks pass in every category (S-5); all three negative controls fail the harness as intended. The measured maximum logit error and intermediate error per category are written into §8 of this document, because the roadmap and Requirements treat 1e-3 and 1e-4 as initial values to revisit after the first real measurement.

## 6. Phase 5 — Generation and CLI (M5)

**Proves:** the engine end to end. **Satisfies:** FR-16 to FR-24, S-4, S-6, S-8.

| # | Task | Cites |
| --- | --- | --- |
| 5.1 | Greedy generation per Specification §8: request validation before any computation (empty prompt, context exceeded), prompt through the forward pass, lowest-ID tie-break, EOS and maximum-length stop, streaming through a callback, decoder flush. | FR-16, FR-17, FR-18, Spec §8 |
| 5.2 | Harness greedy-continuation check including the near-tie stop. | S-4, FR-5 |
| 5.3 | `tracing` events and the `generate` → `step{pos}` → `layer{idx}` spans at the levels in Specification §14. Per-layer statistics computed only when trace is enabled. | FR-22, FR-23, S-8, Spec §14 |
| 5.4 | CLI: `generate` command, prompt from argument or standard input, text-only standard output, errors to standard error with exit codes 1 and 2, `--stats`, `-v` to `-vvv`, `--log-file`. | FR-19 to FR-21, FR-23, FR-24, FR-20, Spec §9 |
| 5.5 | CLI tests that run the built binary: stdout contains only generated text, errors reach stderr with a non-zero exit, `--stats` output appears on stderr, a context-exceeded request is rejected, stdin and argument prompts agree. | FR-19 to FR-21, FR-18, Spec §9 |
| 5.6 | Logging invariance test: the same prompt at default, `-v`, `-vv` and `-vvv` yields identical generated text. | S-8 |
| 5.7 | Reproducibility test: two runs of the same prompt produce logits within S-1 of each other and identical continuations except at near-ties. | S-6 |

**Exit check:** the full `cargo test` passes in every category, including S-4; the CLI generates a recognizable continuation for an English prompt by hand; `-vvv` on a short prompt can be followed through the decode loop by eye (FR-23).

## 7. Phase 6 — Baseline (M6)

**Proves:** the progression has a starting point. **Satisfies:** FR-8 to FR-10, S-7, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 6.1 | **Owner decision first:** choose the benchmark machine. Nothing in this phase is recorded until it is made. | S-7, Requirements §13 |
| 6.2 | Benchmark harness in `benches/` with the two workloads (32-token and 512-token prompts, each followed by 128 generated tokens), run in a fresh child process per repetition, 5 repetitions, median, minimum and maximum. Refuses to run with debug assertions enabled. | FR-8, FR-10, S-7, Spec §12 |
| 6.3 | Metrics: prompt throughput, generation throughput, time to first token, peak resident memory (via `libc`), model load time reported separately. | FR-8, Spec §12 |
| 6.4 | Record writer appending one JSON line per workload to `results/benchmarks.jsonl` with version, commit, dirty-tree flag, CPU, core count, memory, OS, build profile and timestamp. | FR-9, Spec §12 |
| 6.5 | Kernel baseline group: in-house matrix-vector product against `matrixmultiply` at the model's matrix sizes. | HC-2, Spec §12 |
| 6.6 | Run `cargo test`, then `cargo bench` on the benchmark machine, at the default log level. A failing test run stops here. | HC-4, S-7, S-8, Spec §16 |
| 6.7 | Release note `docs/releases/0.1.0.md`: what the release is, why it was expected to work, what was measured, what was learned, and any negative results. There is no previous release to compare against, so the note records this release as the baseline. | Requirements §12 |
| 6.8 | Tag `v0.1.0` after the owner's review of the results and the note. | Spec §16 |

**Exit check:** `results/benchmarks.jsonl` holds the baseline rows, every row carrying its conditions; the release note exists; the tag is applied.

## 8. Measurements to record

Filled in during the phases; they are inputs to the Specification's and Requirements' "tune with the first measured comparison" clauses and are not tuned here.

- Maximum logit error and intermediate error per category (Phase 4 exit) — for S-1 and S-2.
- Near-ties encountered in greedy continuations (Phase 5) — for S-3 and S-4.
- Run-to-run logit difference (task 5.7) — for S-6.

## 9. Upstream feedback

Recorded here and absorbed by the upstream owner at the next bump; the upstream documents are not edited from this one. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Before Phase 1 | Requirements S-3 | The text of S-3 was corrupted after "there". **Resolved:** the owner supplied the wording ("there either token passes") and corrected the Requirements file in place without a version bump, since the intended meaning was unchanged. |
| Task 1.6 | Specification §10 honesty note | The library's hidden states replace the last layer's output with the final-norm output, so the last layer's raw output is not returned. The oracle captures every layer's output with forward hooks (checked equal to the library's own hidden states for all other layers) and stores the final-norm output under its own name, `final_norm_output`. This is stricter than the note: all layer outputs are raw, and the harness compares `final_norm_output` against the engine's normalized output. |
| Task 1.5 | Specification §10 | Per-position top-two gaps are stored for the prompt positions as well as for the continuation, because S-3 needs the near-tie test at every position. |
| Task 2.3 | Requirements S-2; Specification §11 | "Tensor" in S-2 is read as one stored intermediate for one prompt, meaning one layer's `[positions, hidden]` matrix; the largest magnitude is taken over all its positions. A per-position maximum would be stricter. An all-zero reference tensor fails, because a relative error is undefined. |
| Task 2.3 | Requirements S-3, S-4 | A near-tie is a reference gap strictly less than 1e-3 (a gap of exactly 1e-3 is not one). In the greedy comparison the gap before each continuation token decides: the first near-tie stops the comparison, tokens before it must match, and the token at it may be either. |
| Task 3.6 | Specification §6, §19 | To be filled with the tokenizer component types found in `tokenizer.json`, narrowing §6. |
| Task 4.7 | Specification §17 | To be filled with how the negative-control switch is compiled in. |
| Phase 4 exit | Requirements S-1, S-2; Specification §11 | To be filled with measured errors if they suggest changing the thresholds. |

## 10. Open Questions

- **Which machine is the benchmark machine.** Resolver: owner. Blocks task 6.1 and everything after it.
- **Whether the oracle fixtures generated on the development machine are also the ones used on the benchmark machine, or regenerated there.** The Specification only requires the manifest to record the generating machine (§10). Resolver: owner, at task 6.1.
