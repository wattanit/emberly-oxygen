# Emberly Oxygen — Technical Specification

Oct 27, 2026 · @Wattanit

**Version:** 0.1\
**Status:** approved\
**Date:** 2026-10-27\
**Owner:** Wattanit\
**Upstream:** Emberly Oxygen — Requirements 0.1

This document owns HOW Emberly Oxygen is built: layout, data models, subsystems, dependencies, testing strategy and milestones. It defers WHAT and WHY to the Requirements, whose IDs it cites for every choice that satisfies one.

## 1. System layout

A Cargo workspace with two packages: the engine is a library any Rust program can use, and the command-line tool is one consumer of it.

```
emberly-oxygen/             workspace root
  Cargo.toml                workspace manifest
  crates/
    oxygen/                 package `emberly-oxygen` (library)
      src/
        lib.rs
        tensor.rs           tensor type and math kernels (§4)
        loader.rs           safetensors and config reading (§5)
        tokenizer.rs        byte-level BPE encode/decode (§6)
        model.rs            transformer forward pass (§7)
        generate.rs         greedy generation and streaming (§8)
      tests/                correctness harness (§11)
      benches/              benchmark harness (§12)
    oxygen-cli/             package `emberly-oxygen-cli` (binary `emberly-oxygen`)
      src/
        main.rs             entry point: run the command, map errors to exit codes
        cli.rs              argument definitions and the generate command (§9)
      tests/                command-line tests (§9)
  oracle/                   Python reference oracle (§10)
  fixtures/                 expected outputs, git-ignored, one folder per model
  models/                   downloaded model files, git-ignored
  results/                  cumulative benchmark record (§12)
  docs/                     this suite and release notes
```

- The library package is `emberly-oxygen` (Requirements §12) and has no command-line dependencies. The CLI uses only its public API, so anything the CLI does, another program can do.
- The binary is named `emberly-oxygen` (Requirements §12). Its package is `emberly-oxygen-cli`, because two packages in one workspace cannot share a name.
- The CLI holds no engine logic, so the harnesses test the library directly and still cover exactly what the binary runs.

## 2. Execution model

- **Single-threaded and synchronous.** One request, one sequence, one thread, no async runtime. This is the simplest correct model and the baseline later releases measure against; S-6 already permits parallel arithmetic when a release introduces it.
- **Prompt tokens go through the generation path.** The prompt is fed one token at a time through the same forward pass used for generation; batched prompt processing is out of scope (Requirements §2.3). One code path means one thing to verify.
- **Past keys and values are cached.** Each layer stores the key and value vectors of every processed position, so each new token computes only its own row. Recomputing the whole prefix per token would make the 512-token benchmark prompt (§12) quadratically slow and teaches nothing the cache does not; the cache is part of the baseline, not an optimization.
- **Streaming by callback.** Generation calls a caller-supplied function with each completed piece of text (FR-17), so the CLI prints as tokens arrive and the harnesses collect output without a separate code path.

## 3. Data models

**Tensor.** A contiguous `Vec<f32>` with a shape, row-major. f32 only: weights stored in bfloat16 are widened to f32 at load, which is exact, so engine and oracle compute from identical values (FR-2).

**Model configuration.** Parsed from the model's `config.json`. The supported set is checked at load and anything else is rejected naming the field (FR-12):

| Field | Supported value |
| --- | --- |
| `model_type` | `llama` |
| `hidden_act` | `silu` |
| `attention_bias`, `mlp_bias` | `false` |
| `rope_scaling` | `null` |
| `rope_interleaved` | `false` or absent (half-split pairing) |
| `tie_word_embeddings` | `true` |
| `pretraining_tp` | `1` or absent |

Sizes (hidden size, layers, heads, key-value heads, head dimension, intermediate size, vocabulary, context length), `rope_theta` and `rms_norm_eps` are read from the file, not hard-coded, so a model with the same architecture but different sizes needs no code change.

**Weights.** Token embedding table (also used as the output projection, because embeddings are tied); per layer: attention norm, query, key, value and output projections, MLP norm, gate, up and down projections; final norm. Matrices keep their published `[out, in]` layout, so each output element is a dot product over one contiguous row, and no conversion step exists (FR-11).

**Key-value cache.** Per layer, one growing `Vec<f32>` for keys and one for values, each laid out `[position][kv_heads × head_dim]`. Appending a position is a single extend; reading position `t` is a contiguous slice.

**Tokens.** Token IDs are `u32`.

## 4. Tensor operations and kernels

The operation set is exactly what the forward pass needs: matrix-vector product, RMSNorm, softmax, rotary position embedding (RoPE), SiLU, elementwise add and multiply. Each is a plain loop over slices with f32 accumulation in index order, and no `unsafe`, SIMD or threading (HC-1; optimization is out of scope, Requirements §2.3).

- **Matrix-vector product.** Output element `i` is the dot product of weight row `i` with the input, matching the `[out, in]` layout (§3).
- **RMSNorm.** `x · w / sqrt(mean(x²) + eps)`, with `eps` from the configuration.
- **Softmax.** Subtracts the maximum before exponentiating, so large scores cannot overflow.
- **RoPE.** Half-split pairing: dimension `i` rotates with dimension `i + d/2`, at frequency `theta^(-2i/d)`. Frequencies, angles, sines and cosines are computed in f32, not f64, because the reference computes them in f32 and a more precise angle is a different angle.

The reference sums in a different order, so results differ in the last bits; S-1 and S-2 bound that difference.

## 5. Model loading

A model is a directory holding `config.json`, a single `model.safetensors` and `tokenizer.json`, exactly as published (FR-11).

- **Format.** The safetensors file is read in-house (HC-1): an 8-byte little-endian header length, a JSON header mapping each tensor name to dtype, shape and byte range, then raw data. The file is memory-mapped and each tensor copied into an owned f32 tensor.
- **Dtypes.** F32 is copied; BF16 is widened to F32, which is exact. Any other dtype is rejected naming the tensor and dtype (FR-12).
- **Names.** Tensors are found by their published names (`model.embed_tokens.weight`, `model.layers.{i}.self_attn.q_proj.weight`, and so on). A missing tensor, a tensor whose shape disagrees with the configuration, or an unexpected tensor is rejected naming it, because any of these means the file is not the model the configuration describes.
- **Not supported:** weights split across several files. SmolLM2-135M ships as one file; sharding is rejected with an error.

## 6. Tokenizer

The tokenizer is byte-level BPE, read from `tokenizer.json` and implemented in-house (HC-1).

- **Encoding.** Special tokens are matched first. The remaining text is split by the pre-tokenizer that `tokenizer.json` declares, each piece's bytes are mapped to the byte-level alphabet, and BPE merges are applied lowest rank first until no ranked pair remains. Pieces then map to IDs. The splitting pattern is hand-written rather than run through a regex crate: the standard byte-level pattern uses lookahead, which Rust's `regex` crate does not support, and hand-writing it is part of the point.
- **Components.** Only the normalizer, pre-tokenizer, post-processor and decoder types SmolLM2's `tokenizer.json` actually uses are implemented; any other type is rejected naming it. Whether a beginning-of-sequence token is added follows the file's post-processor, so the engine and reference agree by construction (FR-13).
- **Decoding.** IDs map to piece strings, then back to bytes through the inverse byte map (FR-14).
- **Streaming decoder.** Decoded bytes accumulate in a buffer. After each token, the longest valid UTF-8 prefix is emitted and a trailing incomplete sequence of up to 3 bytes is held back (FR-15). Bytes that cannot begin or continue a valid sequence become U+FFFD, as the reference's decode does; at end of generation the held bytes are flushed the same way.

## 7. Forward pass

One call processes one token at position `p` and returns logits over the vocabulary.

1. `x` = embedding row of the token.
2. For each layer:
   1. `h` = RMSNorm(`x`, attention norm weight).
   2. Query, key and value = their projections of `h`; RoPE at position `p` is applied to query and key.
   3. Key and value are appended to this layer's cache.
   4. Each query head attends over cached positions `0..=p` of its key-value head (head index divided by the group size, the number of query heads per key-value head): scores are dot products scaled by `1/sqrt(head_dim)`, softmax, then a weighted sum of values.
   5. `x` += output projection of the concatenated heads.
   6. `h` = RMSNorm(`x`, MLP norm weight); `x` += down(SiLU(gate(`h`)) × up(`h`)).
3. Logits = embedding table × RMSNorm(`x`, final norm weight), using the tied embeddings.

The residual stream `x` is never normalized in place; only the copies fed into attention, the MLP and the output are. An optional capture records `x` after the embedding and after each layer for the correctness harness (FR-1, FR-6); it is off in normal runs.

## 8. Generation

1. Encode the prompt. Reject it before any computation if the prompt is empty or prompt length plus maximum new tokens exceeds `max_position_embeddings` (FR-18).
2. Feed the prompt tokens through the forward pass in order (§2).
3. Take the highest logit as the next token; on an exact tie, the lowest ID wins, as the reference's argmax does (FR-16).
4. Pass the token to the streaming decoder and emit any completed text to the callback (FR-17).
5. Stop after the maximum number of new tokens or when the token is the configuration's `eos_token_id`; flush the decoder.

## 9. Command-line interface

```
emberly-oxygen generate --model <dir> --max-new-tokens <n> [--stats] [-v|-vv|-vvv] [--log-file <path>] [PROMPT]
```

- The prompt is read from standard input when no argument is given (FR-19).
- Generated text goes to standard output and nothing else; errors go to standard error with a non-zero exit code (FR-20).
- `--stats` prints prompt token count, generated token count, time to first token and generation tokens per second to standard error (FR-21). The CLI package's own tests run the built binary and check FR-19 to FR-21.

## 10. Reference oracle

The oracle lives in `oracle/`, a Python project managed by `uv` with a lockfile pinning `torch`, `transformers` and `huggingface_hub`, so expected outputs can be regenerated with the same library versions.

- **Model revision.** The model is downloaded at a pinned commit of its repository; the same files are the engine's model directory, so both read identical bytes (FR-11). The download script is the only step that needs network access.
- **Settings.** f32 on CPU (FR-2), eager attention so no fused kernel changes the arithmetic, gradients disabled.
- **Prompt set.** `oracle/prompts.json` lists each prompt with an ID and a category: `english`, `thai`, `code`, `utf8-split` (prompts whose tokens split multi-byte characters) and `long` (about 512 tokens, the longest benchmark prompt, §12). Each category has at least two prompts (FR-3).
- **Greedy continuation.** 32 tokens, produced by the oracle's own argmax loop rather than a library generate call, so it can record the gap between the two highest logits at every step for near-tie detection (S-3, S-4).
- **Output.** `fixtures/<model-id>/manifest.json` records the model ID and commit, library versions, the generating machine, and per prompt: ID, category, text, token IDs, continuation IDs and text, and top-two gaps. One safetensors file per prompt holds logits `[positions, vocab]`, embeddings `[positions, hidden]` and layer outputs `[layers, positions, hidden]` (FR-1). Fixtures are model-agnostic in shape, so adding a model adds a folder (Requirements §2.2).
- **Honesty note.** In the reference library, the last entry of the per-layer outputs is taken after the final norm, not straight from the last layer. The oracle stores what the library returns and labels that entry, and the harness compares it against the engine's normalized output.

Logits dominate fixture size: about 200 KB per position, so the long prompts are about 100 MB each. Fixtures are therefore kept out of git: \`fixtures/\` is git-ignored and filled by running the oracle locally. What regenerates them is committed instead: the prompt set, the oracle's lockfile and the pinned model commit. Fixtures generated on a different machine may differ from another machine's in the last bits of the reference's own arithmetic; that stays far inside S-1, and the manifest records which machine produced them.

## 11. Correctness harness

Rust integration tests in `tests/`, run with `cargo test`; the test profile is compiled with optimization and debug assertions on, so a full run takes seconds instead of minutes without losing overflow checks (FR-7).

- The harness reads every `fixtures/*/manifest.json`, so a new model's fixtures are picked up without code changes.
- Per prompt it checks encoding (FR-13), decoding (FR-14), logits (S-1), intermediates (S-2), top-1 token (S-3) and greedy continuation (S-4).
- For `utf8-split` prompts it streams the expected continuation through the streaming decoder and checks that the concatenated chunks equal the full decode and that no chunk contains a replacement character the full decode lacks (FR-15).
- It prints a pass/fail table by prompt and by category; any failing category fails the run (S-5). Each failure reports the first diverging position and, for intermediates, the first diverging layer (FR-6).
- Model files are found in `models/<model-id>/` (git-ignored, filled by the oracle's download script). If they are missing, the run fails naming the path; it never downloads (FR-7). Missing fixtures fail the run the same way, naming the oracle command that generates them.

## 12. Benchmark harness

A custom harness in `benches/`, run with `cargo bench`, which builds with the optimized profile. It also refuses to run if debug assertions are enabled, as a second guard (FR-10).

- **Workloads.** A short prompt of 32 tokens and a long prompt of 512 tokens, each followed by 128 generated tokens. Initial values; changing them starts a new comparable series in the record.
- **Metrics (FR-8).**
  - Prompt throughput: prompt tokens divided by prompt time.
  - Generation throughput: generated tokens divided by generation time.
  - Time to first token: from receiving the prompt string to the first generated token.
  - Peak resident memory.

  Model load time is reported separately, so it does not distort time to first token.
- **Isolation.** Each workload runs in a fresh child process, because peak resident memory is a per-process high-water mark and would otherwise carry over.
- **Repetition.** Each workload runs 5 times; median, minimum and maximum are recorded (S-7).
- **Record (FR-9).** One JSON line per workload is appended to `results/benchmarks.jsonl`: package version, commit and whether the tree was dirty, CPU model, core count, memory, OS, build profile, timestamp, and results.
- **Kernel baseline (HC-2).** A separate group times the in-house matrix-vector product against the test-only linear algebra crate at the model's matrix sizes.

## 13. Configuration

There is no configuration file. Run settings come from command-line flags (§9) and model settings from the model's own `config.json` (§3); nothing else is configurable, because nothing else is required.

## 14. Error handling and logging

- The library returns one error type with a variant per failure class: I/O, malformed file, unsupported model or tokenizer component (naming the field and value, FR-12), tensor mismatch (naming the tensor), and request rejected (empty prompt, context exceeded, FR-18).
- User input never causes a panic. Panics are reserved for broken internal invariants, which are bugs.
- The binary prints the error to standard error and exits with code 1; command-line usage errors exit with code 2 (FR-20). Text already streamed before a mid-generation error stays printed; the error follows on standard error.

**Logging.** The library emits events through the `tracing` crate and never prints (FR-22); the CLI installs the subscriber that formats and routes them. Verbosity is set by repeating `-v` (FR-23):

| Flag | Level | Content |
| --- | --- | --- |
| none | warn | warnings and errors |
| `-v` | info | model and configuration summary, load time, prompt token count, run summary with throughput |
| `-vv` | debug | one event per decode step: position, token ID, decoded text, top 5 candidates with logits, step time; prompt-feeding steps are marked as such |
| `-vvv` | trace | per layer within each step: residual-stream norm after attention and after the MLP, and for each head the position receiving the most attention weight |

- Generation, steps and layers are nested spans (`generate` → `step{pos}` → `layer{idx}`), so every event carries its place in the decode loop.
- `--log-file <path>` sends the same events to a file instead of standard error (FR-24). Standard output stays generated text only (FR-20).
- Per-layer statistics are computed only when the trace level is enabled, so lower levels do no extra arithmetic. No statistic feeds back into the computation, so output is identical at every level (S-8). The benchmark harness runs at the default level.
- At trace level a 512-token prompt produces tens of thousands of events. It is meant for short prompts, or with `--log-file`.

## 15. Dependencies

Locked initial set:

| Crate | Used by | Purpose |
| --- | --- | --- |
| `clap` | CLI package only | command-line parsing |
| `tracing-subscriber` | CLI package only | formatting and routing log events |
| `thiserror` | library | error type |
| `tracing` | library | emitting log events and spans |
| `serde`, `serde_json` | library | reading `config.json`, `tokenizer.json` and the safetensors header |
| `memmap2` | library | memory-mapping the weights file |
| `matrixmultiply` | library tests, benches only | kernel oracle and baseline (HC-2) |
| `libc` | library benches only | reading peak resident memory |

All are general application plumbing (HC-1), except `matrixmultiply`, which is a dev-dependency and therefore never linked into the engine binary (HC-2). The oracle's Python dependencies are pinned in its own lockfile (§10).

Acceptance policy: an engine dependency must be general plumbing under HC-1; anything that performs inference work is rejected. Adding any dependency requires a version bump of this section.

## 16. Build and release

- Stable Rust, edition 2024. Release builds use Cargo's default optimized profile; the test profile is raised to full optimization with debug assertions kept (§11).
- Versions follow semantic versioning: before 1.0, a release introducing a new technique bumps the minor version and a fix bumps the patch version (Requirements §12).
- Release steps, in order:
  1. `cargo test` passes (HC-4).
  2. `cargo bench` runs on the benchmark machine and appends to the record.
  3. The release note is written to `docs/releases/<version>.md`.
  4. The commit is tagged `v<version>`.

  Benchmarks follow the correctness run, so no result from a failing build enters the record (HC-4).

## 17. Testing strategy

- **Unit tests per kernel.** Each kernel is checked against hand-computed small cases (softmax sums to 1, RoPE at position 0 is the identity) and, for matrix-vector product, against the test-only linear algebra crate on random inputs (HC-2).
- **Unit tests for the tokenizer.** Byte-map round trips, merge order on known pieces, and the streaming decoder's hold-back on split three-byte characters (FR-15).
- **Correctness harness.** End-to-end comparison against the oracle's fixtures (§11), the gate every release must pass (HC-4).
- **Negative controls.** The harness must be seen to fail. Test-only switches deliberately break the engine (interleaved RoPE pairing, a 2e-3 perturbation of one logit, a dropped final norm), and a test asserts that the harness fails on each. A check that has never failed has not been shown to work.
- Every test names the HC, FR or S item it verifies, so coverage of the Requirements can be listed mechanically.

## 18. Milestones

&#91;embedded content: milestones · M1 to M6, ending in release 0.1.0\]

- **M1 — Oracle.** The oracle generates fixtures for every prompt category from the committed prompt set. *Proves the reference exists.*
- **M2 — Harness.** The harness runs against a stub engine, reports per category, and fails cleanly. *Proves the gate can fail.*
- **M3 — Loading.** Model loading and tokenization pass FR-11 to FR-15. *Proves the engine reads the model as published.*
- **M4 — Forward.** Logits, intermediates and top-1 tokens pass S-1 to S-3 in every category, and the negative controls fail as intended. *Proves the math.*
- **M5 — Generation.** Greedy continuations pass S-4, and the CLI meets FR-16 to FR-21. *Proves the engine end to end.*
- **M6 — Baseline.** First benchmark record on the benchmark machine, release note written, tagged 0.1.0. *Proves the progression has a starting point.*

## 19. Open Items

- **Tokenizer components.** The exact normalizer, pre-tokenizer and decoder types are confirmed from SmolLM2's `tokenizer.json` at M3; §6 is narrowed to match. Resolver: M3 implementation.
- **Benchmark platform.** Peak memory and machine description (§12) assume Linux or macOS. If the benchmark machine runs another OS, §12 changes. Resolver: owner, with the benchmark machine choice (Requirements §13).
