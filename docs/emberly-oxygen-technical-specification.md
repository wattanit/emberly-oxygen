# Emberly Oxygen — Technical Specification

Oct 27, 2026 · @Wattanit

**Version:** 0.3\
**Status:** draft\
**Date:** 2026-11-08\
**Owner:** Wattanit\
**Upstream:** Emberly Oxygen — Requirements 0.4

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
        tensor.rs           tensor type (§3)
        ops.rs              math kernels (§4)
        config.rs           model configuration (§3)
        safetensors.rs      safetensors reading (§5)
        weights.rs          named weights and their checks (§5)
        model.rs            a loaded model (§3)
        tokenizer/          byte-level BPE encode/decode and streaming (§6)
        forward.rs          transformer forward pass (§7)
        kv_cache.rs         key-value cache (§3)
        generate.rs         greedy generation and streaming (§8)
        error.rs            the error type (§14)
        fault.rs            negative-control faults (§17)
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

- **Single-threaded and synchronous.** One request, one sequence, one thread, no async runtime. This is the simplest correct model and the baseline later releases measure against. It stays single-threaded in 0.2 so that the kernel change (§4) is measured alone (FR-25); S-6 already permits parallel arithmetic when a release introduces it.
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
| `eos_token_id` | one whole number inside the vocabulary (some models list several; any other form is rejected as unsupported naming the field, an ID outside the vocabulary as malformed) |

Sizes (hidden size, layers, heads, key-value heads, head dimension, intermediate size, vocabulary, context length), `rope_theta` and `rms_norm_eps` are read from the file, not hard-coded, so a model with the same architecture but different sizes needs no code change.

**Weights.** Token embedding table (also used as the output projection, because embeddings are tied); per layer: attention norm, query, key, value and output projections, MLP norm, gate, up and down projections; final norm. Matrices keep their published `[out, in]` layout, so each output element is a dot product over one contiguous row, and no conversion step exists (FR-11).

**Key-value cache.** Per layer, one growing `Vec<f32>` for keys and one for values, each laid out `[position][kv_heads × head_dim]`. Appending a position is a single extend; reading position `t` is a contiguous slice.

**Tokens.** Token IDs are `u32`.

## 4. Tensor operations and kernels

The operation set is exactly what the forward pass needs: matrix-vector product, RMSNorm, softmax, rotary position embedding (RoPE), SiLU, elementwise add and multiply. Each is a plain loop over slices with f32 accumulation in index order, except the matrix-vector product (below). None uses `unsafe`, SIMD intrinsics or threading (HC-1, FR-25). The other operations stay plain: the matrix-vector product was about 94% of a step in release 0.1.0 and, after the partial sums, is about 86% of a short step and 58% of a long one (release note 0.1.1, estimated from the kernel rows). On long prompts the rest, mostly attention over the cache, is therefore a real share, about 42%; whether to speed it is a later release's decision, taken after profiling.

- **Matrix-vector product.** Output element `i` is the dot product of weight row `i` with the input, matching the `[out, in]` layout (§3). The dot product keeps 16 independent f32 partial sums (FR-25): sum `j` adds the elements whose index is `j` modulo 16, each in index order; the 16 sums are then added to one another sequentially from sum 0, and the final `in mod 16` elements are added in index order after them. The count of 16 was chosen in release 0.1.1 by a sweep of 4, 8, 16 and 32 sums, with the rule fixed before measuring: the lowest estimated kernel time per token, ties within 2% going to the smaller width. It beat 8 sums by about 20% and 32 sums everywhere; it remains tunable with use. Index-order addition was a chain of dependent adds, each waiting for the last, so the processor could not overlap them; independent sums remove the wait and let the compiler use vector registers, from safe code alone. Release 0.1.1 measured the kernel at 6.4× to 8.3× the index-order loop at the model's five shapes.
- **Prohibited in the kernel, for now.** `unsafe`, SIMD intrinsics, threads and tiling stay out (Requirements §2.2): each is its own technique and is measured as its own release. If safe code does not vectorize as expected, the implementation raises the issue with the owner instead of reaching for `unsafe`.
- **RMSNorm.** `x · w / sqrt(mean(x²) + eps)`, with `eps` from the configuration.
- **Softmax.** Subtracts the maximum before exponentiating, so large scores cannot overflow.
- **RoPE.** Half-split pairing: dimension `i` rotates with dimension `i + d/2`, at frequency `theta^(-2i/d)`. Frequencies, angles, sines and cosines are computed in f32, not f64, because the reference computes them in f32 and a more precise angle is a different angle.

The engine computes in f32 and the reference in f64, so engine results differ from the reference in the last bits of f32 rounding, which depends on the order of additions; S-1 and S-2 bound that difference. The partial sums also make the engine's logits differ in the last bits from release 0.1.0's, which were summed in index order; they are still bit-identical from run to run on one machine, because the number of sums and the combining order are fixed (S-6). Release 0.1.1 is therefore compared with 0.1.0 by the harness thresholds and by speed, never by exact equality. Decoded argmax tokens are protected only by S-3 and S-4, whose near-tie rule exists for exactly this reason.

## 5. Model loading

A model is a directory holding `config.json`, a single `model.safetensors` and `tokenizer.json`, exactly as published (FR-11).

- **Format.** The safetensors file is read in-house (HC-1): an 8-byte little-endian header length, a JSON header mapping each tensor name to dtype, shape and byte range, then raw data. The file is memory-mapped and each tensor copied into an owned f32 tensor.
- **Dtypes.** F32 is copied; BF16 is widened to F32, which is exact. Any other dtype is rejected naming the tensor and dtype (FR-12).
- **Names.** Tensors are found by their published names (`model.embed_tokens.weight`, `model.layers.{i}.self_attn.q_proj.weight`, and so on). A missing tensor, a tensor whose shape disagrees with the configuration, or an unexpected tensor is rejected naming it, because any of these means the file is not the model the configuration describes.
- **Not supported:** weights split across several files. SmolLM2-135M ships as one file; sharding is rejected with an error.

## 6. Tokenizer

The tokenizer is byte-level BPE, read from `tokenizer.json` and implemented in-house (HC-1).

- **Encoding.** Special tokens are matched first. The remaining text is split by the pre-tokenizer that `tokenizer.json` declares, each piece's bytes are mapped to the byte-level alphabet, and BPE merges are applied lowest rank first until no ranked pair remains. Pieces then map to IDs. The splitting pattern is hand-written rather than run through a regex crate: the standard byte-level pattern uses lookahead, which Rust's `regex` crate does not support, and hand-writing it is part of the point.
- **Components.** Only the types SmolLM2's `tokenizer.json` actually uses are implemented, and anything else is rejected as an unsupported component naming it (plan 0.1.0 task 3.6). The file declares: model `BPE` with no dropout, unknown token, subword prefix or suffix, byte fallback or `ignore_merges`; normalizer none; pre-tokenizer a sequence of `Digits{individual_digits: true}` then `ByteLevel{add_prefix_space: false, use_regex: true}`; post-processor none, so no beginning-of-sequence token is added and the reference's encoding carries none; decoder `ByteLevel`; no truncation or padding. The vocabulary has 49152 entries and 48900 merges, and 17 added tokens at IDs 0 to 16, all special and all also in the vocabulary (`<|endoftext|>` is ID 0, also the end-of-sequence ID). The engine requires vocabulary IDs to be exactly 0 to n-1 and added tokens to appear in the vocabulary under the same ID. A test pins these facts against the real file (FR-13).
- **Character classes are measured, not borrowed.** Letter, number and white-space ranges for the splitting pattern come from `oracle/unicode_classes.py`, which runs every Unicode scalar value through the reference pre-tokenizer and writes `tokenizer/unicode_classes.rs`, a committed file. Rust's `char::is_alphabetic` is not Unicode `\p{L}` (it counts some Thai vowel signs as letters), and a different Unicode version would shift piece boundaries silently; swapping it in made 557 of 3,979 corpus cases fail. A second oracle output, `oracle/tokenizer_corpus.py`, stores the reference's pieces and token IDs for about 4,000 strings, and the tokenizer tests compare against it.
- **Bytes with no vocabulary entry are dropped silently.** SmolLM2's vocabulary has no entry for 21 of the 256 byte-level characters (bytes 4, 6, 19, 20, 22, 29, 192, 193, 241, 242 and 245 to 255), and the reference drops them, so `"a\x04b"` encodes as `a` and `b` only. The engine does the same because FR-13 requires exact agreement. This is a known loss, not a guarantee of fidelity; whether it should warn or reject is an Open Question of the Requirements.
- **Decoding.** IDs map to piece strings, then back to bytes through the inverse byte map (FR-14). As the oracle decodes (`AutoTokenizer.decode`, default options): special tokens are kept and print as their text, no tokenization-space cleanup is applied, the bytes of all tokens are joined before being read as UTF-8 (so a character split across tokens decodes whole), and bytes forming no valid sequence become U+FFFD, one per maximal invalid subsequence, the same rule as `String::from_utf8_lossy`. A character outside the byte-level alphabet stands for no byte and is skipped. An ID outside the vocabulary is a rejected request.
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
4. If the token is the configuration's `eos_token_id`, stop; the token is returned among the generated IDs, because the oracle's continuation includes it and S-4 compares IDs, but its text is not passed to the callback, since standard output is generated text and the end-of-sequence token is a control signal (FR-17, FR-20).
5. Otherwise pass the token to the streaming decoder and emit any completed text to the callback (FR-17). Stop after the maximum number of new tokens; flush the decoder.

- The last generated token is not fed back through the forward pass, since nothing would use its logits; the context check in step 1 still counts it (FR-18). A maximum of zero new tokens is allowed: the prompt is checked and nothing is computed.
- `generate` returns `Generation { ids, stats }`. `Stats` holds the prompt and generated token counts, time to first token and generation time, with the two throughputs as methods. The CLI may use only the public API (§1), and time to first token cannot be seen through the text callback (the first token may complete no text, or be end-of-sequence), so the library reports it. Time to first token runs from receiving the prompt string, so it includes encoding, to choosing the first token. The first generated token's cost counts as prompt work: prompt throughput is prompt tokens over that time, and generation throughput is the tokens after the first over the time after it, absent for a single token (FR-8, FR-21).

## 9. Command-line interface

```
emberly-oxygen generate --model <dir> --max-new-tokens <n> [--stats] [-v|-vv|-vvv] [--log-file <path>] [PROMPT]
```

- The prompt is read from standard input when no argument is given (FR-19).
- Generated text goes to standard output and nothing else; errors go to standard error with a non-zero exit code (FR-20).
- `--stats` prints, to standard error, the model load time (timed by the CLI around both loads) followed by prompt token count, generated token count, time to first token and generation tokens per second, defined as in §8 (FR-21).
- A write error on standard output, such as a reader closing the pipe, makes the text callback exit with code 1, so generation stops at once instead of running to the maximum (FR-20). The CLI package's own tests run the built binary and check FR-19 to FR-21.

## 10. Reference oracle

The oracle lives in `oracle/`, a Python project managed by `uv` with a lockfile pinning `torch`, `transformers` and `huggingface_hub`, so expected outputs can be regenerated with the same library versions.

- **Model revision.** The model is downloaded at a pinned commit of its repository; the same files are the engine's model directory, so both read identical bytes (FR-11). The download script is the only step that needs network access.
- **Settings.** Two runs on CPU, in f64 and in f32 (FR-2), each loaded from the same published weights (bfloat16 widened exactly, as the engine widens them), with eager attention so no fused kernel changes the arithmetic, and gradients disabled. The oracle asserts each model's parameter dtype and device before running. In the f64 run the oracle forces the two operations the library computes in 32-bit inside any model to run in the model's dtype: RMSNorm (the library casts its input to 32-bit, normalizes and casts back) and the attention softmax (the library passes `dtype=torch.float32`). They are replaced for the duration of that run only, by a context manager that restores the library afterwards, so the f32 run is the unmodified library. Before the f64 run the oracle checks the replacement is active: a float64 RMSNorm of an input that 32-bit rounding would change must match a float64 calculation to 1e-14, and the softmax of a float64 tensor must come back as float64. The rotary tables are not replaced (below). The f64 run is the reference of S-1 to S-4; the f32 run is the library's default precision, kept so the harness can report the engine's distance to it (FR-5, §11) and the disagreement between the two references stays visible.
- **Prompt set.** `oracle/prompts.json` lists each prompt with an ID and a category: `english`, `thai`, `code`, `utf8-split` (prompts whose tokens split multi-byte characters) and `long` (about 512 tokens, the longest benchmark prompt, §12). Each category has at least two prompts (FR-3).
- **Greedy continuation.** 32 tokens per run, produced by the oracle's own argmax loop rather than a library generate call, so it can record the gap between the two highest logits at every step for near-tie detection (S-3, S-4). Each run continues from its own tokens, so the two continuations agree until a step whose gap is small enough for rounding to flip the choice, and may differ after it.
- **Output.** `fixtures/<model-id>/manifest.json` records the model ID and commit, library versions, the generating machine, and per prompt: ID, category, text, token IDs, continuation IDs and text, and top-two gaps. Each prompt has two safetensors files with the same tensor names and shapes: `<id>.safetensors` (f32, the 32-bit run) and `<id>.f64.safetensors` (f64, the reference, stored in 64-bit so the reference is not rounded to 32-bit again). Each holds logits `[positions, vocab]`, embeddings `[positions, hidden]`, layer outputs `[layers, positions, hidden]` and `final_norm_output` (FR-1). The manifest keeps the existing per-prompt fields for the 32-bit run and adds, for the reference, its file name, continuation IDs and text, whether it stopped at the end-of-sequence token, and top-two gaps at the prompt positions and the continuation, all prefixed `f64_`. Fixtures are model-agnostic in shape, so adding a model adds a folder (Requirements §2.2).
- **Honesty note.** In the reference library, the last entry of the library's hidden states is taken after the final norm, not straight from the last layer. The oracle therefore captures every layer's raw output with forward hooks (checked equal to the library's own hidden states for all other layers) and stores the final-norm output separately as `final_norm_output`; the harness compares that against the engine's normalized output.
- **Top-two gaps** are stored for the prompt positions as well as for the continuation, because the near-tie test of S-3 applies at every position. S-3 and S-4 use the f64 gaps.
- **The reference's own limits.** The library computes its rotary sines and cosines in f32 even in a f64 model (it casts the angles to f32 and the results back, in the pinned version), and they are left as the library makes them: they are the model as published, so the f64 reference shares the engine's f32 RoPE tables (§4), where a true f64 angle would be a different angle (at position 5000 its cosine differs by 8.4e-5). The reference is the library's definition of the model in 64-bit arithmetic, not an independent implementation. Measured in plan 0.1.2: the library's unmodified f64 run changed by up to 1.7e-5 in logits with the thread count and differs by up to 7.3e-5 from the run with RMSNorm and softmax in 64-bit, which is why the oracle makes that replacement; with it, a rerun at another thread count differs by about 2e-6 in logits. Any residual noise is measured again whenever the oracle's library versions change.
- **Further outputs.** `oracle/tokenizer_corpus.py` writes `fixtures/<model-id>/tokenizer-corpus.json` (§6), `oracle/unicode_classes.py` writes the committed character classes (§6), and `oracle/rope_reference.py` prints the reference's rotary cosines and sines at positions 1 and 5000 as Rust constants for the RoPE test, because f32 angles must agree with the reference's (§4): at position 5000 an f64 cosine differs from the reference's by 8.4e-5.

Logits dominate fixture size: about 200 KB per position in f32 and twice that in f64, so a long prompt is about 100 MB in f32 and about 200 MB in f64, and the fixtures folder grows from about 0.7 GB to about 2 GB. Fixtures are therefore kept out of git: \`fixtures/\` is git-ignored and filled by running the oracle locally. What regenerates them is committed instead: the prompt set, the oracle's lockfile and the pinned model commit. The f32 fixtures generated on a different machine may differ from another machine's in the rounding of the f32 reference itself, and against which the engine's error in release 0.1.1 reached 99% of S-1; that is why S-1 is judged against the f64 fixtures, whose machine-to-machine differences are expected to be orders of magnitude smaller (measured, plan 0.1.2). The manifest records which machine produced them.

## 11. Correctness harness

Rust integration tests in `tests/`, run with `cargo test`; the test profile is compiled with optimization and debug assertions on, so a full run takes seconds instead of minutes without losing overflow checks (FR-7).

- The harness reads every `fixtures/*/manifest.json`, so a new model's fixtures are picked up without code changes.
- Per prompt it checks encoding (FR-13), decoding (FR-14), logits (S-1), intermediates (S-2), top-1 token (S-3) and greedy continuation (S-4). The numeric checks and the near-tie rule read the f64 fixtures (§10). The engine's f32 values are widened to f64 before the difference is taken, so the comparison adds no rounding of its own and the f64 reference is never narrowed.
- Beside each logit and intermediate error it also measures the engine's distance to the f32 fixtures and reports it, per category, in the same table. This figure never fails a prompt (FR-5). The token encoding and decoding checks read the manifest's existing fields, which both runs share.
- For `utf8-split` prompts it streams the expected continuation through the streaming decoder and checks that the concatenated chunks equal the full decode and that no chunk contains a replacement character the full decode lacks (FR-15).
- A tensor, for S-2, is one layer's `[positions, hidden]` output for one prompt, and its largest magnitude is taken over all positions; a per-position maximum would be stricter. An all-zero reference tensor fails. A near-tie is an f64 reference gap strictly less than 1e-3, and in the greedy comparison the gap before each continuation token decides: the first near-tie stops the comparison, tokens before it must match, and the token at it may be either (S-3, S-4).
- It prints a pass/fail table by prompt and by category; any failing category fails the run (S-5). Each failure reports the first diverging position and, for intermediates, the first diverging layer (FR-6).
- Model files are found in `models/<model-id>/` (git-ignored, filled by the oracle's download script). If they are missing, the run fails naming the path; it never downloads (FR-7). Missing fixtures, f32 or f64, fail the run the same way, naming the oracle command that generates them.

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
- **Kernel baseline (HC-2).** A separate group times the in-house matrix-vector product against the test-only linear algebra crate (`matrixmultiply`, single-threaded) at the model's five matrix shapes. It agrees with both before timing and records one row per shape with ns per call and GFLOP/s. The group is reached through the `bench-kernels` Cargo feature, off by default, which adds the public module `emberly_oxygen::kernels`; only the library's self-referencing dev-dependency enables it, so a normal build never contains it.
- **Reference points for the kernel.** `matrixmultiply` is a weak yardstick for a matrix-vector product: built for matrix-matrix work, it is only about 1.25× faster than the 0.1.0 loop at most shapes and slower at the 49152×576 output projection, because it repacks the matrix on every call. The kernel group therefore also times the 0.1.0 index-order loop and the partial-sum widths swept, 4, 8, 16 and 32 (FR-25), so the speed-up is shown against the engine's own history. A dedicated matrix-vector library is not added as a baseline in 0.2: it would be a new dependency (§15) and the earlier loop already shows the ceiling.
- **Comparability.** The workloads and procedure are unchanged from 0.1.0, so 0.1.0's rows are the previous release's in the record (FR-9, FR-25).
- **Definitions.** Throughput and time to first token are as defined in §8.

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
| `-vv` | debug | one event per forward step: position, the token fed, its text, top 5 candidates with logits, step time; a step whose result is taken also names the chosen token as `next`; prompt-feeding steps are marked as such. Since the last generated token is never fed (§8), a run has prompt tokens plus generated tokens minus one step events |
| `-vvv` | trace | per layer within each step: residual-stream norm after attention and after the MLP, and for each head the position receiving the most attention weight |

- Generation, steps and layers are nested spans (`generate` → `step{pos}` → `layer{idx}`), so every event carries its place in the decode loop.
- `--log-file <path>` sends the same events to a file instead of standard error (FR-24). Standard output stays generated text only (FR-20).
- Per-layer statistics are computed only when the trace level is enabled, so lower levels do no extra arithmetic. No statistic feeds back into the computation, so output is identical at every level (S-8). The benchmark harness runs at the default level.
- Logging costs nothing measurable when no subscriber is installed: the real model's step time was 49.25 ms before logging was added and 49.29 to 49.32 ms after.
- At trace level a 512-token prompt produces tens of thousands of events. It is meant for short prompts, or with `--log-file`.

## 15. Dependencies

Locked initial set:

| Crate | Used by | Purpose |
| --- | --- | --- |
| `clap` 4 | CLI package only | command-line parsing |
| `tracing-subscriber` 0.3, features `fmt` and `std` only | CLI package only | formatting and routing log events, with no `log` bridge and no colour codes in log files |
| `thiserror` | library | error type |
| `tracing`, without its default `attributes` feature | library | emitting log events and spans, with no proc-macro crates |
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

  Benchmarks follow the correctness run, so no result from a failing build enters the record (HC-4). Running straight after the tests leaves the machine warm: in 0.1.0 it made the workloads about 6% slower than on an idle machine. The procedure is kept as written so that every release's rows are taken the same way and stay comparable (S-7); the effect is a known bias, not a correction.

## 17. Testing strategy

- **Unit tests per kernel.** Each kernel is checked against hand-computed small cases (softmax sums to 1, RoPE at position 0 is the identity) and, for matrix-vector product, against the test-only linear algebra crate on random inputs (HC-2).
- **Unit tests for the tokenizer.** Byte-map round trips, merge order on known pieces, and the streaming decoder's hold-back on split three-byte characters (FR-15).
- **Correctness harness.** End-to-end comparison against the oracle's fixtures (§11), the gate every release must pass (HC-4).
- **Negative controls.** The harness must be seen to fail. A Cargo feature, `negative-controls`, off by default, adds the enum `Fault` (`InterleavedRope`, `PerturbedLogit { position }`, `NoFinalNorm`) and `Model::inject_fault`; in a normal build `Fault` has no values, so no faulty arithmetic is compiled in. The library's self-referencing dev-dependency enables the feature, so plain `cargo test` always runs the controls and none is skipped for a missing flag, while `cargo build` never contains them. A test asserts that the harness fails on each fault. With no fault injected the feature changes no result. A check that has never failed has not been shown to work. The new kernel (§4) must leave all three controls failing as before, and so must the move to the f64 reference: the controls are run against it, and the `PerturbedLogit` size is checked to exceed the S-1 limit in force, so that a change of limit cannot leave a control that no longer fails.
- **Kernel tests.** The new matrix-vector product is checked against the 0.1.0 index-order loop and against `matrixmultiply` on random inputs at the model's shapes and at lengths that are not a multiple of 8, including lengths below 8 (the tail), within an absolute tolerance far below S-1 (HC-2, FR-25).
- **Reference tests.** The comparison functions are given a reference value that f32 cannot represent and must measure the error against the unrounded value (S-1, FR-2); the f64 fixtures are checked to have the manifest's shapes and dtype (FR-4); the oracle's own start-up check that the 64-bit RMSNorm and softmax are active (§10) is run whenever the oracle is; and a test shows that the f32 distance never turns a passing prompt into a failing one (FR-5).
- Every test names the HC, FR or S item it verifies, so coverage of the Requirements can be listed mechanically.

## 18. Milestones

&#91;embedded content: milestones · M1 to M7, ending in release 0.1.1\]

- **M1 — Oracle.** The oracle generates fixtures for every prompt category from the committed prompt set. *Proves the reference exists.*
- **M2 — Harness.** The harness runs against a stub engine, reports per category, and fails cleanly. *Proves the gate can fail.*
- **M3 — Loading.** Model loading and tokenization pass FR-11 to FR-15. *Proves the engine reads the model as published.*
- **M4 — Forward.** Logits, intermediates and top-1 tokens pass S-1 to S-3 in every category, and the negative controls fail as intended. *Proves the math.*
- **M5 — Generation.** Greedy continuations pass S-4, and the CLI meets FR-16 to FR-21. *Proves the engine end to end.*
- **M6 — Baseline.** First benchmark record on the benchmark machine, release note written, tagged 0.1.0. *Proves the progression has a starting point.*
- **M7 — Faster kernel.** The matrix-vector product with independent partial sums (§4) passes the unchanged harness in every category and leaves the negative controls failing; the kernel group and both workloads are recorded against 0.1.0's rows; release note written, tagged 0.1.1 (FR-25). *Proves the speed-up is real and costs nothing in correctness.*
- **M8 — Float64 reference.** The oracle writes f64 fixtures beside the f32 ones, the reference's own run-to-run noise is measured, the harness judges S-1 to S-4 against the f64 fixtures and reports the f32 distance beside them, the thresholds are decided by the owner from the measurement, the negative controls fail against the new reference, and the release note is written and tagged 0.1.2 (FR-1, FR-2, FR-5). *Proves the gate measures correctness, not agreement with one library's rounding.*

## 19. Open Items

- **Whether S-1 to S-4's thresholds hold against the f64 reference (Requirements §13).** Resolver: owner, from M8's measurement.

### Resolved since 0.2

- **Which partial-sum width is best (§4).** 16, by the sweep of release 0.1.1 (plan 0.1.1 §4, release note 0.1.1).
- **Whether release 0.1.2 re-runs the benchmark (§16, step 2).** Yes, decided by the owner on 2026-11-08: the version changes, so the record gets rows for it, taken by the same procedure as every release. The engine is unchanged, so the rows also show how repeatable 0.1.1's were; the release note says so and does not claim a speed change.

### Resolved since 0.1

- **Tokenizer components.** SmolLM2's `tokenizer.json` was read at M3; §6 states what it uses.
- **Benchmark platform.** The benchmark machine is an Apple M4 Max on macOS 26.3 (Requirements §13), so §12's Linux-or-macOS assumption holds.

### Feedback absorbed in 0.3

From plan 0.1.1 §5 (G-25). Every row addressed to this document, absorbed or rejected:

| Plan row | Disposition |
| --- | --- |
| Specification 0.2 status | Not a content change. The status is the owner's act and stays `draft`; the pin in the plans follows it. |
| Task 7.2, §4 | Absorbed into §4 (the combining order is named: sequential from sum 0, then the tail in index order). |
| Task 7.5, §10, §11 | Absorbed into §10, §11 and §17, with the owner's wider scope (S-1 to S-4 and the near-tie rule against the f64 reference). The reference's RoPE limit is stated in §10, which the plan row did not mention. |
| Task 8.2, §4, §12 | Absorbed: §4 names 16 sums and the sweep; §12 names the widths swept. |

From plan 0.1.2 §5:

| Plan row | Disposition |
| --- | --- |
| Task 9.4, §10 | Absorbed into §10 (Settings, and the reference's own limits, which 0.3 stated wrongly) and §17. |

### Feedback absorbed in 0.2

From plan 0.1.0 §9 (G-25). Every row addressed to this document, absorbed or rejected:

| Plan row | Disposition |
| --- | --- |
| Task 1.5, §10 | Absorbed into §10 (top-two gaps at every position). |
| Task 1.6, §10 | Absorbed into §10 (hooks, `final_norm_output`). |
| Task 2.3, §11 | Absorbed into §11 (tensor reading, all-zero fails, near-tie rule). |
| Task 3.6, §6 and §19 | Absorbed into §6; the §19 item is resolved. |
| Task 3.8, §6 and §10 | Absorbed into §6 and §10. |
| Task 3.9, §6 | Absorbed into §6 (the silent drop is stated); the warn-or-reject question belongs to Requirements. |
| Task 3.10, §6 and §10 | Absorbed into §6 (special tokens kept, decode rules). |
| Task 4.3, §4 | Absorbed into §4 and §10; no change to the f32 rule. |
| Task 4.7, §17 | Absorbed into §17. |
| Task 5.1, §8 and §3 | Absorbed into §8 (EOS, last token, zero tokens) and §3 (`eos_token_id`). |
| Task 5.3, §12 and §14 | Absorbed into §8 (definitions) and §14 (step events, cost when off). |
| Task 5.4, §1, §8, §9, §14, §15 | Absorbed into §8, §9 and §15. |
| Task 6.5, §1, §12, §17 | Absorbed into §12 and §17 (`bench-kernels`, reference points). The suggestion of a dedicated matrix-vector baseline is rejected for 0.2 with the reason in §12. |
| Task 6.1 | Resolved (§19 ledger). |
