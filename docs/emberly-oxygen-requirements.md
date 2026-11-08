# Emberly Oxygen — Requirements

Oct 27, 2026 · @Wattanit

**Version:** 0.5\
**Status:** approved\
**Date:** 2026-11-08\
**Owner:** Wattanit\
**Companion documents:** Emberly Oxygen — Technical Specification 0.3 (downstream)

This document owns WHAT Emberly Oxygen must do and WHY it exists. It defers HOW it is built (architecture, algorithms, data layouts, dependencies, testing strategy, milestones) to the Technical Specification. There is no Design Guideline: the only human-facing surface is a command-line interface, so naming and output wording are decided here.

## 1. Purpose and Motivation

Emberly Oxygen is a decoder-only LLM inference engine written from scratch in Rust. It exists for three reasons, in priority order:

1. **Understanding by building.** Every component between a model file and generated text is implemented in this project, so that its owner understands each one from the inside. Where speed and understanding conflict, understanding wins.
2. **A measured, documented progression.** The engine starts with the simplest correct implementation and improves through successive releases. Each improvement is measured against the previous release, so the project's history demonstrates what each technique is worth.
3. **Correctness proven, not assumed.** Output is checked against an independent reference implementation of the same model. A faster engine that produces different output is a broken engine, so no performance result counts until correctness holds.

## 2. Scope

### 2.1 In scope

- Loading the SmolLM2-135M base model from its published weights, chosen because it is the smallest widely used Llama-architecture model, so a naive engine runs it in seconds.
- Converting text to tokens and tokens to text with that model's tokenizer.
- Generating text from a prompt on CPU with greedy decoding.
- A command-line interface for generation.
- A reference oracle that produces expected outputs from an independent implementation of the same model.
- A correctness harness that compares engine output against the oracle's expected outputs.
- A benchmark harness that measures speed and memory the same way for every release.
- A faster matrix-vector product, in safe single-threaded code (FR-25). It dominates a decode step, so it is the first place a speed-up can matter.

### 2.2 Explicitly deferred

- **Additional models.** The oracle, its expected outputs and both harnesses must not be specific to SmolLM2-135M: adding a model adds expected outputs, not changes to the harnesses. This door is kept because it costs nothing now and every future model needs it.
- **Further kernel optimization.** Threads, SIMD intrinsics and `unsafe` code, tiling, and reading weights at lower precision inside the kernel. Each enters as its own release by replacing the body of the matrix-vector function, so neither its callers nor the harnesses change. FR-25 takes only the first step, because each technique is measured on its own (§1, purpose 2).

### 2.3 Out of scope

The engine is built naive-first: nothing is designed ahead for a capability until it is planned. The following are expected in future releases and enter this document by version bump when planned; until then no part of the engine is shaped for them.

- Sampling strategies other than greedy, and chat or instruction-following conversation.
- Processing a prompt's tokens in one batched pass.
- Quantized weights or quantized cached state.
- Model families other than the one in scope.
- Speculative decoding and constrained decoding.
- Memory management for long contexts or multiple concurrent sessions.
- GPU execution.
- An OpenAI-compatible HTTP API.

The following are out of scope with no future entry intended:

- **Training or fine-tuning.** The engine runs pre-trained weights only; training requires different machinery and would dilute the project's purpose.
- **Production serving concerns** such as authentication, multi-tenancy and rate limiting. The project is educational; a future OpenAI-compatible API is for interoperability, not for operating a service.
- **Speed parity with established engines.** Progress is measured against this project's own earlier releases.

## 3. Hard Constraints

A release that violates any of these is defective by definition.

- **HC-1 — Built in-house.** The engine's tensor operations, math kernels, model loading, tokenization, transformer computation and generation are implemented in this project. No machine-learning framework, inference library, tokenizer library or linear algebra library is a dependency of the engine, because building these is the project's first purpose (§1). General application crates (command-line parsing, error handling, serialization, memory mapping, logging) are permitted: they are not inference.
- **HC-2 — Test-only oracles.** A linear algebra library may be used by tests and benchmarks as a correctness oracle or performance baseline, and is never linked into the engine binary. Comparing against a mature implementation is how an in-house kernel proves itself; shipping it would defeat HC-1.
- **HC-3 — Python only in the oracle.** The reference oracle is the only Python in the project. The engine, the correctness harness and the benchmark harness run without Python installed, so the engine is self-contained and the reference stays independent of it.
- **HC-4 — Correctness before performance.** A release whose output fails the correctness harness (§5) is defective regardless of its speed. A faster engine that changes the model's output has not improved, so no benchmark result from a failing release is recorded as progress.

## 4. Reference oracle

- **FR-1 — Expected outputs.** For each prompt in the prompt set, the oracle produces: the prompt's token IDs, the logits at every position, a greedy continuation of fixed length, the input embeddings, and the output of every transformer layer. Per-layer outputs exist so that a mismatch can be traced to the first layer that diverges. Every output is produced twice, in 64-bit and in 32-bit floating point (FR-2).
- **FR-2 — Computed in 64-bit and 32-bit floats on CPU.** The oracle computes on CPU in 64-bit floating point, and again in 32-bit, which is the precision the reference library uses by default. Both runs start from identical weights (the published values, widened exactly), which are also the values the engine computes from. The 64-bit run is the reference that §11 judges the engine against; the 32-bit run is reported beside it and judges nothing. The reason: a 32-bit run carries its own rounding, which depends on its order of additions, so a check against it measures agreement with one implementation's rounding and not correctness. In release 0.1.1 the 32-bit reference was measured up to 7.1e-4 in logit from the 64-bit result, farther than the engine at either summation order tried (3.3e-4 and 5.1e-4), so a more accurate engine could fail against it. The 64-bit run computes every operation of the model in 64-bit except the rotary position tables, which stay as the reference library makes them (32-bit): they are part of the model as published, and a more precise angle is a different angle. The library's own 64-bit run is not used as it stands, because it also computes the normalization and the attention softmax in 32-bit; measured in plan 0.1.2, that made its logits depend on the thread count by up to 1.7e-5 and differ by up to 7.3e-5 from the version that does not. Not guaranteed: the reference is independent of the engine but not of the reference library. It carries that library's definition of the model, including its 32-bit rotary tables, so an error in that definition would be shared by the reference and would pass these checks.
- **FR-3 — Prompt coverage.** The prompt set includes English prose, Thai text, source code, prompts whose tokens split multi-byte UTF-8 characters, and at least one prompt near the longest context the benchmarks use. Results are judged per category (§11), so each category must be present.
- **FR-4 — Stored, not rerun.** Expected outputs are generated by running the oracle locally and kept out of version control, because full logits in two precisions make them gigabytes. What regenerates them (the prompt set, the oracle's pinned dependencies and the pinned model revision) is version-controlled. The harnesses never run the oracle, so a test run never depends on Python or on a reference library's version.

## 5. Correctness harness

- **FR-5 — Full comparison.** The harness compares the engine against every stored expected output and reports pass or fail per prompt and per check: token IDs, logits, top-1 token, greedy continuation, and intermediate values, each against the 64-bit reference (FR-2). It also reports, without judging, the engine's distance to the 32-bit reference, so the disagreement between the two references stays visible.
- **FR-6 — Locates divergence.** On failure, the harness reports the first diverging position and, where intermediate values are available, the first diverging layer.
- **FR-7 — One command, offline.** The harness runs with a single command, without network access and without Python.

## 6. Benchmark harness

- **FR-8 — Fixed measurements.** For a fixed set of prompts and generation lengths, the harness measures prompt processing throughput (tokens per second), generation throughput (tokens per second), time to first token, and peak resident memory.
- **FR-9 — Cumulative record.** Each run appends to a cumulative record: release version, commit, machine description, build profile, and results. Releases are comparable only if every result carries its conditions.
- **FR-10 — Optimized builds only.** The harness refuses to record results from a non-optimized build, because unoptimized timings measure the compiler's settings rather than the engine.

## 7. Model loading

- **FR-11 — Published files, unconverted.** The engine loads the model's weights and configuration from the files the model is published in, with no conversion step. The oracle reads the same files, so both sides start from identical bytes.
- **FR-12 — Unsupported models fail loudly.** A model whose configuration the engine does not support is rejected with an error naming the unsupported value. The engine never runs a model it does not fully implement, because partially supported models produce plausible but wrong output.

## 8. Tokenization

- **FR-13 — Exact encoding.** Text encodes to exactly the token IDs the model's reference tokenizer produces, for every prompt in the prompt set. The reference tokenizer silently drops the 21 byte values SmolLM2's vocabulary has no entry for, so the engine drops them too: exact agreement is the requirement, and the loss is silent on both sides (§13).
- **FR-14 — Exact decoding.** Token IDs decode to exactly the text the reference tokenizer produces.
- **FR-15 — No broken characters while streaming.** Streamed output never contains an incomplete UTF-8 sequence: bytes of a partially decoded character are held until the character completes. Thai characters are three bytes and are routinely split across tokens.

## 9. Generation

- **FR-16 — Greedy generation.** Given a prompt and a maximum number of new tokens, the engine generates by always selecting the highest-scoring token, and stops at the maximum or at the model's end-of-sequence token.
- **FR-17 — Streamed output.** Each generated token's text is emitted as soon as it is produced, subject to FR-15.
- **FR-18 — Context limit enforced.** A request whose prompt plus maximum new tokens exceeds the model's context length is rejected with an error. Silent truncation would change the output without telling the user.
- **FR-25 — Faster matrix-vector product.** The matrix-vector product, about 94% of a decode step in 0.1.0 (release note 0.1.1; the 0.1.0 note's 98.6% came from a dry run on a cooler machine), runs faster than in the previous release, in safe single-threaded in-house code (HC-1). The release note reports the kernel's and the engine's speed against the previous release's recorded rows (FR-9) and what the change was worth; there is no numeric target, so a disappointing result is a recorded negative result (§12), not a defect. Output must still pass §11 (HC-4). It stays single-threaded and free of `unsafe` so that this release measures one technique alone.

## 10. Command-line interface and logging

- **FR-19 — Generate command.** A single command takes a model location, a prompt (as an argument or from standard input) and a maximum number of new tokens, and streams generated text to standard output.
- **FR-20 — Clean output stream.** Errors and diagnostics go to standard error only, so standard output contains generated text and nothing else.
- **FR-21 — Run statistics on request.** On request, the command reports token counts, time to first token and generation throughput on standard error.
- **FR-22 — System log.** The engine reports what it does as log events: model loading, tokenization, each generation step, timings and errors. The library never prints; the program using it decides where events go.
- **FR-23 — Verbosity levels.** By default only warnings and errors are logged. A `--verbose` option raises the level step by step: run summary and timings; then each decode step (position, chosen token, top candidates, step time); then per-layer detail inside each step. The highest level lets the decode loop be followed directly.
- **FR-24 — Log to file.** On request, the log is written to a file instead of standard error, because a full trace of a long generation is too large to read in a terminal.

## 11. Stability and quality

The thresholds below are initial values; tune with the first measured comparison and record any change by version bump. S-1 to S-4 are judged against the 64-bit reference (FR-2); the values were set in 0.2 against the 32-bit reference and were measured against the new one and kept in 0.5 (§13).

- **S-1 — Logit agreement.** At every position of every prompt, the largest absolute difference between engine logits and the 64-bit reference's logits is at most 1e-3. The engine computes in 32-bit, so its error is rounding that depends on arithmetic order, which stays well below this; a real bug exceeds it.
- **S-2 — Intermediate agreement.** Every stored intermediate value matches the 64-bit reference's within 1e-4 relative to the largest magnitude in its tensor. A relative bound is used because values grow across layers. A tensor is one layer's output for one prompt, and its largest magnitude, in the 64-bit reference, is taken over all positions; an all-zero reference tensor fails, because a relative error is undefined.
- **S-3 — Top-token agreement.** The highest-scoring token matches the 64-bit reference's at every position, except near-ties where the 64-bit reference's two highest logits differ by less than 1e-3; there either token passes.
- **S-4 — Greedy agreement.** The greedy continuation matches the 64-bit reference's token for token. If a near-tie (S-3) occurs during the continuation, comparison stops at that position and the harness reports it, because the two sequences may legitimately diverge from there.
- **S-5 — Judged per category.** A release passes only if every prompt category (FR-3) passes. Averages are never used, because an aggregate can stay flat while one category fails.
- **S-6 — Reproducible output.** For identical input, the same build on the same machine produces logits that differ between runs by no more than S-1 allows, and greedy continuations that are identical except at near-ties (S-3). Without this, a benchmark or correctness result cannot be reproduced. Bit-identical output is not required, because parallel computation may change the order of additions between runs.
- **S-7 — Benchmark conditions.** Every recorded benchmark runs on one designated machine. Each measurement is repeated and reported as the median with its spread, so a single noisy run is never taken as a result.
- **S-8 — Logging changes nothing.** Generated output is identical at every verbosity level, and recorded benchmarks run at the default level. Logging that altered results or timings would make the trace describe a different run.

## 12. Policies

These are requirements because the project's second purpose (§1), a measured and documented progression, depends on them.

- **Naming.** The project is Emberly Oxygen, part of the Emberly suite; its package and binary are named `emberly-oxygen`.
- **Release versioning.** Releases follow semantic versioning. Each release introduces one main technique, so its effect can be measured and explained on its own.
- **Release notes.** Every release ships a note stating what changed, why it was expected to help, what was measured against the previous release, and what was learned.
- **Negative results kept.** A technique that fails or underperforms is recorded with the reason it did, and the record is never removed. Knowing where a technique stops working is part of understanding it.
- **Original design.** Other engines may be studied, but this engine's design is derived from first principles rather than copied. Copying a structure skips the understanding the project exists to build.

## 13. Open Questions

- **Whether a byte the vocabulary cannot represent should be warned about or rejected instead of silently dropped (FR-13).** Resolver: owner, when a release changes tokenization behavior; the engine and the reference agree today.

### Resolved since 0.4

- **Whether S-1 to S-4's initial thresholds hold against the 64-bit reference (§11).** They hold and are kept, by the owner's decision of 2026-11-08, against a rule agreed before the numbers were seen (keep a limit when the worst error is at most half of it). Release 0.1.2 measured the 16-sum engine's worst logit error at 3.50e-4 (35% of S-1) and its worst relative intermediate error at 9.97e-6 (10% of S-2) against the 64-bit reference; against the 32-bit reference the same engine had reached 99% of S-1. They are not tightened: a real bug smaller than a tighter limit could hide, and the negative controls are sized for the present values (release note 0.1.2).

### Resolved since 0.2

- **Whether S-1 and S-2 describe a real bug, or only distance to the reference library's rounding (§11).** They did not: the 32-bit reference was up to 7.1e-4 from the 64-bit result and a correct engine reached 99% of S-1 against it (release note 0.1.1). Decided by the owner on 2026-11-08: the oracle gains a 64-bit reference (FR-1, FR-2), and S-1 to S-4 are judged against it. The 32-bit comparison stays as a reported figure (FR-5); loosening S-1 was the alternative and was not chosen, because any limit against a 32-bit run would still depend on that run's summation order.

### Resolved since 0.1

- **Which machine is the benchmark machine (S-7).** An Apple M4 Max (16 cores: 12 performance, 4 efficiency; 128 GB; macOS 26.3), named by the owner at plan 0.1.0 task 6.1 (plan §9, §10).
- **Whether the initial thresholds in S-1 to S-4 hold.** They hold and are kept: release 0.1.0 measured a worst logit error of 2.36e-4 against 1e-3 and a worst intermediate error of 5.91e-6 against 1e-4 (release note 0.1.0). Any later change lands by version bump.
- **Exact prompt set, generation lengths and context lengths (FR-3, FR-8).** Technical Specification §10 and §12.

### Feedback absorbed in 0.2

From plan 0.1.0 §9 (G-25). Every row addressed to this document, absorbed or rejected:

| Plan row | Disposition |
| --- | --- |
| Before Phase 1, S-3 | Already resolved in place by the owner in 0.1 (the corrupted sentence was restored; meaning unchanged). |
| Task 2.3, S-2 and S-3 | S-2 absorbed (what a tensor is, all-zero fails). The S-3 and S-4 readings were already the text here ("less than 1e-3", comparison stops at the first near-tie). |
| Task 3.9, FR-13 | Absorbed into FR-13; whether to warn or reject is an Open Question. |
| Task 6.1, §13 machine | Absorbed: resolved ledger above. |
| Task 6.5, HC-2 | Rejected as a requirement change: HC-2 stays as written. The weak baseline is a Specification matter (§12). |
| Phase 4 exit, S-1 and S-2 | Absorbed: thresholds kept, measured errors recorded in the resolved ledger above. |

### Feedback absorbed in 0.5

From plan 0.1.2 §5 (G-25). Every row addressed to this document, absorbed or rejected:

| Plan row | Disposition |
| --- | --- |
| Task 10.7, §13 | Absorbed: the thresholds question moves to the resolved ledger above with the measured errors. |
| (found in review before the 0.1.2 merge) FR-2 | Wording corrected: 0.4 said the reference was "not an independent implementation", which contradicted §1 and §2.1. It is independent of the engine and not of the reference library; FR-2 now says so. |

### Feedback absorbed in 0.4

From plan 0.1.2 §5 (G-25). Every row addressed to this document, absorbed or rejected:

| Plan row | Disposition |
| --- | --- |
| Task 9.4, FR-2 | Absorbed into FR-2: 0.3 said the 64-bit run was "exact in arithmetic", which was wrong, because the library computes normalization, softmax and rotary tables in 32-bit even in a 64-bit model. By the owner's decision of 2026-11-08 the oracle forces normalization and softmax to 64-bit and leaves the rotary tables as published. |
| Before Phase 9, FR-25 | Absorbed into FR-25: the matrix-vector product's share of a 0.1.0 step is 94% by the recorded rows, not 98.6%. |

### Feedback absorbed in 0.3

From plan 0.1.1 §5 (G-25). Every row addressed to this document, absorbed or rejected:

| Plan row | Disposition |
| --- | --- |
| Task 7.5, S-1, S-2, FR-1, FR-2 | Absorbed, by the owner's choice of remedy (1), the 64-bit reference (resolved ledger above). Widened from the plan's suggestion in one respect, decided by the owner on 2026-11-08: S-3 and S-4 are judged against it too, and S-2 as well as S-1, so the engine has one reference. Remedy (2), loosening S-1, was not taken. The thresholds themselves stay open (§13). |
