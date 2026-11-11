# Emberly Oxygen — Requirements

Oct 27, 2026 · @Wattanit

**Version:** 0.9\
**Status:** draft\
**Date:** 2026-11-11\
**Owner:** Wattanit\
**Companion documents:** Emberly Oxygen — Technical Specification 0.7 (downstream)

This document owns WHAT Emberly Oxygen must do and WHY it exists. It defers HOW it is built (architecture, algorithms, data layouts, dependencies, testing strategy, milestones) to the Technical Specification. There is no Design Guideline: the only human-facing surface is a command-line interface, so naming and output wording are decided here.

## 1. Purpose and Motivation

Emberly Oxygen is a decoder-only LLM inference engine written from scratch in Rust. It exists for three reasons, in priority order:

1. **Understanding by building.** Every component between a model file and generated text is implemented in this project, so that its owner understands each one from the inside. Where speed and understanding conflict, understanding wins.
2. **A measured, documented progression.** The engine starts with the simplest correct implementation and improves through successive releases. Each improvement is measured against the previous release, so the project's history demonstrates what each technique is worth.
3. **Correctness proven, not assumed.** Output is checked against an independent reference implementation of the same model. A faster engine that produces different output is a broken engine, so no performance result counts until correctness holds.

## 2. Scope

### 2.1 In scope

- Loading the SmolLM2-135M base model and its instruct variant (SmolLM2-135M-Instruct, same architecture and vocabulary, weights tuned to follow a chat) from their published weights, chosen because it is the smallest widely used Llama-architecture model, so a naive engine runs it in seconds.
- Converting text to tokens and tokens to text with that model's tokenizer.
- Generating text from a prompt on CPU with greedy decoding, or by sampling with temperature, top-k, top-p and a repetition penalty from a seeded random number generator (FR-26 to FR-29). Sampling is the first generation that has no single correct output, so it is judged by the distribution it draws from and by statistics over its draws, not by matching a reference's tokens (S-9, S-10).
- Chat with the instruct model: rendering a conversation in the model's chat template, stopping a reply at the end-of-turn token, keeping the history across turns, and an interactive chat mode in the command-line interface (FR-33 to FR-38). Chat is the first behavior whose output is judged by exact text and tokens of the template and by observable conduct (a reply ends, history is kept, the limit is enforced) rather than by numbers (FR-39, S-11).
- A command-line interface for generation and for chat.
- A reference oracle that produces expected outputs from an independent implementation of the same model.
- A correctness harness that compares engine output against the oracle's expected outputs.
- A benchmark harness that measures speed and memory the same way for every release.
- A faster matrix-vector product, in safe single-threaded code (FR-25; threads follow in FR-43). It dominates a decode step, so it is the first place a speed-up can matter.
- Processing a prompt's tokens in one batched pass instead of one token at a time (FR-40 to FR-42). The prompt is the part of a run that needs no sampling and can be computed for all its positions at once, so it is where batching helps; chat re-reads its whole history every turn (FR-35), which makes the prompt the dominant cost of a long chat.
- Running the matrix products and the attention on several threads, with output identical to the single-threaded engine's (FR-43 to FR-45). A decode step is limited by how fast one core can read the weights: the matrix-vector product reads them at 74.5 GB/s, which is one core's ceiling (69 to 74 GB/s), against 284 GB/s for twelve cores; a prompt is limited by one core's arithmetic, the batched product reaching 38 of the 111 GFLOP/s a core can do. Attention is 4% of a short decode step but grows with the context, to about a third of a step at 512 positions and two thirds at 2000 (profile of 2026-11-11, recorded in plan 0.4.1 §5). More cores are therefore the first place a speed-up remains in both phases.

### 2.2 Explicitly deferred

- **Additional models.** The oracle, its expected outputs and both harnesses must not be specific to SmolLM2-135M: adding a model adds expected outputs, not changes to the harnesses. This door is kept because it costs nothing now and every future model needs it. The first model through it is the instruct variant of SmolLM2-135M, with chat (FR-32).
- **Further kernel optimization.** SIMD intrinsics and `unsafe` code, tiling, and reading weights at lower precision inside the kernel; threads enter with FR-43. Each enters as its own release by replacing the body of the matrix-vector function, so neither its callers nor the harnesses change. FR-25 takes only the first step, because each technique is measured on its own (§1, purpose 2). FR-40 adds a matrix-matrix product for the batched prompt that only reuses each weight row across the prompt's tokens; it was single-threaded, safe and untiled for the same reason, so that no other kernel technique was mixed in with it; threads then arrived on their own (FR-43).
- **Reusing the cache across chat turns.** A turn would then process only its new tokens. It changes what a chat feeds the forward pass (FR-35 renders the whole history), so it is a technique of its own and enters by version bump when planned, through FR-35, which would then say what is kept.

### 2.3 Out of scope

The engine is built naive-first: nothing is designed ahead for a capability until it is planned. The following are expected in future releases and enter this document by version bump when planned; until then no part of the engine is shaped for them.

- Multi-line chat input, saving or loading a conversation, discarding old turns to fit the context (FR-36 rejects instead), and chat templates for model families other than the one in scope. Each enters by version bump when planned.
- Sampling controls beyond FR-26 (for example min-p) and stopping on a text string; neither is planned, and each enters by version bump if measured use shows a need.
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
- **FR-31 — Expected distributions for sampling.** For a fixed set of prompts and sampling settings (FR-26), the oracle produces the probability distribution over the vocabulary after the controls are applied, at the last position of the prompt, computed from the 64-bit logits (FR-2) by the reference library's own logit processors. The oracle does not draw tokens: a draw from a random number generator has no correct value to store, only a distribution it must follow. The distributions are stored, not rerun, under FR-4. Not guaranteed: the processors are the reference library's definition of each control, so a difference between that definition and the textbook one is shared by the reference.

- **FR-39 — Expected chat renderings and replies.** For a fixed set of conversations the oracle produces the text and token IDs the reference library's chat template renders (FR-33), and the instruct model's logits and greedy reply, computed in 64-bit as in FR-2 (the 32-bit run is not repeated for chat: its distance is reported for the base model only, FR-5). The set includes a conversation with no system message, one with a system message, one with several turns, and one in Thai. The renderings are judged (S-11); the logits and replies are reported and judge nothing, because the engine code that computes them is the code S-1 to S-4 already judge on the base model, and judging a second set of weights would repeat gigabytes of expected outputs for no new information about the engine. They are stored, not rerun, under FR-4. Not guaranteed: a defect specific to the instruct weights would show in the reported figures and not fail a release.

## 5. Correctness harness

- **FR-5 — Full comparison.** The harness compares the engine against every stored expected output and reports pass or fail per prompt and per check: token IDs, logits, top-1 token, greedy continuation, and intermediate values, each against the 64-bit reference (FR-2), and the engine's adjusted distribution for each sampling setting against the stored one (FR-31, S-9). It also reports, without judging, the engine's distance to the 32-bit reference, so the disagreement between the two references stays visible, and the instruct model's logits and greedy replies against its own 64-bit reference (FR-39), which judge nothing (§13).
- **FR-6 — Locates divergence.** On failure, the harness reports the first diverging position and, where intermediate values are available, the first diverging layer.
- **FR-7 — One command, offline.** The harness runs with a single command, without network access and without Python.

## 6. Benchmark harness

- **FR-8 — Fixed measurements.** For a fixed set of prompts and generation lengths, the harness measures prompt processing throughput (tokens per second), generation throughput (tokens per second), time to first token, and peak resident memory.
- **FR-9 — Cumulative record.** Each run appends to a cumulative record: release version, commit, machine description, build profile, and results. Releases are comparable only if every result carries its conditions.
- **FR-10 — Optimized builds only.** The harness refuses to record results from a non-optimized build, because unoptimized timings measure the compiler's settings rather than the engine.

## 7. Model loading

- **FR-11 — Published files, unconverted.** The engine loads the model's weights and configuration from the files the model is published in, with no conversion step. The oracle reads the same files, so both sides start from identical bytes.
- **FR-12 — Unsupported models fail loudly.** A model whose configuration the engine does not support is rejected with an error naming the unsupported value. The engine never runs a model it does not fully implement, because partially supported models produce plausible but wrong output.

- **FR-32 — The instruct model loads the same way.** The instruct variant loads through the path of FR-11 and FR-12 with no instruct-specific loading code: it differs from the base model in weights and in the end-of-sequence token its configuration names (the end-of-turn token of the chat template, FR-34), not in architecture. A tokenizer or configuration that differs from the base model's in a way the engine does not implement is rejected under FR-12, so the claim that nothing else differs is checked, not assumed.

## 8. Tokenization

- **FR-13 — Exact encoding.** Text encodes to exactly the token IDs the model's reference tokenizer produces, for every prompt in the prompt set. The reference tokenizer silently drops the 21 byte values SmolLM2's vocabulary has no entry for, so the engine drops them too: exact agreement is the requirement, and the loss is silent on both sides (§13).
- **FR-14 — Exact decoding.** Token IDs decode to exactly the text the reference tokenizer produces.
- **FR-15 — No broken characters while streaming.** Streamed output never contains an incomplete UTF-8 sequence: bytes of a partially decoded character are held until the character completes. Thai characters are three bytes and are routinely split across tokens.

## 9. Generation

- **FR-16 — Greedy generation.** Given a prompt and a maximum number of new tokens, the engine generates by always selecting the highest-scoring token, and stops at the maximum or at the model's end-of-sequence token.
- **FR-17 — Streamed output.** Each generated token's text is emitted as soon as it is produced, subject to FR-15.
- **FR-18 — Context limit enforced.** A request whose prompt plus maximum new tokens exceeds the model's context length is rejected with an error. Silent truncation would change the output without telling the user.
- **FR-25 — Faster matrix-vector product.** The matrix-vector product, about 94% of a decode step in 0.1.0 (release note 0.1.1; the 0.1.0 note's 98.6% came from a dry run on a cooler machine), runs faster than in the previous release, in safe single-threaded in-house code (HC-1). The release note reports the kernel's and the engine's speed against the previous release's recorded rows (FR-9) and what the change was worth; there is no numeric target, so a disappointing result is a recorded negative result (§12), not a defect. Output must still pass §11 (HC-4). It stays single-threaded and free of `unsafe` so that this release measures one technique alone (threads arrive on their own in FR-43).
- **FR-40 — Batched prompt processing.** The engine processes the prompt's tokens in one pass over all positions together, not one token at a time: each position attends to itself and the positions before it in the prompt and to none after (a causal mask), and the pass leaves in the cache the keys and values that one-token-at-a-time processing would have left. The prompt's last position gives the logits the first generated token is chosen from. Generation, sampled generation and chat all start this way, with the same cache and the same limit check (FR-18, FR-36) as before; decoding the tokens that follow is unchanged. The pass is safe in-house code with no tiling (HC-1, §2.2); it ran on one thread in 0.4.0 so that no other kernel technique was mixed in, as FR-25 measured the kernel alone, and from 0.4.1 it runs on several (FR-43).
- **FR-41 — Batched output agrees with the reference.** The batched pass passes the whole of §11 on every prompt of the set, unchanged: S-1 to S-4 against the 64-bit reference, at every position and for every stored intermediate value (HC-4), with the thresholds as they are. The harness judges the batched pass, not the one-token path, because the batched pass is the one generation uses. Batching may change the order of additions, so logits can differ from 0.3.0's in the last bits and greedy continuations from 0.3.0's at near-ties (S-3, S-4); this release is judged against the reference, not against 0.3.0's bits. Not guaranteed: identity with the previous release's output (FR-29).
- **FR-42 — Prompt speed is measured by length.** Prompt throughput and time to first token (FR-8) are recorded for prompts of several lengths, including the 67 and 644 token chat conversations of release 0.3.0's note, beside the one-token-at-a-time path run in the same build on the same machine, so the release note states what batching is worth at each length. There is no numeric target, so a disappointing result is a recorded negative result (§12), not a defect, as in FR-25. The two fixed workloads (FR-8) and their procedure are unchanged, so their rows stay comparable with every earlier release's (FR-9).
- **FR-43 — Threaded products and attention.** The matrix-vector and matrix-matrix products and the attention run on several threads at once, and the output is bit-identical to the single-threaded engine's at every thread count: the same logits, the same tokens, the same cache. Work is divided among the threads by output element (a row of a product, a head or a position of attention) and never inside a sum, so every sum is added in the order the single-threaded engine adds it and no thread count can change a result. The owner decided on 2026-11-11 that this identity is required and not only permitted by S-6: it keeps a seeded replay (FR-28) independent of the thread count, keeps greedy output identical to the previous release's (FR-29), and lets the harness that judged 0.4.0 judge this release with no new tolerance (FR-41). The workers are built from the standard library's threads and synchronization in safe code, with no parallelism crate and no `unsafe`: the worker pool is a technique this project exists to understand (§1, HC-1), and `unsafe` stays a last resort (§2.2). If safe code cannot give the workers the weights and the cache at an acceptable cost, the implementation raises it with the owner instead of using `unsafe`. Not guaranteed: speed. A thread count beyond what the memory system supplies gains nothing, and a product small enough that handing it over costs more than computing it may be slower threaded.
- **FR-44 — The thread count is a setting.** The library and the generate and chat commands take a number of threads (`--threads`, FR-30's manner for an invalid value). The default is chosen by the release's thread sweep (FR-45): 8 in release 0.4.1, or the cores the machine makes available when there are fewer; 1 runs everything on the calling thread, which is the previous release's engine. A count of 0 is rejected with an error naming the option and the value before any computation, because a run on no thread is not a run. The count in use is reported in the run summary (FR-22, FR-23) and recorded with every benchmark row (FR-9), because a speed without it cannot be compared. Output does not depend on it (FR-43).
- **FR-45 — Speed is measured by thread count.** The release reports the two fixed workloads (FR-8) at the default thread count, and prompt speed and decode speed over a sweep of thread counts that includes 1, each against the previous release's recorded rows (FR-9), so the release note states what threads are worth at every count. The release is compared with 0.4.0 directly: no intermediate engine is built to apportion the saving between the products and the attention, as the owner ruled for batching on 2026-11-10; the note gives each operation's share of a step from the profile instead. There is no numeric target, so a disappointing result is a recorded negative result (§12), not a defect, as in FR-25. The workloads and their procedure are unchanged so their rows stay comparable (FR-9); rows recorded before this release carry no thread count and were single-threaded.
- **FR-26 — Sampling controls.** Generation accepts four controls and applies them to the logits of each step in this order, the reference library's order: repetition penalty (FR-27), temperature, top-k, top-p; then the next token is drawn from the resulting distribution. Temperature divides the logits. Top-k keeps the k highest-scoring tokens and every token tied with the k-th, as the reference library does, so more than k can survive. Top-p keeps the smallest set of highest-probability tokens whose probabilities sum to at least p. Each control has a neutral value that changes nothing, and a temperature of 0 means greedy selection (FR-16): the highest raw logit is taken and the other controls are ignored, with a warning if any is not neutral, so that FR-29 holds. A fixed order is required because the controls do not commute: the same settings in another order draw from a different distribution. Sampled generation stops where FR-16 stops greedy generation: at the maximum number of new tokens or at the end-of-sequence token.
- **FR-27 — Repetition penalty.** For every token already in the context, prompt and generated alike, a positive logit is divided by the penalty and a negative logit is multiplied by it, as the reference library does. This definition is chosen over a count-based penalty so that the oracle can check it (FR-31).
- **FR-28 — Seeded generation.** The same build, model, prompt, settings and seed produce the same tokens, and the seed used is reported, so a sampled run can be replayed. Reproduction across releases is not promised: pinning the random number generator forever would bind every future release to one algorithm, and a replay is only needed against the build that produced the run.
- **FR-29 — Greedy is unchanged.** With no sampling option set, or a temperature of 0, the output still passes §11 (S-1 to S-4), and is identical to the previous release's greedy output unless the release changes the order of additions in the arithmetic (FR-25, FR-41), in which case it may differ in the last bits of the logits and at near-ties (S-3, S-4) and is judged by §11 alone. Sampling is added beside the greedy path, not in place of it, so the correctness proven so far keeps its meaning. A release that changes no arithmetic (such as 0.2.0 and 0.3.0) still owes identical output.
- **FR-33 — Chat template.** A conversation is a list of messages, each with a role (system, user or assistant) and text. It is rendered as the instruct model was trained: every message as `<|im_start|>`, the role, a newline, the text, `<|im_end|>` and a newline; when the first message is not a system message, a default system message ("You are a helpful AI assistant named SmolLM, trained by Hugging Face") first; and, for a reply, `<|im_start|>assistant` and a newline at the end. The two markers encode as the single tokens the vocabulary has for them (FR-13), never as the characters they are spelled with. The rendering must equal the reference library's (S-11). The default system message is the model's as published, so a reply is conditioned on the same text as the reference's. Text in a message that spells a marker, typed by the user or generated by the model, also encodes as the marker, as in the reference (decided by the owner, 2026-11-08): a message containing `<|im_end|>` ends its own turn early, and nothing escapes it.
- **FR-34 — A reply ends at the end-of-turn token.** A chat reply stops at the model's end-of-sequence token (FR-16), which for the instruct model is `<|im_end|>`, or at the maximum number of new tokens. The end-of-turn token is neither shown nor streamed as text (FR-17); the history records the turn closed as FR-33 renders it, whether or not the model produced the token, so the next turn's prompt is well formed after a reply cut short by the maximum.
- **FR-35 — History is text.** The conversation is kept as the text of its messages, and every turn renders the whole conversation again (FR-33), as the reference library does, so a reply enters the history as the text the user saw and the oracle can reproduce the prompt of every turn. Generated token IDs are not kept: decoding and encoding again can produce different tokens, and the rendering is the definition of the prompt, not the earlier run's tokens. Not guaranteed: a reply that re-encodes to different tokens than were generated conditions later turns on the re-encoded tokens.
- **FR-36 — The context limit holds across turns.** A turn whose rendered conversation plus maximum new tokens exceeds the model's context length is rejected with an error before anything is generated (FR-18), and the conversation is left as it was, so the user can continue with a shorter message. Dropping the oldest turns is not done: it would change what the model sees without telling the user.
- **FR-37 — Chat samples by default.** A chat generates with the controls of FR-26 set to the instruct model's published example, temperature 0.2 and top-p 0.9 (its model card; its `generation_config.json` has no sampling values), with the other controls neutral. These are initial values; tune with use. A temperature of 0 gives greedy replies (FR-29). One seed serves the whole conversation and is reported once when the conversation starts (FR-28), so the same build, model, seed and sequence of messages replay the same replies.

## 10. Command-line interface and logging

- **FR-19 — Generate command.** A single command takes a model location, a prompt (as an argument or from standard input) and a maximum number of new tokens, and streams generated text to standard output.
- **FR-20 — Clean output stream.** Errors and diagnostics go to standard error only, so standard output contains generated text and nothing else.
- **FR-21 — Run statistics on request.** On request, the command reports token counts, time to first token and generation throughput on standard error.
- **FR-22 — System log.** The engine reports what it does as log events: model loading, tokenization, each generation step, timings and errors. The library never prints; the program using it decides where events go.
- **FR-23 — Verbosity levels.** By default only warnings and errors are logged. A `--verbose` option raises the level step by step: run summary and timings; then each decode step (position, chosen token, top candidates, step time); then per-layer detail inside each step. The highest level lets the decode loop be followed directly.
- **FR-24 — Log to file.** On request, the log is written to a file instead of standard error, because a full trace of a long generation is too large to read in a terminal.
- **FR-30 — Sampling options.** The generate command accepts an option for each control of FR-26 and for the seed (FR-28). A value outside a control's valid range (for example a negative temperature, a top-p outside 0 to 1, or a penalty of 0 or less) is rejected with an error naming the option and the value, in the manner of FR-12: a clamped value would run a different distribution from the one requested.
- **FR-38 — Chat mode.** The command-line interface has an interactive chat mode that takes a model location (and an optional system message, FR-33), reads one user message per line, streams each reply to standard output (FR-17) and keeps the conversation (FR-35) until the input ends or the user types `/exit`. It accepts the sampling options and the seed of FR-30 and the maximum new tokens per reply. Everything that is not a reply, including the prompt marker that invites the next message, goes to standard error (FR-20), so standard output holds the replies and nothing else and a chat can be driven from a pipe. A rejected turn (FR-36) prints its error and the chat continues; an end-of-input stops it with success. Not guaranteed: a message that spans several lines (§2.3).

## 11. Stability and quality

The thresholds below are initial values; tune with the first measured comparison and record any change by version bump. S-1 to S-4 are judged against the 64-bit reference (FR-2); the values were set in 0.2 against the 32-bit reference and were measured against the new one and kept in 0.5 (§13).

- **S-1 — Logit agreement.** At every position of every prompt, the largest absolute difference between engine logits and the 64-bit reference's logits is at most 1e-3. The engine computes in 32-bit, so its error is rounding that depends on arithmetic order, which stays well below this; a real bug exceeds it.
- **S-2 — Intermediate agreement.** Every stored intermediate value matches the 64-bit reference's within 1e-4 relative to the largest magnitude in its tensor. A relative bound is used because values grow across layers. A tensor is one layer's output for one prompt, and its largest magnitude, in the 64-bit reference, is taken over all positions; an all-zero reference tensor fails, because a relative error is undefined.
- **S-3 — Top-token agreement.** The highest-scoring token matches the 64-bit reference's at every position, except near-ties where the 64-bit reference's two highest logits differ by less than 1e-3; there either token passes.
- **S-4 — Greedy agreement.** The greedy continuation matches the 64-bit reference's token for token. If a near-tie (S-3) occurs during the continuation, comparison stops at that position and the harness reports it, because the two sequences may legitimately diverge from there.
- **S-9 — Sampling distribution agreement.** For every prompt and setting in the stored set (FR-31), the largest absolute difference between the engine's adjusted distribution and the stored one is at most 1e-5: the smallest round decimal at least twice the worst error measured on the prompt set, the rule used for S-1, so the worst case uses at most half of it. Every setting is judged, with no exemption for a cutoff near the engine's rounding, because none was found on the prompt set. The engine's distribution is judged, not its draws, because the distribution is the part with a correct answer. Not guaranteed: the value rests on measurements from one machine.
- **S-10 — Draws follow the distribution.** Over a fixed number of draws with a fixed seed, the observed token frequencies are consistent with the engine's adjusted distribution under a stated statistical test. The seed is fixed so that the test is deterministic: a test that can fail by chance is not a test (FR-28). Not guaranteed: a statistical test over finitely many draws cannot rule out a small bias; it catches a wrong distribution, not a slightly wrong one.
- **S-11 — Chat template agreement.** For every conversation in the stored set (FR-39) the rendered text equals the reference's character for character and the token IDs equal the reference's exactly. Both are checked because the text can match while the markers encode as characters (FR-33). There is no tolerance: a template is text, and a difference of one character is a different prompt.
- **S-5 — Judged per category.** A release passes only if every prompt category (FR-3) passes. Averages are never used, because an aggregate can stay flat while one category fails.
- **S-6 — Reproducible output.** For identical input, the same build on the same machine produces logits that differ between runs by no more than S-1 allows, and greedy continuations that are identical except at near-ties (S-3). Without this, a benchmark or correctness result cannot be reproduced. Bit-identical output is not required, because parallel computation may change the order of additions between runs; this engine's own threads do not (FR-43), so its output does not vary with the thread count.
- **S-7 — Benchmark conditions.** Every recorded benchmark runs on one designated machine. Each measurement is repeated and reported as the median with its spread, so a single noisy run is never taken as a result.
- **S-8 — Logging changes nothing.** Generated output is identical at every verbosity level, and recorded benchmarks run at the default level. Logging that altered results or timings would make the trace describe a different run.

## 12. Policies

These are requirements because the project's second purpose (§1), a measured and documented progression, depends on them.

- **Naming.** The project is Emberly Oxygen, part of the Emberly suite; its package and binary are named `emberly-oxygen`.
- **Release versioning.** Releases follow semantic versioning. Each release introduces one main technique, so its effect can be measured and explained on its own. After 0.4.0 each kernel technique is a patch release of the optimisation series (0.4.1, 0.4.2, and so on) so the series reads as one line; a new capability keeps its minor bump (decided by the owner, 2026-11-11).
- **Release notes.** Every release ships a note stating what changed, why it was expected to help, what was measured against the previous release, and what was learned.
- **Negative results kept.** A technique that fails or underperforms is recorded with the reason it did, and the record is never removed. Knowing where a technique stops working is part of understanding it.
- **Original design.** Other engines may be studied, but this engine's design is derived from first principles rather than copied. Copying a structure skips the understanding the project exists to build.

## 13. Open Questions

- **Whether the operations still on one thread after 0.4.1 (normalization, RoPE, SiLU, the elementwise operations, sampling and the final softmax) should be threaded.** Resolver: owner, at the next kernel release, from the share of a step the 0.4.1 profile and rows show they have left.
- **Whether a byte the vocabulary cannot represent should be warned about or rejected instead of silently dropped (FR-13).** Resolver: owner, when a release changes tokenization behavior; the engine and the reference agree today.

### Resolved since 0.6

- **Whether the oracle remains a gate once chat arrives (§1 purpose 3, HC-4).** Decided by the owner on 2026-11-08, at the bump that plans chat: the oracle stays the gate where its answer is exact or already proven, and reports where it would only repeat a proof. HC-4 and S-1 to S-4 keep judging the base model, which exercises every line of engine code the instruct model uses; the chat template and its tokens are judged exactly (S-11); the instruct model's logits and greedy replies are computed and reported, and judge nothing (FR-5, FR-39). The owner's view that tight numeric matching need not extend to later work is honored by not extending it to a second set of weights.

### Resolved since 0.5

- **The tolerance for S-9, and the draws and test for S-10 (§11).** Decided by the owner on 2026-11-08 at the first measurement, by the rule used for S-1. S-9: 1e-5, from a worst error of 1.98e-6 over 60 distributions (release note 0.2.0). S-10: 200,000 draws with a fixed seed and a chi-square test over tokens pooled to at least 5 expected draws, passing at df + 4·sqrt(2·df); measured 9.0 over 12 degrees of freedom (limit 31.6) and 442.7 over 422 (limit 538.2). The S-10 values are initial and live in the Specification (§11).
- **Whether S-9 exempts a setting whose cutoff the engine's rounding could cross.** No: it judges every setting. The engine's error at the position S-9 uses is at most 5.05e-5 in logits and the smallest cutoff margins are 4.3 (top-k) and 49 (top-p) times the matching errors, so no exemption is needed. The question arose in Specification 0.4 §19 and is closed there too.
- **Order of the sampling controls, the repetition penalty definition, and whether a seed reproduces across releases (FR-26 to FR-28).** Decided by the owner on 2026-11-08: the reference library's order; the library's definition; reproducible within a build only. Min-p and stopping on a string were considered and left out (§2.3).

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

### Feedback absorbed in 0.8

From plan 0.4.0 §6 (G-25). Every row addressed to this document, absorbed or rejected:

| Plan row | Disposition |
| --- | --- |
| Task 17.6, §2.2 and FR-40 | Absorbed into §2.2 and FR-40: the two sentences claimed that the release measures batching alone and that the speed-up is the speed-up of batching. Measured, the matrix-matrix loop order alone is no faster than repeated matrix-vector products when one matrix is kept warm in the cache (1.00×), and the release's speed-up was not traced to its sources (plan 0.4.0 §5: about 17% is the output projection computed for the last position only, and the remainder is unattributed). The sentences now say that no other kernel technique is mixed in, which is what the constraint is for. |

### Feedback absorbed in 0.7

From plan 0.3.0 §6 (G-25). Every row addressed to this document, absorbed or rejected:

| Plan row | Disposition |
| --- | --- |
| Task 16.5, FR-36 | Absorbed into FR-36: the chat command fixes the maximum reply length at its start, so a rejected turn can be followed by a shorter message and not by a smaller maximum; the words "or a smaller maximum" are removed. |

### Feedback absorbed in 0.6

From plan 0.2.0 §6 (G-25). Every row addressed to this document, absorbed or rejected:

| Plan row | Disposition |
| --- | --- |
| Before Phase 11, companion pin | Recorded: the header pinned Specification 0.3 when it was approved, before 0.4 existed, and was changed to 0.4 in place at the owner's order. |
| Before Phase 11, FR-26 | Absorbed into FR-26: top-k keeps tokens tied with the k-th. |
| Before Phase 11, FR-26, FR-29 | Absorbed into FR-26: temperature 0 ignores the other controls, with a warning. |
| Before Phase 11, S-9 (the cutoff exemption) | Not needed: S-9 stays "every prompt and setting" and now carries its value (resolved ledger above). |

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
