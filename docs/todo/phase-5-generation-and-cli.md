# Phase 5 — Generation and CLI: To-do

Working checklist for Phase 5 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.0.md) (milestone M5: the engine end to end). Tick items as they land; one task is one commit with a body paragraph and no co-author line, each ending with a passing `cargo test --no-fail-fast` whose **whole summary is read**. IDs cite the Requirements and Technical Specification.

Proves: greedy continuations match the oracle token for token (S-4), the command-line tool streams clean output and reports failures properly, logging changes nothing, and runs are reproducible. Satisfies FR-16 to FR-24, S-4, S-6, S-8.

Layout: `generate` stays in `crates/oxygen/src/generate.rs`; logging events are emitted from the library through `tracing` (Spec §14); the CLI is `crates/oxygen-cli/src/main.rs` (entry point, maps errors to exit codes) and `cli.rs` (arguments and the generate command), per Spec §1. The CLI holds no engine logic.

Dependencies, all already in Spec §15 so no Specification bump: `tracing` in the library; `clap` and `tracing-subscriber` in the CLI package only. Nothing else.

Out of scope: sampling, batching, performance work, the benchmark harness (Phase 6).

## Decision points for the owner

Each has a recommendation; none is decided by this to-do. The first two block work, so they are asked at the task named.

1. **Is the end-of-sequence token printed? (blocks 5.1.)** Spec §8 step 4 passes the chosen token to the streaming decoder, and step 5 then stops on EOS, so read literally the text `<|endoftext|>` would be emitted last. FR-16 and FR-17 do not say. The oracle's continuation includes the EOS id, and S-4 compares ids, so the **returned ids include EOS** either way. *Recommendation:* do not emit EOS's text to the callback, because standard output is meant to be the generated text and EOS is a control signal; record it in plan §9 for Spec §8. None of the 12 fixtures stops at EOS, so the harness will not exercise this; 5.1 has its own tests.
2. **The harness gets about twice as slow. (blocks 5.2.)** The greedy check calls `generate`, which feeds the prompt through the engine again after the logits check already did: about 5,560 steps against 2,596 today, so roughly 270 s instead of 125 s per `cargo test`. *Options:* (a) accept it, consistent with the Phase 4 choice to keep the engine by-the-book; (b) run the prompts' checks in parallel threads **in the harness only** (test code, not the engine, so the engine stays single-threaded and index-order, and results do not change), bringing the wall time to about the longest prompt, roughly a minute; (c) split the prefill between the checks, which would need `generate` to accept a prepared session and is the least preferred because it bends the public API for test economy. *Recommendation:* (b), after 5.2 measures the real figure; stop and ask if it exceeds about 4 minutes.
3. **Stdin and a trailing newline.** `echo hello | emberly-oxygen generate ...` sends `hello\n`. *Recommendation:* read stdin exactly as given, no stripping, so the prompt the engine sees is the bytes it was given; the README says to use `printf` or `echo -n` for no newline, and `--stats` and `-v` show the prompt token count. Invalid UTF-8 on stdin is an error (exit 1), never lossy.
4. **`--max-new-tokens 0`.** *Recommendation:* allowed; the prompt is validated and nothing is generated or printed. `--max-new-tokens` is required (Spec §9 shows no default).

## 5.1 Greedy generation (FR-16, FR-17, FR-18, Spec §8)

- [ ] `Config` gains `eos_token_id`, read from `config.json` (a single whole number; any other form is `Unsupported` naming the field, Spec §3 and FR-12); unit tests on altered copies of the real configuration
- [ ] **Resolve decision point 1 with the owner before writing the stop logic**
- [ ] Validation before any computation: an empty prompt (no tokens after encoding, which also covers text whose every byte the vocabulary lacks) and prompt length plus `max_new_tokens` above the context length are `Rejected`, naming the numbers (FR-18); nothing is computed and nothing is emitted
- [ ] Feed the prompt tokens through `Session::step` in order; take the next token from the last position's logits
- [ ] Argmax with the lowest id winning an exact tie, as the reference's does (FR-16): scan ascending and replace only on a strictly greater logit
- [ ] Stop after `max_new_tokens` tokens or at EOS; the returned ids include the EOS token (decision point 1); `max_new_tokens = 0` returns no ids (decision point 4)
- [ ] Stream each generated token through the `StreamDecoder` and call `on_text` with each non-empty completed chunk as soon as it is produced, then flush the decoder at the end (FR-17, FR-15); `on_text` is never called with an empty string
- [ ] Tests on a tiny model with the real tokenizer's byte alphabet (a model whose vocabulary covers the byte tokens): a model with all-zero weights gives equal logits, so the lowest id (0) is chosen every step; an EOS id that the tiny model emits stops generation with that id last; the maximum length stops it otherwise; an empty prompt and an oversized request are rejected with no step run; streamed chunks joined equal the decode of the ids (and a Thai character split across tokens is never emitted half-finished)
- [ ] Check: no `unsafe`, no threads, no new dependency; every test names FR-16, FR-17, FR-18 or Spec §8

## 5.2 The harness greedy check (S-4, FR-5)

- [ ] The check already exists from Phase 2 (`greedy_check`, with the near-tie stop); confirm it calls the real `generate` with the oracle's continuation length, and that the 12 comparisons pass strictly (no continuation has a near-tie: the smallest top-two gap is 0.0022)
- [ ] The first-diverging position is reported by `greedy_check` (FR-6); a unit test with `generate`'s ids shifted by one token shows the failure line
- [ ] **Measure the full harness time and apply decision point 2**: report the figure to the owner before changing anything; if option (b) is chosen, the prompts run in scoped threads in `tests/correctness.rs`, results are collected in prompt order, and a test shows the table is identical to a sequential run
- [ ] Check: with a real `generate`, the table shows every column passing for every prompt, so every category passes (S-5), and the run exits 0

## 5.3 Logging through `tracing` (FR-22, FR-23, S-8, Spec §14)

- [ ] Add `tracing` to the library; the library never prints (FR-22)
- [ ] Spans nested as `generate` → `step{pos}` → `layer{idx}`, so every event carries its place in the decode loop
- [ ] Events by level, exactly as the Spec §14 table: **info** model and configuration summary and load time (at `Model::load`), prompt token count, run summary with throughput; **debug** one event per decode step with position, token id, decoded text, the top 5 candidates with their logits, and the step time, prompt-feeding steps marked as such; **trace** per layer within each step: residual-stream norm after attention and after the MLP, and for each head the position receiving the most attention weight
- [ ] Per-layer and per-step statistics are computed only when the level is enabled (`tracing::enabled!`), and no statistic feeds back into the computation (S-8); `attention` reports the most-attended position per head only when asked
- [ ] The decoded text and top-5 candidates are built only when debug is enabled
- [ ] Measure that the disabled spans cost nothing visible: the Phase 4 harness-step time is unchanged within noise
- [ ] Tests with a small collecting subscriber (in the test only; `tracing-subscriber` stays CLI-only per Spec §15, so write the minimal `Subscriber` or collect through `tracing`'s own test facilities): at the info level the run summary appears and no per-step events; at debug there is one event per step with the fields above, prompt steps marked; at trace there is one per-layer event per layer per step with a most-attended position for every head
- [ ] Check: every test names FR-22, FR-23, S-8 or Spec §14

## 5.4 The command-line tool (FR-19 to FR-21, FR-23, FR-24, FR-20, Spec §9)

- [ ] `crates/oxygen-cli/Cargo.toml`: depend on the library by path, `clap` (derive), `tracing` and `tracing-subscriber`; the binary stays named `emberly-oxygen`
- [ ] `emberly-oxygen generate --model <dir> --max-new-tokens <n> [--stats] [-v|-vv|-vvv] [--log-file <path>] [PROMPT]`
- [ ] The prompt is the argument, or all of standard input when there is none, read exactly (decision point 3); invalid UTF-8 is an error
- [ ] Standard output carries only the generated text, written and flushed as each chunk arrives (FR-17, FR-20); a closed pipe on standard output is an error exit, not a panic message
- [ ] Errors go to standard error with exit code 1; usage errors (missing `--model`, a non-numeric count, an unknown flag) exit 2 (`clap`'s own usage exit); text already streamed stays printed and the error follows on standard error (Spec §14)
- [ ] `--stats` prints to standard error: prompt token count, generated token count, time to first token and generation tokens per second (FR-21), using Spec §12's definitions (time to first token from receiving the prompt string; model load reported separately). §12 leaves open whether the first token's cost belongs to prompt time or generation time; choose one, state it in the output's help text, and record it in plan §9 for Phase 6 to reuse
- [ ] `-v`, `-vv`, `-vvv` map to info, debug, trace (more `v`s stay at trace), default warn; `tracing-subscriber` formats events including the span path, to standard error
- [ ] `--log-file <path>` sends the same events to that file instead of standard error; the file is created before any work, and failing to create it is an error (exit 1) naming the path; standard output is unaffected (FR-24)
- [ ] Check: `cargo build` has no warnings; the binary's `--help` text is checked by hand

## 5.5 CLI tests that run the built binary (FR-18 to FR-21, Spec §9)

In `crates/oxygen-cli/tests/`, using `env!("CARGO_BIN_EXE_emberly-oxygen")` and the real model; the fixtures supply the expected text.

- [ ] Standard output is exactly the oracle's `continuation_text` for `english-short` with 32 new tokens (FR-19, FR-17, FR-20), and standard error is empty at the default level
- [ ] A prompt on standard input and the same prompt as an argument give identical output
- [ ] `--stats` writes the four figures to standard error and leaves standard output unchanged (FR-21)
- [ ] A request that exceeds the context (`--max-new-tokens 100000`) exits 1 with a message naming the numbers, prints nothing to standard output, and returns quickly because nothing was computed (FR-18)
- [ ] An empty prompt, a missing model directory (message names the path), a model directory with an unsupported `config.json` (message names the field) and invalid UTF-8 on standard input each exit 1 with standard output empty
- [ ] A missing `--model`, a non-numeric `--max-new-tokens` and an unknown flag exit 2
- [ ] `--log-file` leaves standard error empty and the file non-empty at `-vv`; an unwritable path exits 1
- [ ] Check: every test names FR-18 to FR-21 or Spec §9

## 5.6 Logging invariance (S-8)

- [ ] The same prompt at the default level, `-v`, `-vv` and `-vvv`, run through the built binary, gives byte-identical standard output (S-8)
- [ ] The same runs show what each level promises: nothing on standard error at the default level; a run summary at `-v`; one step event per generated token and per prompt token at `-vv`; per-layer events carrying the span path at `-vvv` (FR-23)
- [ ] Check: every test names S-8 or FR-23

## 5.7 Reproducibility (S-6)

- [ ] Two runs of the same prompt in one process and in two separate loads of the model give logits within S-1 of each other (the engine is single-threaded and deterministic, so they are expected to be bit-identical; assert the bound S-6 states, and report whether they are identical) and identical continuations (S-6, S-4)
- [ ] Check: the test names S-6

## Phase exit check

- [ ] The full `cargo test --no-fail-fast` passes in every category, including S-4 (greedy), and exits 0
- [ ] The CLI generates a recognizable continuation for an English prompt, run by hand and shown to the owner
- [ ] `-vvv` on a short prompt can be followed through the decode loop by eye (FR-23), shown to the owner
- [ ] Decision points 1 to 4 are resolved and recorded: 1 in plan §9 for Spec §8, the stats definition in §9 for Spec §12
- [ ] Every test added in this phase names the item it verifies (checked mechanically)
- [ ] Full test time recorded, `cargo build` has no warnings, and the summary of every test binary was read
- [ ] README: usage section for the CLI (the flags, stdin, exit codes, the log file), status line updated
- [ ] Owner reviews the CLI output, the verbosity levels and the tests before Phase 6 starts

## Notes to carry forward

- Prefill computes the output projection (about a fifth of a step's work) for every prompt position and discards all but the last. That is by the book here (Spec §7 returns logits for every call, and the harness needs them), and skipping it during generation is a candidate technique for a later version, to be measured on its own.
- Phase 6 reuses `--stats`'s definitions for the benchmark harness; they should agree with Spec §12.
