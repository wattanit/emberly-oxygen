# Phase 5 — Generation and CLI: To-do

Working checklist for Phase 5 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.0.md) (milestone M5: the engine end to end). Tick items as they land; one task is one commit with a body paragraph and no co-author line, each ending with a passing `cargo test --no-fail-fast` whose **whole summary is read**. IDs cite the Requirements and Technical Specification.

Proves: greedy continuations match the oracle token for token (S-4), the command-line tool streams clean output and reports failures properly, logging changes nothing, and runs are reproducible. Satisfies FR-16 to FR-24, S-4, S-6, S-8.

Layout: `generate` stays in `crates/oxygen/src/generate.rs`; logging events are emitted from the library through `tracing` (Spec §14); the CLI is `crates/oxygen-cli/src/main.rs` (entry point, maps errors to exit codes) and `cli.rs` (arguments and the generate command), per Spec §1. The CLI holds no engine logic.

Dependencies, all already in Spec §15 so no Specification bump: `tracing` in the library; `clap` and `tracing-subscriber` in the CLI package only. Nothing else.

Out of scope: sampling, batching, performance work, the benchmark harness (Phase 6).

## Decision points for the owner

Each has a recommendation; none is decided by this to-do. The first two block work, so they are asked at the task named.

1. **Is the end-of-sequence token printed? (blocks 5.1.)** *Resolved by the owner: the recommendation, EOS's text is not emitted.* Spec §8 step 4 passes the chosen token to the streaming decoder, and step 5 then stops on EOS, so read literally the text `<|endoftext|>` would be emitted last. FR-16 and FR-17 do not say. The oracle's continuation includes the EOS id, and S-4 compares ids, so the **returned ids include EOS** either way. *Recommendation:* do not emit EOS's text to the callback, because standard output is meant to be the generated text and EOS is a control signal; record it in plan §9 for Spec §8. None of the 12 fixtures stops at EOS, so the harness will not exercise this; 5.1 has its own tests.
2. **The harness gets about twice as slow. (blocks 5.2.)** *Resolved by the owner: option (b), after 5.2 measured 293 s.* The greedy check calls `generate`, which feeds the prompt through the engine again after the logits check already did: about 5,560 steps against 2,596 today, so roughly 270 s instead of 125 s per `cargo test`. *Options:* (a) accept it, consistent with the Phase 4 choice to keep the engine by-the-book; (b) run the prompts' checks in parallel threads **in the harness only** (test code, not the engine, so the engine stays single-threaded and index-order, and results do not change), bringing the wall time to about the longest prompt, roughly a minute; (c) split the prefill between the checks, which would need `generate` to accept a prepared session and is the least preferred because it bends the public API for test economy. *Recommendation:* (b), after 5.2 measures the real figure; stop and ask if it exceeds about 4 minutes.
3. **Stdin and a trailing newline.** *Resolved as recommended (5.4): read exactly, the help text says so.* `echo hello | emberly-oxygen generate ...` sends `hello\n`. *Recommendation:* read stdin exactly as given, no stripping, so the prompt the engine sees is the bytes it was given; the README says to use `printf` or `echo -n` for no newline, and `--stats` and `-v` show the prompt token count. Invalid UTF-8 on stdin is an error (exit 1), never lossy.
4. **`--max-new-tokens 0`.** *Done as recommended: allowed, nothing computed after validation (5.1); the CLI exits 0 with no output (5.4).* *Recommendation:* allowed; the prompt is validated and nothing is generated or printed. `--max-new-tokens` is required (Spec §9 shows no default).

## 5.1 Greedy generation (FR-16, FR-17, FR-18, Spec §8)

- [x] `Config` gains `eos_token_id`, read from `config.json` (a single whole number; any other form is `Unsupported` naming the field, Spec §3 and FR-12); unit tests on altered copies of the real configuration. An ID outside the vocabulary is `Malformed`, since generation could never stop on it
- [x] **Resolve decision point 1 with the owner before writing the stop logic.** Resolved: EOS's text is not emitted to the callback; the returned IDs include it. Recorded in plan §9 for Spec §8
- [x] Validation before any computation: an empty prompt (no tokens after encoding, which also covers text whose every byte the vocabulary lacks) and prompt length plus `max_new_tokens` above the context length are `Rejected`, naming the numbers (FR-18); nothing is computed and nothing is emitted
- [x] Feed the prompt tokens through `Session::step` in order; take the next token from the last position's logits
- [x] Argmax with the lowest id winning an exact tie, as the reference's does (FR-16): scan ascending and replace only on a strictly greater logit
- [x] Stop after `max_new_tokens` tokens or at EOS; the returned ids include the EOS token (decision point 1); `max_new_tokens = 0` returns no ids (decision point 4)
- [x] Stream each generated token through the `StreamDecoder` and call `on_text` with each non-empty completed chunk as soon as it is produced, then flush the decoder at the end (FR-17, FR-15); `on_text` is never called with an empty string
- [x] Tests on a tiny model with the real tokenizer's byte alphabet. Two tiny models: all-zero weights for the tie, and a one-layer "bigram" model whose MLP is a lookup table (next token a function of the current one), so a test can name the exact sequence generated, including a Thai character's three bytes and an EOS. The dropped-bytes prompt uses the real tokenizer. Covered (a model whose vocabulary covers the byte tokens): a model with all-zero weights gives equal logits, so the lowest id (0) is chosen every step; an EOS id that the tiny model emits stops generation with that id last; the maximum length stops it otherwise; an empty prompt and an oversized request are rejected with no step run; streamed chunks joined equal the decode of the ids (and a Thai character split across tokens is never emitted half-finished)
- [x] Check: no `unsafe`, no threads, no new dependency; every test names FR-16, FR-17, FR-18 or Spec §8
- [x] `tests/stub_smoke.rs` removed, as its own header said once the harness covers generation; with the real `generate` the harness's greedy column passes for all 12 prompts (5.2 confirms and measures it: the harness now takes about 293 s)

## 5.2 The harness greedy check (S-4, FR-5)

- [x] The check already exists from Phase 2 (`greedy_check`, with the near-tie stop); confirm it calls the real `generate` with the oracle's continuation length, and that the 12 comparisons pass strictly (no continuation has a near-tie: the smallest top-two gap is 0.0022). Confirmed: the call is `harness.rs` `greedy`, and the table shows `pass`, not `tie`, in the greedy column for all 12
- [x] The first-diverging position is reported by `greedy_check` (FR-6); a unit test with `generate`'s ids shifted by one token shows the failure line. In `tests/correctness.rs`; english-short's continuation starts with two newline tokens, so the shifted sequence first differs at index 1, and the test computes the position rather than assuming 0
- [x] **Measure the full harness time and apply decision point 2**: report the figure to the owner before changing anything; if option (b) is chosen, the prompts run in scoped threads in `tests/correctness.rs`, results are collected in prompt order, and a test shows the table is identical to a sequential run. Measured with a sequential harness after 5.1: 293 s (was 125 s). The owner chose option (b): one scoped thread per prompt, joined in manifest order; the table identity is checked on the three shortest prompts, errors included. The harness now takes about 61 s (16-core Apple M5, 12 performance cores), set by the longest prompts, and the whole `cargo test --no-fail-fast` about 68 s
- [x] Check: with a real `generate`, the table shows every column passing for every prompt, so every category passes (S-5), and the run exits 0. Worst errors are unchanged from Phase 4, to the last digit

## 5.3 Logging through `tracing` (FR-22, FR-23, S-8, Spec §14)

- [x] Add `tracing` to the library; the library never prints (FR-22). Version 0.1.44 with `default-features = false, features = ["std"]`, which drops the proc-macro `attributes` feature: the only new crates are `tracing-core` and `pin-project-lite`
- [x] Spans nested as `generate` → `step{pos}` → `layer{idx}`, so every event carries its place in the decode loop. `generate` is an info span, `step` a debug span created by `generate` around each forward step, `layer` a trace span in the forward pass; a direct `Session::step` (as the harness makes) has layer spans only
- [x] Events by level, exactly as the Spec §14 table: **info** model and configuration summary and load time (at `Model::load`), prompt token count, run summary with throughput; **debug** one event per decode step with position, token id, decoded text, the top 5 candidates with their logits, and the step time, prompt-feeding steps marked as such (one event per forward step: `phase` is `prompt` or `generate`, `token` and `text` are the token fed, and a step whose result is taken also carries `next` and `next_text`, so every prompt token appears once as `token` and every generated token once as `next`; the last generated token is never fed, plan §9 task 5.1, so a run has prompt + generated − 1 step events); **trace** per layer within each step: residual-stream norm after attention and after the MLP, and for each head the position receiving the most attention weight
- [x] Per-layer and per-step statistics are computed only when the level is enabled (`tracing::enabled!`), and no statistic feeds back into the computation (S-8); `attention` reports the most-attended position per head only when asked
- [x] The decoded text and top-5 candidates are built only when debug is enabled
- [x] Measure that the disabled spans cost nothing visible: the Phase 4 harness-step time is unchanged within noise. A new ignored measurement, `forward::tests::step_speed` (five rounds of 128 real-model steps), gave a median of 49.25 ms per step before the change and 49.29 to 49.32 ms after (+0.1%), inside the before-runs' own spread of 49.22 to 49.65
- [x] Tests with a small collecting subscriber (`src/test_log.rs`, test-only; it asks for every callsite each time so collectors on parallel test threads do not decide for one another) (in the test only; `tracing-subscriber` stays CLI-only per Spec §15, so write the minimal `Subscriber` or collect through `tracing`'s own test facilities): at the info level the run summary appears and no per-step events; at debug there is one event per step with the fields above, prompt steps marked; at trace there is one per-layer event per layer per step with a most-attended position for every head
- [x] Also tested: S-8 on the forward pass, with logits and captures bit-identical with no subscriber and at warn, debug and trace, and the trace norms checked against the captured layer outputs; the most-attended position per head on a hand-built case, the earliest winning a tie; the model-load event's fields; top-5 ordering on ties
- [x] The info run summary uses the throughput definitions that `--stats` will reuse (5.4): time to first token runs from receiving the prompt string to choosing the first token, prompt throughput is prompt tokens over that time, and generation throughput is the tokens after the first over the time after it (none for a single token). Recorded in plan §9 for Spec §12
- [x] Check: every test names FR-22, FR-23, S-8 or Spec §14

## 5.4 The command-line tool (FR-19 to FR-21, FR-23, FR-24, FR-20, Spec §9)

- [x] `crates/oxygen-cli/Cargo.toml`: depend on the library by path, `clap` (derive), `tracing` and `tracing-subscriber`; the binary stays named `emberly-oxygen`. `tracing` turned out not to be needed directly (the level type comes from `tracing-subscriber`), so it is not a direct dependency. clap 4.6.7; tracing-subscriber 0.3.23 with only `fmt` and `std`, so no `log` bridge and no ANSI colours, which keeps log files plain
- [x] `emberly-oxygen generate --model <dir> --max-new-tokens <n> [--stats] [-v|-vv|-vvv] [--log-file <path>] [PROMPT]`
- [x] The prompt is the argument, or all of standard input when there is none, read exactly (decision point 3); invalid UTF-8 is an error
- [x] Standard output carries only the generated text, written and flushed as each chunk arrives (FR-17, FR-20); a closed pipe on standard output is an error exit, not a panic message. The write callback prints the error and exits 1 at once, so a closed pipe stops generation instead of running to the maximum (`| head -c 5` on 400 tokens returns in 0.4 s)
- [x] Errors go to standard error with exit code 1; usage errors (missing `--model`, a non-numeric count, an unknown flag) exit 2 (`clap`'s own usage exit); text already streamed stays printed and the error follows on standard error (Spec §14)
- [x] `--stats` prints to standard error: prompt token count, generated token count, time to first token and generation tokens per second (FR-21), using Spec §12's definitions (time to first token from receiving the prompt string; model load reported separately). §12 leaves open whether the first token's cost belongs to prompt time or generation time; choose one, state it in the output's help text, and record it in plan §9 for Phase 6 to reuse. Chosen in 5.3 for the info run summary (the first token's cost is the prompt's, plan §9); `--stats` reports the same figures and states the definition in its help text. To give the CLI those figures through the public API only (Spec §1), `generate` now returns `Generation { ids, stats }`, where `Stats` holds the counts, time to first token and generation time with the two throughputs as methods; the run summary is computed from the same `Stats`, so the log and `--stats` agree by construction. Model load time is timed by the CLI around `Model::load` and `Tokenizer::load` and printed first. The report starts on a new line after an empty one, under a `---------- stats ----------` divider (added at the owner's request), because generated text has no trailing newline and on a terminal the two streams share a line
- [x] `-v`, `-vv`, `-vvv` map to info, debug, trace (more `v`s stay at trace), default warn; `tracing-subscriber` formats events including the span path, to standard error
- [x] `--log-file <path>` sends the same events to that file instead of standard error; the file is created before any work, and failing to create it is an error (exit 1) naming the path; standard output is unaffected (FR-24)
- [x] Log readability, found by running `-v` to `-vvv` by hand: timings are logged in milliseconds to three decimals and throughputs to two, and the trace norms in f32's shortest form, instead of raw f64 values such as `48.932334000000004`
- [x] Test infrastructure, found while running the unit tests repeatedly: per-test subscribers (`with_default`) on parallel test threads race in `tracing`'s callsite-interest cache, so a test could silently receive no events (about 1 run in 10, then 134 in 300 with collectors serialized). `test_log` now installs one global router subscriber once, which hands events to the current thread's collector, and rebuilds the interest cache before each collection: 0 failures in 1,000 runs of the unit tests
- [x] Checked by hand on the release binary: help texts; a run with `--stats`; `-v`, `-vv` and `-vvv`; `--log-file` (standard error empty, 7 lines in the file at `-vv` for a 4-token run) and an unwritable log path (exit 1, path named); context exceeded, empty prompt and missing model directory (exit 1, nothing on standard output); `--max-new-tokens 0` (exit 0, no output); standard input and argument giving identical output; invalid UTF-8 on standard input (exit 1); missing `--model`, a non-numeric count and an unknown flag (exit 2); a closed pipe (exit 1)
- [x] Check: `cargo build` has no warnings; the binary's `--help` text is checked by hand

## 5.5 CLI tests that run the built binary (FR-18 to FR-21, Spec §9)

In `crates/oxygen-cli/tests/`, using `env!("CARGO_BIN_EXE_emberly-oxygen")` and the real model; the fixtures supply the expected text.

Done in `crates/oxygen-cli/tests/cli.rs`: 10 tests, about 5 s. `serde_json` (already in Spec §15) is a dev-dependency of the CLI package to read the fixture manifest; it is not linked into the binary.

- [x] Standard output is exactly the oracle's `continuation_text` for `english-short` with 32 new tokens (FR-19, FR-17, FR-20), and standard error is empty at the default level
- [x] A prompt on standard input and the same prompt as an argument give identical output
- [x] `--stats` writes the four figures to standard error and leaves standard output unchanged (FR-21); the prompt token count is checked against the fixture's (30)
- [x] A request that exceeds the context (`--max-new-tokens 100000`) exits 1 with a message naming the numbers, prints nothing to standard output, and returns quickly because nothing was computed (FR-18). Shown deterministically: the run is at `-vv`, the model-load event is logged, and no step event is; a loose 5 s bound is checked too
- [x] An empty prompt (as an argument and as empty standard input), a missing model directory (message names the path), a model directory with an unsupported `config.json` (message names the field) and invalid UTF-8 on standard input each exit 1 with standard output empty
- [x] A missing `--model`, a non-numeric `--max-new-tokens` and an unknown flag exit 2
- [x] `--log-file` leaves standard error empty and the file non-empty at `-vv`; an unwritable path exits 1. The file is checked to hold the run summary and a `step{pos=0}` span, and standard output to equal a run without logging
- [x] Check: every test names FR-18 to FR-21 or Spec §9

## 5.6 Logging invariance (S-8)

- [x] The same prompt at the default level, `-v`, `-vv` and `-vvv`, run through the built binary, gives byte-identical standard output (S-8)
- [x] The same runs show what each level promises: nothing on standard error at the default level; a run summary at `-v`; at `-vv` one step event per forward step, every prompt token appearing once as `token` and every generated token once as `next` (5.3); per-layer events carrying the span path at `-vvv` (FR-23)
- [x] Done in `crates/oxygen-cli/tests/logging.rs` (2 tests, about 2 s) with "The quick brown fox" and 6 new tokens. At `-vv` the step events are checked in position order with their phase, `text`, `top5` and `step_ms`; `next` appears exactly on the steps whose result is taken, and each generated token but the last is the next step's `token`. At `-vvv` there are exactly 30 layer events per step, each with the full `generate:step{pos}:layer{idx}` path, both norms and 9 most-attended positions no later than the step's own. The run helpers moved to `tests/common/mod.rs`, shared with `cli.rs`
- [x] Check: every test names S-8 or FR-23

## 5.7 Reproducibility (S-6)

- [x] Two runs of the same prompt in one process and in two separate loads of the model give logits within S-1 of each other (the engine is single-threaded and deterministic, so they are expected to be bit-identical; assert the bound S-6 states, and report whether they are identical) and identical continuations (S-6, S-4)
- [x] Done in `crates/oxygen/tests/reproducibility.rs` on `english-short`: a second run on the same load and a run on a second load are bit-identical over all 30 positions (largest difference 0, S-1 asserted) and the 16-token continuations are identical, compared exactly. Recorded in plan §8. The test takes about 12 s, and since Cargo runs test binaries one after another, the whole `cargo test --no-fail-fast` now takes about 87 s
- [x] Check: the test names S-6

## Phase exit check

- [x] The full `cargo test --no-fail-fast` passes in every category, including S-4 (greedy), and exits 0
- [x] The CLI generates a recognizable continuation for an English prompt, run by hand and shown to the owner. Prepared: "Photosynthesis is the process by which plants" with 60 new tokens gives " convert light energy into chemical energy. The process is a complex one, involving the use of chlorophyll, a green pigment, ..." at about 20 tokens/s; shown to the owner, who accepted it
- [x] `-vvv` on a short prompt can be followed through the decode loop by eye (FR-23), shown to the owner. Prepared on "Paris is" with 2 new tokens: for each position the 30 layer lines, then the step line with its top 5 and the chosen `next` (" a", then " city"), then the run summary; shown to the owner, who accepted it
- [x] Decision points 1 to 4 are resolved and recorded: 1 in plan §9 for Spec §8, the stats definition in §9 for Spec §12 (task 5.3 row; 2 to 4 are marked resolved at the top of this file)
- [x] Every test added in this phase names the item it verifies (checked mechanically): a script lists every `#[test]` added since the Phase 5 to-do commit and the IDs in the comment above it; 38 new tests, none without an ID
- [x] Full test time recorded, `cargo build` has no warnings, and the summary of every test binary was read: `cargo test --no-fail-fast` takes about 85 s and exits 0 (the harness 61 s, reproducibility 12 s, negative controls 5 s, the CLI's two files 7 s); no warnings with or without the `negative-controls` feature; all 13 test binaries' summaries read, 196 tests passing, 2 ignored (the two speed measurements)
- [x] README: usage section for the CLI (the flags, stdin, exit codes, the log file), status line updated
- [x] Owner reviews the CLI output, the verbosity levels and the tests before Phase 6 starts. Accepted; the one change asked for, a divider before the `--stats` report, is made

## Notes to carry forward

- Prefill computes the output projection (about a fifth of a step's work) for every prompt position and discards all but the last. That is by the book here (Spec §7 returns logits for every call, and the harness needs them), and skipping it during generation is a candidate technique for a later version, to be measured on its own.
- Phase 6 reuses `--stats`'s definitions for the benchmark harness; they should agree with Spec §12. They are `emberly_oxygen::Stats` (plan §9, tasks 5.3 and 5.4), so the benchmark can take them from `generate` directly.
- `cargo test` takes about 85 s; the reproducibility test adds 12 s of it because Cargo runs test binaries one after another.
- Per-test `tracing` subscribers race on parallel test threads; library tests that read log events must go through `test_log::collect` (task 5.4).
