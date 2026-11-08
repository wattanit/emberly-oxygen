# Emberly Oxygen

A decoder-only LLM inference engine written from scratch in Rust, to understand every part by building it. It starts naive and improves one version at a time; the first target is SmolLM2-135M on CPU. The foundation documents in `docs/` are the source of truth: [Requirements](docs/emberly-oxygen-requirements.md), [Technical Specification](docs/emberly-oxygen-technical-specification.md), and the [roadmap](docs/high-level-roadmap.md) for direction.

Status: release 0.3.0 is built and measured on the branch `release/0.3.0`, awaiting the owner's review and the tag; the last tagged release is 0.2.0 (`v0.2.0`). 0.3.0 adds chat with the instruct model, SmolLM2-135M-Instruct: the chat template (written in Rust and checked against the reference library's, text and token ids exactly), replies that end at the end-of-turn token, a conversation kept across turns with the context limit enforced on the whole of it, chat sampling defaults (temperature 0.2, top-p 0.9, from the model card) with one seeded generator per conversation, and an interactive `chat` command. The engine's arithmetic, the greedy path and sampling are unchanged (a guard test pins their output). The instruct model's numbers against the float64 reference (logits within 9e-5, every greedy reply identical) are reported and judge nothing; the template is judged. A turn costs the whole conversation, since nothing is cached across turns: 0.55 s to the first token at 67 prompt tokens and 6.5 s at 644. The kernels contain no `unsafe`, no intrinsics and no threads; the only `unsafe` in the engine is the memory-map call in the model loader. See the [0.3.0 release note](docs/releases/0.3.0.md), the [0.2.0](docs/releases/0.2.0.md), [0.1.2](docs/releases/0.1.2.md), [0.1.1](docs/releases/0.1.1.md) and [0.1.0](docs/releases/0.1.0.md) notes for the history.

## Setup on a new machine

You need:

- **Rust**, stable, recent enough for edition 2024 (1.85 or later), via [rustup](https://rustup.rs).
- **[uv](https://docs.astral.sh/uv/)**, which installs the pinned Python (3.12) and the oracle's dependencies itself. No other Python setup is needed.
- **Network access**, for the model download only. Everything else runs offline.
- About **3 GB of free disk**: the two models are about 270 MB each, and the fixtures about 2 GB for the base model (each prompt is stored in float32 and float64) and 100 MB for the instruct model's chat conversations.

```sh
git clone <repo-url> emberly-oxygen
cd emberly-oxygen

# 1. Oracle environment (one time; reproduces the pinned versions from uv.lock)
cd oracle
uv sync --frozen

# 2. Download the model into models/smollm2-135m/
uv run python download.py

# 3. Generate the expected outputs into fixtures/smollm2-135m/ (about 100 seconds: float32 and float64)
uv run python generate.py

# 4. Generate the sampling distributions into the same folder (about 1 second, 23 MB; reads step 3's float64 files)
uv run python sampling_reference.py

# 5. Check the prompt set against the model's tokenizer (optional)
uv run python validate_prompts.py

# 6. Generate the tokenizer's comparison corpus (about 1 MB, a few seconds)
uv run python tokenizer_corpus.py

# 7. Download the instruct model into models/smollm2-135m-instruct/ and generate its chat fixtures (about 15 seconds, 100 MB)
uv run python download.py smollm2-135m-instruct
uv run python chat_reference.py
cd ..

# 8. Build and test the engine
cargo test
```

`models/` and `fixtures/` are git-ignored. The prompt set, the oracle's lockfile and the pinned model commit are committed, so steps 2 to 4 and 7 rebuild them on any machine.

## Oracle scripts

Run them from `oracle/` with `uv run python <script>`. They find the repository root themselves.

| Script | Arguments | What it does |
| --- | --- | --- |
| `download.py` | `smollm2-135m` (the default) or `smollm2-135m-instruct` | Fetches `config.json`, `model.safetensors` and `tokenizer.json` of the chosen model, `HuggingFaceTB/SmolLM2-135M` or `HuggingFaceTB/SmolLM2-135M-Instruct`, at its pinned commit into `models/<model-id>/` (the instruct model also gets `tokenizer_config.json`, which only the oracle reads, for the chat template). Checks the commit and file sizes. Safe to re-run: files already present are not downloaded again. The download is anonymous, so a stale Hugging Face token on the machine does not matter. |
| `generate.py` | `--out DIR` (optional) | Runs the model through Hugging Face `transformers` on CPU twice, in float64 (the reference the correctness checks judge by; RMSNorm and softmax are forced to float64, the rotary tables stay as the library makes them) and in float32 (the library's default, reported against), and writes two safetensors files per prompt (`<id>.safetensors`, `<id>.f64.safetensors`) plus `manifest.json`. Without `--out` it writes to `fixtures/smollm2-135m/`. Needs the model from `download.py`. |
| `sampling_reference.py` | none | Run after `generate.py`. Reads the stored float64 logits (it does not run the model again) and writes, per prompt, the distribution the reference library's repetition penalty, temperature, top-k and top-p processors give for each setting in `sampling_settings.json` (`<id>.sampling.safetensors`, about 23 MB in all, in about 1 s), and adds a `sampling` entry to the manifest. Used by the sampling checks (FR-31). |
| `chat_reference.py` | `--out DIR` (optional) | Needs the instruct model from `download.py smollm2-135m-instruct`. Renders each conversation in `conversations.json` with the reference library's chat template, encodes it, runs the instruct model in float64 (the same reference setup as `generate.py`, in `reference_common.py`), and writes `chat.json` and one `<id>.f64.safetensors` per conversation (the logits at every prompt position) to `fixtures/smollm2-135m-instruct/`, with a greedy reply of up to 64 tokens. That folder has no `manifest.json`, so the correctness checks S-1 to S-4 never judge the instruct model; its template is judged (S-11) and its logits and replies are reported only. |
| `validate_prompts.py` | none | Prints token counts per prompt and fails if `oracle/prompts.json` breaks a category rule. |
| `tokenizer_corpus.py` | none | Runs the reference tokenizer over about 4,000 strings (the prompts, edge cases and seeded random mixtures of Thai, white space, contractions, digits and emoji) and writes the expected pre-tokenizer pieces and token IDs, plus the oracle's decoded text for every string, every single token and 4,000 random ID sequences, to `fixtures/smollm2-135m/tokenizer-corpus.json`. The engine's tokenizer tests compare against it. Needs the model from `download.py`. |
| `unicode_classes.py` | none | Measures which characters the reference treats as letters, numbers and white space, by running every Unicode scalar value through its pre-tokenizer, and writes `crates/oxygen/src/tokenizer/unicode_classes.rs`. That file is committed; re-run the script only after changing the model revision. |
| `rope_reference.py` | none | Prints the reference's rotary-embedding cosines and sines for SmolLM2 (positions 1 and 5000) as Rust constants. They are pasted into the engine's RoPE test, so the script is only needed to regenerate them. Needs the model from `download.py`. |

`--out` is for comparing runs: for example, `uv run python generate.py --out /tmp/run2` writes a second set that you can diff against the first. The engine's tests only read `fixtures/smollm2-135m/`.

### Changing things

- **Prompts:** edit `oracle/prompts.json`, run `validate_prompts.py`, then re-run `generate.py`. Fixtures are not versioned, so regenerate them after any change.
- **Model revision:** each model's pinned commit is in `MODELS` in `oracle/download.py`. Changing one invalidates every stored fixture of that model.
- **Conversations:** edit `oracle/conversations.json` and re-run `chat_reference.py`.
- **Another machine's fixtures:** the float32 files can differ from yours by the library's own rounding, up to about 7e-4 in logits from the float64 result, which is why the checks judge by the float64 files; those differed by about 2e-6 between thread counts on one machine and have not been compared across machines. `manifest.json` records which machine and library versions produced them.

## Engine

```sh
cargo build                   # debug build
cargo test                    # every test (optimized, debug assertions on); about 42 s of test time, of which the harness running the 12 prompts in parallel threads takes about 15
cargo test --no-fail-fast     # same, but keeps running other test files after one fails
cargo build --release         # optimized build; the binary is target/release/emberly-oxygen
```

`cargo test --test correctness -- --nocapture` prints the harness's table for a passing run: each category's worst error against the float64 reference (what S-1 to S-4 judge) and, in separate columns labelled as reported only, against the float32 reference. `cargo test --test correctness chat_harness -- --nocapture` prints the chat template check (S-11) and the instruct model's reported-only figures.

The correctness harness never downloads files and never runs Python: if `models/` or `fixtures/` is missing, it fails and names the command that creates it.

## Command-line tool

```sh
emberly-oxygen generate --model <DIR> --max-new-tokens <N> [--temperature <T>] [--top-k <K>] [--top-p <P>] [--repetition-penalty <R>] [--seed <N>] [--stats] [-v|-vv|-vvv] [--log-file <PATH>] [PROMPT]
```

For example, after `cargo build --release`:

```sh
target/release/emberly-oxygen generate --model models/smollm2-135m --max-new-tokens 32 "The quick brown fox"
```

By default it generates greedily (always the highest-scoring token); with `--temperature` above 0 it samples. It streams the text to standard output as it is produced. Standard output carries the generated text and nothing else; everything else goes to standard error.

| Option | What it does |
| --- | --- |
| `--model <DIR>` | Model directory with `config.json`, `model.safetensors` and `tokenizer.json`. Required. |
| `--max-new-tokens <N>` | The most tokens to generate. Required. Generation also stops at the end-of-sequence token, which is not printed. `0` checks the prompt and prints nothing. |
| `--temperature <T>` | Divides the logits by `T`. `0`, the default, is greedy selection, and then the other sampling options are ignored with a warning. Above `0` the run is sampled and its seed is printed. |
| `--top-k <K>` | Sample only among the `K` highest-scoring tokens; tokens tied with the `K`-th stay. `0`, the default, is off. |
| `--top-p <P>` | Sample only among the smallest set of most likely tokens whose probabilities add up to at least `P`, `0 < P <= 1`. `1`, the default, is off. |
| `--repetition-penalty <R>` | For every token already in the prompt or the output, divide a positive score and multiply a negative one by `R`, as the reference library does; `R > 0`. `1`, the default, is off. |
| `--seed <N>` | Seed of the random number generator. The same seed, prompt, settings and build give the same text; the same seed across releases is not promised. When sampling without one, a seed is taken from the clock. Sampling always prints `seed: <N>` to standard error before any text, so a run can be replayed. |
| `PROMPT` | The prompt. Without it, all of standard input is read exactly as given, so `echo hello \| ...` sends `hello` with a trailing newline; use `printf` or `echo -n` to leave it out. Input that is not UTF-8 is an error. |
| `--stats` | Prints to standard error: model load time, prompt and generated token counts, time to first token and generation throughput. The first generated token counts as prompt work: time to first token runs from receiving the prompt to choosing the first token, and throughput is the tokens after the first per second after it. |
| `-v`, `-vv`, `-vvv` | Log to standard error: `-v` model summary, prompt token count and a run summary with timings; `-vv` adds one line per decode step (position, token, its text, the top 5 candidates, step time; `next` is the token chosen); `-vvv` adds one line per layer per step (residual-stream norms and the position each head attends to most). Each line shows where it happened, for example `generate:step{pos=3}:layer{idx=0}`. By default only warnings and errors are logged. |
| `--log-file <PATH>` | Writes the log to this file instead of standard error. The file is created before any work. `-vvv` writes 30 lines per token, so use it for long runs. |

Controls apply in the reference library's order: repetition penalty, temperature, top-k, top-p; then one token is drawn. An invalid value (a negative temperature, a top-p outside 0 to 1, a penalty of 0 or less) is an error that names the option and the value.

Exit codes: `0` on success; `1` on an error (missing or unsupported model files, an empty prompt, an invalid sampling value, a prompt plus `--max-new-tokens` longer than the model's context, input that is not UTF-8, a log file that cannot be created, or standard output closed early), with the message on standard error and any text already generated left printed; `2` on a usage error such as a missing option or an unknown flag.

### Chat

```sh
emberly-oxygen chat --model <DIR> [--system <TEXT>] [--max-new-tokens <N>] [--temperature <T>] [--top-k <K>] [--top-p <P>] [--repetition-penalty <R>] [--seed <N>] [-v|-vv|-vvv] [--log-file <PATH>]
```

For example, after `cargo build --release` and the instruct model's download:

```sh
target/release/emberly-oxygen chat --model models/smollm2-135m-instruct
```

It reads one message per line from standard input and streams each reply to standard output, followed by a newline. It keeps the conversation: every turn renders the whole history again in the model's chat template. Standard output carries the replies and nothing else; the prompt marker `> `, errors and the seed go to standard error, so a chat can be driven from a pipe (`printf 'What is the capital of France?\n' | ...`). It ends at end of input or a line that is exactly `/exit`; blank lines send nothing.

| Option | What it does |
| --- | --- |
| `--model <DIR>` | Model directory of an instruct model. Required. |
| `--system <TEXT>` | A system message to start the conversation with, instead of the model's default ("You are a helpful AI assistant named SmolLM, trained by Hugging Face"). |
| `--max-new-tokens <N>` | The most tokens in one reply; default 256. A reply also ends at the model's end-of-turn token, which is not printed. |
| `--temperature <T>` | Default **0.2**, the instruct model's published example. `0` gives greedy replies. |
| `--top-p <P>` | Default **0.9**, or `1` (off) when `--temperature 0` is given and this is not, so asking for greedy chat is not answered with a warning about a control you never set. |
| `--top-k <K>`, `--repetition-penalty <R>`, `--seed <N>` | As for `generate`. One generator, built from the seed, serves the whole conversation; when sampling without a seed, one is taken from the clock and `seed: <N>` is printed once to standard error before the first prompt, so the conversation can be replayed. |
| `-v`, `-vv`, `-vvv`, `--log-file <PATH>` | As for `generate`; each turn is logged like a generation, and the position restarts at 0 each turn because the whole conversation is processed again. |

A turn whose conversation plus `--max-new-tokens` would not fit the model's context (8192 positions) is rejected before anything is computed: the error goes to standard error and the chat continues with the conversation as it was. A message that spells a chat marker such as `<|im_end|>` is read by the tokenizer as the marker, as in the reference library, so it ends its own turn. The exit codes are those of `generate`: `0` at end of input or `/exit`, `1` on an error (a missing model, an invalid sampling value, input that is not UTF-8), `2` on a usage error.

## Benchmark

```sh
cargo bench                   # about 5 minutes; appends to results/benchmarks.jsonl
cargo bench --bench benchmarks -- --dry-run   # the same, printing the rows instead of appending them (the plain `cargo bench -- --dry-run` fails: the library's own test binary rejects the flag)
```

The harness runs two workloads, a 32-token and a 512-token prompt each followed by 128 generated tokens (texts in `crates/oxygen/benches/prompts.json`), every run in a fresh process: one warm-up and five recorded runs, reported as median, minimum and maximum. It measures time to first token, prompt and generation throughput, model load time and peak memory, and then times the engine's matrix-vector product at each of the model's matrix shapes against `matrixmultiply`, the 0.1.0 index-order loop and partial-sum widths 4, 8, 16 and 32. Each row carries its conditions: version, commit, whether the tree was dirty, machine, build profile and time. It refuses a build with debug assertions and needs neither network nor Python.

It also times, in its own process, the matrix-vector kernel at every shape in the model and, since 0.2.0, one sampling step on a vocabulary-sized logit vector for each control (`cargo bench --bench benchmarks -- --sampling-only --dry-run` runs just that group). The two workloads stay greedy, so their rows remain comparable with every earlier release's; chat has no benchmark group of its own, because a turn's cost is the forward pass over the whole history, which the 512-token workload already measures per token.

Recorded results come from the benchmark machine only (an Apple M4 Max), after `cargo test` has passed; rows from anywhere else, or with `"dirty": true`, are not results.

## Repository layout

```
crates/oxygen/       the engine library (package emberly-oxygen)
crates/oxygen-cli/   the command-line tool (binary emberly-oxygen)
oracle/              Python reference oracle: the only Python in the project
fixtures/            expected outputs, git-ignored
models/              downloaded model files, git-ignored
results/             cumulative benchmark record
docs/                requirements, specification, plan, roadmap, per-phase to-dos, release notes
```

## License

AGPL-3.0-or-later. See `LICENSE`.
