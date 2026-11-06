# Emberly Oxygen

A decoder-only LLM inference engine written from scratch in Rust, to understand every part by building it. It starts naive and improves one version at a time; the first target is SmolLM2-135M on CPU. The foundation documents in `docs/` are the source of truth: [Requirements](docs/emberly-oxygen-requirements.md), [Technical Specification](docs/emberly-oxygen-technical-specification.md), and the [roadmap](docs/high-level-roadmap.md) for direction.

Status: release 0.1.0, the baseline. The six phases of its plan are done (the oracle, the correctness harness, model loading and the tokenizer, the forward pass, generation with the command-line tool, and the baseline benchmark) apart from the owner's review and the `v0.1.0` tag. The engine matches the oracle on every check in every prompt category, and its first benchmark rows are in `results/benchmarks.jsonl`. See the [release note](docs/releases/0.1.0.md) for what was built, measured and learned.

## Setup on a new machine

You need:

- **Rust**, stable, recent enough for edition 2024 (1.85 or later), via [rustup](https://rustup.rs).
- **[uv](https://docs.astral.sh/uv/)**, which installs the pinned Python (3.12) and the oracle's dependencies itself. No other Python setup is needed.
- **Network access**, for the model download only. Everything else runs offline.
- About **1 GB of free disk**: the model files are about 270 MB and the fixtures about 670 MB.

```sh
git clone <repo-url> emberly-oxygen
cd emberly-oxygen

# 1. Oracle environment (one time; reproduces the pinned versions from uv.lock)
cd oracle
uv sync --frozen

# 2. Download the model into models/smollm2-135m/
uv run python download.py

# 3. Generate the expected outputs into fixtures/smollm2-135m/ (about 35 seconds)
uv run python generate.py

# 4. Check the prompt set against the model's tokenizer (optional)
uv run python validate_prompts.py

# 5. Generate the tokenizer's comparison corpus (about 1 MB, a few seconds)
uv run python tokenizer_corpus.py
cd ..

# 6. Build and test the engine
cargo test
```

`models/` and `fixtures/` are git-ignored. The prompt set, the oracle's lockfile and the pinned model commit are committed, so steps 2 and 3 rebuild them on any machine.

## Oracle scripts

Run them from `oracle/` with `uv run python <script>`. They find the repository root themselves.

| Script | Arguments | What it does |
| --- | --- | --- |
| `download.py` | none | Fetches `config.json`, `model.safetensors` and `tokenizer.json` of `HuggingFaceTB/SmolLM2-135M` at the pinned commit into `models/smollm2-135m/`. Checks the commit and file sizes. Safe to re-run: files already present are not downloaded again. The download is anonymous, so a stale Hugging Face token on the machine does not matter. |
| `generate.py` | `--out DIR` (optional) | Runs the model through Hugging Face `transformers` in float32 on CPU and writes one safetensors file per prompt plus `manifest.json`. Without `--out` it writes to `fixtures/smollm2-135m/`. Needs the model from `download.py`. |
| `validate_prompts.py` | none | Prints token counts per prompt and fails if `oracle/prompts.json` breaks a category rule. |
| `tokenizer_corpus.py` | none | Runs the reference tokenizer over about 4,000 strings (the prompts, edge cases and seeded random mixtures of Thai, white space, contractions, digits and emoji) and writes the expected pre-tokenizer pieces and token IDs, plus the oracle's decoded text for every string, every single token and 4,000 random ID sequences, to `fixtures/smollm2-135m/tokenizer-corpus.json`. The engine's tokenizer tests compare against it. Needs the model from `download.py`. |
| `unicode_classes.py` | none | Measures which characters the reference treats as letters, numbers and white space, by running every Unicode scalar value through its pre-tokenizer, and writes `crates/oxygen/src/tokenizer/unicode_classes.rs`. That file is committed; re-run the script only after changing the model revision. |
| `rope_reference.py` | none | Prints the reference's rotary-embedding cosines and sines for SmolLM2 (positions 1 and 5000) as Rust constants. They are pasted into the engine's RoPE test, so the script is only needed to regenerate them. Needs the model from `download.py`. |

`--out` is for comparing runs: for example, `uv run python generate.py --out /tmp/run2` writes a second set that you can diff against the first. The engine's tests only read `fixtures/smollm2-135m/`.

### Changing things

- **Prompts:** edit `oracle/prompts.json`, run `validate_prompts.py`, then re-run `generate.py`. Fixtures are not versioned, so regenerate them after any change.
- **Model revision:** the pinned commit is `REVISION` in `oracle/download.py`. Changing it invalidates every stored fixture.
- **Another machine's fixtures:** they can differ from yours in the last bits of the reference's own arithmetic, far inside the test tolerances. `manifest.json` records which machine and library versions produced them.

## Engine

```sh
cargo build                   # debug build
cargo test                    # every test (optimized, debug assertions on); about 90 s, of which the harness running the 12 prompts in parallel threads takes about 60
cargo test --no-fail-fast     # same, but keeps running other test files after one fails
cargo build --release         # optimized build; the binary is target/release/emberly-oxygen
```

The correctness harness never downloads files and never runs Python: if `models/` or `fixtures/` is missing, it fails and names the command that creates it.

## Command-line tool

```sh
emberly-oxygen generate --model <DIR> --max-new-tokens <N> [--stats] [-v|-vv|-vvv] [--log-file <PATH>] [PROMPT]
```

For example, after `cargo build --release`:

```sh
target/release/emberly-oxygen generate --model models/smollm2-135m --max-new-tokens 32 "The quick brown fox"
```

It generates greedily (always the highest-scoring token) and streams the text to standard output as it is produced. Standard output carries the generated text and nothing else; everything else goes to standard error.

| Option | What it does |
| --- | --- |
| `--model <DIR>` | Model directory with `config.json`, `model.safetensors` and `tokenizer.json`. Required. |
| `--max-new-tokens <N>` | The most tokens to generate. Required. Generation also stops at the end-of-sequence token, which is not printed. `0` checks the prompt and prints nothing. |
| `PROMPT` | The prompt. Without it, all of standard input is read exactly as given, so `echo hello \| ...` sends `hello` with a trailing newline; use `printf` or `echo -n` to leave it out. Input that is not UTF-8 is an error. |
| `--stats` | Prints to standard error: model load time, prompt and generated token counts, time to first token and generation throughput. The first generated token counts as prompt work: time to first token runs from receiving the prompt to choosing the first token, and throughput is the tokens after the first per second after it. |
| `-v`, `-vv`, `-vvv` | Log to standard error: `-v` model summary, prompt token count and a run summary with timings; `-vv` adds one line per decode step (position, token, its text, the top 5 candidates, step time; `next` is the token chosen); `-vvv` adds one line per layer per step (residual-stream norms and the position each head attends to most). Each line shows where it happened, for example `generate:step{pos=3}:layer{idx=0}`. By default only warnings and errors are logged. |
| `--log-file <PATH>` | Writes the log to this file instead of standard error. The file is created before any work. `-vvv` writes 30 lines per token, so use it for long runs. |

Exit codes: `0` on success; `1` on an error (missing or unsupported model files, an empty prompt, a prompt plus `--max-new-tokens` longer than the model's context, input that is not UTF-8, a log file that cannot be created, or standard output closed early), with the message on standard error and any text already generated left printed; `2` on a usage error such as a missing option or an unknown flag.

## Benchmark

```sh
cargo bench                   # about 5 minutes; appends to results/benchmarks.jsonl
cargo bench -- --dry-run      # the same, printing the rows instead of appending them
```

The harness runs two workloads, a 32-token and a 512-token prompt each followed by 128 generated tokens (texts in `crates/oxygen/benches/prompts.json`), every run in a fresh process: one warm-up and five recorded runs, reported as median, minimum and maximum. It measures time to first token, prompt and generation throughput, model load time and peak memory, and then times the engine's matrix-vector product against `matrixmultiply` at each of the model's matrix shapes. Each row carries its conditions: version, commit, whether the tree was dirty, machine, build profile and time. It refuses a build with debug assertions and needs neither network nor Python.

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
