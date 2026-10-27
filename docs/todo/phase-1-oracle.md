# Phase 1 — Oracle: To-do

Working checklist for Phase 1 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.0.md) (milestone M1: the reference exists). Tick items as they land; one task is one commit. IDs cite the Requirements and Technical Specification.

## 1.1 Workspace skeleton (Spec §1, §16)

- [x] Root `Cargo.toml` as a workspace with members `crates/oxygen` and `crates/oxygen-cli`
- [x] Set edition 2024 and the stable toolchain (no `rust-toolchain.toml`; owner has not asked for pinning)
- [x] Test profile: `opt-level = 3` with `debug-assertions = true` and `overflow-checks = true` (Spec §11)
- [x] `crates/oxygen`: package `emberly-oxygen`, empty `lib.rs`
- [x] `crates/oxygen-cli`: package `emberly-oxygen-cli`, binary named `emberly-oxygen`, `main.rs` that compiles and does nothing yet
- [x] `.gitignore`: `target/`, `fixtures/`, `models/`, `oracle/.venv/`
- [x] Create `results/` with a placeholder so it is tracked; `fixtures/` and `models/` are created by the oracle scripts
- [x] AGPL `LICENSE` file (roadmap decision)
- [x] Check: `cargo build` and `cargo test` pass on the empty workspace

## 1.2 Oracle project (HC-3, FR-4, Spec §10)

- [ ] `oracle/` initialised with `uv`; Python project metadata with a stated minimum Python
- [ ] Add `torch`, `transformers`, `huggingface_hub` (and `safetensors`, `numpy` if the generator needs them); commit `uv.lock`
- [ ] Install `torch` as a CPU build so no GPU kernel can change the arithmetic (FR-2)
- [ ] Check: `uv sync` from a clean checkout reproduces the environment; record library versions for the manifest

## 1.3 Download script (FR-11, FR-4)

- [ ] Choose and record the pinned commit of the SmolLM2-135M repository (`HuggingFaceTB/SmolLM2-135M`); verify it is the base model, not the instruct variant
- [ ] `oracle/download.py` fetches exactly `config.json`, `model.safetensors` and `tokenizer.json` into `models/<model-id>/`
- [ ] Fails if the downloaded commit differs from the pin; verifies the files exist afterwards
- [ ] Record the model ID, commit and file sizes in the script's output
- [ ] Decide how `<model-id>` is spelled (for example, `smollm2-135m`) and use it for both `models/` and `fixtures/`
- [ ] Check: re-running is a no-op when files are already present

## 1.4 Prompt set (FR-3, Spec §10)

- [ ] `oracle/prompts.json` with `id`, `category`, `text` per prompt
- [ ] `english`: at least two prompts, including one short (about 32 tokens, the short benchmark workload)
- [ ] `thai`: at least two prompts
- [ ] `code`: at least two prompts in different languages or styles
- [ ] `utf8-split`: at least two prompts whose tokens split a multi-byte character; confirm by decoding token by token and seeing a partial sequence
- [ ] `long`: at least two prompts of about 512 tokens (the long benchmark workload); confirm the token count with the reference tokenizer
- [ ] Check: a small validation script lists prompt count and token count per category

## 1.5 Fixture generator (FR-1, FR-2, S-3, S-4, Spec §10)

- [ ] `oracle/generate.py` loads the model from `models/<model-id>/` in float32 on CPU, eager attention, `torch.no_grad`
- [ ] Per prompt: tokenize with the reference tokenizer exactly as the model's files declare, and save the token IDs
- [ ] Forward pass with hidden states returned; save logits `[positions, vocab]`, embeddings `[positions, hidden]`, layer outputs `[layers, positions, hidden]`
- [ ] Greedy continuation of 32 tokens by the script's own argmax loop (not a library `generate` call); on exact ties take the lowest ID
- [ ] Record the gap between the top two logits at every continuation step, and for the prompt positions too if cheap
- [ ] One safetensors file per prompt in `fixtures/<model-id>/`
- [ ] Check: the continuation's first step agrees with the argmax of the last prompt-position logits

## 1.6 Final-norm labelling (FR-1, Spec §10 honesty note)

- [ ] Confirm in the installed `transformers` version that the last entry of the returned hidden states is taken after the final norm
- [ ] Store that entry under a distinct name (for example, `final_norm_output`) rather than as the last layer, so the harness cannot compare it against the wrong engine tensor
- [ ] Check: layer count in the file is the model's layer count, with the post-norm tensor stored separately

## 1.7 Manifest (FR-1, FR-4, Spec §10)

- [ ] `fixtures/<model-id>/manifest.json` with model ID and commit, library versions, generating machine, and a timestamp
- [ ] Per prompt: ID, category, text, token IDs, continuation IDs and text, top-two gaps, and the name of its safetensors file
- [ ] The manifest alone is enough for the harness to find every file (Spec §11)

## Phase exit check

- [ ] Generator runs end to end and produces a manifest plus one safetensors file per prompt, for all five categories
- [ ] A second run on the same machine reproduces token IDs and continuations exactly; note the largest difference in logits between runs
- [ ] The `long` fixtures are about 512 tokens; record the largest fixture size
- [ ] `git status` shows no fixtures or model files; the prompt set, lockfile, pinned commit and scripts are committed
- [ ] Owner reviews the prompt set and the manifest before Phase 2 starts

## Notes to carry forward

- Requirements S-3 has corrupted text; its intended wording is needed before task 2.3, not before this phase.
- The size of the long fixtures (about 100 MB each) determines disk needs for the benchmark machine; note the actual number.
