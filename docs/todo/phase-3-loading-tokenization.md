# Phase 3 — Loading and tokenization: To-do

Working checklist for Phase 3 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.0.md) (milestone M3: the engine reads the model as published). Tick items as they land; one task is one commit, each ending with a passing `cargo test --no-fail-fast`. IDs cite the Requirements and Technical Specification.

Proves: encoding, decoding and streaming match the oracle; the loader accepts the published files and rejects anything else by name. Satisfies FR-11 to FR-15, HC-1.

Layout: library code in `crates/oxygen/src/` (new modules `config`, `tensor`, `safetensors`, `weights`, and `tokenizer/` split by concern as it grows); tests that need the real model stay in `crates/oxygen/tests/`. Unit tests build tiny inputs in memory so they run without `models/`.

Facts already checked, to be confirmed and recorded in 3.6: SmolLM2's `tokenizer.json` has `model.type` = BPE (49152 vocab entries, 48900 merges, 17 added tokens), `normalizer` = null, `pre_tokenizer` = Sequence of `Digits{individual_digits: true}` then `ByteLevel{add_prefix_space: false, use_regex: true}`, `post_processor` = null (so no BOS is added), `decoder` = ByteLevel.

Dependencies: `memmap2` is the only addition (needed by 3.4), and it is already in Spec §15, so no Specification bump. Nothing else.

## 3.1 Error type (FR-12, Spec §14)

The type landed with the stub in task 2.1, so this task is a check, not new code.

- [x] Confirm `Error` has exactly the five Spec §14 classes (I/O, malformed file, unsupported component, tensor mismatch, request rejected) and no more
- [x] Add a unit test per variant that its message names what Spec §14 says it names (path, component and value, tensor name)
- [x] Check: no later task in this phase adds a variant; one that seems to need it goes to the upstream-feedback table instead

## 3.2 Configuration parsing and supported-set check (FR-12, Spec §3)

- [x] `config.rs`: typed `Config` read from `config.json` with sizes (hidden, layers, heads, kv heads, head dim, intermediate, vocab, context length), `rope_theta` and `rms_norm_eps` taken from the file, never hard-coded
- [x] Head dimension: use the file's value when present, else hidden size divided by heads; reject a hidden size that does not divide evenly, naming the fields
- [x] Supported-set check for every row of the Spec §3 table (`model_type`, `hidden_act`, `attention_bias`, `mlp_bias`, `rope_scaling`, `rope_interleaved`, `tie_word_embeddings`, `pretraining_tp`), with absent fields handled as the table says
- [x] Rejection is `Error::Unsupported` naming the field and the value found
- [x] Reject inconsistent sizes (heads not a multiple of kv heads, zero sizes) as `Malformed`, naming the field
- [x] Check: unit tests load the real `config.json` and then altered copies, one per supported-set row, asserting the message contains the field name and the value; a test for each absent-field case

## 3.3 Tensor type (HC-1, Spec §3)

- [x] `tensor.rs`: owned `Vec<f32>` plus row-major shape; constructor that rejects a length that disagrees with the shape product
- [x] Only what loading needs: shape, data, row access for `[out, in]` matrices. No operations; kernels arrive in Phase 4
- [x] Check: unit tests for construction, mismatch rejection, and row slices

## 3.4 Safetensors reader (FR-11, FR-12, HC-1, Spec §5)

- [x] Add `memmap2` to the library's dependencies
- [x] `safetensors.rs`: 8-byte little-endian header length, JSON header, memory map; validate that the header length fits the file and that every byte range lies inside the data region and does not overlap another
- [x] F32 copied; BF16 widened to F32 (shift left by 16 bits); any other dtype rejected naming the tensor and dtype
- [x] Reject weights split across several files (a `model.safetensors.index.json` beside the file, or a header that signals sharding) with an error saying sharding is not supported
- [x] Ignore the `__metadata__` header entry; reject duplicate or malformed entries as `Malformed`
- [x] A truncated or hostile file never panics (Spec §14): header length of `u64::MAX`, ranges past the end, shape/byte-length disagreement
- [x] Check: unit tests build tiny files in memory (written to a temp file for the map), including a BF16 widening exactness case (every BF16 pattern of a sample set maps to the same f32 bits as the reference widening) and each rejection above
- [x] Check: the real `models/smollm2-135m/model.safetensors` opens and lists tensor names and shapes (test skips with a clear message only if the model directory is absent, matching the harness's loud-failure rule from 2.7: it fails naming the download command instead of skipping)

## 3.5 Weight binding (FR-12, Spec §3, §5)

- [x] `weights.rs`: build the expected name list from the `Config` (`model.embed_tokens.weight`, per layer `self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`, both norms, `model.norm.weight`)
- [x] Check every tensor's shape against the config; a mismatch is `TensorMismatch` naming the tensor, expected and found shape
- [x] A missing tensor and an unexpected tensor (one in the file but not in the expected list) are both `TensorMismatch` naming it; with tied embeddings, an `lm_head.weight` in the file is unexpected unless the Specification says otherwise: check what the real file contains first and record it
- [x] `Model::load` loads config and weights through this path; the stub's `load` is replaced, the rest of `Model` stays a stub until Phase 4
- [x] Check: tests with an in-memory file missing one tensor, one extra tensor and one mis-shaped tensor, each asserting the name in the message; the real model binds without error
- Result of the `lm_head.weight` check: the real file has 272 tensors and none is `lm_head.weight`, so an `lm_head.weight` is rejected as unexpected, consistent with Spec §3 (tied embeddings); no Specification change.

## 3.6 Record the tokenizer's component types (Spec §6, §19)

- [x] Read `tokenizer.json` and confirm the facts at the top of this file: model type, normalizer, pre-tokenizer sequence and options, post-processor, decoder, added tokens (list their IDs and whether each is `special`)
- [x] Record the result in the plan's §9 upstream-feedback table (the row for task 3.6 already exists), narrowing Spec §6 and resolving the first open item in §19
- [x] Note which options are implemented and which are rejected: anything outside what SmolLM2 uses (for example a non-null normalizer, `add_prefix_space: true`, a post-processor of any type) is `Unsupported` naming the type
- [x] Check: the recorded text matches the file; a test parses the real `tokenizer.json` and asserts the same facts so a model-revision change cannot silently invalidate them

## 3.7 Byte-to-unicode map (FR-13, FR-14, Spec §6, §17)

- [x] The standard byte-level alphabet: printable bytes map to themselves, the rest to code points from U+0100 upward in byte order; and its inverse
- [x] Check: unit tests round-trip all 256 bytes; the map is a bijection onto 256 distinct characters; spot values (space → `Ġ`, newline → `Ċ`)
- [x] Check: every vocabulary entry of the real `tokenizer.json` is made only of characters in the alphabet (apart from added tokens)

## 3.8 Pre-tokenizer splitting (FR-13, Spec §6)

Hand-written, no regex crate (Spec §6). Two stages in the order the file declares: digits split individually, then the ByteLevel pattern.

- [x] Digit stage: every digit becomes its own piece
- [x] ByteLevel stage: the GPT-2 style pattern, written as a scanner over characters: contractions (`'s 't 're 've 'm 'll 'd`), optional-space-plus-letters, optional-space-plus-other-symbols, whitespace runs with the trailing-space lookahead rule (whitespace not followed by non-whitespace, so the last space attaches to the next word), and `use_regex: true` honoured
- [x] Unicode letter and number classification for Thai and other scripts: **decided** not to approximate with `char` methods. The classes are measured from the reference over every scalar value (`oracle/unicode_classes.py`) and committed as `unicode_classes.rs`; recorded in plan §9
- [x] Never panic on any `&str`, including empty input and lone combining marks
- [x] Check: unit tests on constructed strings: contractions, digits (`a123` → `a`,`1`,`2`,`3`), punctuation runs, whitespace runs (spaces, tabs, newlines, trailing spaces), Thai, emoji, mixed
- [x] Check: concatenating the pieces always reproduces the input exactly (every Unicode scalar value, and every corpus string including all prompts)
- [x] Check: the splitter agrees with the reference on all 3,979 strings of `oracle/tokenizer_corpus.py`'s output (prompts, edge cases, 4,000 seeded random mixtures); a deliberate `is_alphabetic` substitution fails 557 of them

## 3.9 BPE encoding (FR-13, Spec §6)

- [x] Load vocabulary (token to ID) and merges (pair to rank) from `tokenizer.json`; accept both merge representations the file format allows (a `"a b"` string or a two-element array), rejecting anything else as `Malformed`
- [x] Special-token matching first: added tokens marked special are matched in the raw text before splitting, longest match first; everything between matches goes through pre-tokenizer and BPE
- [x] BPE per piece: map bytes to the alphabet, repeatedly merge the adjacent pair with the lowest rank until no ranked pair remains; piece strings then map to IDs; a character with no vocabulary entry is dropped, exactly as the reference does (21 bytes have none; recorded in plan §9); merges that name tokens missing from the vocabulary are `Malformed`
- [x] Post-processor: SmolLM2 has none, so no BOS is added; the code follows whatever 3.6 recorded and rejects other post-processors
- [x] Empty text encodes to an empty list here; rejecting an empty prompt is the generation layer's job (FR-18), not the tokenizer's
- [x] Check: unit tests on a tiny hand-built vocabulary with known merges (rank order decides, ties, no merges available); one test for special tokens appearing mid-text
- [x] Check: encode every `oracle/prompts.json` prompt and compare to the fixture IDs exactly (the real check is 3.12)
- [x] Check: the engine's IDs equal the reference's on all 3,997 corpus strings (which include every prompt, special-token text and the characters the vocabulary lacks); making ties go to the rightmost pair fails 294 of them

## 3.10 Decoding (FR-14, Spec §6)

- [x] IDs to piece strings, pieces to bytes through the inverse map, bytes to a string; invalid UTF-8 in the whole output becomes U+FFFD exactly as the reference's decode does (verify against the oracle with a constructed ID list that splits a character, and record the result)
- [x] Special tokens: decide whether they are printed as their text, matching what the oracle's decode does for the fixtures, and write down the reading
- [x] An ID outside the vocabulary is `Rejected` naming the ID, never a panic
- [x] Check: unit tests for ASCII, Thai, emoji, a special token, an out-of-range ID, and an empty list

## 3.11 Streaming decoder (FR-15, Spec §6, §17)

- [x] `StreamDecoder` holding a byte buffer: after each token push, emit the longest valid UTF-8 prefix and hold back an incomplete trailing sequence of at most 3 bytes
- [x] Bytes that cannot begin or continue a valid sequence become U+FFFD immediately and are not held; `flush` emits whatever is held the same way
- [x] Emitted text concatenated over a whole stream equals one-shot `decode` of the same IDs (property test over every prompt's fixture IDs and continuation IDs)
- [x] Never emit an incomplete sequence at any chunk boundary
- [x] Check: unit test splits a three-byte Thai character across every possible chunk boundary (1+2, 2+1, 1+1+1) and asserts nothing is emitted until the character completes
- [x] Check: invalid-byte cases: a stray continuation byte, a truncated four-byte sequence followed by ASCII, a flush with held bytes

## 3.12 Wire into the harness (FR-13 to FR-15, Spec §11)

- [x] `Tokenizer::load` and `Model::load` use the real implementations; the correctness harness's encode and decode checks now call them
- [x] Add the streaming check to the harness: feed the fixture IDs of each `utf8-split` prompt one at a time through the `StreamDecoder`, assert no emitted chunk ends inside a character and the total matches decode; show it as its own column in the pass/fail table
- [x] The logits, top-1, intermediates and greedy-continuation checks are left running against the stub and still fail
- [x] Check: `cargo test --no-fail-fast` prints the table with the encode, decode and streaming columns passing for all five categories and the other columns failing, exit code non-zero

## Phase exit check

- [ ] For every prompt in every category, the harness's encoding and decoding checks pass, and the streaming check passes on `utf8-split` prompts (plan §4)
- [ ] The logit, intermediate and continuation checks still fail, because the forward pass is still the stub
- [ ] An unsupported-configuration test shows the error names the field; the same for an unsupported tokenizer component and a mis-shaped tensor
- [ ] Every test names the HC, FR or S item it verifies
- [ ] Plan §9 holds the task 3.6 row filled in; any decision that changes HOW (for example the `lm_head.weight` reading in 3.5, the Thai character classification in 3.8) is recorded there and not decided here
- [ ] `cargo test --no-fail-fast` time recorded, and `cargo build` has no new warnings
- [ ] Owner reviews the table output and the tokenizer tests before Phase 4 starts

## Notes to carry forward

- Phase 4 replaces the rest of `Model` (the forward pass); the loader built here is what it receives.
- Plain `cargo test` stops at the first failing test binary; keep using `--no-fail-fast` until the correctness harness passes.
