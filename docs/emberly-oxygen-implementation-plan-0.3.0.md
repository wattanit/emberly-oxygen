# Emberly Oxygen — Implementation Plan for Release 0.3.0

Nov 8, 2026 · @Wattanit

**Target release:** 0.3.0\
**Owner:** Wattanit\
**Built against:** Emberly Oxygen — Requirements 0.7 (upstream; approved on 2026-11-08), Emberly Oxygen — Technical Specification 0.5 (upstream; approved on 2026-11-08)

This is a living working document, not part of the SFD suite: it has no document version or status and is edited in place as work proceeds. It expands the Technical Specification's milestones M11 and M12 into three phases with ordered tasks and exit checks. It owns sequencing and task breakdown only; WHAT stays in the Requirements and HOW stays in the Specification, and every task cites the IDs it satisfies.

Phases are numbered on from 0.2.0's (Phases 11 to 13), so their to-do files in `docs/todo/` keep unique names.

**What this release is.** One feature: chat with the instruct model, SmolLM2-135M-Instruct (FR-32 to FR-39, S-11): the chat template, a reply that stops at `<|im_end|>`, history kept across turns, the context limit enforced, chat sampling defaults, and an interactive `chat` command. The greedy and sampled paths do not change (FR-29). The oracle's role was decided with the Requirements: it judges the template and its tokens exactly and only reports the instruct model's numbers (Requirements §13).

## 1. Working rules

All of 0.2.0's rules hold. The ones this release leans on, and what it adds:

- **Phase order is fixed.** Phase 15 does not start until Phase 14's exit check passes (the template cannot be judged against fixtures that do not exist); Phase 16 does not start until Phase 15's does.
- **The engine's arithmetic is not touched.** No change to `forward.rs`, `ops.rs`, the tokenizer, `sampling.rs` or the existing harness checks. `generate.rs` may be refactored so a chat turn can run the step loop with a generator it owns (Specification §8), and `generate` and `generate_with` must produce the same IDs as before (FR-29). If a task seems to need a change to the forward pass or the tokenizer, it stops and the issue goes to the owner.
- **The instruct numbers judge nothing.** The reported-only table (Specification §11) never fails a conversation or a release. If it shows something alarming, that is reported to the owner and recorded in §5, not turned into a threshold by the implementation.
- **Tests before code.** Each new unit (the renderer, the S-11 check, the conversation, the CLI loop) gets its tests first, seen to fail before the code exists. The S-11 check also gets its negative controls before the real renderer is trusted (Specification §17).
- **One task, one commit.** Each task ends with a passing `cargo test --no-fail-fast` whose whole summary is read (Phases 15 and 16), or with the oracle command it names (Phase 14), and a commit with a body paragraph and **no attribution line**: the owner's standing rule, in every project. Docs-only commits carry the `[docs]` prefix.
- **Every test names the HC, FR or S item it verifies** (Specification §17).
- **No new Rust dependency, and no new Python dependency** (Specification §15). The instruct model needs one new download, at the pinned revision; `models/` and `fixtures/` stay git-ignored.
- **Throwaway scripts stay out of the repository.** A script written to check something once (the by-hand rendering of the template in task 14.4) lives in the scratchpad; its result goes in §5 with how it was produced.
- **Approved documents are not edited without the owner's permission.** What the work finds goes in §6. A small fix is written into the approved documents in place only when the owner says so; a large miss means rolling back to before the work, fixing the document and re-approving, as the owner has set out. No version bump absorbs a mistake: a bump comes only with new scope (Requirements §2.3).
- **The SFD is settled before any merge.** Task 16.11 comes before the question of merging: §6's rows written into the documents this release was built against as the owner permits, a compliance review, the owner's approval of the statuses, pins current.
- **Work happens on `release/0.3.0`.** The owner pushes; nothing is pushed, tagged or merged to `develop` or `main` without the owner's explicit order at that moment (standing rule, 2026-11-06).

## 2. Phase 14 — Chat reference (first part of M11)

**Proves:** the oracle writes the instruct model's rendered conversations, token ids, f64 logits and greedy replies; the template's text was checked by hand against the published template; and the harness can read the files. **Satisfies:** FR-39, FR-4, FR-11, HC-3.

| # | Task | Cites |
| --- | --- | --- |
| 14.1 | `oracle/download.py` takes a model argument: repository and pinned revision per model, the base model unchanged, the instruct model at `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Download it into `models/smollm2-135m-instruct/`. Record that its `tokenizer.json` equals the base's as JSON and which `config.json` fields differ (Specification §3). | FR-11, FR-32, HC-3 |
| 14.2 | Factor the f64 context manager and its start-up check out of `oracle/generate.py` into a module both scripts import, with no change to the base model's fixtures: rerun `generate.py`'s start-up check and confirm the stored manifest is unchanged. | FR-2, HC-3 |
| 14.3 | `oracle/conversations.json`: `no-system`, `with-system`, `multi-turn`, `thai` (Specification §10; initial, tune with use). | FR-39 |
| 14.4 | `oracle/chat_reference.py`: renders each conversation with `apply_chat_template`, encodes, asserts the ids equal `apply_chat_template(tokenize=True)`, runs the f64 model, stores logits at every prompt position and a greedy reply of up to 64 tokens with gaps, writes `chat.json` and the safetensors files. In the scratchpad, write each rendering by hand from the template text in `tokenizer_config.json` and compare it with the library's. | FR-39, FR-4, HC-3 |
| 14.5 | Run the script. Record the time, file sizes, each conversation's token count, whether each reply stopped at `<|im_end|>` and its length, and the replies' text. | FR-39, FR-4 |
| 14.6 | The harness reads `chat.json` and the files; tests first: a missing folder, a missing file, a wrong shape and a missing field each fail with a message naming the command that generates them. Manifest discovery is unchanged and passes over the instruct folder. | FR-4, FR-7 |

**Exit check:** all four conversations have rendering, ids, logits and a reply; the hand-written renderings equal the library's; the base model's fixtures are unchanged; the harness reads the files; `cargo test` still passes untouched.

## 3. Phase 15 — Template and the S-11 gate (second part of M11)

**Proves:** the engine's template and encoding equal the reference's exactly, the check can fail, the instruct model loads, and its numbers are reported. **Satisfies:** FR-32, FR-33, FR-39, S-11, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 15.1 | Pin the instruct facts of Specification §3 in a test against the real files (tokenizer equal to the base's, config differences, `eos_token_id` 2), and that the model loads and `<|im_start|>`, `<|im_end|>` encode as IDs 1 and 2. | FR-32, FR-33, FR-12 |
| 15.2 | `chat.rs`: `Role`, `Message`, `render`. Tests first against hand-written strings: no system message, system first, system not first, several turns, empty content, the generation prompt present and absent. | FR-33 |
| 15.3 | The S-11 check in the harness, taking the renderer and encoder as arguments: text and ids judged separately, mismatches located. Tests first with hand-made conversations. | S-11, FR-39 |
| 15.4 | Chat negative controls in the tests: no default system, no newline after `<|im_end|>`, markers spelled out as characters; S-11 must fail on each (the last on tokens only), and pass with the real functions. | HC-4, S-11, Spec §17 |
| 15.5 | The reported-only table: logit difference per conversation and greedy reply agreement up to the first near-tie, printed beside the other tables and read by no verdict. Tests first: a failing reported figure does not fail the run. | FR-5, FR-39 |
| 15.6 | Measure: the table's numbers per conversation, what the engine's replies say beside the reference's, and S-11 on all four. Recorded in §5. | FR-39, S-11 |

**Exit check:** S-11 passes on every conversation with the real functions and fails on each control; the reported table is printed and judges nothing; the greedy and sampling harness passes untouched.

## 4. Phase 16 — Chat and release (M12)

**Proves:** conversations, the chat command and the defaults work, greedy and sampling are unchanged, and the release is measured and documented. **Satisfies:** FR-32 to FR-38, FR-8, FR-9, S-8, HC-4.

| # | Task | Cites |
| --- | --- | --- |
| 16.1 | Refactor `generate.rs` so the step loop can run with a generator the caller owns; `generate` and `generate_with` call it with a fresh one. Tests first: the existing greedy and sampled outputs are identical before and after. | FR-29, FR-28 |
| 16.2 | `Chat::new` and `Chat::turn` (Specification §8). Tests first on the real instruct model: reply ends at `<|im_end|>` and it is not streamed; a turn appends a user and an assistant message equal to the text streamed; a reply cut by a tiny maximum is closed by the next rendering; the second prompt contains both earlier messages; temperature 0 equals `generate` on the rendered text. | FR-34, FR-35, FR-29 |
| 16.3 | The context limit: a turn that cannot fit is rejected before any computation and leaves the messages and generator as they were; a later turn works. Tests first. | FR-36, FR-18 |
| 16.4 | Seeded chat: the same seed and messages replay the same replies, another seed does not; one generator serves all turns. Tests first. | FR-37, FR-28 |
| 16.5 | The `chat` command (Specification §9): the loop, `/exit`, blank lines, the marker and seed on standard error, the defaults including top-p 1 when only `--temperature 0` is given. CLI tests first against the built binary. | FR-38, FR-37, FR-20 |
| 16.6 | `-vv` events per turn; a test shows output identical at every verbosity in chat. | S-8, FR-23 |
| 16.7 | The full gate: `cargo test --no-fail-fast`, whole summary read; record the time. | HC-4 |
| 16.8 | Bump the workspace version to 0.3.0 and confirm a clean tree. | FR-9, Spec §16 |
| 16.9 | Run `cargo test --no-fail-fast`, then `cargo bench` straight after, at the default log level (Specification §16). A failing test run stops here. | FR-8, FR-9, S-7 |
| 16.10 | Record the comparison with 0.2.0's rows in §5 (the greedy workloads should be unchanged; say so, do not claim a speed change), plus the time of a chat turn at a few history lengths to show it grows (Specification §2). | FR-8, FR-9 |
| 16.11 | Release note `docs/releases/0.3.0.md` and the README. Then **settle the SFD before the merge question:** write §6's rows into the documents as the owner permits, run the sfd-standard compliance review over both, check every pin here and in the release note, and ask the owner to approve the statuses. | Requirements §12, G-14, G-16, G-24, G-25 |
| 16.12 | Owner's review of the rows and the note. Then **stop and ask** before merging or tagging `v0.3.0`: neither happens without the owner's explicit order. | Spec §16 |

**Exit check:** greedy and sampled output identical to 0.2.0's; chat tests and the command's tests pass on the real instruct model; `results/benchmarks.jsonl` holds the 0.3.0 rows from a clean tree; the release note exists; both foundation documents are approved and pinned.

## 5. Measurements to record

Filled in as the phases land. Machine: the benchmark machine (Apple M4 Max), as before. Baseline: 304 tests at 0.2.0.

- **Task 14.1 (done).** The instruct model is in `models/smollm2-135m-instruct/` at revision `12fd25f`: `config.json` 861 bytes, `model.safetensors` 269,060,552 bytes (the base model's size to the byte), `tokenizer.json` 2,104,556 bytes (also the base's size), and `tokenizer_config.json` 3,764 bytes. Its `tokenizer.json` is equal to the base model's as JSON; `config.json` differs in `bos_token_id` (0 → 1), `eos_token_id` (0 → 2), `pad_token_id` (absent → 2), `mlp_bias` (absent → false), `transformers_version` and a `transformers.js_config` block, as Specification §3 says. The base model downloads exactly as before.
- **Task 14.2 (done).** `oracle/reference_common.py` now holds `machine`, `top2_gaps`, `float64_pure_ops` and its start-up check; `generate.py` imports them. Reran `generate.py` into the scratchpad (about 4 minutes): all 24 tensor files, f32 and f64, are **bit-identical** to the stored ones (worst difference 0), and the manifest differs only in `generated_at` and in the `sampling` entry and `sampling_file` fields, which `sampling_reference.py` added later. The base fixtures are unchanged.
- **Tasks 14.4 and 14.5 (done).** `chat_reference.py` ran in 14 s (the model is not rerun for base prompts; four short conversations). Hand check: a scratchpad script (`hand_template.py`) writes each rendering from the template text in `tokenizer_config.json` without Jinja, and all four equal the library's text character for character; the script's own assertions also hold (encoding the rendered text equals `apply_chat_template(tokenize=True)`, `<|im_start|>` is ID 1, `<|im_end|>` is ID 2, the model's `eos_token_id` is 2).

  | Conversation | Positions | f64 logits file | Reply | Stopped at `<|im_end|>` | Smallest top-two gap, prompt / reply |
  | --- | --- | --- | --- | --- | --- |
  | `no-system` | 37 | 14.5 MB | 8 tokens: "The capital of France is Paris." | yes | 0.0221 / 0.257 |
  | `with-system` | 29 | 11.4 MB | 64 tokens, cut by the maximum (a boiling-egg answer) | no | 0.0252 / 0.00435 |
  | `multi-turn` | 93 | 36.6 MB | 7 tokens: "My name is Ploy." | yes | 0.0252 / 0.838 |
  | `thai` | 110 | 43.3 MB | 64 tokens, cut by the maximum, inside a character (the decoded text ends in U+FFFD) | no | 0.0185 / 0.481 |

  The folder is **101 MB** (four logits files and `chat.json`, 21 KB). No reply step has a top-two gap under 1e-3, so no near-tie falls inside any reply. Two of the four replies hit the 64-token maximum and not `<|im_end|>`, so the harness will see both endings; the Thai one is also a reply cut in the middle of a character, which exercises streaming.
- **Task 14.6 (done).** The harness reads `chat.json` and the logits files (`discover_chat`, `ChatFixtures::load_logits`). Eight tests, written first and seen to fail against stubs: missing chat fixtures, the two discoveries ignoring each other's folders, a missing field, a missing logits file, a wrong shape, a missing logits tensor or an F32 file, an exact read of values f32 cannot hold, and the real fixtures' consistency (four conversations, shapes `[positions, 49152]`, reply ends at `<|im_end|>` exactly when it stopped there). **312 tests pass, 0 fail** (304 at 0.2.0, so 8 new); the greedy and sampling harness is untouched.
- **Phase 14 exit check (passed).** All four conversations have rendering, ids, logits and a reply; the hand-written renderings equal the library's; the base fixtures are unchanged (bit-identical rerun); the harness reads the files; `cargo test` passes with no existing test changed.

## 6. Upstream feedback

Recorded here for the upstream owner; the approved documents are not edited from this plan without the owner's permission. This section is a permanent record.

| Raised | Upstream item | What and why |
| --- | --- | --- |
| Task 14.5 | Specification §10 | §10 estimates a conversation at "about 100 to 300 tokens" and the folder at "tens of megabytes". The four conversations are 29 to 110 positions and the folder is 101 MB, because each f64 position is 0.39 MB. The sentence should give the measured size. |
| Task 14.1 | Specification §10 | The oracle needs a fourth file for the instruct model, `tokenizer_config.json`, which holds the chat template; the engine never reads it and §5's model directory (three files) is unchanged. `download.py` fetches it as an oracle-only file. §10 should say the instruct download includes it. |

## 7. Open Questions

None at the start of the release.
