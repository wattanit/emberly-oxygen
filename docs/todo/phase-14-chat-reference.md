# Phase 14 — Chat reference: To-do

Working checklist for Phase 14 of the [implementation plan](../emberly-oxygen-implementation-plan-0.3.0.md) (the first part of milestone M11). Tick items as they land; one task is one commit, tests first.

Proves: the oracle writes the instruct model's conversations, rendered text, token ids, f64 logits and greedy replies, and the harness can read them. Satisfies FR-39, FR-4, FR-11, HC-3..

Layout: `oracle/download.py`, `oracle/conversations.json`, `oracle/chat_reference.py`, the shared f64 module, harness reading under `crates/oxygen/tests/common/`. Behavior: Specification §10.

Dependencies: none new. Out of scope: the renderer and the S-11 check (Phase 15); any engine change.

## 14.1 Download the instruct model (FR-11, FR-32)

- [x] Model argument in `download.py`; fetch at the pinned revision; record the tokenizer and config comparison.

## 14.2 Share the f64 setup (FR-2)

- [ ] Move the context manager and start-up check to a module both scripts import; confirm the base fixtures are unchanged.

## 14.3 Conversations (FR-39)

- [ ] `conversations.json` with `no-system`, `with-system`, `multi-turn`, `thai`.

## 14.4 Chat reference script (FR-39, FR-4)

- [ ] `chat_reference.py` writes `chat.json` and the f64 logits; renderings checked by hand in the scratchpad.

## 14.5 Run (FR-4)

- [ ] Record time, sizes, token counts, reply endings and text.

## 14.6 Harness reads the files (FR-4, FR-7)

- [ ] Tests first: missing folder, file, shape, field; then the reader.
