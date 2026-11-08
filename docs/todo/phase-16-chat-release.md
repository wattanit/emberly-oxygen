# Phase 16 — Chat and release: To-do

Working checklist for Phase 16 of the [implementation plan](../emberly-oxygen-implementation-plan-0.3.0.md) (milestone M12). Tick items as they land; one task is one commit, tests first.

Proves: conversations, the chat command and the defaults work, greedy and sampling are unchanged, and the release is measured and documented. Satisfies FR-32 to FR-38, FR-8, FR-9, S-8, HC-4..

Layout: `crates/oxygen/src/generate.rs`, `chat.rs`, `crates/oxygen-cli/src/cli.rs`, tests, `docs/releases/0.3.0.md`. Behavior: Specification §8, §9, §12, §16.

Dependencies: none new. Out of scope: reusing the cache across turns, multi-line input, templates for other models (Requirements §2.3).

## 16.1 Step loop with a caller's generator (FR-29, FR-28)

- [x] Tests first: existing outputs identical; then the refactor.

## 16.2 Chat and turn (FR-34, FR-35, FR-29)

- [x] Tests first on the real instruct model; then `Chat`.

## 16.3 Context limit (FR-36, FR-18)

- [x] Rejected before computation, state unchanged, later turn works.

## 16.4 Seeded chat (FR-37, FR-28)

- [x] Same seed replays, another differs, one generator across turns.

## 16.5 Chat command (FR-38, FR-37, FR-20)

- [x] CLI tests first against the binary; then the loop and defaults.

## 16.6 Logging (S-8, FR-23)

- [x] `-vv` per-turn events; output identical at every level.

## 16.7 Full gate (HC-4)

- [ ] `cargo test --no-fail-fast`, read the whole summary.

## 16.8 Version (FR-9)

- [ ] 0.3.0, clean tree.

## 16.9 Test then bench (FR-8, FR-9, S-7)

- [ ] Tests, then `cargo bench` straight after.

## 16.10 Compare (FR-8, FR-9)

- [ ] Against 0.2.0's rows; turn time against history length.

## 16.11 Release note and SFD settlement

- [ ] Note and README; §6 rows into the documents as the owner permits; compliance review; pins; ask for approval.

## 16.12 Owner's review; stop before merge

- [ ] Ask before merging or tagging `v0.3.0`.
