# Phase 19 — The prefill sweep and the release: To-do

Working checklist for Phase 19 of the [implementation plan](../emberly-oxygen-implementation-plan-0.4.0.md) (milestone M14). Tick items as they land; one task is one commit, tests first.

Proves: what batching is worth, apart from the skipped logits, and that decode did not change. Satisfies FR-42, FR-8, FR-9, S-7, HC-4.

Layout: `crates/oxygen/benches/`, `results/benchmarks.jsonl`, `docs/releases/0.4.0.md`. Behavior: Specification §12, §16.

Dependencies: none new. Out of scope: reusing the cache across turns, attention speed-ups, anything in v4 (Requirements §2.2).

## 19.1 Prefill sweep group (FR-42)

- [ ] Tests first for the support code: fixed prompts, agreement check, row format.
- [ ] Then the group: lengths 32, 67, 128, 512, 644; three arms.

## 19.2 Version (FR-9)

- [ ] Bump the workspace version to 0.4.0; clean tree.

## 19.3 Test, then bench (FR-8, S-7)

- [ ] `cargo test --no-fail-fast`, then `cargo bench` straight after, at the default log level.

## 19.4 Record the comparison (FR-42)

- [ ] Sweep per length and arm; chat first token at 67 and 644 tokens; the two workloads; decode throughput (A/B against `v0.3.0` if drift matters); peak memory; name each effect and any negative result.

## 19.5 Release note and README

- [ ] `docs/releases/0.4.0.md`; README status line and the stale 0.3.0 branch wording.

## 19.6 Settle the SFD

- [ ] Feedback rows into the documents as the owner permits; compliance review; pins; the owner approves the statuses.

## 19.7 Owner's review

- [ ] Review the rows and the note; then stop and ask before merging or tagging `v0.4.0`.
