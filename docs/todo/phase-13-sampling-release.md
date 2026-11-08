# Phase 13 — Sampling and release: To-do

Working checklist for Phase 13 of the [implementation plan](../emberly-oxygen-implementation-plan-0.2.0.md) (milestone M10). Tick items as they land; one task is one commit.

Proves: seeded generation and the CLI work, greedy is unchanged, draws follow the distribution, and the release is measured and documented. Satisfies FR-26 to FR-30, FR-8, FR-9, S-8, S-10, HC-4.

Layout: `generate.rs`, `crates/oxygen-cli/src/cli.rs`, `crates/oxygen/benches/`, `docs/releases/0.2.0.md`, `README.md`.

Dependencies: none new. Out of scope: chat, the instruct model, any change to the forward pass.

## Decision points for the owner

1. **Seed line. (applies to 13.3.)** *Recommendation:* print `seed: <n>` to standard error whenever the run samples, whether or not `--seed` was given, so the run can always be replayed (Specification §8).

## 13.1 Generation (FR-26, FR-28, FR-29)

- [x] Tests first: same seed same tokens; different seed different tokens; temperature 0 with all controls set equals default; greedy harness untouched. Then `generate` takes `Sampling`.

## 13.2 Draw test (S-10)

- [x] Chi-square test as in Specification §11, shown to fail on draws from another distribution; record the statistic and limit.

## 13.3 CLI (FR-19, FR-20, FR-30)

- [x] Options and the seed line; binary tests for invalid value, clean stdout, replay.

## 13.4 Logging (S-8, FR-23)

- [x] `-vv` additions at that level only; test identical output at every level when sampling.

## 13.5 Full gate (HC-4)

- [ ] `cargo test --no-fail-fast`, whole summary read; record the time.

## 13.6 Sampling cost benchmark (FR-8)

- [ ] The group in `benches/`; the workloads stay greedy.

## 13.7 Version (FR-9)

- [ ] Workspace version 0.2.0, clean tree.

## 13.8 Benchmark (FR-8, FR-9, S-7)

- [ ] Tests, then `cargo bench` straight after, default log level.

## 13.9 Comparison (FR-8, FR-9)

- [ ] Rows against 0.1.2's; no speed claim; the cost of a sampled step beside a forward step.

## 13.10 Release note and README (Requirements §12)

- [ ] `docs/releases/0.2.0.md`, README status.

## 13.11 Settle the SFD (G-14, G-16, G-24, G-25)

- [ ] Requirements 0.7 and Specification 0.5 absorb the plan's section 6; compliance review; pins current; the owner approves the statuses.

## 13.12 Owner's review; stop and ask

- [ ] The owner reviews; then ask before merging or tagging `v0.2.0`.

**Exit check:** as in the plan's section 4.
