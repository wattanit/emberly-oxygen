# Phase 15 — Template and the S-11 gate: To-do

Working checklist for Phase 15 of the [implementation plan](../emberly-oxygen-implementation-plan-0.3.0.md) (the second part of milestone M11). Tick items as they land; one task is one commit, tests first.

Proves: the engine's template and encoding equal the reference's exactly, the check can fail, and the instruct numbers are reported. Satisfies FR-32, FR-33, FR-39, S-11, HC-4..

Layout: `crates/oxygen/src/chat.rs`, harness additions and tests under `crates/oxygen/tests/`. Behavior: Specification §8 and §11.

Dependencies: none new. Out of scope: `Chat`, the command (Phase 16).

## 15.1 Instruct facts (FR-32, FR-33)

- [ ] Test against the real files; model loads; markers encode as IDs 1 and 2.

## 15.2 Renderer (FR-33)

- [ ] Tests first on hand-written strings; then `render`.

## 15.3 S-11 check (S-11, FR-39)

- [ ] Tests first; text and ids judged separately; renderer and encoder as arguments.

## 15.4 Negative controls (HC-4, S-11)

- [ ] Three faulty renderers/encoders fail S-11; the real ones pass.

## 15.5 Reported-only table (FR-5, FR-39)

- [ ] Tests first: reported figures never fail a run; then the table.

## 15.6 Measure

- [ ] Record the reported figures and S-11 results in plan §5.
