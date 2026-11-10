# Phase 17 — The matrix-matrix product and the batched pass: To-do

Working checklist for Phase 17 of the [implementation plan](../emberly-oxygen-implementation-plan-0.4.0.md) (first part of milestone M13). Tick items as they land; one task is one commit, tests first.

Proves: `matmul` equals the matrix-vector product row by row, `prefill` gives what `step` gives at every position within the bound, the mask is causal, and the loop order is worth trying. Satisfies FR-40, FR-41, HC-1, HC-2.

Layout: `crates/oxygen/src/ops.rs`, `forward.rs`, `fault.rs`. Behavior: Specification §4, §7, §17.

Dependencies: none new. Out of scope: tiling, register blocking, threads, SIMD, `unsafe`, batching `attention`, reusing the cache (Requirements §2.2).

## 17.1 `matmul` (FR-40, HC-2)

- [x] Tests first: equal to `matvec` row by row; equal to `matrixmultiply` within tolerance at the five shapes and at odd sizes; one row; lengths below the partial-sum width.
- [x] Then `matmul`, weight row outermost, calling the existing dot product.

## 17.2 `Logits` and `prefill`'s checks (FR-40, FR-18)

- [x] Tests first on the tiny model: unknown token, empty prompt, prompt over the context each rejected before anything is touched; cache and capture as they were.
- [x] Then `Logits` and the checks.

## 17.3 `prefill`'s body (FR-40, FR-41)

- [x] Tests first: equal to `step` at every position within 1e-5 (logits, capture, cache); causal (a later token changes nothing before it, bit for bit); one-token prompt; prompt filling the context; `Last` equals the last row of `All`.
- [x] Then the body (Specification §7).

## 17.4 Faults in `prefill` (HC-4)

- [x] Tests first: each of the four faults changes `prefill`'s output; no fault, no change.
- [x] Then `UnmaskedAttention` and the existing three in `prefill`.

## 17.5 Measure agreement (FR-41)

- [x] Largest logit and intermediate difference between `prefill` and `step` per prompt; bit-identical or not; record in §5.
- [x] Set the bound of Specification §11 with the owner: it stays at 1e-5 (a guard; the measured difference is 0).

## 17.6 Dry run of the loop order (FR-42)

- [x] Time `step`, `step` without the unembedding, and `prefill` over 512 tokens, three runs each; record in §5.
- [x] If the batched arm is not clearly faster than the second, stop and raise it with the owner.
