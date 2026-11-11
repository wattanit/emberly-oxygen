# Phase 23 — The blocked kernel: To-do

Working checklist for Phase 23 of the [implementation plan](../emberly-oxygen-implementation-plan-0.4.2.md) (first part of milestone M17). Tick items as they land; one task is one commit, tests first (23.3 is a measurement and commits nothing but its §5 record).

Proves: a safe blocked kernel computes exactly the index-order fused sum at every shape, edge and thread count, and it is as fast in the engine as in the profile. Satisfies FR-46, HC-1, HC-2.

Layout: `crates/oxygen/src/ops.rs`. Behavior: Specification §4, §17.

Dependencies: none new. Out of scope: `unsafe`, SIMD intrinsics, a packed copy of the weights, changes to `matvec`, `matmul`, the attention or the decode step.

## 23.1 Tests for the blocked kernel (FR-46, HC-2)

- [x] Exact equality to a plain index-order fused sum at the model's shapes, at counts that do not divide the tile, are smaller than it, or are 1, and at lengths below and above the vector width.
- [x] Within tolerance of `matrixmultiply`; a 16-partial-sum sum must differ in some bit; a tile-edge test marks every output element.

## 23.2 `matmul_blocked` (FR-46)

- [x] Input packing into `[k][NR]` panels; 4 × 16 tile with `mul_add`; leftover rows and inputs in the same order; safe code, weights in place.

## 23.3 Measure in the library (FR-46, FR-48)

- [ ] Rate of `matmul`, `matmul_blocked` and sgemm at the three shapes, 128 inputs, one thread; tile sweep in the real code; record in §5.
- [ ] If 4 × 16 is not clearly fastest or the rate is far under 118 to 120, stop and take it to the owner.

## 23.4 Pooled `matmul_blocked` (FR-46, FR-43)

- [x] Tests first: exactly equal to the single-threaded blocked product at counts 1, 2, 3, 7, 16, at the model's shapes and odd sizes.
- [x] Then row ranges of whole tile multiples, packed input shared, small products on the caller.
