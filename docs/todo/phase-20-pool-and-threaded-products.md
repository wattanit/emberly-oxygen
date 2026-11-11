# Phase 20 — The pool and the threaded products: To-do

Working checklist for Phase 20 of the [implementation plan](../emberly-oxygen-implementation-plan-0.4.1.md) (first part of milestone M15). Tick items as they land; one task is one commit, tests first (20.1 is a measurement and commits nothing but its §5 record).

Proves: threads are worth building at these shapes, the pool hands workers what they read in safe code, and a threaded `matvec` and `matmul` equal the single-threaded ones exactly. Satisfies FR-43, FR-44, HC-1, HC-2.

Layout: `crates/oxygen/src/pool.rs`, `ops.rs`, `model.rs`. Behavior: Specification §2, §4, §19.

Dependencies: none new. Out of scope: `unsafe`, SIMD intrinsics, tiling, a parallelism crate, threaded attention (Phase 21).

## 20.1 Measure threads and safe sharing (FR-43)

- [x] Hand-over cost to a persistent worker against scoped threads per call, at 1, 2, 4, 8, 16 workers.
- [x] The two safe ways to let workers read the key-value cache, at the real shapes.
- [x] Product time at each step shape under 1 to 16 threads; the small-product threshold.
- [x] Record in §5; take the open item of Specification §19 to the owner.

## 20.2 The pool (FR-43, FR-44)

- [x] Tests first: each range runs once; 1 runs on the caller and spawns nothing; 0 rejected naming `threads`; workers joined on drop; a worker panic reaches the caller.
- [x] Then `pool.rs`.

## 20.3 Threaded `matvec` and `matmul` (FR-43, HC-2)

- [x] Tests first: exactly equal to single-threaded at counts 1, 2, 3, 7, default; the five shapes; one row; fewer rows than workers.
- [x] Then contiguous row ranges, the caller taking one, the small-product threshold.

## 20.4 `Model::load_with_threads` (FR-44)

- [ ] Tests first: count stored; 0 rejected naming `threads`; runs at 1.
- [ ] Then the loader and the pool's place in the session.
