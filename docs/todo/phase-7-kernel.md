# Phase 7 — Kernel: To-do

Working checklist for Phase 7 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.1.md) (the first half of milestone M7: the partial-sum matrix-vector kernel is correct). Tick items as they land; one task is one commit with a body paragraph, each ending with a passing `cargo test --no-fail-fast` whose **whole summary is read**. IDs cite the Requirements 0.2 and Technical Specification 0.2.

Proves: the engine with the partial-sum kernel passes the unchanged correctness harness in every category and leaves the negative controls failing. Satisfies FR-25, HC-1, HC-4, S-1 to S-6, S-8.

Layout: the kernel lives in `crates/oxygen/src/ops.rs`, next to the other operations. The `kernels` module in `crates/oxygen/src/lib.rs` (the `bench-kernels` feature, off in a normal build, on in tests and benches through the library's self-referencing dev-dependency) gains `matvec_index_order` (task 7.1) and `matvec_lanes::<N>` (task 7.3), so tests and the Phase 8 benchmark reach every variant. The kernel tests are `crates/oxygen/tests/kernel.rs` (decision point 3). Plan §4's table is where the measurements go.

Dependencies: none new. `matrixmultiply` is already a dev-dependency; random inputs come from a small seeded generator written in the test file, not a crate.

Out of scope: everything FR-25 and Spec §4 exclude: threads, `unsafe`, SIMD intrinsics, tiling, `target-cpu` or other build flags, and a change to any operation other than the matrix-vector product. The width sweep, the benchmark and the release note are Phase 8. If safe code does not vectorize or does not speed up, the work stops and the issue goes to the owner (Spec §4).

## Decision points for the owner

Each has a recommendation; none is decided by this to-do. Points 1 and 2 blocked 7.2 and were resolved by the owner (2026-11-07): both recommendations.

1. **The order the partial sums are combined in. (blocks 7.2.)** Spec §4 fixes "one fixed order" but does not name it, and the order-pinning test needs a specific one. *Recommendation:* sequentially, lane 0 first: `((l0 + l1) + l2) + … + l(N-1)`, then the tail elements added in index order. It is the easiest to state and to hand-compute, and the combine happens once per row (about 70 times for a 576-column row), so a tree would buy nothing measurable. Recorded in plan §5, suggesting §4 name it.
2. **The tolerance for "agrees with the reference". (blocks 7.2.)** The tests compare the new kernel with the index-order loop and with `matrixmultiply`, and the two sides differ in the last bits by design. *Recommendation:* per output element, `|a - b| <= 64 · f32::EPSILON · Σ|w·x|` over that row, a standard summation-error bound scaled by how much the row's terms could have cancelled (about 7.6e-6 times the row's absolute sum). It is far below S-1's absolute 1e-3 for the model's values. A looser bound needs the owner's word.
3. **Where the kernel tests live.** *Recommendation:* `crates/oxygen/tests/kernel.rs`, through the public `kernels` module, rather than unit tests in `ops.rs`. The tests then see exactly what the benchmark sees, and the existing small matrix-vector unit tests in `ops.rs` stay beside the kernel they exercise.
4. **What counts as "vectorized". (applies to 7.4.)** *Recommendation:* with the plain release build (no `RUSTFLAGS`, no `target-cpu=native`, so the result does not depend on this machine), the kernel's inner loop contains NEON vector multiply and add or fused multiply-add instructions (`fmla.4s`, or `fmul.4s` with `fadd.4s`) operating on the loaded weights. If it does not, stop and raise it: the fix would be an intrinsic, a build flag or `unsafe`, each the owner's call.
5. **What counts as a noticeable change in the worst errors. (applies to 7.5.)** The partial sums are expected to change the last bits, and the error against the reference could move either way. *Recommendation:* a category whose worst logit or intermediate error is more than twice 0.1.0's (plan 0.1.1 §4) is explained and shown to the owner before the phase exits, even if it is still far inside S-1 and S-2. Staying inside the thresholds is required either way (HC-4).
6. **Pushing, tagging and merging.** *Standing rule of the owner:* nothing is pushed, tagged or merged to `develop` or `main` without the owner's explicit order at the time. Phase 7 touches none of these; the work stays committed on `release/0.1.1`.

## 7.1 Keep the 0.1.0 loop as a reference (FR-25, HC-2, Spec §12, §17)

- [x] Add `kernels::matvec_index_order`, the 0.1.0 loop copied unchanged, available only with `bench-kernels`; it stays a separate function from the engine's `matvec`, so changing the engine's kernel never changes the reference Added in `lib.rs`, self-contained (it does not call `ops`), with the 0.1.0 assertions
- [x] The engine's own `matvec` is untouched in this task
- [x] The existing matrix-vector unit tests in `ops.rs` still pass; one new test (names FR-25) checks that `matvec_index_order` and the engine's current `matvec` give identical bits on a seeded input, which holds only until task 7.3 and is then replaced by 7.2's tests `tests/kernel.rs` holds `index_order_reference_matches_the_engine_until_the_kernel_changes` (names FR-25): seeded inputs at 7×13, 192×576 and 64×1536, identical bits; the full run passes, and the existing `ops.rs` tests are among the 130 library tests that pass
- [x] A normal `cargo build` and `cargo doc --no-deps` do not contain the new function (as for `bench-kernels` in 0.1.0: documentation lists it only with `--features bench-kernels`) Checked from a clean `target/doc`: a normal `cargo doc -p emberly-oxygen` has no `kernels` module and no warning; with `--features bench-kernels` it lists `kernels::matvec_index_order`. (A plain `cargo doc --no-deps` for the workspace prints one output-filename-collision warning between the library and the CLI binary, which is there before this change too)

## 7.2 Kernel tests first (FR-25, S-6, Spec §4, §17)

- [x] **Resolve decision points 1 and 2 with the owner before writing these.** Resolved: sequential lane order, and the 64·epsilon·Σ|w·x| bound
- [x] Add a stub `kernels::matvec_lanes::<N>` (const generic number of sums) that is not yet implemented, so the tests compile and fail for the right reason. The stub, in `lib.rs`, fills the output with zeros: wrong values, no panic on valid input, as Phase 2's stub did
- [x] **Agreement:** on seeded random inputs at the model's five shapes (576×576, 192×576, 1536×576, 576×1536, 49152×576), every output element of `matvec_lanes` at N = 4, 8 and 16 is within decision point 2's bound of both `matvec_index_order` and `matrixmultiply`'s single-column product (HC-2). `fr25_agrees_with_the_references_at_the_models_shapes`
- [x] **Every length from 0 to 40** as the column count (with a few rows), at each N, so lengths below N (only a tail), exactly N, N+1 and not divisible by N are all covered; zero rows and zero columns are valid and give empty or zero outputs. `fr25_every_length_from_0_to_40` (five rows each; zero rows checked too)
- [x] **Exact on exact inputs:** with small-integer weights and inputs, where every partial sum is exactly representable, all variants give the same bits, which separates a wrong index from rounding. `fr25_exact_on_exact_inputs`, at 11 column counts from 0 to 576, and for the engine's own kernel too
- [x] **Reproducible (S-6):** two calls on the same input are bit-identical. `s6_calls_are_bit_identical`, at 577 columns so every width has a tail. On the zero stub it passes trivially; it means something only from 7.3
- [x] **Order pins.** One test pins the partial-sum structure: with 16 columns, an input of all ones and weights `[1e8, 1 ×7, -1e8, 1 ×7]`, the index-order loop gives 7 and eight partial sums give 14 (lane 0 cancels to 0, each other lane is 2), hand-computed; it must fail against the 0.1.0 loop. A second test pins the combining order of decision point 1, with an input built so that a different combining order (a pairwise tree, or last lane first) gives different bits, hand-computed in the test's comment. `fr25_order_pin_partial_sum_structure` pins all of 4, 8 and 16 sums (13, 14 and 7, each hand-worked in the test's comment; 7.2's plan figure of 14 is the 8-sum case) and the engine's kernel (14); `fr25_order_pin_combining_order` uses 8 columns `[1e8, 3 ×7]`, where lane-first order gives 100000000, last-lane-first 100000024 and a pairwise tree 100000016
- [x] Panics on mismatched lengths and non-2-D weights are tested for `matvec_lanes` as `ops.rs` already does for `matvec` (Spec §14). three `should_panic` tests, expecting the messages `2-D`, `input length` and `output length`
- [x] Check, with the finding written down. Against the zero stub 8 of the 10 tests in `kernel.rs` fail (all the new ones except reproducibility, which passes on zeros). Pointing the stub at the 0.1.0 loop instead: agreement, every-length, exactness, reproducibility and the panic tests pass, as they should, and the structure pin fails (`left: [7.0]`, `right: [13.0]`). **The combining-order pin passes against the 0.1.0 loop**, and cannot fail there: with one element per lane, lane-first combining is index order. Its job is to fail for a different combining order, which is checked in task 7.3 by temporarily changing the order. This corrects the plan's wording, which said both pins fail against the 0.1.0 loop. Every test names FR-25, S-6 or HC-2

## 7.3 The partial-sum kernel (FR-25, HC-1, Spec §4)

- [x] `matvec_lanes::<N>` in safe Rust in `ops.rs`: for each row, `N` independent f32 sums over `chunks_exact(N)` of the row and the input, combined in decision point 1's order, then the `len mod N` tail elements added in index order; the assertions and panics of the current `matvec` are kept. `ops::matvec_lanes::<N>`: `chunks_exact(N)` on the row and the input, an `[f32; N]` of sums, a sequential combine from lane 0, then the remainder
- [x] The engine's `matvec` becomes `matvec_lanes` at N = 8 (initial, Spec §4) through one named constant, so Phase 8 changes one line; no caller and no part of the forward pass changes. `ops::LANES = 8`, and `matvec` is `matvec_lanes::<LANES>`; `forward.rs` is untouched
- [x] The combining order is written in the doc comment and the doc comment names Spec §4. In the function's doc comment, which cites Spec §4 and `tests/kernel.rs`
- [x] The combining-order pin is shown able to fail: temporarily combine the lanes last-first (then as a pairwise tree) and check `fr25_order_pin_combining_order` fails each time with the values in its comment; revert each. Combining last-first makes the pin fail with `100000024` for `100000000` (and the structure pin with `8` for `7`); combining as a pairwise tree gives `100000016` (printed `100000020.0`, the shortest decimal for that f32) and `0` for `7`. Both reverted; the pin does fail for a different order
- [x] All of 7.2's tests pass; `matvec_index_order` is unchanged and 7.1's temporary equality test is removed. All 9 kernel tests pass, `matvec_index_order` is unchanged, and the temporary equality test is gone from `kernel.rs`
- [x] The module comment at the top of `ops.rs` no longer says every operation adds in index order: it says all but the matrix-vector product do, with the reason (Spec §4). Now says every operation but the matrix-vector product adds in index order
- [x] No `unsafe`, no new dependency, no warning in a normal build, the test build, or with the features on. No `unsafe` and no new dependency in the engine; no warning in the debug, release or test builds. Whole suite: 204 tests in 14 test binaries all `ok`, 30 s (it was 92 s): the correctness harness takes 14.5 s where it took 61 s, and the reproducibility test 2.4 s where it took 11 s. The harness errors are not recorded yet: that is task 7.5

## 7.4 Did it vectorize? (FR-25, HC-1, Spec §4, §15)

- [ ] **Apply decision point 4.** Emit the release assembly of the library (for example `cargo rustc --release -p emberly-oxygen --lib -- --emit asm`) with no build flags, find the kernel's inner loop, and record in plan §4 what it contains (which vector instructions, how many sums per iteration)
- [ ] If the loop is scalar, **stop and raise it with the owner**; do not reach for `unsafe`, intrinsics or a build flag (Spec §4)
- [ ] `grep -rn unsafe crates/oxygen/src` shows none (the benchmark harness's two commented blocks are outside the engine), and `Cargo.lock` and the manifests show no new dependency
- [ ] A quick `emberly-oxygen generate --stats` run on the short benchmark prompt with the release build gives a first estimate of the speed-up. It is written here and **not recorded as a result**: it is a single run, outside the benchmark's procedure (Spec §12). Record the 0.1.0 comparison point it is read against (about 19 generation tokens/s)

## 7.5 The full gate (HC-4, S-1 to S-6, S-8, Spec §11, §17)

- [ ] `cargo test --no-fail-fast`, the whole summary read: every test binary, the number of tests, and the time
- [ ] Per category, the worst logit error (S-1) and worst intermediate error (S-2) recorded in plan §4 next to 0.1.0's; apply decision point 5 to any category more than twice 0.1.0's
- [ ] The top-token and greedy checks (S-3, S-4) pass in every category; record whether any near-tie was reached (none was in 0.1.0)
- [ ] The three negative controls still fail the harness where they should (Spec §17); the test that asserts it passes
- [ ] Run-to-run logits (S-6) are still bit-identical on one load and on a second load of the model (the existing reproducibility test); the largest difference is recorded
- [ ] Output is identical at every verbosity level (S-8, the existing test passes) and the step time with logging off is unchanged by the kernel's presence in the logging path (no change to the logging code in this phase)
- [ ] A test that failed only because it asserted a bit-exact 0.1.0 value is not edited to pass: the test, the old and new values and the reason go to the owner, who decides whether it asserts the right thing (S-6 permits last-bit differences; the harness thresholds do not change)
- [ ] `cargo build` has no warnings in any configuration; every test added in this phase names the HC, FR or S item it verifies (checked mechanically, as in Phase 6)

## Phase exit check

- [ ] The harness passes every prompt in every category with the worst errors recorded in plan §4 and, where a category moved by more than decision point 5's margin, explained (S-1 to S-5)
- [ ] All three negative controls fail as before (Spec §17)
- [ ] The kernel tests pass, including both order pins, and the order pin was seen to fail against the 0.1.0 loop (task 7.2)
- [ ] The vectorization evidence is recorded (task 7.4), and the engine contains no `unsafe` and no new dependency
- [ ] S-6 reproducibility and S-8 identical-output tests pass
- [ ] Test time and the whole summary of every binary recorded; no warnings
- [ ] The owner has reviewed the tests and the evidence; nothing is pushed, tagged or merged
