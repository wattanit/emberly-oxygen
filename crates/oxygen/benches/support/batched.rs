//! The batched kernel rows' pure helpers (FR-48, Spec §12), unit-tested from
//! `tests/bench_support.rs`.

use serde_json::{json, Value};

/// Input rows of every batched kernel timing: a prompt of 128 tokens.
pub const INPUTS: usize = 128;

/// The shapes of the batched products, as `[rows, columns]` with the
/// projections that use them: the same shapes the prompt pass runs on every
/// layer (the 192-row key and value projections are too small to tile well and
/// are not timed on their own).
pub const SHAPES: [(usize, usize, &str); 3] = [
    (1536, 576, "MLP gate and up"),
    (576, 1536, "MLP down"),
    (576, 576, "query and attention output"),
];

/// The kernels a row times, in the order of the `kernels` of its record: the
/// 0.4.1 loop, the blocked kernel, and the library baseline (HC-2).
pub const KERNELS: [&str; 3] = ["matmul", "blocked", "matrixmultiply"];

/// Floating-point operations of one product: a multiply and an add for every
/// element of the weight and every input row.
pub fn flops(rows: usize, cols: usize, inputs: usize) -> f64 {
    2.0 * (rows * cols * inputs) as f64
}

/// The blocked kernel's definition written out plainly (FR-46): every element
/// a sum over the shared dimension in index order, taken in chunks of 64
/// products, each chunk summed with fused multiply-adds and the chunk sums
/// added in order; laid out `[input][row]`. The benchmark requires the kernel
/// to equal it exactly before timing it, so no rate is recorded for a kernel
/// that computes something else.
pub fn chunked_reference(weight: &[f32], rows: usize, cols: usize, input: &[f32], count: usize) -> Vec<f32> {
    let mut out = vec![0f32; count * rows];
    for t in 0..count {
        for i in 0..rows {
            let mut total = 0f32;
            for start in (0..cols).step_by(64) {
                let mut chunk = 0f32;
                for k in start..(start + 64).min(cols) {
                    chunk = weight[i * cols + k].mul_add(input[t * cols + k], chunk);
                }
                total += chunk;
            }
            out[t * rows + i] = total;
        }
    }
    out
}

/// The row fields of one shape: the group, what was timed, and the conditions
/// that make a rate comparable (one thread, 128 inputs).
pub fn fields(rows: usize, cols: usize, role: &str) -> Value {
    json!({
        "group": "kernels",
        "kernel": "matmul",
        "threads": 1,
        "rows": rows,
        "cols": cols,
        "inputs": INPUTS,
        "role": role,
        "variants": KERNELS,
        "baseline": "matrixmultiply sgemm, single-threaded",
    })
}
