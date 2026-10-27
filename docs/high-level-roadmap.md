# Emberly Oxygen — High-Level Roadmap

Oct 27, 2026 · @Wattanit

Reference only. The SFD suite is the source of truth for scope and decisions; where the two disagree, the SFD wins.

## Purpose and principles

Emberly Oxygen (codename) is a decoder-only LLM inference engine built from scratch in Rust, to understand every part by building it. The engine starts naive and improves one version at a time, and the documented history of those versions is itself a product of the project.

- **Naive first.** Each version implements the simplest thing that works for its goal. No abstraction is added for a future version's needs; it is added when that version arrives.
- **One idea per version.** Each version introduces one main technique, so its effect can be measured and explained in isolation.
- **Correctness is proven against an oracle.** Every version must reproduce a reference implementation's output for the target model before any speed claim counts.
- **Improvement is measured, not asserted.** Every version records the same benchmarks on the same machine, so progress across versions is directly comparable.
- **Rewrites are expected.** AI agents write the code, so reimplementation is cheap. The tests and benchmarks, not the code, are the durable assets.
- **Original design.** KatGPT-RS was the learning reference, but this engine's design starts from first principles and does not copy its structure.

## Scope and non-goals

The engine covers everything between a model file on disk and generated text on screen: weight loading, tokenizer, forward pass, KV cache, sampling, decoding strategies, and their optimizations.

**Built in-house:** tensor type and math kernels, model loading, tokenizer, transformer forward pass, KV cache, samplers, decoding strategies, constraint systems, quantization.

**Allowed external crates:** general application plumbing (CLI, error handling, serialization, memory-mapped files, logging, benchmarking). The engine uses its own math kernels; a linear algebra crate is permitted only as a test oracle and performance baseline.

**Allowed outside Rust:** one Python script using Hugging Face `transformers` to generate reference fixtures. It is the oracle, not part of the engine.

**Non-goals:**

- Training or fine-tuning of any kind; the engine only runs pre-trained weights.
- Production serving features (auth, API compatibility, multi-tenant scheduling) unless a later version chooses them.
- Competing with llama.cpp, vLLM or similar on speed. Speed gains are measured against our own earlier versions.

## Rules every version follows

A version is done only when all four of these hold.

1. **Correctness gate.** All reference tests from v0 still pass: tokenizer IDs match exactly, logits match within tolerance, greedy continuations match token for token. A version that is deliberately lossy (for example, quantization) reports its measured deviation instead of passing silently, and is judged per prompt category, not only on averages.
2. **Benchmark record.** The fixed benchmark suite is run in release mode on the reference machine and appended to the version results table: prefill tok/s, decode tok/s, time to first token, peak memory.
3. **Version note.** A short document per version: what changed, why it should help, what was measured, and what was learned. This is the narrative of the project.
4. **Negative results kept.** An idea that fails or underperforms gets a written explanation of why, kept alongside the successes.

## Version overview

&#91;embedded content: version roadmap · v0 to v9, one re-plan gate\]

Each version adds one main technique on top of the last. v0 to v3 build a correct, usable engine; the order of v4 onward is decided again once v3 is done and measured.

## Version details

Versions v0 to v3 are firm. Later versions are a provisional order and will be re-planned once v3 is done.

### v0: Oracle and harness

**Goal:** be able to prove correctness and measure speed before any engine code exists.

- Python reference script: loads SmolLM2-135M, runs a fixed prompt set, saves token IDs, full logits per position, a greedy continuation, and intermediate tensors (embeddings, first layers' outputs) as fixtures.
- Rust test harness that loads the fixtures and compares against engine output with defined tolerances.
- Benchmark harness with fixed prompts and context lengths, recording results with the machine spec.

**Exit:** fixtures committed; harness runs and fails cleanly against a stub engine.

### v1: Naive engine on SmolLM2

**Goal:** correct output, with the simplest code possible.

- CPU only. Own tensor type (f32 only) and naive math kernels: matmul, RMSNorm, softmax, RoPE, SiLU.
- safetensors loader and config parsing for SmolLM2.
- Byte-level BPE tokenizer, including streaming detokenization that buffers incomplete UTF-8 sequences.
- Llama-architecture forward pass, one token at a time; plain `Vec` KV cache.
- Greedy decoding and a minimal CLI.

**Exit:** all v0 tests pass; first benchmark row recorded as the baseline.

### v2: Sampling and chat

**Goal:** usable, non-deterministic generation.

- Temperature, top-k, top-p, repetition penalty.
- Chat template and stop conditions, using the instruct variant of SmolLM2.
- Seeded RNG so sampled runs are reproducible in tests.

**Exit:** distribution-level tests for samplers; greedy path unchanged.

### v3: Batched prefill

**Goal:** process the whole prompt in one pass instead of token by token.

- Forward pass over many positions at once, with a causal mask.
- Matrix-matrix instead of matrix-vector work during prefill.

**Exit:** identical outputs; measured time-to-first-token improvement.

### v4: CPU performance

**Goal:** make the kernels fast on the reference machine.

- Cache-friendly memory layout, tiled matmul, SIMD, multithreading.
- Each optimization measured separately in the version note.

### v5: Weight quantization

**Goal:** lower memory and higher speed by storing weights in fewer bits.

- int8, then 4-bit block quantization, with dequantization inside the kernels.
- First deliberately lossy version: deviation measured per prompt category.

### v6: Second model family

**Goal:** generalize the forward pass for the first time.

- Add Qwen2.5 (QKV bias, different tokenizer and chat template); the first model with usable Thai.
- Configurable components are introduced here, driven by the real differences between two families.

### v7: Speculative decoding

**Goal:** more than one token per expensive forward pass.

- Chain speculation first, with a smaller model of the same family as drafter (they share a tokenizer), then tree speculation with a tree attention mask.
- Leviathan accept/reject with the residual distribution; distribution-equality tests.

### v8: Constrained decoding

**Goal:** guaranteed-valid output for structured formats.

- Token masking from an incremental constraint state machine; first target JSON, then a code-syntax DFA.
- Integrated with the v7 tree builder.

### v9: KV cache memory

**Goal:** support long contexts and multiple sessions within a memory budget.

- Paged KV cache, KV quantization, prefix sharing across sessions.

### Later candidates

Not yet ordered: Gemma 2 support (sandwich norms, soft-capping, sliding window), Mixture of Experts, alternative attention with fixed-size state, GPU backend.

## Relationship to SFD

This roadmap is a reference for direction, not part of the SFD suite and not a source of truth. The SFD suite is written for v0 and v1 first, then version-bumped as each later version is planned in detail.

- **Requirements Document:** the WHAT of the current version, taken from its section here.
- **Design Guideline:** principles and conventions that hold across versions, starting with this roadmap's rules.
- **Technical Specification:** the HOW of the current version (data layouts, algorithms, test tolerances).
- **Downstream:** implementation plan, phase to-do lists, and test cases cite those IDs; the v0 fixtures become the first test cases.

## Decisions

**Decided**

| Decision | Choice |
| --- | --- |
| Project name | Emberly Oxygen (codename) |
| License | AGPL, matching the Emberly suite |
| Hardware target for v1 | CPU only |
| Linear algebra | Own kernels only; a crate may serve as test oracle and benchmark baseline |
| Benchmark machine | One fixed machine runs every benchmark |
| Logit tolerance | Max absolute logit error ≤ 1e-3 at every position (starting value) |

The tolerance assumes the reference runs in f32 on CPU, so both sides use identical upcast weights and differ only in summation order. Alongside it: intermediate tensors must match within 1e-4 relative to each tensor's largest magnitude; the top-1 token must match at every position, except near-ties where the reference's top-2 gap is under 1e-3; greedy continuations must match token for token. The value is revisited after the first real measurement in v1.

**Open**

| Decision | Options | Needed by |
| --- | --- | --- |
| Which benchmark machine | To be chosen | Before the v1 baseline is recorded |
| GPU backend | Planned after the CPU versions; timing undecided | Not scheduled |
