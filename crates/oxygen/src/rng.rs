//! The seeded random number generator (Spec §8, FR-28): xoshiro256** (Blackman
//! and Vigna, public domain), its 256-bit state filled from the 64-bit seed by
//! SplitMix64, as the authors recommend. Written in this project (HC-1): the
//! generator is part of what the project exists to understand. The algorithm
//! is the Specification's choice, not the Requirements': FR-28 promises
//! reproduction within a build only, so a later release may replace it.

pub(crate) struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut splitmix = || {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Rng {
            state: [splitmix(), splitmix(), splitmix(), splitmix()],
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// A uniform value in `[0, 1)` from the top 53 bits of one output, so every
    /// value is exactly representable and 1.0 can never be returned.
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectors from the authors' reference C code (xoshiro256** 1.0 and
    // SplitMix64, public domain), compiled in the scratchpad for task 12.1.
    // They verify the generator is the published algorithm (HC-1, FR-28).
    const SEED_0: [u64; 5] = [
        0x99ec5f36cb75f2b4, 0xbf6e1f784956452a, 0x1a5f849d4933e6e0, 0x6aa594f1262d2d2c, 0xbba5ad4a1f842e59,
    ];
    const SEED_1: [u64; 5] = [
        0xb3f2af6d0fc710c5, 0x853b559647364cea, 0x92f89756082a4514, 0x642e1c7bc266a3a7, 0xb27a48e29a233673,
    ];
    const SEED_BIG: [u64; 5] = [
        0x9e32cfb5bb93eebb, 0x16006bd9d4ac0014, 0x8ada5d6d34b6538e, 0x7c327ca32346a238, 0xc43a6d6a3492ced2,
    ];

    fn first_five(seed: u64) -> [u64; 5] {
        let mut rng = Rng::new(seed);
        std::array::from_fn(|_| rng.next_u64())
    }

    #[test]
    fn fr28_outputs_equal_the_published_reference_for_known_seeds() {
        assert_eq!(first_five(0), SEED_0);
        assert_eq!(first_five(1), SEED_1);
        assert_eq!(first_five(0xDEADBEEFCAFEF00D), SEED_BIG);
    }

    #[test]
    fn fr28_the_uniform_is_the_top_53_bits_scaled_into_zero_to_one() {
        let mut rng = Rng::new(1);
        let expected = [0.70292183315885048, 0.52043661993885693, 0.5741057000197225];
        for want in expected {
            assert_eq!(rng.next_f64(), want);
        }
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u), "{u}");
        }
    }

    #[test]
    fn fr28_the_same_seed_repeats_and_different_seeds_differ() {
        assert_eq!(first_five(42), first_five(42));
        assert_ne!(first_five(42), first_five(43));
    }
}
