pub use core_crypto::commons::math::decomposition::DecompositionLevel;
use core_crypto::commons::numeric::UnsignedInteger;
use core_crypto::commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
use dyn_stack::PodStack;
use std::iter::Map;
use std::slice::IterMut;
use tfhe::core_crypto;
use tfhe::core_crypto::commons::math::random::RandomGenerator;
use tfhe::core_crypto::prelude::Gaussian;
use tfhe::core_crypto::seeders::new_seeder;
use tfhe::prelude::CastInto;
use tfhe_csprng::generators::SoftwareRandomGenerator;

// copied from src/commons/math/decomposition/*.rs
// in order to avoid allocations
const NOISE_STD_DEV: f64 = 4.5; // Example value

pub struct TensorSignedDecompositionLendingIter<'buffers, Scalar: UnsignedInteger> {
    // The base log of the decomposition
    base_log: usize,
    // The current level
    current_level: usize,
    // A mask which allows to compute the mod B of a value. For B=2^4, this guy is of the form:
    // ...0001111
    mod_b_mask: Scalar,
    // The internal states of each decomposition
    states: &'buffers mut [Scalar],
    // A flag which stores whether the iterator is a fresh one (for the recompose method).
    fresh: bool,
}

impl<'buffers, Scalar: UnsignedInteger> TensorSignedDecompositionLendingIter<'buffers, Scalar> {
    #[inline]
    pub(crate) fn new(
        input: impl Iterator<Item = Scalar>,
        base_log: DecompositionBaseLog,
        level: DecompositionLevelCount,
        stack: &'buffers mut PodStack,
    ) -> (Self, &'buffers mut PodStack) {
        let (states, stack) = stack.collect_aligned(aligned_vec::CACHELINE_ALIGN, input);
        (
            TensorSignedDecompositionLendingIter {
                base_log: base_log.0,
                current_level: level.0,
                mod_b_mask: (Scalar::ONE << base_log.0) - Scalar::ONE,
                states,
                fresh: true,
            },
            stack,
        )
    }

    // inlining this improves perf of external product by about 25%, even in LTO builds
    #[inline]
    #[allow(
        clippy::type_complexity,
        reason = "The type complexity would require a pub type = ...; \
        but impl Trait is not stable in pub type so we tell clippy to leave us alone"
    )]
    pub fn next_term<'short>(
        &'short mut self,
    ) -> Option<(
        DecompositionLevel,
        DecompositionBaseLog,
        Map<IterMut<'short, Scalar>, impl FnMut(&'short mut Scalar) -> Scalar>,
    )> {
        // The iterator is not fresh anymore.
        self.fresh = false;
        // We check if the decomposition is over
        if self.current_level == 0 {
            return None;
        }
        let current_level = self.current_level;
        let base_log = self.base_log;
        let mod_b_mask = self.mod_b_mask;
        self.current_level -= 1;

        Some((
            DecompositionLevel(current_level),
            DecompositionBaseLog(self.base_log),
            self.states
                .iter_mut()
                .map(move |state| decompose_one_level(base_log, state, mod_b_mask)),
        ))
    }
}

fn decompose_one_level<S: UnsignedInteger>(base_log: usize, state: &mut S, mod_b_mask: S) -> S {
    let res = *state & mod_b_mask;
    let carry = decomposition_bit_trick(res, *state, base_log);
    let res1 = res.wrapping_sub(carry << base_log);

    // let gaussian = S::ZERO;
    // let res2 = gaussian.wrapping_mul(S::ONE << base_log).wrapping_add(res1);

    // Define the standard deviation for the Gaussian noise.
    // This value should be chosen based on your scheme's security parameters.

    // Note: This is inefficient as it creates a new generator for each decomposition.
    // A better approach would be to pass a mutable generator down the call stack.
    let mut seeder = new_seeder();
    let mut generator = RandomGenerator::<SoftwareRandomGenerator>::new(seeder.seed());

    let random = generator.random_from_distribution::<u128, _>(Gaussian {
        mean: 0.0,
        std: NOISE_STD_DEV,
    });
    let gaussian_noise = random.cast_into();
    // The final term is the signed digit with the added noise.
    let res2 = res1.wrapping_add(gaussian_noise);

    *state -= res2;
    *state >>= base_log;

    res2
}
#[inline(always)]
fn decomposition_bit_trick<Scalar: UnsignedInteger>(
    res: Scalar,
    state: Scalar,
    base_log: usize,
) -> Scalar {
    ((res.wrapping_sub(Scalar::ONE) | state >> base_log) & res) >> (base_log - 1)
}

// fn decompose_one_level<S: UnsignedInteger>(
//     base_log: usize,
//     state: &mut S,
//     mod_b_mask: S,
// ) -> S {
//     let res = *state & mod_b_mask;
//     *state >>= base_log;
//     let carry = decomposition_bit_trick(res, *state, base_log);
//     *state += carry;
//     res.wrapping_sub(carry << base_log)
// }

// #[inline(always)]
// fn decomposition_bit_trick<Scalar: UnsignedInteger>(
//     res: Scalar,
//     state: Scalar,
//     base_log: usize,
// ) -> Scalar {
//     ((res.wrapping_sub(Scalar::ONE) | state) & res) >> (base_log - 1)
// }
