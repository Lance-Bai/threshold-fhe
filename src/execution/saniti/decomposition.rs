pub use core_crypto::commons::math::decomposition::DecompositionLevel;
use core_crypto::commons::numeric::UnsignedInteger;
use core_crypto::commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
use dyn_stack::PodStack;
use std::iter::Map;
use std::ops::Neg;
use std::slice::IterMut;
use tfhe::core_crypto;
use tfhe::core_crypto::commons::math::random::RandomGenerator;
use tfhe::core_crypto::seeders::new_seeder;
use tfhe::prelude::CastInto;
use tfhe_csprng::generators::SoftwareRandomGenerator;

// copied from src/commons/math/decomposition/*.rs
// in order to avoid allocations
// const NOISE_STD_DEV: f64 = 2269928957.75; // (2^31.08) should divide by 2^\beta later
// const NOISE_STD_DEV: f64 = 1048576.0; // 2^20
const NOISE_STD_DEV: f64 = 536870912.0; //2^29 yes
// const NOISE_STD_DEV: f64 = 1073741824.0; //2^30 no
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

        // let mut state0 = self.states[0].clone();
        // println!("in  state: {:064b}",state0);
        // let dec = decompose_one_level(base_log, &mut state0, mod_b_mask);
        // println!("out state: {:064b}",state0);
        // println!("decompse : {:064b}",dec);

        Some((
            DecompositionLevel(current_level),
            DecompositionBaseLog(self.base_log),
            self.states
                .iter_mut()
                .map(move |state| decompose_one_level(base_log, state, mod_b_mask)),
        ))
    }
}

pub fn decompose_one_level<S: UnsignedInteger>(base_log: usize, state: &mut S, mod_b_mask: S) -> S {
    // println!("\n
    //     \tstage i:{:064b}", state);

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
    let (a, _): (f64, f64) = generator.random_gaussian(0_f64, NOISE_STD_DEV);

    // let gaussian_noise: S = a.cast_into();
    // let gaussian_noise:S = float_to_u128_fraction(a).cast_into();
    let gaussian_noise: S = float_to_u64_fraction2(a);

    // JUST FOR TEST
    // println!("noise is {}->{:?}", a, gaussian_noise.into_signed());
    // let gaussian_noise = S::ONE;

    let res2 = gaussian_noise
        .wrapping_div(S::ONE << base_log)
        .wrapping_mul(S::ONE << base_log)
        .wrapping_add(res1);
    // println!("
    //     \t + a_i: {:064b}\n
    //     \tout:    {:064b}",
    //     res1,
    //     res2
    // );
    // let res2 = gaussian_noise
    //     .wrapping_add(res1);
    // println!(
    //     "gaussian_noise: {:?}+{:?}->{:?}",
    //     gaussian_noise.into_signed(),
    //     res1.into_signed(),
    //     res2.into_signed()
    // );

    *state = state.wrapping_sub(res2);
    let high = state.shr(S::BITS - 1); // high = 0 or 1
    let shifted = state.shr(base_log);
    let fill_mask = if high == S::ONE {
        (!S::ZERO).shl(S::BITS - base_log)
    } else {
        S::ZERO
    };
    *state = shifted | fill_mask;

    // println!("\n
    //     \tstage o:{:064b}", state);
    res2
}

// fn decompose_one_level<S: UnsignedInteger>(base_log: usize, state: &mut S, mod_b_mask: S) -> S {
//     let res = *state & mod_b_mask;
//     let carry = decomposition_bit_trick(res, *state, base_log);
//     let res1 = res.wrapping_sub(carry << base_log);

//     // let gaussian = S::ZERO;
//     // let res2 = gaussian.wrapping_mul(S::ONE << base_log).wrapping_add(res1);

//     // Define the standard deviation for the Gaussian noise.
//     // This value should be chosen based on your scheme's security parameters.

//     // Note: This is inefficient as it creates a new generator for each decomposition.
//     // A better approach would be to pass a mutable generator down the call stack.
//     let mut seeder = new_seeder();
//     let mut generator = RandomGenerator::<SoftwareRandomGenerator>::new(seeder.seed());

//     let std_dev = NOISE_STD_DEV / 2_f64.powf(base_log.cast_into());
//     let (a, _): (f64, f64) = generator.random_gaussian(0_f64, std_dev);

//     // let gaussian_noise: S = a.cast_into();
//     // let gaussian_noise:S = float_to_u128_fraction(a).cast_into();
//     let gaussian_noise: S = float_to_u64_fraction(a);

//     // JUST FOR TEST
//     // println!("noise is {}->{:?}", a, gaussian_noise.into_signed());
//     // let gaussian_noise = S::ONE;

//     let res2 = gaussian_noise
//         .wrapping_mul(S::ONE << base_log)
//         .wrapping_add(res1);
//     // println!(
//     //     "gaussian_noise: {:?}->{:?}->{:?}",
//     //     gaussian_noise.into_signed(),
//     //     gaussian_noise
//     //         .wrapping_mul(S::ONE << base_log)
//     //         .into_signed(),
//     //     res2.into_signed()
//     // );
//     // let res2 = gaussian_noise
//     //     .wrapping_add(res1);
//     // println!(
//     //     "gaussian_noise: {:?}+{:?}->{:?}",
//     //     gaussian_noise.into_signed(),
//     //     res1.into_signed(),
//     //     res2.into_signed()
//     // );

//     *state = state.wrapping_sub(res2);
//     *state >>= base_log;

//     res2
// }
#[inline(always)]
fn decomposition_bit_trick<Scalar: UnsignedInteger>(
    res: Scalar,
    state: Scalar,
    base_log: usize,
) -> Scalar {
    ((res.wrapping_sub(Scalar::ONE) | state >> base_log) & res) >> (base_log - 1)
}

// #[inline(always)]
// fn float_to_u64_fraction<S: UnsignedInteger>(x: f64) -> S {
//     let frac = if x.abs() > 0.5 { 0.0 } else { x };
//     let max_v: f64 = S::MAX.cast_into();
//     let scaled: f64;
//     if frac >= 0.0 {
//         scaled = frac * (max_v + 1.0);
//         scaled.cast_into()
//     } else {
//         scaled = (0.0 - frac) * (max_v + 1.0);
//         S::MAX - scaled.cast_into()
//     }
// }

#[inline(always)]
fn float_to_u64_fraction2<S: UnsignedInteger>(x: f64) -> S {
    if x >= 0.0 {
        x.cast_into()
    } else {
        S::MAX - x.neg().cast_into()
    }
}
