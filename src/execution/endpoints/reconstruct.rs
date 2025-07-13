use std::num::Wrapping;

use num_traits::AsPrimitive;
use tfhe::{core_crypto::prelude::UnsignedInteger, integer::block_decomposition::BlockRecomposer, shortint::ClassicPBSParameters};

use crate::{
    algebra::{base_ring::{Z128, Z64},galois_rings::common::ResiduePoly, structure_traits::Zero},
    error::error_handler::anyhow_error_and_log,
    execution::tfhe_internals::{
        parameters::AugmentedCiphertextParameters, switch_and_squash::from_expanded_msg,
    },
};

/// Reconstructs a vector of plaintexts from raw, opened ciphertexts,
/// by using the contant term of the `openeds` and mapping it down
/// to the message space of a ciphertext block.
pub fn reconstruct_message<const EXTENSION_DEGREE: usize>(
    openeds: Option<Vec<ResiduePoly<Z128, EXTENSION_DEGREE>>>,
    params: &ClassicPBSParameters,
) -> anyhow::Result<Vec<Z128>> {
    let total_mod_bits = params.total_block_bits() as usize;
    // shift
    let mut out = Vec::new();
    match openeds {
        Some(openeds) => {
            for opened in openeds {
                let v_scalar = opened.to_scalar()?;
                out.push(from_expanded_msg(v_scalar.0, total_mod_bits));
            }
        }
        _ => {
            return Err(anyhow_error_and_log(
                "Right shift not possible - no opened value".to_string(),
            ))
        }
    };
    Ok(out)
}

pub fn reconstruct_message_64<const EXTENSION_DEGREE: usize>(
    openeds: Option<Vec<ResiduePoly<Z64, EXTENSION_DEGREE>>>,
    params: &ClassicPBSParameters,
) -> anyhow::Result<Vec<Z64>> {
    let total_mod_bits = params.total_block_bits() as usize;
    // shift
    let mut out = Vec::new();
    match openeds {
        Some(openeds) => {
            for opened in openeds {
                let v_scalar = opened.to_scalar()?;
                out.push(from_msg(v_scalar.0, total_mod_bits));
            }
        }
        _ => {
            return Err(anyhow_error_and_log(
                "Right shift not possible - no opened value".to_string(),
            ))
        }
    };
    Ok(out)
}

fn from_msg<Scalar: UnsignedInteger + AsPrimitive<u64>>(
    raw_plaintext: Scalar,
    message_and_carry_mod_bits: usize,
) -> Z64 {
    // delta = q/t where t is the amount of plain text bits
    // Observe that t includes the message and carry bits as well as the padding bit (hence the + 1)
    let delta_pad_bits = (Scalar::BITS as u64) - (message_and_carry_mod_bits as u64 + 1_u64);

    // Observe that in certain situations the computation of b-<a,s> may be negative
    // Concretely this happens when the message encrypted is 0 and randomness ends up being negative.
    // We cannot simply do the standard modulo operation then, as this would mean the message becomes
    // 2^message_mod_bits instead of 0 as it should be.
    // However the maximal negative value it can have (without a general decryption error) is delta/2
    // which we can compute as 1 << delta_pad_bits, since the padding already halves the true delta
    if raw_plaintext.as_() > Scalar::MAX.as_() - (1 << delta_pad_bits) {
        Z64::ZERO
    } else {
        // compute delta / 2
        let delta_pad_half = 1 << (delta_pad_bits - 1);

        // add delta/2 to kill the negative noise, note this does not affect the message.
        // and then divide by delta
        let raw_msg = raw_plaintext.as_().wrapping_add(delta_pad_half) >> delta_pad_bits;
        Wrapping(raw_msg % (1 << message_and_carry_mod_bits))
    }
}


/// Reconstructs a vector of plaintexts from raw, opened ciphertexts
/// and mapping it down to the message space of a ciphertext block.
/// Unlike the function [reconstruct_message], every term in `openeds`
/// is used for the reconstruction and at most `num_blocks` terms will
/// be used.
pub fn reconstruct_packed_message<const EXTENSION_DEGREE: usize>(
    openeds: Option<Vec<ResiduePoly<Z128, EXTENSION_DEGREE>>>,
    params: &ClassicPBSParameters,
    num_blocks: usize,
) -> anyhow::Result<Vec<Z128>> {
    let total_mod_bits = params.total_block_bits() as usize;
    let mut processed_blocks = 0;
    let mut out = Vec::new();
    match openeds {
        Some(openeds) => {
            for opened in openeds {
                for coef in opened.coefs {
                    out.push(from_expanded_msg(coef.0, total_mod_bits));
                    processed_blocks += 1;
                    if processed_blocks >= num_blocks {
                        break;
                    }
                }
            }
        }
        _ => return Err(anyhow_error_and_log("No opened value".to_string())),
    };

    if processed_blocks < num_blocks {
        return Err(anyhow_error_and_log(format!(
            "expected to process {num_blocks} but only processed {processed_blocks}"
        )));
    }
    Ok(out)
}

/// Helper function that takes a vector of decrypted plaintexts (each of [bits_in_block] plaintext bits)
/// and combine them into the integer message of many bits.
pub fn combine_decryptions<T>(bits_in_block: u32, decryptions: Vec<Z128>) -> anyhow::Result<T>
where
    T: tfhe::integer::block_decomposition::Recomposable
        + tfhe::core_crypto::commons::traits::CastFrom<u128>,
{
    let mut recomposer = BlockRecomposer::<T>::new(bits_in_block);

    for block in decryptions {
        if !recomposer.add_unmasked(block.0) {
            // End of T::BITS reached no need to try more
            // recomposition
            break;
        };
    }
    Ok(recomposer.value())
}

pub fn combine_decryptions_64<T>(bits_in_block: u32, decryptions: Vec<Z64>) -> anyhow::Result<T>
where
    T: tfhe::integer::block_decomposition::Recomposable
        + tfhe::core_crypto::commons::traits::CastFrom<u64>,
{
    let mut recomposer = BlockRecomposer::<T>::new(bits_in_block);

    for block in decryptions {
        if !recomposer.add_unmasked(block.0) {
            // End of T::BITS reached no need to try more
            // recomposition
            break;
        };
    }
    Ok(recomposer.value())
}

#[test]
fn test_recomposer() {
    use crate::algebra::structure_traits::FromU128;
    let out =
        combine_decryptions::<tfhe::integer::U256>(1, vec![Z128::from_u128(1), Z128::from_u128(3)])
            .unwrap();
    assert_eq!(out, tfhe::integer::U256::from((3, 0)));

    let out =
        combine_decryptions::<tfhe::integer::U256>(1, vec![Z128::from_u128(0), Z128::from_u128(7)])
            .unwrap();
    assert_eq!(out, tfhe::integer::U256::from((2, 0)));

    let out = combine_decryptions::<u64>(1, vec![Z128::from_u128(0), Z128::from_u128(0)]).unwrap();
    assert_eq!(out, 0_u64);

    let out = combine_decryptions::<u32>(2, vec![Z128::from_u128(3), Z128::from_u128(11)]).unwrap();
    assert_eq!(out, 15_u32);

    let out = combine_decryptions::<u16>(3, vec![Z128::from_u128(1), Z128::from_u128(1)]).unwrap();
    assert_eq!(out, 9_u16);
}
