use aligned_vec::CACHELINE_ALIGN;
use itertools::izip;
use tfhe::{
    boolean::prelude::{DecompositionBaseLog, DecompositionLevelCount},
    core_crypto::{
        commons::math::decomposition::DecompositionLevel,
        fft_impl::{
            common::pbs_modulus_switch,
            fft128::{crypto::ggsw::update_with_fmadd, math::fft::Fft128View},
        },
        prelude::{
            extract_lwe_sample_from_glwe_ciphertext,
            polynomial_algorithms::{
                polynomial_wrapping_monic_monomial_div_assign,
                polynomial_wrapping_monic_monomial_mul_assign,
            },
            programmable_bootstrap_f128_lwe_ciphertext,
            programmable_bootstrap_f128_lwe_ciphertext_mem_optimized_requirement,
            ComputationBuffers, Container, ContainerMut, ContiguousEntityContainer,
            ContiguousEntityContainerMut, Fft128, Fourier128GgswCiphertext,
            Fourier128LweBootstrapKey, GlweCiphertext, GlweCiphertextMutView, GlweCiphertextView,
            LweCiphertext, MonomialDegree, SignedDecomposer, Split, UnsignedTorus,
        },
    },
    prelude::CastInto,
};

use dyn_stack::PodStack;

use crate::execution::saniti::decomposition::TensorSignedDecompositionLendingIter;
pub fn santi_programmable_bootstrap_f128_lwe_ciphertext<
    Scalar,
    InputCont,
    OutputCont,
    AccCont,
    KeyCont,
>(
    input: &LweCiphertext<InputCont>,
    output: &mut LweCiphertext<OutputCont>,
    accumulator: &GlweCiphertext<AccCont>,
    fourier_bsk: &Fourier128LweBootstrapKey<KeyCont>,
) where
    // CastInto required for PBS modulus switch which returns a usize
    Scalar: UnsignedTorus + CastInto<usize>,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
    AccCont: Container<Element = Scalar>,
    KeyCont: Container<Element = f64>,
{
    assert_eq!(input.ciphertext_modulus(), output.ciphertext_modulus());
    assert_eq!(
        output.ciphertext_modulus(),
        accumulator.ciphertext_modulus()
    );

    let mut buffers = ComputationBuffers::new();

    let fft = Fft128::new(fourier_bsk.polynomial_size());
    let fft = fft.as_view();

    buffers.resize(
        programmable_bootstrap_f128_lwe_ciphertext_mem_optimized_requirement::<Scalar>(
            fourier_bsk.glwe_size(),
            fourier_bsk.polynomial_size(),
            fft,
        )
        .unwrap()
        .unaligned_bytes_required(),
    );

    let stack = buffers.stack();

    saniti_programmable_bootstrap_f128_lwe_ciphertext_mem_optimized(
        input,
        output,
        accumulator,
        fourier_bsk,
        fft,
        stack,
    );
}

fn saniti_programmable_bootstrap_f128_lwe_ciphertext_mem_optimized<
    Scalar,
    InputCont,
    OutputCont,
    AccCont,
    KeyCont,
>(
    input: &LweCiphertext<InputCont>,
    output: &mut LweCiphertext<OutputCont>,
    accumulator: &GlweCiphertext<AccCont>,
    fourier_bsk: &Fourier128LweBootstrapKey<KeyCont>,
    fft: Fft128View<'_>,
    stack: &mut PodStack,
) where
    // CastInto required for PBS modulus switch which returns a usize
    Scalar: UnsignedTorus + CastInto<usize>,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
    AccCont: Container<Element = Scalar>,
    KeyCont: Container<Element = f64>,
{
    saniti_bootstrap(
        fourier_bsk.as_view(),
        output.as_mut_view(),
        input.as_view(),
        accumulator.as_view(),
        fft,
        stack,
    );
}

fn saniti_bootstrap<Scalar: UnsignedTorus + CastInto<usize>>(
    fourier_bsk: Fourier128LweBootstrapKey<&[f64]>,
    mut lwe_out: LweCiphertext<&mut [Scalar]>,
    lwe_in: LweCiphertext<&[Scalar]>,
    accumulator: GlweCiphertext<&[Scalar]>,
    fft: Fft128View<'_>,
    stack: &mut PodStack,
) {
    let (local_accumulator_data, stack) =
        stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());
    let mut local_accumulator = GlweCiphertextMutView::from_container(
        local_accumulator_data,
        accumulator.polynomial_size(),
        accumulator.ciphertext_modulus(),
    );

    saniti_blind_rotate_assign(
        fourier_bsk,
        local_accumulator.as_mut_view(),
        lwe_in.as_view(),
        fft,
        stack,
    );
    // fourier_bsk.blind_rotate_assign(&mut local_accumulator.as_mut_view(), &lwe_in, fft, stack);
    extract_lwe_sample_from_glwe_ciphertext(&local_accumulator, &mut lwe_out, MonomialDegree(0));
}

fn saniti_blind_rotate_assign<Scalar: UnsignedTorus + CastInto<usize>>(
    fourier_bsk: Fourier128LweBootstrapKey<&[f64]>,
    mut lut: GlweCiphertext<&mut [Scalar]>,
    lwe: LweCiphertext<&[Scalar]>,
    fft: Fft128View<'_>,
    stack: &mut PodStack,
) {
    let lwe = lwe.as_ref();
    let (lwe_body, lwe_mask) = lwe.split_last().unwrap();

    let lut_poly_size = lut.polynomial_size();
    let ciphertext_modulus = lut.ciphertext_modulus();
    assert!(ciphertext_modulus.is_compatible_with_native_modulus());
    let monomial_degree = pbs_modulus_switch(*lwe_body, lut_poly_size);

    lut.as_mut_polynomial_list()
        .iter_mut()
        .for_each(|mut poly| {
            polynomial_wrapping_monic_monomial_div_assign(
                &mut poly,
                MonomialDegree(monomial_degree),
            );
        });

    // We initialize the ct_0 used for the successive cmuxes
    let mut ct0 = lut;

    for (lwe_mask_element, bootstrap_key_ggsw) in
        izip!(lwe_mask.iter(), fourier_bsk.into_ggsw_iter())
    {
        if *lwe_mask_element != Scalar::ZERO {
            let stack = &mut *stack;
            // We copy ct_0 to ct_1
            let (ct1, stack) = stack.collect_aligned(CACHELINE_ALIGN, ct0.as_ref().iter().copied());
            let mut ct1 = GlweCiphertextMutView::from_container(
                ct1,
                ct0.polynomial_size(),
                ct0.ciphertext_modulus(),
            );

            // We rotate ct_1 by performing ct_1 <- ct_1 * X^{a_hat}
            for mut poly in ct1.as_mut_polynomial_list().iter_mut() {
                polynomial_wrapping_monic_monomial_mul_assign(
                    &mut poly,
                    MonomialDegree(pbs_modulus_switch(*lwe_mask_element, lut_poly_size)),
                );
            }

            // ct1 is re-created each loop it can be moved, ct0 is already a view, but
            // as_mut_view is required to keep borrow rules consistent
            saniti_cmux(&mut ct0, &mut ct1, &bootstrap_key_ggsw, fft, stack);
        }
    }

    if !ciphertext_modulus.is_native_modulus() {
        // When we convert back from the fourier domain, integer values will contain up to
        // about 100 MSBs with information. In our representation of power of 2
        // moduli < native modulus we fill the MSBs and leave the LSBs
        // empty, this usage of the signed decomposer allows to round while
        // keeping the data in the MSBs
        let signed_decomposer = SignedDecomposer::new(
            DecompositionBaseLog(ciphertext_modulus.get_custom_modulus().ilog2() as usize),
            DecompositionLevelCount(1),
        );
        ct0.as_mut()
            .iter_mut()
            .for_each(|x| *x = signed_decomposer.closest_representable(*x));
    }
}

fn saniti_cmux<Scalar, ContCt0, ContCt1, ContGgsw>(
    ct0: &mut GlweCiphertext<ContCt0>,
    ct1: &mut GlweCiphertext<ContCt1>,
    ggsw: &Fourier128GgswCiphertext<ContGgsw>,
    fft: Fft128View<'_>,
    stack: &mut PodStack,
) where
    Scalar: UnsignedTorus,
    ContCt0: ContainerMut<Element = Scalar>,
    ContCt1: ContainerMut<Element = Scalar>,
    ContGgsw: Container<Element = f64>,
{
    fn implementation<Scalar: UnsignedTorus>(
        mut ct0: GlweCiphertext<&mut [Scalar]>,
        mut ct1: GlweCiphertext<&mut [Scalar]>,
        ggsw: Fourier128GgswCiphertext<&[f64]>,
        fft: Fft128View<'_>,
        stack: &mut PodStack,
    ) {
        for (c1, c0) in izip!(ct1.as_mut(), ct0.as_ref()) {
            *c1 = c1.wrapping_sub(*c0);
        }
        saniti_add_external_product_assign(&mut ct0, &ggsw, &ct1, fft, stack);
    }

    implementation(
        ct0.as_mut_view(),
        ct1.as_mut_view(),
        ggsw.as_view(),
        fft,
        stack,
    );
}
fn saniti_add_external_product_assign<Scalar, ContOut, ContGgsw, ContGlwe>(
    out: &mut GlweCiphertext<ContOut>,
    ggsw: &Fourier128GgswCiphertext<ContGgsw>,
    glwe: &GlweCiphertext<ContGlwe>,
    fft: Fft128View<'_>,
    stack: &mut PodStack,
) where
    Scalar: UnsignedTorus,
    ContOut: ContainerMut<Element = Scalar>,
    ContGgsw: Container<Element = f64>,
    ContGlwe: Container<Element = Scalar>,
{
    fn implementation<Scalar: UnsignedTorus>(
        mut out: GlweCiphertext<&mut [Scalar]>,
        ggsw: Fourier128GgswCiphertext<&[f64]>,
        glwe: GlweCiphertext<&[Scalar]>,
        fft: Fft128View<'_>,
        stack: &mut PodStack,
    ) {
        // we check that the polynomial sizes match
        debug_assert_eq!(ggsw.polynomial_size(), glwe.polynomial_size());
        debug_assert_eq!(ggsw.polynomial_size(), out.polynomial_size());
        // we check that the glwe sizes match
        debug_assert_eq!(ggsw.glwe_size(), glwe.glwe_size());
        debug_assert_eq!(ggsw.glwe_size(), out.glwe_size());

        debug_assert_eq!(glwe.ciphertext_modulus(), out.ciphertext_modulus());

        let align = CACHELINE_ALIGN;
        let fourier_poly_size = ggsw.polynomial_size().to_fourier_polynomial_size().0;
        let ciphertext_modulus = glwe.ciphertext_modulus();

        // we round the input mask and body
        let decomposer = SignedDecomposer::<Scalar>::new(
            ggsw.decomposition_base_log(),
            ggsw.decomposition_level_count(),
        );

        let (output_fft_buffer_re0, stack) =
            stack.make_aligned_raw::<f64>(fourier_poly_size * ggsw.glwe_size().0, align);
        let (output_fft_buffer_re1, stack) =
            stack.make_aligned_raw::<f64>(fourier_poly_size * ggsw.glwe_size().0, align);
        let (output_fft_buffer_im0, stack) =
            stack.make_aligned_raw::<f64>(fourier_poly_size * ggsw.glwe_size().0, align);
        let (output_fft_buffer_im1, substack0) =
            stack.make_aligned_raw::<f64>(fourier_poly_size * ggsw.glwe_size().0, align);

        // output_fft_buffer is initially uninitialized, considered to be implicitly zero, to avoid
        // the cost of filling it up with zeros. `is_output_uninit` is set to `false` once
        // it has been fully initialized for the first time.
        let mut is_output_uninit = true;

        {
            // ------------------------------------------------------ EXTERNAL PRODUCT IN FOURIER
            // DOMAIN In this section, we perform the external product in the fourier
            // domain, and accumulate the result in the output_fft_buffer variable.
            let (mut decomposition, substack1) = TensorSignedDecompositionLendingIter::new(
                glwe.as_ref()
                    .iter()
                    .map(|s| decomposer.init_decomposer_state(*s)),
                decomposer.base_log(),
                decomposer.level_count(),
                substack0,
            );

            // We loop through the levels (we reverse to match the order of the decomposition
            // iterator.)
            for ggsw_decomp_matrix in ggsw.into_levels() {
                // We retrieve the decomposition of this level.
                let (glwe_level, glwe_decomp_term, substack2) =
                    collect_next_term(&mut decomposition, substack1, align);
                let glwe_decomp_term = GlweCiphertextView::from_container(
                    &*glwe_decomp_term,
                    ggsw.polynomial_size(),
                    ciphertext_modulus,
                );
                debug_assert_eq!(ggsw_decomp_matrix.decomposition_level(), glwe_level);

                // For each level we have to add the result of the vector-matrix product between the
                // decomposition of the glwe, and the ggsw level matrix to the output. To do so, we
                // iteratively add to the output, the product between every line of the matrix, and
                // the corresponding (scalar) polynomial in the glwe decomposition:
                //
                //                ggsw_mat                        ggsw_mat
                //   glwe_dec   | - - - - | <        glwe_dec   | - - - - |
                //  | - - - | x | - - - - |         | - - - | x | - - - - | <
                //    ^         | - - - - |             ^       | - - - - |
                //
                //        t = 1                           t = 2                     ...

                for (ggsw_row, glwe_poly) in izip!(
                    ggsw_decomp_matrix.into_rows(),
                    glwe_decomp_term.as_polynomial_list().iter()
                ) {
                    let len = fourier_poly_size;
                    let stack = &mut *substack2;
                    let (fourier_re0, stack) = stack.make_aligned_raw::<f64>(len, align);
                    let (fourier_re1, stack) = stack.make_aligned_raw::<f64>(len, align);
                    let (fourier_im0, stack) = stack.make_aligned_raw::<f64>(len, align);
                    let (fourier_im1, _) = stack.make_aligned_raw::<f64>(len, align);
                    // We perform the forward fft transform for the glwe polynomial
                    fft.forward_as_integer(
                        fourier_re0,
                        fourier_re1,
                        fourier_im0,
                        fourier_im1,
                        glwe_poly.as_ref(),
                    );
                    // Now we loop through the polynomials of the output, and add the
                    // corresponding product of polynomials.
                    update_with_fmadd(
                        output_fft_buffer_re0,
                        output_fft_buffer_re1,
                        output_fft_buffer_im0,
                        output_fft_buffer_im1,
                        ggsw_row,
                        fourier_re0,
                        fourier_re1,
                        fourier_im0,
                        fourier_im1,
                        is_output_uninit,
                        fourier_poly_size,
                    );

                    // we initialized `output_fft_buffer, so we can set this to false
                    is_output_uninit = false;
                }
            }
        }

        // --------------------------------------------  TRANSFORMATION OF RESULT TO STANDARD DOMAIN
        // In this section, we bring the result from the fourier domain, back to the standard
        // domain, and add it to the output.
        //
        // We iterate over the polynomials in the output.
        if !is_output_uninit {
            for (mut out, fourier_re0, fourier_re1, fourier_im0, fourier_im1) in izip!(
                out.as_mut_polynomial_list().iter_mut(),
                output_fft_buffer_re0.into_chunks(fourier_poly_size),
                output_fft_buffer_re1.into_chunks(fourier_poly_size),
                output_fft_buffer_im0.into_chunks(fourier_poly_size),
                output_fft_buffer_im1.into_chunks(fourier_poly_size),
            ) {
                fft.add_backward_as_torus(
                    out.as_mut(),
                    fourier_re0,
                    fourier_re1,
                    fourier_im0,
                    fourier_im1,
                    substack0,
                );
            }
        }
    }

    implementation(
        out.as_mut_view(),
        ggsw.as_view(),
        glwe.as_view(),
        fft,
        stack,
    );
}

fn collect_next_term<'a, Scalar: UnsignedTorus>(
    decomposition: &mut TensorSignedDecompositionLendingIter<'_, Scalar>,
    substack1: &'a mut PodStack,
    align: usize,
) -> (DecompositionLevel, &'a mut [Scalar], &'a mut PodStack) {
    let (glwe_level, _, glwe_decomp_term) = decomposition.next_term().unwrap();
    let (glwe_decomp_term, substack2) = substack1.collect_aligned(align, glwe_decomp_term);
    (glwe_level, glwe_decomp_term, substack2)
}

#[cfg(test)]
mod test {
    use tfhe::{
        boolean::prelude::{GlweDimension, LweDimension, PolynomialSize, StandardDev},
        core_crypto::{
            prelude::{
                allocate_and_encrypt_new_lwe_ciphertext,
                convert_standard_lwe_bootstrap_key_to_fourier_128, decrypt_lwe_ciphertext,
                generate_programmable_bootstrap_glwe_lut,
                par_allocate_and_generate_new_seeded_lwe_bootstrap_key, EncryptionRandomGenerator,
                Gaussian, GlweCiphertextOwned, GlweSecretKey, LweBootstrapKeyOwned,
                LweCiphertextOwned, LweSecretKey, Plaintext, SecretRandomGenerator,
            },
            seeders::new_seeder,
        },
        shortint::CiphertextModulus,
    };
    use tfhe_csprng::generators::DefaultRandomGenerator;

    use super::*;
    #[test]
    fn test_saniti_programmable_bootstrap_f128_lwe_ciphertext() {
        let small_lwe_dimension = LweDimension(742);
        let glwe_dimension = GlweDimension(1);
        let polynomial_size = PolynomialSize(2048);
        let lwe_noise_distribution = Gaussian::from_dispersion_parameter(
            StandardDev(0.000007069849454709433 * 0.000007069849454709433),
            0.0,
        );
        let glwe_noise_distribution = Gaussian::from_dispersion_parameter(
            StandardDev(0.00000000000000029403601535432533 * 0.00000000000000029403601535432533),
            0.0,
        );
        let pbs_base_log = DecompositionBaseLog(23);
        let pbs_level = DecompositionLevelCount(1);
        let ciphertext_modulus = CiphertextModulus::new_native();

        // Request the best seeder possible, starting with hardware entropy sources and falling back to
        // /dev/random on Unix systems if enabled via cargo features
        let mut boxed_seeder = new_seeder();
        // Get a mutable reference to the seeder as a trait object from the Box returned by new_seeder
        let seeder = boxed_seeder.as_mut();

        // Create a generator which uses a CSPRNG to generate secret keys
        let mut secret_generator =
            SecretRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed());

        // Create a generator which uses two CSPRNGs to generate public masks and secret encryption
        // noise
        let mut encryption_generator =
            EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);

        println!("Generating keys...");

        // Generate an LweSecretKey with binary coefficients
        let small_lwe_sk =
            LweSecretKey::generate_new_binary(small_lwe_dimension, &mut secret_generator);

        // Generate a GlweSecretKey with binary coefficients
        let glwe_sk = GlweSecretKey::generate_new_binary(
            glwe_dimension,
            polynomial_size,
            &mut secret_generator,
        );

        // Create a copy of the GlweSecretKey re-interpreted as an LweSecretKey
        let big_lwe_sk = glwe_sk.clone().into_lwe_secret_key();

        // Generate the seeded bootstrapping key to show how to handle entity decompression,
        // we use the parallel variant for performance reason
        let std_bootstrapping_key = par_allocate_and_generate_new_seeded_lwe_bootstrap_key(
            &small_lwe_sk,
            &glwe_sk,
            pbs_base_log,
            pbs_level,
            glwe_noise_distribution,
            ciphertext_modulus,
            seeder,
        );

        // We decompress the bootstrapping key
        let std_bootstrapping_key: LweBootstrapKeyOwned<u64> =
            std_bootstrapping_key.decompress_into_lwe_bootstrap_key();

        // Create the empty bootstrapping key in the Fourier domain
        let mut fourier_bsk = Fourier128LweBootstrapKey::new(
            std_bootstrapping_key.input_lwe_dimension(),
            std_bootstrapping_key.glwe_size(),
            std_bootstrapping_key.polynomial_size(),
            std_bootstrapping_key.decomposition_base_log(),
            std_bootstrapping_key.decomposition_level_count(),
        );

        // Use the conversion function (a memory optimized version also exists but is more complicated
        // to use) to convert the standard bootstrapping key to the Fourier domain
        convert_standard_lwe_bootstrap_key_to_fourier_128(&std_bootstrapping_key, &mut fourier_bsk);
        // We don't need the standard bootstrapping key anymore
        drop(std_bootstrapping_key);

        // Our 4 bits message space
        let message_modulus = 1u64 << 2;

        // Our input message
        let input_message = 2u64;

        // Delta used to encode 4 bits of message + a bit of padding on u128
        let delta = (1_u64 << 63) / message_modulus;

        // Apply our encoding
        let plaintext = Plaintext(input_message * delta);

        // Allocate a new LweCiphertext and encrypt our plaintext
        let lwe_ciphertext_in: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(
            &small_lwe_sk,
            plaintext,
            lwe_noise_distribution,
            ciphertext_modulus,
            &mut encryption_generator,
        );

        // Now we will use a PBS to compute a multiplication by 2, it is NOT the recommended way of
        // doing this operation in terms of performance as it's much more costly than a multiplication
        // with a cleartext, however it resets the noise in a ciphertext to a nominal level and allows
        // to evaluate arbitrary functions so depending on your use case it can be a better fit.

        // Generate the accumulator for our multiplication by 2 using a simple closure
        let accumulator: GlweCiphertextOwned<u64> = generate_programmable_bootstrap_glwe_lut(
            polynomial_size,
            glwe_dimension.to_glwe_size(),
            message_modulus as usize,
            ciphertext_modulus,
            delta,
            |x: u64| x,
        );

        // Allocate the LweCiphertext to store the result of the PBS
        let mut pbs_multiplication_ct = LweCiphertext::new(
            0u64,
            big_lwe_sk.lwe_dimension().to_lwe_size(),
            ciphertext_modulus,
        );
        println!("Computing PBS...");
        santi_programmable_bootstrap_f128_lwe_ciphertext(
            &lwe_ciphertext_in,
            &mut pbs_multiplication_ct,
            &accumulator,
            &fourier_bsk,
        );

        // Decrypt the PBS multiplication result
        let pbs_multiplication_plaintext: Plaintext<u64> =
            decrypt_lwe_ciphertext(&big_lwe_sk, &pbs_multiplication_ct);

        // Create a SignedDecomposer to perform the rounding of the decrypted plaintext
        // We pass a DecompositionBaseLog of 5 and a DecompositionLevelCount of 1 indicating we want to
        // round the 5 MSB, 1 bit of padding plus our 4 bits of message
        let signed_decomposer =
            SignedDecomposer::new(DecompositionBaseLog(5), DecompositionLevelCount(1));

        // Round and remove our encoding
        let pbs_multiplication_result: u64 =
            signed_decomposer.closest_representable(pbs_multiplication_plaintext.0) / delta;

        println!("Checking result...");
        assert_eq!(input_message, pbs_multiplication_result);
        println!(
            "Multiplication via PBS result is correct! Expected {input_message}, got {pbs_multiplication_result}"
        );
    }
}
