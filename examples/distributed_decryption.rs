//! To use the latest version of threshold-fhe in your project,
//! you first need to add it as a dependency in your `Cargo.toml`:
//!
//! ```
//! threshold_fhe = { git = "https://github.com/zama-ai/threshold-fhe.git" }
//! ```
//!
//! This is an example where we setup a testing runtime that runs 4 parties on the same machine.
//! You can run it with `cargo run -F testing --example distributed_decryption`.
use aes_prng::AesRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use tfhe::{
    boolean::prelude::{DecompositionBaseLog, DecompositionLevelCount, StandardDev},
    core_crypto::{
        prelude::{
            allocate_and_generate_new_lwe_keyswitch_key,
            convert_standard_lwe_bootstrap_key_to_fourier_128, generate_programmable_bootstrap_glwe_lut, keyswitch_lwe_ciphertext,
            par_allocate_and_generate_new_seeded_lwe_bootstrap_key, EncryptionRandomGenerator,
            Fourier128LweBootstrapKey, Gaussian, GlweCiphertextOwned, LweBootstrapKeyOwned,
            LweCiphertext,
        },
        seeders::new_seeder,
    },
    integer::IntegerCiphertext,
    set_server_key,
    shortint::CiphertextModulus,
    FheUint8,
};
use tfhe_csprng::generators::DefaultRandomGenerator;
use threshold_fhe::{
    algebra::{galois_rings::degree_4::ResiduePolyF4Z64, structure_traits::Ring},
    execution::{
        endpoints::decryption::{threshold_decrypt64, DecryptionMode},
        runtime::test_runtime::{generate_fixed_identities, DistributedTestRuntime},
        saniti::pbs::santi_programmable_bootstrap_f128_lwe_ciphertext,
        tfhe_internals::{
            parameters::BC_PARAMS_SAM_SNS,
            test_feature::{gen_key_set, keygen_all_party_shares, KeySet},
            utils::expanded_encrypt,
        },
    },
    networking::NetworkMode,
};

fn main() {
    let num_parties = 8;
    let threshold = 2;
    let mut rng = AesRng::from_entropy();
    let mut boxed_seeder = new_seeder();
    // Get a mutable reference to the seeder as a trait object from the Box returned by new_seeder
    let seeder = boxed_seeder.as_mut();

    // Generate the keys normally, we'll secret share them later.
    let keyset: KeySet = gen_key_set(BC_PARAMS_SAM_SNS, &mut rng);
    set_server_key(keyset.public_keys.server_key.clone());

    let pbs_base_log = DecompositionBaseLog(15);
    let pbs_level = DecompositionLevelCount(2);
    let ks_base_log = DecompositionBaseLog(8);
    let ks_level = DecompositionLevelCount(4);

    let ciphertext_modulus = CiphertextModulus::new_native();
    let glwe_noise_distribution = Gaussian::from_dispersion_parameter(
        StandardDev(0.0000000000000000029403601535432533 * 0.00000000000000029403601535432533),
        0.0,
    );
    let lwe_noise_distribution = Gaussian::from_dispersion_parameter(
        StandardDev(0.00000007069849454709433 * 0.000007069849454709433),
        0.0,
    );
    let message_modulus = 1_u64 << 4;
    let delta = (1_u64 << 63) / message_modulus;

    let lwe_secret_key = keyset.get_raw_lwe_client_key();
    let lwe_dimension = lwe_secret_key.lwe_dimension();
    let glwe_secret_key = keyset.get_raw_glwe_client_key();

    let big_lwe_sk = glwe_secret_key.clone().into_lwe_secret_key();

    // Generate the seeded bootstrapping key to show how to handle entity decompression,
    // we use the parallel variant for performance reason
    let std_bootstrapping_key = par_allocate_and_generate_new_seeded_lwe_bootstrap_key(
        &lwe_secret_key,
        &glwe_secret_key,
        pbs_base_log,
        pbs_level,
        glwe_noise_distribution,
        ciphertext_modulus,
        seeder,
    );

    let mut encryption_generator =
        EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
    let ksk = allocate_and_generate_new_lwe_keyswitch_key(
        &big_lwe_sk,
        &lwe_secret_key,
        ks_base_log,
        ks_level,
        lwe_noise_distribution,
        ciphertext_modulus,
        &mut encryption_generator,
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
    let glwe_dimension = std_bootstrapping_key.glwe_size().to_glwe_dimension();
    let polynomial_size = std_bootstrapping_key.polynomial_size();

    // Use the conversion function (a memory optimized version also exists but is more complicated
    // to use) to convert the standard bootstrapping key to the Fourier domain
    convert_standard_lwe_bootstrap_key_to_fourier_128(&std_bootstrapping_key, &mut fourier_bsk);
    // We don't need the standard bootstrapping key anymore
    drop(std_bootstrapping_key);

    let glwe_secret_key_sns_as_lwe = keyset.sns_secret_key.key.clone();
    let params = keyset.sns_secret_key.params;

    let key_shares = keygen_all_party_shares(
        lwe_secret_key,
        glwe_secret_key,
        glwe_secret_key_sns_as_lwe,
        params,
        &mut rng,
        num_parties,
        threshold,
    )
    .unwrap();

    // Encrypt a message and extract the raw ciphertexts.
    let message = rng.gen::<u8>();

    let ct: FheUint8 = expanded_encrypt(&keyset.public_keys.public_key, message, 8).unwrap();
    let (mut raw_ct, _id, _tag) = ct.into_raw_parts();
    // let decrypted_message: u8 = ct.decrypt(&keyset.client_key);

    // Setup the test runtime.
    // Using Sync because threshold_decrypt64 encompasses both online and offline
    let identities = generate_fixed_identities(num_parties);
    let mut runtime = DistributedTestRuntime::<
        ResiduePolyF4Z64,
        { ResiduePolyF4Z64::EXTENSION_DEGREE },
    >::new(identities.clone(), threshold as u8, NetworkMode::Sync, None);

    let keyset_ck = Arc::new(keyset.public_keys.sns_key.clone().unwrap());
    runtime.conversion_keys = Some(keyset_ck.clone());
    runtime.setup_sks(key_shares);

    // first do santity to keep safe
    // Allocate the LweCiphertext to store the result of the PBS
    let mut pbs_multiplication_ct = LweCiphertext::new(
        0u64,
        big_lwe_sk.lwe_dimension().to_lwe_size(),
        ciphertext_modulus,
    );
    let mut ks_ct = LweCiphertext::new(0u64, lwe_dimension.to_lwe_size(), ciphertext_modulus);

    for (i,e) in raw_ct.blocks_mut().iter_mut().enumerate() {
        let lwe_ciphertext_in: LweCiphertext<Vec<u64>> = e.ct.clone();
        
        keyswitch_lwe_ciphertext(&ksk, &lwe_ciphertext_in, &mut ks_ct);
        let accumulator: GlweCiphertextOwned<u64> = generate_programmable_bootstrap_glwe_lut(
            polynomial_size,
            glwe_dimension.to_glwe_size(),
            message_modulus as usize,
            ciphertext_modulus,
            delta,
            |x: u64| x,
        );

        println!("Computing Santiti PBS for block {i} ...");
        santi_programmable_bootstrap_f128_lwe_ciphertext(
            &ks_ct,
            &mut pbs_multiplication_ct,
            &accumulator,
            &fourier_bsk,
        );


        e.ct = pbs_multiplication_ct.clone();
    }

    let result = threshold_decrypt64(&runtime, &raw_ct, DecryptionMode::Saniti, &ksk).unwrap();

    for (i, v) in result {
        println!("identity: {i}, message: {message} -> result: {v}");
        // assert_eq!(v.0 as u8, message);
    }
    println!("Done")
}
