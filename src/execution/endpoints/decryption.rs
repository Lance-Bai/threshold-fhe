// Re-export the non-wasm module
#[cfg(feature = "non-wasm")]
pub use super::decryption_non_wasm::*;

#[derive(
    Copy,
    Clone,
    Default,
    serde::Serialize,
    serde::Deserialize,
    derive_more::Display,
    Debug,
    clap::ValueEnum,
)]
pub enum DecryptionMode {
    /// nSmall Noise Flooding, this is the default
    #[default]
    NoiseFloodSmall,
    /// nLarge Noise Flooding
    NoiseFloodLarge,
    /// nSmall Bit Decomposition
    BitDecSmall,
    /// nLarge Bit Decomposition
    BitDecLarge,
    /// saniti
    Saniti
}

impl DecryptionMode {
    pub fn as_str_name(&self) -> &'static str {
        match self {
            DecryptionMode::NoiseFloodSmall => "NoiseFloodSmall",
            DecryptionMode::NoiseFloodLarge => "NoiseFloodLarge",
            DecryptionMode::BitDecSmall => "BitDecSmall",
            DecryptionMode::BitDecLarge => "BitDecLarge",
            DecryptionMode::Saniti => "Saniti",
        }
    }
}
