# Threshold FHE Example based on Zama's Code

This project demonstrates a **Threshold Fully Homomorphic Encryption (FHE)** system, built on top of Zama's `concrete` or `tfhe` libraries. It explores distributed decryption where the secret key is shared among multiple parties.

## 🔐 Security

Security is achieved using **santiti** — a threshold cryptographic approach that ensures no single party can decrypt the ciphertext alone. Each party holds a share of the secret key and must cooperate in the decryption process.

## 🚀 Run Instructions

To execute the distributed threshold decryption example, use the following command:

```bash
cargo run --release -F testing --example distributed_decryption

