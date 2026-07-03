//! Trait definitions for cryptographic keypairs.
//!
//! `Keypair` (signing + public identity) and `ExportableKeypair` (also
//! private-key export) are intentionally split, so a caller can receive
//! a handle that can sign but cannot exfiltrate the private material.
//! `SealedKeypair` wraps any `ExportableKeypair` and *drops* the export
//! capability at the type level.

use crate::CryptoError;

/// A type that can produce digital signatures.
pub trait Signer: Send + Sync {
    /// The JWT algorithm identifier (e.g., `"ES256"` or `"ES256K"`).
    fn jwt_alg(&self) -> &str;

    /// Sign a message, returning the raw compact signature bytes (64 bytes: R || S).
    ///
    /// The message is SHA-256 hashed internally before signing.
    fn sign(&self, msg: &[u8]) -> Result<Vec<u8>, CryptoError>;
}

/// A type that can verify digital signatures.
pub trait Verifier: Send + Sync {
    /// Verify a signature against a message.
    ///
    /// The message is SHA-256 hashed internally before verification.
    /// By default, requires low-S normalized compact signatures.
    fn verify(&self, msg: &[u8], sig: &[u8]) -> Result<bool, CryptoError>;

    /// Verify a signature, optionally allowing malleable (high-S or DER) signatures.
    fn verify_malleable(&self, msg: &[u8], sig: &[u8]) -> Result<bool, CryptoError>;
}

/// A full keypair that can sign and provide its DID.
pub trait Keypair: Signer {
    /// Return the `did:key:z...` string for this keypair's public key.
    fn did(&self) -> String;

    /// Return the compressed public key bytes (33 bytes).
    fn public_key_compressed(&self) -> Vec<u8>;
}

/// A keypair that can export its private key.
pub trait ExportableKeypair: Keypair {
    /// Export the raw private key bytes.
    fn export_private_key(&self) -> Vec<u8>;
}

/// A keypair wrapper that can sign and identify itself but *cannot*
/// export its private key at the type level. Pass the inner keypair in
/// at construction time, then hand out the sealed wrapper to any code
/// that should not be able to exfiltrate the secret.
///
/// The sealed wrapper owns the inner keypair, so the original value is
/// consumed. If the caller holds onto the original they can still
/// export it — the sealing only applies to what's handed out.
///
/// Signing and identity still work:
///
/// ```
/// use proto_blue_crypto::{Keypair, P256Keypair, SealedKeypair, Signer};
///
/// let kp = P256Keypair::generate();
/// let sealed = SealedKeypair::seal(kp);
///
/// let sig = sealed.sign(b"hello").unwrap();
/// assert_eq!(sig.len(), 64);
/// assert!(sealed.did().starts_with("did:key:z"));
/// ```
///
/// But `export_private_key` does not compile against a sealed wrapper:
///
/// ```compile_fail
/// use proto_blue_crypto::{ExportableKeypair, P256Keypair, SealedKeypair};
/// let kp = P256Keypair::generate();
/// let sealed = SealedKeypair::seal(kp);
/// // Must not compile: SealedKeypair does not implement ExportableKeypair.
/// let _ = sealed.export_private_key();
/// ```
pub struct SealedKeypair<K: Keypair> {
    inner: K,
}

impl<K: Keypair> SealedKeypair<K> {
    /// Wrap a keypair to drop its export capability at the type level.
    /// Accepts any `Keypair` — including non-exportable ones — so you
    /// can re-seal something that's already sealed (no-op semantically).
    pub const fn seal(inner: K) -> Self {
        Self { inner }
    }
}

impl<K: Keypair> Signer for SealedKeypair<K> {
    fn jwt_alg(&self) -> &str {
        self.inner.jwt_alg()
    }
    fn sign(&self, msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        self.inner.sign(msg)
    }
}

impl<K: Keypair> Keypair for SealedKeypair<K> {
    fn did(&self) -> String {
        self.inner.did()
    }
    fn public_key_compressed(&self) -> Vec<u8> {
        self.inner.public_key_compressed()
    }
}

// Deliberately no `impl ExportableKeypair for SealedKeypair<K>` — that's
// the whole point. The test suite includes a compile-time check that
// verifies this.

/// A type that can produce digital signatures asynchronously.
///
/// Parallel to [`Signer`], but the signing operation returns a `Future`.
/// Intended for consumers whose signing transport is inherently async —
/// holder-mediated signing over a channel (e.g. did:web account holders
/// signing from a browser or native app), hardware devices behind async
/// I/O, or remote signing services.
///
/// Every existing [`Signer`] is trivially an `AsyncSigner` via a blanket
/// impl below — code holding a local `P256Keypair` or `K256Keypair` can be
/// used through the `AsyncSigner` interface with zero changes.
///
/// # Contract
///
/// Implementers must preserve the same semantics as [`Signer::sign`]:
/// the message is SHA-256 hashed internally before signing, and the
/// returned bytes are the 64-byte compact ECDSA signature (R || S,
/// low-S normalized). The `jwt_alg` method must return the JWT
/// algorithm identifier corresponding to the underlying curve (e.g.
/// `"ES256"` for P-256, `"ES256K"` for secp256k1).
///
/// # Example
///
/// ```ignore
/// use proto_blue_crypto::{AsyncSigner, P256Keypair};
///
/// let kp = P256Keypair::generate();
/// // Any Signer is trivially an AsyncSigner via blanket impl:
/// let sig = <P256Keypair as AsyncSigner>::sign_async(&kp, b"hello").await.unwrap();
/// assert_eq!(sig.len(), 64);
/// ```
#[cfg(feature = "async-signer")]
#[async_trait::async_trait]
pub trait AsyncSigner: Send + Sync {
    /// The JWT algorithm identifier (e.g., `"ES256"` or `"ES256K"`).
    fn jwt_alg(&self) -> &str;

    /// Sign a message asynchronously, returning the raw compact signature
    /// bytes (64 bytes: R || S).
    ///
    /// The message is SHA-256 hashed internally before signing.
    async fn sign_async(&self, msg: &[u8]) -> Result<Vec<u8>, CryptoError>;
}

/// Blanket impl: every synchronous [`Signer`] is trivially an
/// [`AsyncSigner`]. Enables gradual adoption — existing code that takes
/// a sync `Signer` can be lifted to `AsyncSigner` without changing the
/// underlying keypair type.
#[cfg(feature = "async-signer")]
#[async_trait::async_trait]
impl<S: Signer> AsyncSigner for S {
    fn jwt_alg(&self) -> &str {
        Signer::jwt_alg(self)
    }

    async fn sign_async(&self, msg: &[u8]) -> Result<Vec<u8>, CryptoError> {
        Signer::sign(self, msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::P256Keypair;

    #[test]
    fn sealed_keypair_can_sign_and_identify() {
        let kp = P256Keypair::generate();
        let did_before = kp.did();
        let compressed_before = kp.public_key_compressed();

        let sealed = SealedKeypair::seal(kp);
        assert_eq!(Signer::jwt_alg(&sealed), "ES256");
        assert_eq!(sealed.did(), did_before);
        assert_eq!(sealed.public_key_compressed(), compressed_before);

        // Signing still works.
        let sig = sealed.sign(b"hello sealed world").unwrap();
        assert_eq!(sig.len(), 64);
    }

    /// Sealing an already-sealed keypair is a no-op — you can re-wrap
    /// indefinitely without losing signing capability or changing the DID.
    #[test]
    fn reseal_is_a_noop() {
        let kp = P256Keypair::generate();
        let did = kp.did();
        let sealed = SealedKeypair::seal(kp);
        let resealed = SealedKeypair::seal(sealed);
        assert_eq!(resealed.did(), did);
        assert_eq!(Signer::jwt_alg(&resealed), "ES256");
    }

    #[cfg(feature = "async-signer")]
    #[tokio::test]
    async fn blanket_async_signer_impl_p256() {
        let kp = P256Keypair::generate();
        let msg = b"async signer test";
        let sync_sig = Signer::sign(&kp, msg).unwrap();
        let async_sig = <P256Keypair as AsyncSigner>::sign_async(&kp, msg)
            .await
            .unwrap();
        // ECDSA is nondeterministic; can't compare signatures directly.
        // Both should be 64 bytes compact.
        assert_eq!(sync_sig.len(), 64);
        assert_eq!(async_sig.len(), 64);
        // Both jwt_alg calls should return the same value.
        assert_eq!(
            <P256Keypair as Signer>::jwt_alg(&kp),
            <P256Keypair as AsyncSigner>::jwt_alg(&kp)
        );
    }

    #[cfg(feature = "async-signer")]
    #[tokio::test]
    async fn blanket_async_signer_impl_sealed() {
        // SealedKeypair<P256Keypair> also gets the blanket impl.
        let kp = P256Keypair::generate();
        let sealed = SealedKeypair::seal(kp);
        let sig = <SealedKeypair<P256Keypair> as AsyncSigner>::sign_async(&sealed, b"sealed test")
            .await
            .unwrap();
        assert_eq!(sig.len(), 64);
    }
}
