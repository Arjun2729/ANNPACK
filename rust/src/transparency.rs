//! Offline verification of Sigsum transparency-log proofs for a release-state
//! statement. Requires the `transparency-log` feature.
//!
//! # What this binds
//!
//! A [`ChannelState`](crate::release::ChannelState) statement is already
//! signed by a release-state key and verified as such
//! (`verify_channel_state`). That proves who asserted the statement. It does
//! not prove the assertion was made publicly, and only once -- a compromised
//! or dishonest publisher could sign two different statements at the same
//! sequence number and show each to a different verifier, and neither
//! verifier would see the other's copy. A Sigsum proof closes exactly that
//! gap: it shows the statement's own digest
//! ([`statement_digest_bytes`](crate::release::statement_digest_bytes)) was
//! logged in a public, append-only Merkle tree, at a tree state a configured
//! quorum of witnesses has cosigned. Two conflicting statements at the same
//! sequence number would each need their own log entry, and comparing
//! entries across independent observers is what a Step 9 equivocation
//! monitor does -- this module only checks one proof against one statement,
//! not history.
//!
//! # What this does NOT do
//!
//! ANNPack does not operate a transparency log, submit statements to one, or
//! run a witness. This module verifies a proof the publisher already
//! obtained from a real Sigsum log. It does not detect equivocation itself --
//! that requires comparing multiple independently observed log entries,
//! which is a monitoring concern, not a per-statement verification concern.
//! It also does not establish that a log entry is *recent*: a genuine,
//! fully-witnessed proof for an old, superseded statement verifies exactly as
//! well as one for the current statement. Fresh inclusion of an old release
//! statement does not prove that statement is the latest release --
//! `TransparencyEvidence::Verified` answers "was this publicly logged and
//! witnessed," never "is this current." Currency is `release::Currency`'s
//! question, answered independently.
//!
//! # Trust configuration
//!
//! The operator supplies a Sigsum policy file (the real `sigsum-go` policy
//! syntax: `log`/`witness`/`group`/`quorum` lines,
//! <https://git.glasklar.is/sigsum/core/sigsum-go/-/blob/main/doc/policy.md>)
//! naming which log and witness keys are trusted and what quorum is
//! required. This module never fetches or updates that file. Witnesses
//! strengthen observation consistency; they do not replace a trusted clock
//! or durable monotonic state -- `authorized_current_witnessed` requires the
//! ordinary `authorized_current` checks in addition to transparency evidence,
//! never instead of them (`policy::evaluate_policy`).

#[cfg(feature = "transparency-log")]
use crate::error::{AdyarError, Result};
#[cfg(feature = "transparency-log")]
use crate::policy::TransparencyEvidence;
#[cfg(feature = "transparency-log")]
use crate::release::ChannelState;

#[cfg(feature = "transparency-log")]
pub const MAX_PROOF_BYTES: usize = 64 * 1024;
#[cfg(feature = "transparency-log")]
pub const MAX_POLICY_BYTES: usize = 64 * 1024;

/// Parse the operator-supplied Sigsum trust policy: which log and witness
/// keys are trusted, and what witness quorum is required. Never fetched or
/// refreshed by this function; updating it is a deliberate operational act,
/// exactly as replacing GitHub's Sigstore trusted-root snapshot is
/// (`attestation.rs`). Private: the `sigsum` crate's types never cross this
/// module's public boundary, matching how `attestation.rs` never exposes
/// `sigstore-verify`'s types either -- callers get ANNPack's own
/// [`TransparencyReport`], not a third-party library's shapes.
/// Callers must check `text.len() <= MAX_POLICY_BYTES` first --
/// `verify_transparency`, this function's only caller, does so before either
/// input is parsed, so a caller who oversized only one of the two inputs
/// gets an error naming that one specifically.
#[cfg(feature = "transparency-log")]
fn parse_transparency_policy(text: &str) -> Result<sigsum::Policy> {
    sigsum::Policy::parse(text).map_err(|error| {
        AdyarError::InvalidFormat(format!("malformed transparency policy: {error}"))
    })
}

/// Parse a Sigsum proof bundle: the native ASCII checkpoint/inclusion-proof
/// document a log returns when a signature is submitted to it. Callers must
/// check `text.len() <= MAX_PROOF_BYTES` first; see
/// [`parse_transparency_policy`].
#[cfg(feature = "transparency-log")]
fn parse_transparency_proof(text: &str) -> Result<sigsum::SigsumSignature> {
    sigsum::SigsumSignature::from_ascii(text).map_err(|error| {
        AdyarError::InvalidFormat(format!("malformed transparency proof: {error}"))
    })
}

/// Parse a hex-encoded Ed25519 public key into the type `sigsum::verify`
/// expects. Kept separate from ANNPack's own key parsing (`trust::`) because
/// a Sigsum leaf signer identity is a distinct trust domain from any ANNPack
/// role, even when an operator chooses to reuse the same physical key for
/// both purposes.
#[cfg(feature = "transparency-log")]
fn parse_signer_key(hex_key: &str) -> Result<sigsum::PublicKey> {
    let bytes = hex::decode(hex_key)
        .map_err(|_| AdyarError::InvalidInput("signer key is not valid hex".into()))?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| AdyarError::InvalidInput("signer key must be 32 bytes".into()))?;
    Ok(bytes.into())
}

#[cfg(feature = "transparency-log")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransparencyReport {
    pub evidence: TransparencyEvidence,
    /// The statement digest (hex) this proof was checked against, so a
    /// caller can confirm the proof was checked against the statement they
    /// expected, not silently substituted for another.
    pub statement_digest: String,
    pub issues: Vec<String>,
}

/// Verify that `proof_text` shows `statement`'s digest logged and witnessed,
/// signed by one of `trusted_signer_hex_keys`, per `policy_text`.
///
/// `trusted_signer_hex_keys` should be the release-state role's authorized
/// keys from the caller's trust root (hex-encoded Ed25519 public keys,
/// `trust::role_public_keys`) -- the same keys already required to have
/// produced the statement's own ANNPack signature. A Sigsum proof from an
/// untrusted key proves nothing about this statement's authority; the two
/// signatures are checked independently and never merged into one identity,
/// even when an operator's key material happens to be reused across both.
///
/// Never returns `Err` for a well-formed but unsatisfying proof -- a missing
/// witness quorum, an untrusted log, or a mismatched signer is exactly the
/// case `TransparencyEvidence::Insufficient` exists to report without
/// treating the caller's request itself as malformed. `Err` is reserved for
/// what could not even be parsed and evaluated: a malformed proof, a
/// malformed policy, or a malformed signer key.
#[cfg(feature = "transparency-log")]
pub fn verify_transparency(
    statement: &ChannelState,
    proof_text: &str,
    trusted_signer_hex_keys: &[String],
    policy_text: &str,
) -> Result<TransparencyReport> {
    // Both size limits are checked before either input is parsed, so a
    // caller who oversized only one of the two gets an error naming that
    // one, not an unrelated parse failure from whichever input happens to be
    // checked first.
    if proof_text.len() > MAX_PROOF_BYTES {
        return Err(AdyarError::InvalidFormat(
            "transparency proof exceeds size limit".into(),
        ));
    }
    if policy_text.len() > MAX_POLICY_BYTES {
        return Err(AdyarError::InvalidFormat(
            "transparency policy exceeds size limit".into(),
        ));
    }

    let proof = parse_transparency_proof(proof_text)?;
    let policy = parse_transparency_policy(policy_text)?;
    let trusted_signers = trusted_signer_hex_keys
        .iter()
        .map(|hex_key| parse_signer_key(hex_key))
        .collect::<Result<Vec<_>>>()?;

    let digest_bytes = crate::release::statement_digest_bytes(statement)?;
    let statement_digest = hex::encode(digest_bytes);
    let message = sigsum::Hash::new(digest_bytes);

    match sigsum::verify(&message, &proof, &trusted_signers, &policy) {
        Ok(()) => Ok(TransparencyReport {
            evidence: TransparencyEvidence::Verified,
            statement_digest,
            issues: Vec::new(),
        }),
        Err(error) => Ok(TransparencyReport {
            evidence: TransparencyEvidence::Insufficient,
            statement_digest,
            issues: vec![error.to_string()],
        }),
    }
}

#[cfg(all(test, feature = "transparency-log"))]
mod tests {
    use super::*;
    use crate::release::{CHANNEL_STATE_SCHEMA_V1, ChannelState, CurrentRelease};
    use ed25519_dalek::{Signer, SigningKey};
    use sha2::{Digest, Sha256};

    fn statement() -> ChannelState {
        ChannelState {
            schema: CHANNEL_STATE_SCHEMA_V1.into(),
            publisher: "example.com".into(),
            corpus: "support-manual".into(),
            channel: "production".into(),
            sequence: 4,
            issued_at: "2026-08-06T00:00:00Z".into(),
            valid_until: "2026-08-06T01:00:00Z".into(),
            current: CurrentRelease {
                version: "4.3.0".into(),
                artifact_root: "aa".repeat(32),
            },
            superseded: Vec::new(),
            revoked: Vec::new(),
            signatures: Vec::new(),
        }
    }

    fn sha256(data: &[u8]) -> [u8; 32] {
        Sha256::digest(data).into()
    }

    fn base64_standard(bytes: &[u8]) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn ed25519_keypair(seed: u8) -> (SigningKey, [u8; 32]) {
        let key = SigningKey::from_bytes(&[seed; 32]);
        let public = key.verifying_key().to_bytes();
        (key, public)
    }

    /// A genuine, self-consistent, single-leaf Sigsum proof: real Ed25519
    /// signatures over the exact byte sequences `sigsum-rs` (pinned `=0.3.0`)
    /// reconstructs during verification, and a real SHA-256 Merkle tree of
    /// size 1 (root_hash == leaf_hash, empty inclusion path -- RFC 9162 §2.1.1
    /// with zero intermediate nodes). Every convention here -- the
    /// `"sigsum.org/v1/tree-leaf\0"` leaf-signing prefix, the
    /// `"sigsum.org/v1/tree/{keyhash}\n{size}\n{base64(root)}\n"` tree-head
    /// message, the `"cosignature/v1\ntime {t}\n"` cosignature wrapper, and
    /// the `0x00`/`0x01` Merkle domain-separation prefixes -- is taken
    /// directly from `verify.rs` and `merkle/mod.rs` in the sigsum-rs source,
    /// not approximated, and is validated by the round-trip in
    /// `a_genuine_proof_verifies` below.
    struct Fixture {
        proof_ascii: String,
        signer_public_key_hex: String,
        policy_text: String,
    }

    fn build_fixture(digest_bytes: [u8; 32], witness_timestamp: u64) -> Fixture {
        let (signer_key, signer_public) = ed25519_keypair(1);
        let (log_key, log_public) = ed25519_keypair(2);
        let (witness_key, witness_public) = ed25519_keypair(3);

        // `verify_transparency` passes `Hash::new(digest_bytes)` as `message`;
        // `sigsum::verify` internally re-hashes it (`Hash::new(message)`), so
        // the value actually signed and logged is SHA256(SHA256(digest)).
        let message = sha256(&digest_bytes);
        let checksum = sha256(&message);

        let leaf_keyhash = sha256(&signer_public);
        let leaf_signed_bytes = [b"sigsum.org/v1/tree-leaf\x00".as_slice(), &checksum].concat();
        let leaf_signature = signer_key.sign(&leaf_signed_bytes).to_bytes();

        // One-leaf tree: root_hash == leaf_hash, empty inclusion path.
        let leaf_bytes = [checksum.as_slice(), &leaf_signature, &leaf_keyhash].concat();
        let leaf_hash = sha256(&[[0x00u8].as_slice(), &leaf_bytes].concat());
        let root_hash = leaf_hash;
        let size: u64 = 1;

        let log_keyhash = sha256(&log_public);
        let tree_head_message = format!(
            "sigsum.org/v1/tree/{}\n{}\n{}\n",
            hex::encode(log_keyhash),
            size,
            base64_standard(&root_hash),
        );
        let sth_signature = log_key.sign(tree_head_message.as_bytes()).to_bytes();

        let witness_keyhash = sha256(&witness_public);
        let cosigned_message =
            format!("cosignature/v1\ntime {witness_timestamp}\n{tree_head_message}");
        let cosignature = witness_key.sign(cosigned_message.as_bytes()).to_bytes();

        let proof_ascii = format!(
            "version=2\nlog={}\nleaf={} {}\n\nsize={}\nroot_hash={}\nsignature={}\ncosignature={} {} {}\n\nleaf_index=0\n",
            hex::encode(log_keyhash),
            hex::encode(leaf_keyhash),
            hex::encode(leaf_signature),
            size,
            hex::encode(root_hash),
            hex::encode(sth_signature),
            hex::encode(witness_keyhash),
            witness_timestamp,
            hex::encode(cosignature),
        );

        let policy_text = format!(
            "log {}\nwitness w {}\nquorum w\n",
            hex::encode(log_public),
            hex::encode(witness_public),
        );

        Fixture {
            proof_ascii,
            signer_public_key_hex: hex::encode(signer_public),
            policy_text,
        }
    }

    #[test]
    fn a_genuine_proof_verifies() {
        let statement = statement();
        let digest = crate::release::statement_digest_bytes(&statement).unwrap();
        let fixture = build_fixture(digest, 1_700_000_000);

        let report = verify_transparency(
            &statement,
            &fixture.proof_ascii,
            std::slice::from_ref(&fixture.signer_public_key_hex),
            &fixture.policy_text,
        )
        .unwrap();

        assert_eq!(
            report.evidence,
            TransparencyEvidence::Verified,
            "{:?}",
            report.issues
        );
        assert_eq!(report.statement_digest, hex::encode(digest));
        assert!(report.issues.is_empty());
    }

    #[test]
    fn a_proof_for_a_different_statement_is_insufficient() {
        let statement = statement();
        let digest = crate::release::statement_digest_bytes(&statement).unwrap();
        // Build the fixture against a different digest than the statement's own.
        let fixture = build_fixture(sha256(b"a different statement entirely"), 1_700_000_000);

        let report = verify_transparency(
            &statement,
            &fixture.proof_ascii,
            std::slice::from_ref(&fixture.signer_public_key_hex),
            &fixture.policy_text,
        )
        .unwrap();

        assert_eq!(report.evidence, TransparencyEvidence::Insufficient);
        assert_eq!(report.statement_digest, hex::encode(digest));
        assert!(!report.issues.is_empty());
    }

    #[test]
    fn an_untrusted_signer_is_insufficient() {
        let statement = statement();
        let digest = crate::release::statement_digest_bytes(&statement).unwrap();
        let fixture = build_fixture(digest, 1_700_000_000);
        let (_, other_public) = ed25519_keypair(9);

        // Caller's trusted-signer list does not include the key that actually
        // produced the leaf signature: a genuinely valid, well-witnessed
        // proof from a key nobody authorized must still be reported as
        // insufficient, exactly as an untrusted builder key is for local
        // Ed25519 provenance and an untrusted repository is for GitHub
        // attestations.
        let report = verify_transparency(
            &statement,
            &fixture.proof_ascii,
            &[hex::encode(other_public)],
            &fixture.policy_text,
        )
        .unwrap();

        assert_eq!(report.evidence, TransparencyEvidence::Insufficient);
    }

    #[test]
    fn an_untrusted_log_is_insufficient() {
        let statement = statement();
        let digest = crate::release::statement_digest_bytes(&statement).unwrap();
        let fixture = build_fixture(digest, 1_700_000_000);

        // A policy that trusts a witness but never lists the log that
        // actually issued this tree head.
        let (_, other_log_public) = ed25519_keypair(99);
        let policy_text = fixture.policy_text.replacen(
            &hex::encode(ed25519_keypair(2).1),
            &hex::encode(other_log_public),
            1,
        );

        let report = verify_transparency(
            &statement,
            &fixture.proof_ascii,
            std::slice::from_ref(&fixture.signer_public_key_hex),
            &policy_text,
        )
        .unwrap();

        assert_eq!(report.evidence, TransparencyEvidence::Insufficient);
    }

    #[test]
    fn a_quorum_of_zero_trusted_witnesses_is_insufficient() {
        let statement = statement();
        let digest = crate::release::statement_digest_bytes(&statement).unwrap();
        let fixture = build_fixture(digest, 1_700_000_000);

        // A policy that trusts the log but requires a witness that never
        // cosigned this checkpoint: the quorum the policy demands is not met,
        // even though the log's own tree-head signature is genuine.
        let (_, log_public) = ed25519_keypair(2);
        let (_, unrelated_witness_public) = ed25519_keypair(42);
        let policy_text = format!(
            "log {}\nwitness w {}\nquorum w\n",
            hex::encode(log_public),
            hex::encode(unrelated_witness_public),
        );

        let report = verify_transparency(
            &statement,
            &fixture.proof_ascii,
            std::slice::from_ref(&fixture.signer_public_key_hex),
            &policy_text,
        )
        .unwrap();

        assert_eq!(report.evidence, TransparencyEvidence::Insufficient);
    }

    #[test]
    fn a_tampered_root_hash_is_insufficient() {
        let statement = statement();
        let digest = crate::release::statement_digest_bytes(&statement).unwrap();
        let fixture = build_fixture(digest, 1_700_000_000);

        // Flip the root_hash's last hex nibble. The inclusion proof
        // (root_hash == leaf_hash for a one-leaf tree) and the log's own
        // tree-head signature both cover the true root_hash, so any change
        // here must break at least one check.
        let line_start = fixture.proof_ascii.find("root_hash=").unwrap();
        let value_start = line_start + "root_hash=".len();
        let mut bytes = fixture.proof_ascii.into_bytes();
        let last_nibble = value_start + 63;
        bytes[last_nibble] = if bytes[last_nibble] == b'0' {
            b'1'
        } else {
            b'0'
        };
        let tampered = String::from_utf8(bytes).unwrap();

        let report = verify_transparency(
            &statement,
            &tampered,
            std::slice::from_ref(&fixture.signer_public_key_hex),
            &fixture.policy_text,
        )
        .unwrap();

        assert_eq!(report.evidence, TransparencyEvidence::Insufficient);
    }

    #[test]
    fn oversized_proof_is_rejected_before_parsing() {
        let statement = statement();
        let oversized = "a".repeat(MAX_PROOF_BYTES + 1);
        let error = verify_transparency(&statement, &oversized, &[], "quorum none\n").unwrap_err();
        assert!(error.to_string().contains("exceeds size limit"));
    }

    #[test]
    fn oversized_policy_is_rejected_before_parsing() {
        let statement = statement();
        let oversized = "a".repeat(MAX_POLICY_BYTES + 1);
        let error = verify_transparency(&statement, "", &[], &oversized).unwrap_err();
        assert!(error.to_string().contains("exceeds size limit"));
    }

    #[test]
    fn malformed_signer_key_is_rejected() {
        let statement = statement();
        let digest = crate::release::statement_digest_bytes(&statement).unwrap();
        let fixture = build_fixture(digest, 1_700_000_000);
        let error = verify_transparency(
            &statement,
            &fixture.proof_ascii,
            &["not hex".into()],
            &fixture.policy_text,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not valid hex"));
    }
}
