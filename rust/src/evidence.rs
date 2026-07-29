//! Standalone evidence receipts: offline, pack-free proof that a cited passage
//! existed unmodified in a known immutable artifact.
//!
//! A Core evidence envelope names `(pack_root, passage_id, passage_hash)`. That
//! is sufficient only for a verifier that already holds the pack. A **receipt**
//! closes the chain without the pack and without the network:
//!
//! ```text
//! passage JSON
//!   -> BLAKE3("ANNPACK3-PASSAGE-EVIDENCE\0" || json)      = leaf
//!   -> Merkle path over passage leaves in ordinal order   = passage_merkle_root
//!   -> equals manifest.passage_merkle_root                (manifest bytes)
//!   -> BLAKE3(manifest bytes) equals its directory entry hash
//!   -> BLAKE3("ANNPACK3-CONTENT-ROOT\0" || entries)       = pack_root
//!   -> Ed25519 signature over the pack root               (optional)
//! ```
//!
//! Every step is recomputed from bytes carried in the receipt, so a verifier
//! trusts only the publisher's key — never this implementation and never a
//! hosted service.
//!
//! `passage_merkle_root` is additionally a **logical content root**: it commits
//! to canonicalized passage records, independent of DEFLATE settings, block
//! packing, and section layout. Two builders that agree on ingestion and
//! chunking produce the same `passage_merkle_root` even when their artifact
//! roots differ. See ADR-0003.

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};
use crate::format::{DIRECTORY_ENTRY_SIZE, HEADER_SIZE};

/// Domain separator for interior Merkle nodes. Leaves use the Core passage
/// evidence separator, so a leaf can never be reinterpreted as an interior node.
const NODE_CONTEXT: &[u8] = b"ANNPACK3-EVIDENCE-NODE\0";
/// Domain separator for Core passage evidence hashes (the Merkle leaves).
pub const PASSAGE_EVIDENCE_CONTEXT: &[u8] = b"ANNPACK3-PASSAGE-EVIDENCE\0";
/// Domain separator for the Ed25519 signature message.
const SIGNATURE_CONTEXT: &[u8] = b"ANNPACK3-SIGNATURE\0";
/// Domain separator for the artifact content root.
const CONTENT_ROOT_CONTEXT: &[u8] = b"ANNPACK3-CONTENT-ROOT\0";

/// The Core passage evidence hash: BLAKE3 over the exact stored passage record.
pub fn passage_evidence_hash(passage_json: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PASSAGE_EVIDENCE_CONTEXT);
    hasher.update(passage_json);
    *hasher.finalize().as_bytes()
}

fn node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(NODE_CONTEXT);
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

/// Merkle root over passage evidence hashes in deterministic corpus order.
///
/// Binary tree, pairwise from the left. A level with an odd node count promotes
/// the final node unchanged to the next level (it is not duplicated, which would
/// admit the classic duplicate-leaf ambiguity). A single leaf is its own root.
/// An empty leaf set has no root; `build` rejects empty corpora.
pub fn merkle_root(leaves: &[[u8; 32]]) -> Option<[u8; 32]> {
    if leaves.is_empty() {
        return None;
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut index = 0;
        while index + 1 < level.len() {
            next.push(node(&level[index], &level[index + 1]));
            index += 2;
        }
        if index < level.len() {
            next.push(level[index]);
        }
        level = next;
    }
    Some(level[0])
}

/// One sibling on the path from a leaf to the Merkle root.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProofStep {
    /// Lowercase hex of the sibling hash.
    pub sibling: String,
    /// True when the sibling is the left operand (this subtree is the right one).
    pub sibling_is_left: bool,
}

/// Inclusion path for `index` within `leaves`. Promoted (odd) nodes contribute
/// no step, exactly mirroring `merkle_root`.
pub fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Result<Vec<ProofStep>> {
    if index >= leaves.len() {
        return Err(AnnpackError::InvalidInput(format!(
            "passage ordinal {index} exceeds the {} leaves in the tree",
            leaves.len()
        )));
    }
    let mut steps = Vec::new();
    let mut level = leaves.to_vec();
    let mut position = index;
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut cursor = 0;
        while cursor + 1 < level.len() {
            if position == cursor {
                steps.push(ProofStep {
                    sibling: hex::encode(level[cursor + 1]),
                    sibling_is_left: false,
                });
            } else if position == cursor + 1 {
                steps.push(ProofStep {
                    sibling: hex::encode(level[cursor]),
                    sibling_is_left: true,
                });
            }
            next.push(node(&level[cursor], &level[cursor + 1]));
            cursor += 2;
        }
        if cursor < level.len() {
            next.push(level[cursor]);
        }
        position /= 2;
        level = next;
    }
    Ok(steps)
}

/// Replay an inclusion path, returning the reconstructed root.
pub fn apply_proof(leaf: &[u8; 32], proof: &[ProofStep]) -> Result<[u8; 32]> {
    let mut current = *leaf;
    for step in proof {
        let sibling = decode_hash(&step.sibling, "proof sibling")?;
        current = if step.sibling_is_left {
            node(&sibling, &current)
        } else {
            node(&current, &sibling)
        };
    }
    Ok(current)
}

/// A self-contained, offline-verifiable citation receipt.
///
/// Carries every byte a verifier needs: the passage record, its inclusion path,
/// the manifest that commits the Merkle root, and the directory that commits the
/// manifest and produces the artifact root. Nothing is fetched and no service is
/// trusted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceReceipt {
    pub schema: String,
    pub pack: String,
    /// Artifact root: BLAKE3 over the non-signature directory entries.
    pub pack_root: String,
    /// Logical content root: Merkle root over passage evidence hashes.
    pub passage_merkle_root: String,
    pub source_revision: Option<String>,
    pub passage_id: String,
    pub passage_hash: String,
    pub passage_ordinal: u32,
    pub canonical_url: Option<String>,
    /// The exact stored passage record bytes, base64. Hashing these must
    /// reproduce `passage_hash`.
    pub passage_record_b64: String,
    /// Inclusion path from the passage leaf to `passage_merkle_root`.
    pub inclusion_proof: Vec<ProofStep>,
    /// Manifest section bytes, base64. Committed by the directory.
    pub manifest_bytes_b64: String,
    /// Full section directory, base64. Its non-signature entries produce
    /// `pack_root`.
    pub directory_b64: String,
    /// Section ID of the manifest, so the verifier can locate its entry.
    pub manifest_section_id: u32,
    /// Section ID of the Documents section. Lets the verifier locate its
    /// directory entry to authenticate `canonical_url`. Present in
    /// `annpack-receipt-v2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents_section_id: Option<u32>,
    /// The Documents section's exact stored (as-compressed) bytes, base64.
    /// Hashing these reproduces the section's directory-entry hash — which
    /// `pack_root` already commits — and the document whose ID matches the
    /// passage record then reproduces `canonical_url`. Present in
    /// `annpack-receipt-v2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents_bytes_b64: Option<String>,
    /// Optional publisher signature over the artifact root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<ReceiptSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptSignature {
    pub algorithm: String,
    pub public_key: String,
    pub signature: String,
    pub key_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<String>,
}

/// The outcome of verifying a receipt. Each claim is reported separately so a
/// consumer never conflates integrity, authenticity, and identity trust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptVerification {
    /// The passage record hashes to `passage_hash`.
    pub passage_hash_matches: bool,
    /// The inclusion path reproduces `passage_merkle_root`.
    pub inclusion_proof_valid: bool,
    /// The manifest commits that Merkle root.
    pub manifest_commits_merkle_root: bool,
    /// The manifest bytes hash to the manifest section's directory entry.
    pub manifest_matches_directory: bool,
    /// The directory reproduces `pack_root`.
    pub directory_matches_pack_root: bool,
    /// The receipt's `passage_id` and `passage_ordinal` match the authenticated
    /// passage record, so its labels cannot misidentify the proven passage.
    pub passage_metadata_matches: bool,
    /// The receipt's `source_revision` matches the authenticated manifest.
    pub source_revision_matches: bool,
    /// The receipt's `pack` (name@version) matches the authenticated manifest.
    pub pack_matches: bool,
    /// The receipt's `canonical_url` is reproduced from the authenticated
    /// Documents section, or the receipt makes no URL claim. A URL claim with no
    /// Documents section to back it fails, which also blocks a downgrade that
    /// simply drops the section.
    pub canonical_url_matches: bool,
    /// A signature over `pack_root` verified. `false` when unsigned.
    pub signature_valid: bool,
    /// Always false unless the caller supplied a trusted key that matched.
    pub identity_trusted: bool,
    /// True only when every integrity claim above holds.
    pub verified: bool,
    pub issues: Vec<String>,
}

fn decode_hash(value: &str, label: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(value)
        .map_err(|_| AnnpackError::InvalidFormat(format!("{label} is not valid hex")))?;
    bytes
        .try_into()
        .map_err(|_| AnnpackError::InvalidFormat(format!("{label} is not 32 bytes")))
}

fn b64_decode(value: &str, label: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|_| AnnpackError::InvalidFormat(format!("{label} is not valid base64")))
}

pub fn b64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Recompute the artifact root from raw directory bytes, skipping signature
/// entries exactly as the writer does.
fn root_from_directory(directory: &[u8]) -> Result<[u8; 32]> {
    if !directory.len().is_multiple_of(DIRECTORY_ENTRY_SIZE) {
        return Err(AnnpackError::InvalidFormat(
            "receipt directory is misaligned".into(),
        ));
    }
    const SIGNATURE_TYPE: u16 = 10;
    let mut hasher = blake3::Hasher::new();
    hasher.update(CONTENT_ROOT_CONTEXT);
    for entry in directory.chunks_exact(DIRECTORY_ENTRY_SIZE) {
        let section_type = u16::from_le_bytes([entry[4], entry[5]]);
        if section_type != SIGNATURE_TYPE {
            hasher.update(entry);
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

/// Locate a directory entry by section ID, returning
/// `(stored_hash, type, logical_length)`.
fn directory_entry(directory: &[u8], section_id: u32) -> Option<([u8; 32], u16, u64)> {
    for entry in directory.chunks_exact(DIRECTORY_ENTRY_SIZE) {
        if u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]) == section_id {
            let mut hash = [0_u8; 32];
            hash.copy_from_slice(&entry[44..76]);
            let logical_length =
                u64::from_le_bytes(entry[28..36].try_into().expect("slice length"));
            return Some((
                hash,
                u16::from_le_bytes([entry[4], entry[5]]),
                logical_length,
            ));
        }
    }
    None
}

/// Verify a receipt offline. Needs no pack, no network, and no trust in the
/// implementation that produced it.
///
/// `trusted_public_key` is an optional externally-supplied publisher key. A
/// cryptographically valid signature never by itself sets `identity_trusted`.
pub fn verify_receipt(
    receipt: &EvidenceReceipt,
    trusted_public_key: Option<&str>,
) -> Result<ReceiptVerification> {
    let mut issues = Vec::new();

    let record = b64_decode(&receipt.passage_record_b64, "passage record")?;
    let computed_leaf = passage_evidence_hash(&record);
    let declared_leaf = decode_hash(&receipt.passage_hash, "passage_hash")?;
    let passage_hash_matches = computed_leaf == declared_leaf;
    if !passage_hash_matches {
        issues.push("passage record does not hash to the declared passage_hash".into());
    }

    let declared_merkle = decode_hash(&receipt.passage_merkle_root, "passage_merkle_root")?;
    let replayed = apply_proof(&computed_leaf, &receipt.inclusion_proof)?;
    let inclusion_proof_valid = replayed == declared_merkle;
    if !inclusion_proof_valid {
        issues.push("inclusion proof does not reproduce the declared passage_merkle_root".into());
    }

    let manifest_bytes = b64_decode(&receipt.manifest_bytes_b64, "manifest bytes")?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    let manifest_commits_merkle_root = manifest
        .get("passage_merkle_root")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|value| value.eq_ignore_ascii_case(&receipt.passage_merkle_root));
    if !manifest_commits_merkle_root {
        issues.push("manifest does not commit the receipt's passage_merkle_root".into());
    }

    let directory = b64_decode(&receipt.directory_b64, "directory")?;
    let manifest_matches_directory = match directory_entry(&directory, receipt.manifest_section_id)
    {
        Some((stored_hash, section_type, _)) => {
            if section_type != 1 {
                issues.push("manifest_section_id does not reference a manifest section".into());
                false
            } else {
                let actual = *blake3::hash(&manifest_bytes).as_bytes();
                let matches = actual == stored_hash;
                if !matches {
                    issues.push("manifest bytes do not match their directory entry hash".into());
                }
                matches
            }
        }
        None => {
            issues.push("directory contains no entry for manifest_section_id".into());
            false
        }
    };

    let declared_root = decode_hash(&receipt.pack_root, "pack_root")?;
    let computed_root = root_from_directory(&directory)?;
    let directory_matches_pack_root = computed_root == declared_root;
    if !directory_matches_pack_root {
        issues.push("directory does not reproduce the declared pack_root".into());
    }

    // Bind the receipt's descriptive and provenance fields to authenticated
    // bytes. Steps above prove the passage record is in the signed artifact;
    // these prove the receipt's labels (which passage, revision, pack, URL)
    // describe *that* record and artifact, not an attacker's substitution.
    let record_value: Option<serde_json::Value> = serde_json::from_slice(&record).ok();
    let passage_metadata_matches = record_value.as_ref().is_some_and(|value| {
        value.get("id").and_then(serde_json::Value::as_str) == Some(receipt.passage_id.as_str())
            && value.get("ordinal").and_then(serde_json::Value::as_u64)
                == Some(u64::from(receipt.passage_ordinal))
    });
    if !passage_metadata_matches {
        issues.push("passage_id or passage_ordinal does not match the authenticated record".into());
    }

    let source_revision_matches = manifest
        .get("source_revision")
        .and_then(serde_json::Value::as_str)
        == receipt.source_revision.as_deref();
    if !source_revision_matches {
        issues.push("source_revision does not match the authenticated manifest".into());
    }

    let pack_matches = match (
        manifest.get("name").and_then(serde_json::Value::as_str),
        manifest.get("version").and_then(serde_json::Value::as_str),
    ) {
        (Some(name), Some(version)) => receipt.pack == format!("{name}@{version}"),
        _ => false,
    };
    if !pack_matches {
        issues.push("pack does not match the authenticated manifest name@version".into());
    }

    let canonical_url_matches =
        verify_canonical_url(receipt, &directory, record_value.as_ref(), &mut issues);

    let (signature_valid, identity_trusted) =
        verify_receipt_signature(receipt, &declared_root, trusted_public_key, &mut issues);

    let verified = passage_hash_matches
        && inclusion_proof_valid
        && manifest_commits_merkle_root
        && manifest_matches_directory
        && directory_matches_pack_root
        && passage_metadata_matches
        && source_revision_matches
        && pack_matches
        && canonical_url_matches;

    Ok(ReceiptVerification {
        passage_hash_matches,
        inclusion_proof_valid,
        manifest_commits_merkle_root,
        manifest_matches_directory,
        directory_matches_pack_root,
        passage_metadata_matches,
        source_revision_matches,
        pack_matches,
        canonical_url_matches,
        signature_valid,
        identity_trusted,
        verified,
        issues,
    })
}

/// Authenticate `canonical_url` against the Documents section carried in the
/// receipt. Returns true when the URL is reproduced from an authentic document,
/// or when the receipt makes no URL claim. A URL claim with no backing Documents
/// section fails — which also blocks a downgrade that drops the section to
/// smuggle a forged URL past the check.
///
/// This is the one verification step that needs zlib inflation rather than only
/// BLAKE3/Ed25519/base64; a minimal verifier MAY skip it and MUST then report
/// `canonical_url` as unauthenticated rather than as covered by `verified`.
fn verify_canonical_url(
    receipt: &EvidenceReceipt,
    directory: &[u8],
    record: Option<&serde_json::Value>,
    issues: &mut Vec<String>,
) -> bool {
    let Some(declared_url) = receipt.canonical_url.as_deref() else {
        // No URL claim: nothing to authenticate and no attribution to forge.
        return true;
    };
    let (Some(section_id), Some(stored_b64)) = (
        receipt.documents_section_id,
        receipt.documents_bytes_b64.as_deref(),
    ) else {
        issues.push(
            "canonical_url is present but the receipt carries no Documents section to authenticate it"
                .into(),
        );
        return false;
    };
    let stored = match b64_decode(stored_b64, "documents section") {
        Ok(bytes) => bytes,
        Err(_) => {
            issues.push("documents section is not valid base64".into());
            return false;
        }
    };
    let Some((entry_hash, section_type, logical_length)) = directory_entry(directory, section_id)
    else {
        issues.push("directory contains no entry for documents_section_id".into());
        return false;
    };
    const DOCUMENTS_TYPE: u16 = 2;
    if section_type != DOCUMENTS_TYPE {
        issues.push("documents_section_id does not reference a Documents section".into());
        return false;
    }
    if *blake3::hash(&stored).as_bytes() != entry_hash {
        issues.push("documents section bytes do not match their directory entry hash".into());
        return false;
    }
    let Ok(limit) = usize::try_from(logical_length) else {
        issues.push("documents section logical length exceeds address space".into());
        return false;
    };
    let logical = match miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&stored, limit) {
        Ok(bytes) if bytes.len() == limit => bytes,
        _ => {
            issues.push("documents section failed to decompress to its committed length".into());
            return false;
        }
    };
    let Ok(documents) = serde_json::from_slice::<serde_json::Value>(&logical) else {
        issues.push("documents section is not valid JSON".into());
        return false;
    };
    let Some(document_id) = record
        .and_then(|value| value.get("document_id"))
        .and_then(serde_json::Value::as_str)
    else {
        issues.push("passage record carries no document_id to resolve canonical_url".into());
        return false;
    };
    let Some(document) = documents.as_array().and_then(|docs| {
        docs.iter()
            .find(|doc| doc.get("id").and_then(serde_json::Value::as_str) == Some(document_id))
    }) else {
        issues.push(
            "no authenticated document matches the passage's document_id for canonical_url".into(),
        );
        return false;
    };
    let base = document.get("url").and_then(serde_json::Value::as_str);
    let anchor = record
        .and_then(|value| value.get("anchor"))
        .and_then(serde_json::Value::as_str);
    if compose_canonical_url(base, anchor).as_deref() != Some(declared_url) {
        issues.push("canonical_url is not reproduced by the authenticated document".into());
        return false;
    }
    true
}

/// Reproduce the builder's citation URL: append the passage anchor as a fragment
/// only when the base URL carries none. Must match `citation_url` in search.
fn compose_canonical_url(base: Option<&str>, anchor: Option<&str>) -> Option<String> {
    let base = base?;
    match anchor {
        Some(anchor) if !anchor.is_empty() && !base.contains('#') => {
            Some(format!("{base}#{anchor}"))
        }
        _ => Some(base.to_string()),
    }
}

#[cfg(feature = "signing")]
fn verify_receipt_signature(
    receipt: &EvidenceReceipt,
    root: &[u8; 32],
    trusted_public_key: Option<&str>,
    issues: &mut Vec<String>,
) -> (bool, bool) {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let Some(envelope) = &receipt.signature else {
        if trusted_public_key.is_some() {
            issues.push("a trusted key was supplied but the receipt is unsigned".into());
        }
        return (false, false);
    };
    if envelope.algorithm != "Ed25519" {
        issues.push(format!(
            "unsupported signature algorithm {:?}",
            envelope.algorithm
        ));
        return (false, false);
    }
    let (Ok(public_key), Ok(signature_bytes)) = (
        decode_hash(&envelope.public_key, "signature public key"),
        hex::decode(&envelope.signature),
    ) else {
        issues.push("signature envelope is malformed".into());
        return (false, false);
    };
    let Ok(signature_bytes): std::result::Result<[u8; 64], _> = signature_bytes.try_into() else {
        issues.push("signature is not 64 bytes".into());
        return (false, false);
    };
    let Ok(verifying_key) = VerifyingKey::from_bytes(&public_key) else {
        issues.push("signature public key is not a valid Ed25519 key".into());
        return (false, false);
    };
    if blake3::hash(&public_key).to_hex().to_string() != envelope.key_id {
        issues.push("signature key_id does not match its public key".into());
        return (false, false);
    }
    let mut message = SIGNATURE_CONTEXT.to_vec();
    message.extend_from_slice(root);
    let signature = Signature::from_bytes(&signature_bytes);
    if verifying_key.verify(&message, &signature).is_err() {
        issues.push("signature does not verify over the artifact root".into());
        return (false, false);
    }
    // Cryptographic validity is not identity trust. Only an externally supplied
    // key binding can establish the latter.
    let identity_trusted = trusted_public_key
        .is_some_and(|trusted| trusted.eq_ignore_ascii_case(&envelope.public_key));
    if trusted_public_key.is_some() && !identity_trusted {
        issues.push("receipt signature does not use the supplied trusted key".into());
    }
    (true, identity_trusted)
}

#[cfg(not(feature = "signing"))]
fn verify_receipt_signature(
    receipt: &EvidenceReceipt,
    _root: &[u8; 32],
    _trusted_public_key: Option<&str>,
    issues: &mut Vec<String>,
) -> (bool, bool) {
    if receipt.signature.is_some() {
        issues.push("receipt is signed but this build lacks signature support".into());
    }
    (false, false)
}

/// Byte offset of the section directory within an artifact, read from its header.
pub fn directory_span(header: &[u8]) -> Result<(u64, u64)> {
    if header.len() < HEADER_SIZE {
        return Err(AnnpackError::InvalidFormat("truncated header".into()));
    }
    let offset = u64::from_le_bytes(header[24..32].try_into().expect("slice length"));
    let length = u64::from_le_bytes(header[32..40].try_into().expect("slice length"));
    Ok((offset, length))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[test]
    fn single_leaf_is_its_own_root() {
        assert_eq!(merkle_root(&[leaf(1)]), Some(leaf(1)));
    }

    #[test]
    fn empty_tree_has_no_root() {
        assert_eq!(merkle_root(&[]), None);
    }

    #[test]
    fn every_leaf_proves_against_the_root() {
        for count in 1..=33_usize {
            let leaves: Vec<[u8; 32]> = (0..count).map(|index| leaf(index as u8)).collect();
            let root = merkle_root(&leaves).unwrap();
            for index in 0..count {
                let proof = merkle_proof(&leaves, index).unwrap();
                assert_eq!(
                    apply_proof(&leaves[index], &proof).unwrap(),
                    root,
                    "leaf {index} of {count} failed to prove"
                );
            }
        }
    }

    #[test]
    fn a_forged_leaf_does_not_prove() {
        let leaves: Vec<[u8; 32]> = (0..8).map(|index| leaf(index as u8)).collect();
        let root = merkle_root(&leaves).unwrap();
        let proof = merkle_proof(&leaves, 3).unwrap();
        assert_ne!(apply_proof(&leaf(200), &proof).unwrap(), root);
    }

    #[test]
    fn odd_levels_promote_rather_than_duplicate() {
        // Duplicating the final node would make a 3-leaf tree collide with a
        // 4-leaf tree whose last two leaves are equal. Promotion must not.
        let three = vec![leaf(1), leaf(2), leaf(3)];
        let four = vec![leaf(1), leaf(2), leaf(3), leaf(3)];
        assert_ne!(merkle_root(&three), merkle_root(&four));
    }

    #[test]
    fn proof_index_is_bounds_checked() {
        assert!(merkle_proof(&[leaf(1)], 1).is_err());
    }
}
