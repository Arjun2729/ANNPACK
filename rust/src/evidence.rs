//! Standalone evidence receipts: offline, pack-free proof that a cited passage
//! existed unmodified in a known immutable artifact.

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};
use crate::format::{
    DECOMPRESSION_RATIO_FLOOR, DECOMPRESSION_RATIO_LIMIT, DIRECTORY_ENTRY_SIZE, HEADER_SIZE,
    MAX_SECTION_SIZE,
};

const NODE_CONTEXT: &[u8] = b"ANNPACK3-EVIDENCE-NODE\0";
pub const PASSAGE_EVIDENCE_CONTEXT: &[u8] = b"ANNPACK3-PASSAGE-EVIDENCE\0";
// Only the signing-enabled verifier uses this; a no-signing build (the WASM
// target) reports `signature_valid: false` without ever forming the message.
#[cfg(feature = "signing")]
const SIGNATURE_CONTEXT: &[u8] = b"ANNPACK3-SIGNATURE\0";
const CONTENT_ROOT_CONTEXT: &[u8] = b"ANNPACK3-CONTENT-ROOT\0";
const RECEIPT_SCHEMA_V2: &str = "annpack-receipt-v2";
const MANIFEST_TYPE: u16 = 1;
const DOCUMENTS_TYPE: u16 = 2;
const SIGNATURE_TYPE: u16 = 10;
const CODEC_NONE: u16 = 0;
const CODEC_DEFLATE: u16 = 1;
const MAX_RECEIPT_DIRECTORY_BYTES: usize = DIRECTORY_ENTRY_SIZE * 16_384;
const MAX_RECEIPT_JSON_BYTES: usize = 16 * 1024 * 1024;

/// Maximum receipt file size the reference CLI reads.
///
/// The per-field limits above bound each embedded blob once the document has
/// been parsed. This bounds the file *before* it is read, so a hostile receipt
/// cannot make the verifier allocate its whole length first. The value leaves
/// room for the largest plausible honest receipt: a passage record and manifest
/// at their own limits, the directory at its limit, and a stored Documents
/// section for a large corpus.
pub const MAX_RECEIPT_FILE_BYTES: u64 = 64 * 1024 * 1024;

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProofStep {
    pub sibling: String,
    pub sibling_is_left: bool,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceReceipt {
    pub schema: String,
    pub pack: String,
    pub pack_root: String,
    pub passage_merkle_root: String,
    pub source_revision: Option<String>,
    pub passage_id: String,
    pub passage_hash: String,
    pub passage_ordinal: u32,
    pub canonical_url: Option<String>,
    pub passage_record_b64: String,
    pub inclusion_proof: Vec<ProofStep>,
    pub manifest_bytes_b64: String,
    pub directory_b64: String,
    pub manifest_section_id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents_section_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents_bytes_b64: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptVerification {
    pub passage_hash_matches: bool,
    pub inclusion_proof_valid: bool,
    pub manifest_commits_merkle_root: bool,
    pub manifest_matches_directory: bool,
    pub directory_matches_pack_root: bool,
    pub passage_metadata_matches: bool,
    pub source_revision_matches: bool,
    pub pack_matches: bool,
    pub canonical_url_matches: bool,
    pub signature_valid: bool,
    pub identity_trusted: bool,
    pub verified: bool,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct ReceiptDirectoryEntry {
    section_id: u32,
    section_type: u16,
    codec: u16,
    stored_length: u64,
    logical_length: u64,
    hash: [u8; 32],
}

fn decode_hash(value: &str, label: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(value)
        .map_err(|_| AnnpackError::InvalidFormat(format!("{label} is not valid hex")))?;
    bytes
        .try_into()
        .map_err(|_| AnnpackError::InvalidFormat(format!("{label} is not 32 bytes")))
}

fn b64_decode_limited(value: &str, label: &str, max_decoded: usize) -> Result<Vec<u8>> {
    use base64::Engine;
    let max_encoded = max_decoded
        .checked_add(2)
        .and_then(|value| value.checked_div(3))
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| AnnpackError::InvalidFormat(format!("{label} size limit overflow")))?;
    if value.len() > max_encoded {
        return Err(AnnpackError::InvalidFormat(format!(
            "{label} exceeds receipt size limit"
        )));
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|_| AnnpackError::InvalidFormat(format!("{label} is not valid base64")))?;
    if decoded.len() > max_decoded {
        return Err(AnnpackError::InvalidFormat(format!(
            "{label} exceeds receipt size limit"
        )));
    }
    Ok(decoded)
}

pub fn b64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn parse_directory(directory: &[u8]) -> Result<Vec<ReceiptDirectoryEntry>> {
    if directory.is_empty()
        || directory.len() > MAX_RECEIPT_DIRECTORY_BYTES
        || !directory.len().is_multiple_of(DIRECTORY_ENTRY_SIZE)
    {
        return Err(AnnpackError::InvalidFormat(
            "receipt directory is empty, oversized, or misaligned".into(),
        ));
    }

    let mut entries = Vec::with_capacity(directory.len() / DIRECTORY_ENTRY_SIZE);
    let mut previous_id = None;
    for raw in directory.as_chunks::<DIRECTORY_ENTRY_SIZE>().0 {
        if raw[76..80].iter().any(|byte| *byte != 0) {
            return Err(AnnpackError::InvalidFormat(
                "receipt directory reserved bytes must be zero".into(),
            ));
        }
        let section_id = u32::from_le_bytes(raw[0..4].try_into().expect("slice length"));
        if previous_id.is_some_and(|previous| section_id <= previous) {
            return Err(AnnpackError::InvalidFormat(
                "receipt directory section IDs must be strictly increasing".into(),
            ));
        }
        previous_id = Some(section_id);

        let stored_length = u64::from_le_bytes(raw[20..28].try_into().expect("slice length"));
        let logical_length = u64::from_le_bytes(raw[28..36].try_into().expect("slice length"));
        if stored_length > MAX_SECTION_SIZE || logical_length > MAX_SECTION_SIZE {
            return Err(AnnpackError::InvalidFormat(format!(
                "receipt section {section_id} exceeds size limit"
            )));
        }

        let mut hash = [0_u8; 32];
        hash.copy_from_slice(&raw[44..76]);
        entries.push(ReceiptDirectoryEntry {
            section_id,
            section_type: u16::from_le_bytes(raw[4..6].try_into().expect("slice length")),
            codec: u16::from_le_bytes(raw[8..10].try_into().expect("slice length")),
            stored_length,
            logical_length,
            hash,
        });
    }
    Ok(entries)
}

fn root_from_directory(directory: &[u8]) -> Result<[u8; 32]> {
    parse_directory(directory)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(CONTENT_ROOT_CONTEXT);
    for raw in directory.as_chunks::<DIRECTORY_ENTRY_SIZE>().0 {
        let section_type = u16::from_le_bytes(raw[4..6].try_into().expect("slice length"));
        if section_type != SIGNATURE_TYPE {
            hasher.update(raw);
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

fn directory_entry(
    entries: &[ReceiptDirectoryEntry],
    section_id: u32,
) -> Option<ReceiptDirectoryEntry> {
    entries
        .binary_search_by_key(&section_id, |entry| entry.section_id)
        .ok()
        .map(|index| entries[index])
}

pub fn verify_receipt(
    receipt: &EvidenceReceipt,
    trusted_public_key: Option<&str>,
) -> Result<ReceiptVerification> {
    if receipt.schema != RECEIPT_SCHEMA_V2 {
        return Err(AnnpackError::Unsupported(format!(
            "receipt schema {:?}; this verifier supports {RECEIPT_SCHEMA_V2}",
            receipt.schema
        )));
    }
    if receipt.inclusion_proof.len() > 64 {
        return Err(AnnpackError::InvalidFormat(
            "receipt inclusion proof exceeds 64 steps".into(),
        ));
    }

    let mut issues = Vec::new();
    let record = b64_decode_limited(
        &receipt.passage_record_b64,
        "passage record",
        MAX_RECEIPT_JSON_BYTES,
    )?;
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

    let manifest_bytes = b64_decode_limited(
        &receipt.manifest_bytes_b64,
        "manifest bytes",
        MAX_RECEIPT_JSON_BYTES,
    )?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    let manifest_commits_merkle_root = manifest
        .get("passage_merkle_root")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|value| value.eq_ignore_ascii_case(&receipt.passage_merkle_root));
    if !manifest_commits_merkle_root {
        issues.push("manifest does not commit the receipt's passage_merkle_root".into());
    }

    let directory = b64_decode_limited(
        &receipt.directory_b64,
        "directory",
        MAX_RECEIPT_DIRECTORY_BYTES,
    )?;
    let entries = parse_directory(&directory)?;
    let manifest_matches_directory = match directory_entry(&entries, receipt.manifest_section_id) {
        Some(entry) if entry.section_type == MANIFEST_TYPE => {
            let matches = manifest_bytes.len() as u64 == entry.stored_length
                && entry.stored_length == entry.logical_length
                && entry.codec == CODEC_NONE
                && *blake3::hash(&manifest_bytes).as_bytes() == entry.hash;
            if !matches {
                issues.push(
                    "manifest bytes, lengths, or codec do not match their directory entry".into(),
                );
            }
            matches
        }
        Some(_) => {
            issues.push("manifest_section_id does not reference a manifest section".into());
            false
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

    let record_value: serde_json::Value = serde_json::from_slice(&record)?;
    let passage_metadata_matches = record_value.get("id").and_then(serde_json::Value::as_str)
        == Some(receipt.passage_id.as_str())
        && record_value
            .get("ordinal")
            .and_then(serde_json::Value::as_u64)
            == Some(u64::from(receipt.passage_ordinal));
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

    let canonical_url_matches = verify_canonical_url(receipt, &entries, &record_value, &mut issues);

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

fn verify_canonical_url(
    receipt: &EvidenceReceipt,
    entries: &[ReceiptDirectoryEntry],
    record: &serde_json::Value,
    issues: &mut Vec<String>,
) -> bool {
    let Some(declared_url) = receipt.canonical_url.as_deref() else {
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
    let Some(entry) = directory_entry(entries, section_id) else {
        issues.push("directory contains no entry for documents_section_id".into());
        return false;
    };
    if entry.section_type != DOCUMENTS_TYPE {
        issues.push("documents_section_id does not reference a Documents section".into());
        return false;
    }

    let max_stored = match usize::try_from(entry.stored_length) {
        Ok(value) => value,
        Err(_) => {
            issues.push("documents section stored length exceeds address space".into());
            return false;
        }
    };
    let stored = match b64_decode_limited(stored_b64, "documents section", max_stored) {
        Ok(bytes) => bytes,
        Err(error) => {
            issues.push(error.to_string());
            return false;
        }
    };
    if stored.len() as u64 != entry.stored_length {
        issues.push("documents section length does not match its directory entry".into());
        return false;
    }
    if *blake3::hash(&stored).as_bytes() != entry.hash {
        issues.push("documents section bytes do not match their directory entry hash".into());
        return false;
    }

    let logical = match decode_committed_section(entry, &stored) {
        Ok(bytes) => bytes,
        Err(error) => {
            issues.push(error.to_string());
            return false;
        }
    };
    let Ok(documents) = serde_json::from_slice::<serde_json::Value>(&logical) else {
        issues.push("documents section is not valid JSON".into());
        return false;
    };
    let Some(document_id) = record
        .get("document_id")
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
    let anchor = record.get("anchor").and_then(serde_json::Value::as_str);
    if compose_canonical_url(base, anchor).as_deref() != Some(declared_url) {
        issues.push("canonical_url is not reproduced by the authenticated document".into());
        return false;
    }
    true
}

fn decode_committed_section(entry: ReceiptDirectoryEntry, stored: &[u8]) -> Result<Vec<u8>> {
    if entry.stored_length > MAX_SECTION_SIZE || entry.logical_length > MAX_SECTION_SIZE {
        return Err(AnnpackError::InvalidFormat(
            "documents section exceeds size limit".into(),
        ));
    }
    let logical_length = usize::try_from(entry.logical_length).map_err(|_| {
        AnnpackError::InvalidFormat("documents section exceeds address space".into())
    })?;
    match entry.codec {
        CODEC_NONE => {
            if entry.stored_length != entry.logical_length || stored.len() != logical_length {
                return Err(AnnpackError::InvalidFormat(
                    "uncompressed documents section has inconsistent lengths".into(),
                ));
            }
            Ok(stored.to_vec())
        }
        CODEC_DEFLATE => {
            if entry.stored_length == 0 {
                return Err(AnnpackError::InvalidFormat(
                    "compressed documents section is empty".into(),
                ));
            }
            if entry.logical_length > DECOMPRESSION_RATIO_FLOOR
                && entry.logical_length
                    > entry
                        .stored_length
                        .saturating_mul(DECOMPRESSION_RATIO_LIMIT)
            {
                return Err(AnnpackError::InvalidFormat(
                    "documents section exceeds decompression-ratio limit".into(),
                ));
            }
            let decoded =
                miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(stored, logical_length)
                    .map_err(|error| {
                        AnnpackError::InvalidFormat(format!(
                            "documents section deflate decode failed: {error:?}"
                        ))
                    })?;
            if decoded.len() != logical_length {
                return Err(AnnpackError::InvalidFormat(format!(
                    "documents section decompressed to {}, expected {logical_length} bytes",
                    decoded.len()
                )));
            }
            Ok(decoded)
        }
        other => Err(AnnpackError::Unsupported(format!(
            "documents section codec {other}"
        ))),
    }
}

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

    fn entry(codec: u16, stored: u64, logical: u64) -> ReceiptDirectoryEntry {
        ReceiptDirectoryEntry {
            section_id: 2,
            section_type: DOCUMENTS_TYPE,
            codec,
            stored_length: stored,
            logical_length: logical,
            hash: [0; 32],
        }
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
                assert_eq!(apply_proof(&leaves[index], &proof).unwrap(), root);
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
        let three = vec![leaf(1), leaf(2), leaf(3)];
        let four = vec![leaf(1), leaf(2), leaf(3), leaf(3)];
        assert_ne!(merkle_root(&three), merkle_root(&four));
    }

    #[test]
    fn proof_index_is_bounds_checked() {
        assert!(merkle_proof(&[leaf(1)], 1).is_err());
    }

    #[test]
    fn codec_zero_uses_stored_bytes_directly() {
        let bytes = br#"[{"id":"doc"}]"#;
        assert_eq!(
            decode_committed_section(
                entry(CODEC_NONE, bytes.len() as u64, bytes.len() as u64),
                bytes,
            )
            .unwrap(),
            bytes
        );
    }

    #[test]
    fn codec_zero_rejects_length_mismatch() {
        let bytes = b"{}";
        assert!(decode_committed_section(entry(CODEC_NONE, 2, 3), bytes).is_err());
    }

    #[test]
    fn deflate_round_trips_with_bound() {
        let logical = br#"[{"id":"doc","url":"https://example.test"}]"#;
        let stored = miniz_oxide::deflate::compress_to_vec_zlib(logical, 6);
        assert_eq!(
            decode_committed_section(
                entry(CODEC_DEFLATE, stored.len() as u64, logical.len() as u64),
                &stored,
            )
            .unwrap(),
            logical
        );
    }

    #[test]
    fn unsupported_codec_fails() {
        assert!(decode_committed_section(entry(99, 2, 2), b"{}").is_err());
    }

    #[test]
    fn excessive_decompression_ratio_fails_before_inflate() {
        assert!(
            decode_committed_section(
                entry(
                    CODEC_DEFLATE,
                    1,
                    DECOMPRESSION_RATIO_FLOOR + DECOMPRESSION_RATIO_LIMIT + 1,
                ),
                &[0],
            )
            .is_err()
        );
    }
}
