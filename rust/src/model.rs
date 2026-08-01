use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub type Hash32 = [u8; 32];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub source_revision: Option<String>,
    pub base_url: Option<String>,
    pub created_at: Option<String>,
    pub document_count: u64,
    pub passage_count: u64,
    pub capabilities: Vec<String>,
    pub embedding_profiles: Vec<EmbeddingProfile>,
    pub policy: PackPolicy,
    pub dependencies: Vec<PackDependency>,
    /// Logical content root: Merkle root over per-passage evidence hashes in
    /// deterministic corpus order. Unlike the artifact root it does not commit
    /// to compression settings, block packing, or section layout, so it is
    /// stable across builders that agree on ingestion and chunking. It is what
    /// makes a standalone evidence receipt verifiable without the pack.
    /// Manifest section format 2 and later always populate it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passage_merkle_root: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceDescriptor>,
    /// ANN-7/ANN-8/ANN-9 provenance: one record per derived section, recording
    /// the offline generator and the pinned sidecar digest the build consumed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derived_inputs: Vec<DerivedInput>,
    /// ANN-10 fat-pack descriptor. Order is the deterministic fallback order and
    /// the final entry MUST be the Core lexical profile.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retrieval_profiles: Vec<RetrievalProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DerivedInput {
    pub kind: String,
    pub section_id: u32,
    pub generator: String,
    pub model: String,
    pub revision: String,
    /// Filtering/quantization parameters, recorded verbatim from the sidecar.
    pub params: BTreeMap<String, String>,
    /// BLAKE3 hex digest of the pinned sidecar the build consumed.
    pub sidecar_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetrievalProfile {
    pub id: String,
    pub kind: String,
    pub section_ids: Vec<u32>,
    pub requires: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceDescriptor {
    pub format: String,
    pub version: Option<String>,
    pub digest_algorithm: String,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PackPolicy {
    pub license: Option<String>,
    pub access: AccessClass,
    pub redistributable: Option<bool>,
    pub expires_at: Option<String>,
    pub policy_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payment: Option<PaymentTerms>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encryption: Option<EncryptionDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PaymentTerms {
    pub currency: String,
    pub amount_micros: u64,
    pub unit: String,
    pub discovery_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncryptionDescriptor {
    pub scheme: String,
    pub key_id: String,
    pub license_url: Option<String>,
    pub encrypted_section_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AccessClass {
    #[default]
    Public,
    Authenticated,
    Licensed,
    OrganizationRestricted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackDependency {
    pub name: String,
    pub version_requirement: String,
    pub root_hash: Option<String>,
    pub discovery_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingProfile {
    pub id: String,
    pub model: String,
    pub revision: String,
    pub dimensions: u32,
    pub dtype: String,
    pub pooling: String,
    pub normalized: bool,
    pub query_prefix: Option<String>,
    pub document_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<EmbeddingRuntime>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingRuntime {
    pub library: String,
    pub library_version: String,
    pub weights_dtype: String,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Document {
    pub id: String,
    pub source_path: String,
    pub title: String,
    pub url: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Passage {
    pub id: String,
    pub document_id: String,
    pub ordinal: u32,
    pub heading_path: Vec<String>,
    pub anchor: Option<String>,
    pub text: String,
    pub source_byte_start: Option<u64>,
    pub source_byte_end: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LexicalDictionary {
    pub passage_lengths: Vec<u32>,
    pub average_passage_length: f64,
    pub terms: BTreeMap<String, PostingMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostingMeta {
    pub offset: u64,
    pub length: u64,
    pub document_frequency: u32,
}

/// ANN-7 / ANN-8 term overlay (section type 13). A weighted inverted index over
/// generated or vocabulary-space terms, decoupled from raw passage text.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TermOverlaySection {
    /// `expansion-v1` (ANN-7) or `splade-v1` (ANN-8).
    pub kind: String,
    pub generator: String,
    pub model: String,
    pub revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocabulary: Option<OverlayVocabulary>,
    /// Lexicographically ordered map: term -> [[passage_ordinal, weight], ...],
    /// ordinals strictly increasing, weights non-negative integers.
    pub terms: BTreeMap<String, Vec<(u32, u32)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OverlayVocabulary {
    pub id: String,
    pub size: u32,
    pub quantization: String,
    pub scale: f64,
}

/// ANN-9 anchor set (section type 14): canonical reference inputs shipped in the
/// pack so any model can embed them and compute comparable coordinates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnchorSetSection {
    pub space_id: String,
    pub anchors: Vec<String>,
}

/// ANN-9 anchor coordinates (section type 15, derived): each passage's quantized
/// similarity to every anchor, in deterministic corpus order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnchorCoordinatesSection {
    pub space_id: String,
    pub metric: String,
    pub quantization: String,
    pub scale: f64,
    pub coordinates: Vec<Vec<i32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorProfileSection {
    pub profile: EmbeddingProfile,
    pub passage_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IvfIndex {
    pub algorithm: String,
    pub distance: String,
    pub dimensions: u32,
    pub default_probes: u32,
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<Vec<u32>>,
}

/// An Ed25519 signature over the artifact root, plus descriptive fields.
///
/// The signature covers exactly `UTF8("ANNPACK3-SIGNATURE\0") || artifact_root`
/// and nothing else. Of the fields below, only `algorithm`, `public_key`,
/// `signature`, `signed_root` and `key_id` participate in verification:
/// `signed_root` is compared against the reader's own root, `key_id` against
/// BLAKE3 of the public key, and the rest form or check the signature itself.
///
/// Every remaining field is **unauthenticated metadata**. A signature section is
/// excluded from the artifact root, so nothing in this envelope beyond the root
/// binding above is covered by any hash or signature, and anyone who can rewrite
/// the bytes can rewrite those fields without invalidating the signature. No
/// runtime security decision may depend on them. Authenticating them would take
/// a signed-envelope format, which this release does not define.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignatureEnvelope {
    pub algorithm: String,
    pub public_key: String,
    pub signature: String,
    pub signed_root: String,
    pub key_id: String,
    /// Unauthenticated. A self-declared string; never evidence of identity.
    pub identity: Option<String>,
    /// Unauthenticated. Not enforced anywhere: expiry is not implemented.
    pub expires_at: Option<String>,
    /// Unauthenticated. A hint for an operator, not a checked reference.
    pub transparency_log_url: Option<String>,
    /// Unauthenticated. Revocation is unimplemented; see ADR-0004.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revocation_url: Option<String>,
    /// Unauthenticated. Recorded provenance, not a verified attestation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_attestation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredPassageIndex {
    pub codec: String,
    pub records: Vec<StoredRecord>,
    pub blocks: Vec<StoredBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredRecord {
    pub id: String,
    pub block: u32,
    pub offset: u32,
    pub length: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredBlock {
    pub offset: u64,
    pub stored_length: u64,
    pub logical_length: u64,
    pub hash: String,
}
