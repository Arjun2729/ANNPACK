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
    /// ANN-7/ANN-8 provenance: one record per derived section, recording
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
    /// Inline term table. Populated only in a format-1 lexical index, where the
    /// whole dictionary is one deflated section. A format-2 pack leaves this
    /// empty and stores terms in independently addressable blocks instead; see
    /// [`LexicalBlockIndex`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub terms: BTreeMap<String, PostingMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostingMeta {
    pub offset: u64,
    pub length: u64,
    pub document_frequency: u32,
}

/// One independently deflated, independently hashed run of bytes inside a
/// section whose section-level codec is `None`.
///
/// This is the same shape the passage-data section has always used. Generalizing
/// it to the lexical index is what makes a term lookup a bounded range read
/// instead of a whole-section download: the block's own hash authenticates it,
/// so a reader never has to materialize the section to trust a part of it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexBlock {
    /// Byte offset of the stored block, relative to the start of its section.
    pub offset: u64,
    pub stored_length: u64,
    pub logical_length: u64,
    /// Lowercase hex BLAKE3 over the *stored* (deflated) block bytes.
    pub hash: String,
    /// First term in this block, in the dictionary's sort order. Present on
    /// dictionary blocks only; it is the sparse index a reader binary-searches
    /// to find which single block can contain a term.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_term: Option<String>,
}

/// Random-access layout for the lexical index (section format 2).
///
/// `dictionary` blocks partition the sorted term table; `postings` blocks
/// partition the posting-list byte stream. A query resolves a term by binary
/// searching `dictionary` on `first_term`, fetching that one block, then
/// fetching only the postings blocks its byte range touches.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LexicalBlockIndex {
    pub dictionary: Vec<IndexBlock>,
    pub postings: Vec<IndexBlock>,
}

/// Random-access layout for the passage record table (passage index format 2).
///
/// `records` blocks hold fixed-width records in passage-ordinal order, so a
/// reader seeks straight to an ordinal without scanning. `ids` blocks hold the
/// same records keyed by passage id and sorted by it, which is the only way to
/// answer `get_passage(id)` without reading the whole table. Both are needed
/// because the two lookups have different orders and search uses only the first.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecordBlockIndex {
    /// Fixed-width record stride, in bytes. Records are
    /// `32-byte id || u32 block || u32 offset || u32 length`, little-endian.
    pub stride: u32,
    /// Records per block, uniform except in the final block.
    pub per_block: u32,
    pub records: Vec<IndexBlock>,
    /// Sorted by passage id. Each block's `first_term` carries its first id as
    /// lowercase hex, reusing the sparse-search field the dictionary uses.
    pub ids: Vec<IndexBlock>,
}

/// One dictionary block's payload: a contiguous run of the sorted term table.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictionaryBlock {
    pub terms: BTreeMap<String, PostingMeta>,
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
    /// Inline record table. Populated only in passage index format 1; a format-2
    /// pack leaves it empty and uses [`RecordBlockIndex`] instead.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub records: Vec<StoredRecord>,
    pub blocks: Vec<StoredBlock>,
    /// Random-access layout for the passage record table. Absent in a format-1
    /// pack, whose records are inline in this section. Present in format 2,
    /// where `records` above is empty and the table lives in its own section.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_blocks: Option<RecordBlockIndex>,
    /// Random-access layout for the lexical index. Absent in a format-1 pack,
    /// whose dictionary and postings are whole deflated sections. Present in
    /// format 2, where they are block-addressable.
    ///
    /// It lives here because this section is already the index-of-indexes: it
    /// is small, it is fetched once at open, and it is what a reader consults
    /// before issuing any other range request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical_blocks: Option<LexicalBlockIndex>,
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
