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
    pub builder: String,
    pub document_count: u64,
    pub passage_count: u64,
    pub capabilities: Vec<String>,
    pub embedding_profiles: Vec<EmbeddingProfile>,
    pub policy: PackPolicy,
    pub dependencies: Vec<PackDependency>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceDescriptor>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignatureEnvelope {
    pub algorithm: String,
    pub public_key: String,
    pub signature: String,
    pub signed_root: String,
    pub key_id: String,
    pub identity: Option<String>,
    pub expires_at: Option<String>,
    pub transparency_log_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revocation_url: Option<String>,
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
