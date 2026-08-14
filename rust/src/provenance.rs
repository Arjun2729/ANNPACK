//! Build provenance: cryptographic proof of how an artifact was built.
//!
//! An artifact's own signature (`signing.rs`) proves a key produced these bytes.
//! It says nothing about which source, which builder, or which execution
//! produced them. Provenance binds that chain externally, using a standard
//! envelope rather than a fifth bespoke ANNPack signature format:
//!
//! ```text
//! source repository + revision
//!   -> exact consumed source bytes    (authenticated by the artifact, format 4+)
//!   -> exact builder executable
//!   -> build execution
//!   -> ANNPack artifact root
//!   -> distributed .annpack file
//! ```
//!
//! # What this proves, and what it does not
//!
//! A verified statement establishes: a named builder, running a named
//! executable, produced the exact distributed file whose internal artifact root
//! is `X`, from source bytes whose digest is `Y`. For a format-4 artifact `Y` is
//! independently authenticated by the artifact itself (`ADR-0005`); the
//! verifier recomputes it and compares, rather than trusting the statement.
//!
//! It does not establish that the source was correct, that the repository was
//! uncompromised, or that the workflow was trustworthy merely because it had a
//! name. **`repository` and `revision` are always reported as carried claims**,
//! never as verified: a signature proves who asserted them, not that they are
//! historically true. Conflating "signed" with "true" here is the mistake this
//! module exists to make impossible to make silently.
//!
//! Release authorization is a separate question, answered by `release.rs`. This
//! module does not read or write channel state, and a composed report keeps the
//! two sets of claims side by side rather than merged.
//!
//! # Why DSSE and in-toto rather than a bespoke envelope
//!
//! Every other ANNPack signature (`trust.rs`, `release.rs`, `signing.rs`,
//! `evidence.rs`) signs a canonical re-serialization of its own document,
//! because those documents are ANNPack's to define. A build-provenance
//! statement is meant to be produced and consumed by tooling outside ANNPack —
//! CI systems, SLSA verifiers, artifact registries — so it uses their standard
//! instead of inventing one: [DSSE](https://github.com/secure-systems-lab/dsse)
//! as the envelope, an
//! [in-toto Statement v1](https://in-toto.io/Statement/v1) as the payload.
//!
//! This changes what gets signed. DSSE signs the **exact payload bytes**
//! present in the envelope via Pre-Authentication Encoding (PAE), not a
//! recomputed serialization of a parsed struct. That is deliberate: it is what
//! stops a signature validated against one payload from being silently
//! re-associated with a different one, and it is why verification here
//! recomputes PAE over the base64-decoded bytes rather than re-serializing the
//! parsed [`Statement`].
//!
//! # Builder identity is not publisher identity
//!
//! The key that signs a provenance statement is a **builder** key. It is
//! disjoint from every [`crate::trust`] role. Using an artifact-signing or
//! release-state key to sign provenance does not make that key a trusted
//! builder; trust here comes only from a caller-supplied list of builder key
//! ids, checked independently of any [`crate::trust::TrustRoot`].

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::conformance::SourceBinding;
use crate::error::{AdyarError, Result};
use crate::format::PackReader;

pub const STATEMENT_TYPE: &str = "https://in-toto.io/Statement/v1";
pub const PREDICATE_TYPE: &str = "https://annpack.dev/attestations/build/v1";
pub const DSSE_PAYLOAD_TYPE: &str = "application/vnd.in-toto+json";

const MAX_SUBJECTS: usize = 1;
const MAX_SIGNATURES: usize = 16;
const MAX_ENVELOPE_BYTES: u64 = 1024 * 1024;
const MAX_PARAMETER_ENTRIES: usize = 64;

// ---------------------------------------------------------------------------
// in-toto Statement
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubjectDigest {
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Subject {
    pub name: String,
    pub digest: SubjectDigest,
}

/// FROZEN WIRE KEYS. The `annpack_` field names below are serialized into the
/// signed in-toto predicate, and the signature covers those exact bytes.
/// Renaming them — or adding a `serde(rename)` that diverges from them —
/// invalidates every attestation ever issued. They name a predicate version,
/// not a project, and they change when the predicate does.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuilderIdentityClaim {
    /// Workflow, workload, or operator identity. Free text; not itself a trust
    /// decision, which is made by the caller-supplied trusted-key list.
    pub id: String,
    pub annpack_version: String,
    /// SHA-256 of the exact executable that performed the build, when the
    /// creator could identify it. `None` under a mode that does not require it
    /// (`ADR-0006`), never a fabricated placeholder.
    pub annpack_binary_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceClaim {
    pub repository: String,
    pub revision: String,
    /// The digest of the exact consumed source bytes. For a format-4 artifact
    /// this is read from the artifact's own authenticated descriptor, never
    /// supplied independently by the caller — see [`create_build_provenance`].
    pub tree_digest: String,
    pub tree_digest_algorithm: String,
    /// Resolved input format the digest was computed under (`markdown`,
    /// `okf`). Absent for a legacy artifact with no authenticated descriptor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuildExecution {
    pub invocation_id: String,
    pub started_at: String,
    pub finished_at: String,
    /// Opt-in only. Nothing is captured by default, so a caller must choose to
    /// record anything here — see the module's privacy note in
    /// `spec/PROVENANCE-v1.md`.
    #[serde(default)]
    pub parameters: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub environment: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub platform: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locked: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdyarClaim {
    pub artifact_root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_content_root: Option<String>,
    pub manifest_format_version: u16,
    /// Whether `source.tree_digest` is authenticated by the artifact itself
    /// (`authenticated`) or only asserted by the builder
    /// (`absent_legacy_artifact`), mirroring [`SourceBinding`]. Recorded inside
    /// the signed predicate so a verifier does not have to trust the creator's
    /// account of which case applied — it is checked independently anyway, but
    /// disagreement between the claim and the recheck is itself informative.
    pub source_binding: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuildPredicate {
    pub builder: BuilderIdentityClaim,
    pub source: SourceClaim,
    pub build: BuildExecution,
    pub annpack: AdyarClaim,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Statement {
    #[serde(rename = "_type")]
    pub statement_type: String,
    pub subject: Vec<Subject>,
    #[serde(rename = "predicateType")]
    pub predicate_type: String,
    pub predicate: BuildPredicate,
}

// ---------------------------------------------------------------------------
// DSSE envelope
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DsseSignature {
    /// Sigstore bundles commonly omit `keyid` when the verification material
    /// carries exactly one certificate. DSSE defines it as optional; retain
    /// the empty-string representation used by ANNPack's local signing path.
    #[serde(default)]
    pub keyid: String,
    pub sig: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Envelope {
    /// Base64-encoded statement bytes. This exact string, decoded, is what PAE
    /// covers — never a re-serialization of a parsed [`Statement`].
    pub payload: String,
    #[serde(rename = "payloadType")]
    pub payload_type: String,
    pub signatures: Vec<DsseSignature>,
}

/// DSSE Pre-Authentication Encoding: `"DSSEv1" SP LEN(type) SP type SP LEN(body) SP body`.
///
/// `SP` is one ASCII space; `LEN` is ASCII decimal. Binding both the payload
/// type and its length into the signed bytes is what stops a valid signature
/// over one (type, payload) pair from being replayed against another.
#[cfg(feature = "signing")]
pub(crate) fn pae(payload_type: &str, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + payload_type.len() + 32);
    out.extend_from_slice(b"DSSEv1 ");
    out.extend_from_slice(payload_type.len().to_string().as_bytes());
    out.push(b' ');
    out.extend_from_slice(payload_type.as_bytes());
    out.push(b' ');
    out.extend_from_slice(payload.len().to_string().as_bytes());
    out.push(b' ');
    out.extend_from_slice(payload);
    out
}

#[cfg(feature = "signing")]
pub(crate) fn b64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub(crate) fn b64_decode(value: &str, max: usize) -> Result<Vec<u8>> {
    use base64::Engine;
    if value.len() > max.saturating_mul(4).saturating_div(3).saturating_add(4) {
        return Err(AdyarError::InvalidFormat(
            "DSSE payload exceeds size limit".into(),
        ));
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|_| AdyarError::InvalidFormat("DSSE payload is not valid base64".into()))?;
    if decoded.len() > max {
        return Err(AdyarError::InvalidFormat(
            "DSSE payload exceeds size limit".into(),
        ));
    }
    Ok(decoded)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

pub struct BuildProvenanceInput<'a> {
    pub artifact_path: &'a std::path::Path,
    pub repository: String,
    pub revision: String,
    pub builder_id: String,
    /// Path to the exact executable that performed the build, hashed by this
    /// function. `None` when the mode does not require builder-binary binding;
    /// never a digest the caller computed elsewhere and merely asserts.
    pub builder_binary_path: Option<&'a std::path::Path>,
    pub invocation_id: String,
    pub started_at: String,
    pub finished_at: String,
    pub parameters: std::collections::BTreeMap<String, String>,
    pub environment: std::collections::BTreeMap<String, String>,
    pub platform: Option<String>,
    pub locked: Option<bool>,
}

/// Build an unsigned provenance statement for a completed artifact.
///
/// Every binding fact — the distributed file's digest, the artifact root, the
/// logical content root, the manifest format version, the source digest, and
/// (when a path is given) the builder binary's digest — is derived here from
/// the artifact and the executable, never accepted as a caller-supplied string.
/// There is deliberately no parameter through which a caller could pass a
/// source digest that disagrees with the artifact's own: the only source digest
/// this function can record is the one it reads out of the artifact, or, for a
/// legacy artifact, the one the artifact's own ingestion-time computation
/// would have produced is unavailable at all, and this function refuses rather
/// than inventing one — see the `AbsentLegacyArtifact` branch below.
///
/// Refuses to create provenance for an artifact that fails integrity
/// verification. Signed provenance for bytes that do not verify would assert a
/// build chain for content that is not even self-consistent.
pub fn create_build_provenance(input: BuildProvenanceInput<'_>) -> Result<Statement> {
    let reader = PackReader::open_path(input.artifact_path)?;
    reader.verify_all()?; // integrity gate before any claim is recorded

    let manifest = reader.manifest()?;
    let manifest_entry = reader.entry(reader.header.manifest_section_id)?;
    let conformance = crate::conformance::inspect_conformance(&reader)?;

    let file_bytes = std::fs::read(input.artifact_path)?;
    let subject_name = input
        .artifact_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact.adyar".to_string());

    let (tree_digest, format, source_binding) = match conformance.source_binding {
        SourceBinding::Authenticated => {
            let source = manifest.source.as_ref().ok_or_else(|| {
                AdyarError::Integrity(
                    "conformance reported an authenticated source binding but the manifest \
                     carries no source descriptor"
                        .into(),
                )
            })?;
            (
                source.digest.clone(),
                Some(source.format.clone()),
                "authenticated",
            )
        }
        SourceBinding::AbsentLegacyArtifact => {
            return Err(AdyarError::Unsupported(
                "this artifact predates manifest format 4 and carries no authenticated source \
                 digest to record; create provenance with an explicit legacy source digest is \
                 not supported by this function, since any digest offered here would be an \
                 unverifiable builder claim indistinguishable from a fabricated one -- use \
                 `create_legacy_build_provenance` to record that limitation explicitly"
                    .into(),
            ));
        }
        SourceBinding::Malformed | SourceBinding::UnsupportedVersion => {
            return Err(AdyarError::Integrity(
                "artifact's source binding is malformed or its manifest format is unsupported; \
                 refusing to create provenance for content that is not self-consistent"
                    .into(),
            ));
        }
    };

    let annpack_binary_sha256 = input
        .builder_binary_path
        .map(|path| -> Result<String> { Ok(sha256_hex(&std::fs::read(path)?)) })
        .transpose()?;

    Ok(Statement {
        statement_type: STATEMENT_TYPE.to_string(),
        subject: vec![Subject {
            name: subject_name,
            digest: SubjectDigest {
                sha256: sha256_hex(&file_bytes),
            },
        }],
        predicate_type: PREDICATE_TYPE.to_string(),
        predicate: BuildPredicate {
            builder: BuilderIdentityClaim {
                id: input.builder_id,
                annpack_version: env!("CARGO_PKG_VERSION").to_string(),
                annpack_binary_sha256,
            },
            source: SourceClaim {
                repository: input.repository,
                revision: input.revision,
                tree_digest,
                tree_digest_algorithm: "blake3".to_string(),
                format,
            },
            build: BuildExecution {
                invocation_id: input.invocation_id,
                started_at: input.started_at,
                finished_at: input.finished_at,
                parameters: input.parameters,
                environment: input.environment,
                platform: input.platform,
                locked: input.locked,
            },
            annpack: AdyarClaim {
                artifact_root: reader.root_hex(),
                logical_content_root: manifest.passage_merkle_root.clone(),
                manifest_format_version: manifest_entry.format_version,
                source_binding: source_binding.to_string(),
            },
        },
    })
}

/// Build provenance for an artifact whose manifest predates format 4.
///
/// The resulting statement's source binding is honestly `absent_legacy_artifact`
/// and `tree_digest` is a builder-supplied assertion the artifact cannot
/// corroborate. Split from [`create_build_provenance`] so that the common case
/// cannot accidentally take the weaker path: a caller has to name the function
/// that admits the weaker claim.
pub fn create_legacy_build_provenance(
    input: BuildProvenanceInput<'_>,
    asserted_source_digest: String,
) -> Result<Statement> {
    let reader = PackReader::open_path(input.artifact_path)?;
    reader.verify_all()?;
    let manifest = reader.manifest()?;
    let manifest_entry = reader.entry(reader.header.manifest_section_id)?;
    let conformance = crate::conformance::inspect_conformance(&reader)?;
    if conformance.source_binding != SourceBinding::AbsentLegacyArtifact {
        return Err(AdyarError::InvalidInput(
            "this artifact is not a legacy artifact; use create_build_provenance so its \
             authenticated source digest is recorded instead of a builder assertion"
                .into(),
        ));
    }

    let file_bytes = std::fs::read(input.artifact_path)?;
    let subject_name = input
        .artifact_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact.adyar".to_string());
    let annpack_binary_sha256 = input
        .builder_binary_path
        .map(|path| -> Result<String> { Ok(sha256_hex(&std::fs::read(path)?)) })
        .transpose()?;

    Ok(Statement {
        statement_type: STATEMENT_TYPE.to_string(),
        subject: vec![Subject {
            name: subject_name,
            digest: SubjectDigest {
                sha256: sha256_hex(&file_bytes),
            },
        }],
        predicate_type: PREDICATE_TYPE.to_string(),
        predicate: BuildPredicate {
            builder: BuilderIdentityClaim {
                id: input.builder_id,
                annpack_version: env!("CARGO_PKG_VERSION").to_string(),
                annpack_binary_sha256,
            },
            source: SourceClaim {
                repository: input.repository,
                revision: input.revision,
                tree_digest: asserted_source_digest,
                tree_digest_algorithm: "blake3".to_string(),
                format: None,
            },
            build: BuildExecution {
                invocation_id: input.invocation_id,
                started_at: input.started_at,
                finished_at: input.finished_at,
                parameters: input.parameters,
                environment: input.environment,
                platform: input.platform,
                locked: input.locked,
            },
            annpack: AdyarClaim {
                artifact_root: reader.root_hex(),
                logical_content_root: manifest.passage_merkle_root.clone(),
                manifest_format_version: manifest_entry.format_version,
                source_binding: "absent_legacy_artifact".to_string(),
            },
        },
    })
}

// ---------------------------------------------------------------------------
// Signing
// ---------------------------------------------------------------------------

/// Sign a statement, producing a DSSE envelope.
///
/// Signs PAE over the exact serialized statement bytes. Those bytes, base64
/// encoded, become `payload`; nothing about the payload is re-derived at
/// verification time, so there is exactly one representation of "what was
/// signed" rather than a struct that might not round-trip identically.
#[cfg(feature = "signing")]
pub fn sign_provenance(statement: &Statement, secret_key: &[u8; 32]) -> Result<Envelope> {
    use ed25519_dalek::{Signer, SigningKey};

    let payload_bytes = serde_json::to_vec(statement)?;
    let signing_key = SigningKey::from_bytes(secret_key);
    let (key_id, _) = crate::trust::key_identity(secret_key);
    let signature = signing_key.sign(&pae(DSSE_PAYLOAD_TYPE, &payload_bytes));

    Ok(Envelope {
        payload: b64_encode(&payload_bytes),
        payload_type: DSSE_PAYLOAD_TYPE.to_string(),
        signatures: vec![DsseSignature {
            keyid: key_id,
            sig: hex::encode(signature.to_bytes()),
        }],
    })
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingStatus {
    Verified,
    Mismatched,
    Missing,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuilderIdentity {
    /// A valid signature exists from a key in the caller's trusted-builder list.
    Trusted,
    /// A valid signature exists, but not from a trusted key.
    Untrusted,
    /// No trusted-builder list was supplied, so trust was not evaluated.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvelopeSignature {
    /// A valid signature exists from a key in the caller's trusted-builder list.
    Valid,
    /// Either a signature exists but none validated against a trusted key, or
    /// no trusted keys were supplied at all to check against. DSSE carries a
    /// `keyid` but not the public key, so cryptographic validity can only be
    /// checked against candidate keys the caller supplies; this module does
    /// not report "valid against some unknown key" as a distinct state.
    Invalid,
    Unsigned,
}

/// `repository` and `revision` are always `Carried`: a signature proves who
/// asserted them, never that they are historically true. There is no `Verified`
/// variant for this type, on purpose — adding one would be the exact
/// conflation this module exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    Carried,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceDigestBinding {
    /// The artifact's own authenticated descriptor agrees with the statement.
    Authenticated,
    /// The artifact predates format 4; the statement's digest, if any, is an
    /// unverifiable builder claim.
    AbsentLegacyArtifact,
    /// The artifact is format 4 and its authenticated digest disagrees with the
    /// statement.
    Mismatched,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Completeness {
    /// Every binding verified, including an artifact-authenticated source digest.
    Complete,
    /// Every binding verified except source digest, which is legacy-carried.
    PartialLegacySourceBinding,
    Invalid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildProvenanceVerification {
    pub predicate_type_supported: bool,
    /// Exactly one subject, unambiguously naming the distributed file. A
    /// dedicated field rather than folding this into `issues`, so a CLI
    /// classifier can dispatch on it instead of matching issue text.
    pub subject_valid: bool,
    pub envelope_signature: EnvelopeSignature,
    pub builder_identity: BuilderIdentity,
    /// Distinct trusted-or-not key ids that produced a valid signature.
    pub signer_key_ids: Vec<String>,
    pub artifact_integrity: BindingStatus,
    pub distributed_file_digest: BindingStatus,
    pub artifact_root_binding: BindingStatus,
    pub logical_root_binding: BindingStatus,
    pub source_digest_binding: SourceDigestBinding,
    /// `Verified`/`Mismatched` only when a builder binary path was supplied to
    /// re-derive it; otherwise `Missing` is not right, since the claim itself
    /// may be present — reported as `Unsupported` to mean "not checked".
    pub builder_binary_binding: BindingStatus,
    pub builder_version_binding: BindingStatus,
    pub repository_claim: ClaimStatus,
    pub revision_claim: ClaimStatus,
    pub completeness: Completeness,
    pub verified: bool,
    pub assumptions: Vec<String>,
    pub issues: Vec<String>,
}

/// Whether `public_key` produced a valid signature in `envelope` over
/// `payload_bytes`, and if so its key id.
#[cfg(feature = "signing")]
pub(crate) fn check_signer(
    envelope: &Envelope,
    payload_bytes: &[u8],
    public_key_hex: &str,
) -> Option<String> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let public_bytes = hex::decode(public_key_hex).ok()?;
    let public_bytes: [u8; 32] = public_bytes.try_into().ok()?;
    let verifying_key = VerifyingKey::from_bytes(&public_bytes).ok()?;
    let expected_key_id = blake3::hash(&public_bytes).to_hex().to_string();
    let message = pae(&envelope.payload_type, payload_bytes);
    for signature in &envelope.signatures {
        if signature.keyid != expected_key_id {
            continue;
        }
        let Ok(signature_bytes) = hex::decode(&signature.sig) else {
            continue;
        };
        let Ok(signature_bytes): std::result::Result<[u8; 64], _> = signature_bytes.try_into()
        else {
            continue;
        };
        if verifying_key
            .verify(&message, &Signature::from_bytes(&signature_bytes))
            .is_ok()
        {
            return Some(expected_key_id);
        }
    }
    None
}

#[cfg(not(feature = "signing"))]
pub(crate) fn check_signer(
    _envelope: &Envelope,
    _payload_bytes: &[u8],
    _public_key_hex: &str,
) -> Option<String> {
    None
}

/// Verify a build-provenance envelope against a distributed artifact.
///
/// `trusted_builder_keys` are hex Ed25519 public keys. They are unrelated to
/// any [`crate::trust::TrustRoot`] role; a caller who wants an artifact-signing
/// key to also count as a trusted builder must list it here explicitly, which
/// is what makes "an artifact key used as a builder key without authorization"
/// (a required adversarial case) fail rather than silently succeed.
///
/// `builder_binary_path`, when supplied, is hashed and its `--version` output
/// captured to independently check the two builder-identity claims that would
/// otherwise be merely carried.
pub fn verify_build_provenance(
    envelope: &Envelope,
    artifact_path: &std::path::Path,
    trusted_builder_keys: &[String],
    builder_binary_path: Option<&std::path::Path>,
) -> Result<BuildProvenanceVerification> {
    let mut issues = Vec::new();
    let mut assumptions = Vec::new();

    if envelope.signatures.len() > MAX_SIGNATURES {
        return Err(AdyarError::InvalidFormat(
            "provenance envelope carries more signatures than the limit".into(),
        ));
    }
    if envelope.payload_type != DSSE_PAYLOAD_TYPE {
        issues.push(format!(
            "unsupported DSSE payload type {:?}",
            envelope.payload_type
        ));
    }
    let payload_bytes = b64_decode(&envelope.payload, MAX_ENVELOPE_BYTES as usize)?;
    let statement: Statement = serde_json::from_slice(&payload_bytes)?;

    let predicate_type_supported =
        statement.statement_type == STATEMENT_TYPE && statement.predicate_type == PREDICATE_TYPE;
    if !predicate_type_supported {
        issues.push(format!(
            "unsupported statement or predicate type: {:?} / {:?}",
            statement.statement_type, statement.predicate_type
        ));
    }
    let subject_valid = statement.subject.len() == MAX_SUBJECTS;
    if !subject_valid {
        issues.push(format!(
            "statement names {} subjects; exactly {MAX_SUBJECTS} is required, since an \
             ambiguous subject cannot be bound to one distributed file",
            statement.subject.len()
        ));
    }
    if statement.predicate.build.parameters.len() > MAX_PARAMETER_ENTRIES
        || statement.predicate.build.environment.len() > MAX_PARAMETER_ENTRIES
    {
        issues.push("build parameters or environment exceed the entry limit".into());
    }

    // DSSE carries a `keyid` but not the public key itself, so cryptographic
    // validity can only be checked against candidate keys supplied out of band.
    // Those candidates are exactly the caller's trusted-builder list: this
    // module has no notion of "some valid signature from an unknown key" to
    // fall back to, and does not invent one. A signature from a real but
    // untrusted key is therefore indistinguishable here from a forged one; both
    // are `Untrusted` / `Invalid`, which is the correct, conservative answer.
    let mut valid_trusted = Vec::new();
    for candidate in trusted_builder_keys {
        if let Some(key_id) = check_signer(envelope, &payload_bytes, candidate) {
            valid_trusted.push(key_id);
        }
    }
    if trusted_builder_keys.is_empty() {
        assumptions.push(
            "no trusted-builder keys supplied: signature cryptographic validity could not be \
             checked against any key, so builder identity is unknown"
                .into(),
        );
    }

    let envelope_signature = if envelope.signatures.is_empty() {
        EnvelopeSignature::Unsigned
    } else if !valid_trusted.is_empty() {
        EnvelopeSignature::Valid
    } else {
        if !trusted_builder_keys.is_empty() {
            issues.push("no signature validated against a trusted builder key".into());
        }
        EnvelopeSignature::Invalid
    };

    let builder_identity = if trusted_builder_keys.is_empty() {
        BuilderIdentity::Unknown
    } else if !valid_trusted.is_empty() {
        BuilderIdentity::Trusted
    } else {
        BuilderIdentity::Untrusted
    };

    // --- Artifact-derived facts, recomputed independently of the statement ---
    let reader = PackReader::open_path(artifact_path)?;
    let artifact_integrity = match reader.verify_all() {
        Ok(_) => BindingStatus::Verified,
        Err(error) => {
            issues.push(format!("artifact integrity failed: {error}"));
            BindingStatus::Mismatched
        }
    };

    let file_bytes = std::fs::read(artifact_path)?;
    let actual_file_digest = sha256_hex(&file_bytes);
    let distributed_file_digest = match statement.subject.first() {
        Some(subject) if subject.digest.sha256 == actual_file_digest => BindingStatus::Verified,
        Some(_) => {
            issues.push("distributed file digest does not match the statement's subject".into());
            BindingStatus::Mismatched
        }
        None => BindingStatus::Missing,
    };

    let actual_root = reader.root_hex();
    let artifact_root_binding = if statement.predicate.annpack.artifact_root == actual_root {
        BindingStatus::Verified
    } else {
        issues.push("artifact root does not match the statement".into());
        BindingStatus::Mismatched
    };

    let manifest = reader.manifest().ok();
    let logical_root_binding = match (
        &statement.predicate.annpack.logical_content_root,
        manifest
            .as_ref()
            .and_then(|manifest| manifest.passage_merkle_root.clone()),
    ) {
        (None, None) => BindingStatus::Missing,
        (Some(claimed), Some(actual)) if *claimed == actual => BindingStatus::Verified,
        (Some(_), Some(_)) => {
            issues.push("logical content root does not match the statement".into());
            BindingStatus::Mismatched
        }
        _ => {
            issues.push(
                "logical content root present in one of the artifact/statement but not the other"
                    .into(),
            );
            BindingStatus::Mismatched
        }
    };

    let conformance = crate::conformance::inspect_conformance(&reader)?;
    let source_digest_binding = match conformance.source_binding {
        SourceBinding::Authenticated => {
            let actual_digest = manifest
                .as_ref()
                .and_then(|manifest| manifest.source.as_ref())
                .map(|source| source.digest.clone());
            match actual_digest {
                Some(actual) if actual == statement.predicate.source.tree_digest => {
                    SourceDigestBinding::Authenticated
                }
                Some(_) => {
                    issues.push(
                        "artifact's authenticated source digest does not match the statement"
                            .into(),
                    );
                    SourceDigestBinding::Mismatched
                }
                None => SourceDigestBinding::Missing,
            }
        }
        SourceBinding::AbsentLegacyArtifact => {
            assumptions.push(
                "artifact predates manifest format 4: source digest is a builder claim the \
                 artifact cannot corroborate"
                    .into(),
            );
            SourceDigestBinding::AbsentLegacyArtifact
        }
        SourceBinding::Malformed | SourceBinding::UnsupportedVersion => {
            issues.push("artifact's source binding is malformed or unsupported".into());
            SourceDigestBinding::Missing
        }
    };

    // --- Builder-identity facts, checked only when a binary was supplied ---
    let (builder_binary_binding, builder_version_binding) = match builder_binary_path {
        None => {
            assumptions.push(
                "no builder binary supplied: builder version and executable digest are carried \
                 claims, not independently checked"
                    .into(),
            );
            (BindingStatus::Unsupported, BindingStatus::Unsupported)
        }
        Some(path) => {
            let binary_bytes = std::fs::read(path)?;
            let actual_digest = sha256_hex(&binary_bytes);
            let binary_binding = match &statement.predicate.builder.annpack_binary_sha256 {
                Some(claimed) if *claimed == actual_digest => BindingStatus::Verified,
                Some(_) => {
                    issues.push("builder executable digest does not match the statement".into());
                    BindingStatus::Mismatched
                }
                None => BindingStatus::Missing,
            };

            let version_binding = match std::process::Command::new(path).arg("--version").output() {
                Ok(output) if output.status.success() => {
                    let reported = String::from_utf8_lossy(&output.stdout);
                    if reported.contains(&statement.predicate.builder.annpack_version) {
                        BindingStatus::Verified
                    } else {
                        issues.push(format!(
                            "builder binary reports version output {reported:?}, statement \
                             claims {:?}",
                            statement.predicate.builder.annpack_version
                        ));
                        BindingStatus::Mismatched
                    }
                }
                _ => {
                    issues.push(
                        "could not invoke the supplied builder binary to check its version".into(),
                    );
                    BindingStatus::Mismatched
                }
            };
            (binary_binding, version_binding)
        }
    };

    let repository_claim = if statement.predicate.source.repository.trim().is_empty() {
        ClaimStatus::Missing
    } else {
        ClaimStatus::Carried
    };
    let revision_claim = if statement.predicate.source.revision.trim().is_empty() {
        ClaimStatus::Missing
    } else {
        ClaimStatus::Carried
    };

    let hard_bindings_ok = predicate_type_supported
        && subject_valid
        && matches!(envelope_signature, EnvelopeSignature::Valid)
        && matches!(builder_identity, BuilderIdentity::Trusted)
        && artifact_integrity == BindingStatus::Verified
        && distributed_file_digest == BindingStatus::Verified
        && artifact_root_binding == BindingStatus::Verified
        && logical_root_binding != BindingStatus::Mismatched
        && !matches!(builder_binary_binding, BindingStatus::Mismatched)
        && !matches!(builder_version_binding, BindingStatus::Mismatched);

    let completeness = if !hard_bindings_ok {
        Completeness::Invalid
    } else {
        match source_digest_binding {
            SourceDigestBinding::Authenticated => Completeness::Complete,
            SourceDigestBinding::AbsentLegacyArtifact => Completeness::PartialLegacySourceBinding,
            SourceDigestBinding::Mismatched | SourceDigestBinding::Missing => Completeness::Invalid,
        }
    };

    let verified = matches!(
        completeness,
        Completeness::Complete | Completeness::PartialLegacySourceBinding
    );

    Ok(BuildProvenanceVerification {
        predicate_type_supported,
        subject_valid,
        envelope_signature,
        builder_identity,
        signer_key_ids: valid_trusted,
        artifact_integrity,
        distributed_file_digest,
        artifact_root_binding,
        logical_root_binding,
        source_digest_binding,
        builder_binary_binding,
        builder_version_binding,
        repository_claim,
        revision_claim,
        completeness,
        verified,
        assumptions,
        issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "signing")]
    #[test]
    fn pae_binds_both_type_and_length() {
        let a = pae("text/plain", b"ab");
        let b = pae("text/plain", b"a");
        assert_ne!(a, b);
        let c = pae("text/plainX", b"ab");
        assert_ne!(a, c);
    }

    #[cfg(feature = "signing")]
    #[test]
    fn pae_matches_the_dsse_specification_definition() {
        // Built directly from the definition in secure-systems-lab/dsse
        // (protocol.md): PAE(type, body) = "DSSEv1" SP LEN(type) SP type SP
        // LEN(body) SP body. An earlier version of this test asserted a
        // by-hand length (30) for a 29-byte type string, which was simply
        // wrong -- a fabricated-looking expectation that was never computed.
        let payload_type = b"http://example.com/HelloWorld";
        let body = b"hello world";
        let mut expected = Vec::new();
        expected.extend_from_slice(b"DSSEv1 ");
        expected.extend_from_slice(payload_type.len().to_string().as_bytes());
        expected.push(b' ');
        expected.extend_from_slice(payload_type);
        expected.push(b' ');
        expected.extend_from_slice(body.len().to_string().as_bytes());
        expected.push(b' ');
        expected.extend_from_slice(body);

        assert_eq!(
            pae("http://example.com/HelloWorld", b"hello world"),
            expected
        );
        // Pinned once independently computed, so a future accidental change to
        // pae() is still caught even if this test's own construction were bugged.
        assert_eq!(
            pae("http://example.com/HelloWorld", b"hello world"),
            b"DSSEv1 29 http://example.com/HelloWorld 11 hello world".to_vec()
        );
    }

    #[cfg(feature = "signing")]
    #[test]
    fn base64_round_trips() {
        let payload = b"{\"a\":1}";
        let encoded = b64_encode(payload);
        assert_eq!(b64_decode(&encoded, 1024).unwrap(), payload);
    }

    #[test]
    fn oversized_base64_is_refused_before_decoding() {
        let huge = "A".repeat(10_000_000);
        assert!(b64_decode(&huge, 1024).is_err());
    }
}
