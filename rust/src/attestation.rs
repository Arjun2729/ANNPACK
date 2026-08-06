//! Offline verification of GitHub-issued Sigstore attestation bundles.
//!
//! # What this module does
//!
//! Parses a [Sigstore bundle](https://raw.githubusercontent.com/sigstore/protobuf-specs/main/protos/sigstore_bundle.proto)
//! (`application/vnd.dev.sigstore.bundle.v0.3+json`), extracts the leaf
//! certificate's GitHub Actions OIDC claims — issuer, workflow identity,
//! repository, commit, ref — from the real Fulcio extension OIDs registered at
//! `1.3.6.1.4.1.57264.1.*`, and matches them against caller-supplied builder
//! policy (allowed repositories, workflow refs, issuers). It then reuses
//! [`crate::provenance`]'s existing, already-tested predicate-binding checks
//! (artifact root, source digest, distributed-file digest) against the
//! predicate carried inside the bundle's DSSE envelope.
//!
//! Cryptographic verification is delegated to the exactly pinned
//! `sigstore-verify` stack.  The verifier is given artifact bytes, an exported
//! bundle and an operator-supplied trusted-root snapshot; this module performs
//! no network access and has no embedded-root fallback.  Authenticated GitHub
//! claims are extracted and policy is evaluated only after that verification
//! succeeds.

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};
use crate::provenance::{BindingStatus, BuildPredicate, Envelope, SourceDigestBinding};

const MAX_BUNDLE_BYTES: usize = 4 * 1024 * 1024;
const MAX_CERTIFICATE_BYTES: usize = 16 * 1024;

/// Real, current (non-deprecated) Fulcio extension OIDs, from
/// <https://github.com/sigstore/fulcio/blob/main/docs/oid-info.md>. Deprecated
/// GitHub-specific OIDs (`.1` through `.6`) are intentionally not read: a
/// certificate issued today carries the generic-provider OIDs below.
#[cfg(feature = "github-attestation")]
mod oid {
    pub const ISSUER_V2: &str = "1.3.6.1.4.1.57264.1.8";
    pub const BUILD_SIGNER_URI: &str = "1.3.6.1.4.1.57264.1.9";
    pub const BUILD_SIGNER_DIGEST: &str = "1.3.6.1.4.1.57264.1.10";
    pub const RUNNER_ENVIRONMENT: &str = "1.3.6.1.4.1.57264.1.11";
    pub const SOURCE_REPOSITORY_URI: &str = "1.3.6.1.4.1.57264.1.12";
    pub const SOURCE_REPOSITORY_DIGEST: &str = "1.3.6.1.4.1.57264.1.13";
    pub const SOURCE_REPOSITORY_REF: &str = "1.3.6.1.4.1.57264.1.14";
    pub const SOURCE_REPOSITORY_OWNER_URI: &str = "1.3.6.1.4.1.57264.1.16";
}

// ---------------------------------------------------------------------------
// Sigstore bundle (application/vnd.dev.sigstore.bundle.v0.3+json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct SigstoreBundle {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    #[serde(rename = "verificationMaterial")]
    pub verification_material: VerificationMaterial,
    #[serde(rename = "dsseEnvelope")]
    pub dsse_envelope: Envelope,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VerificationMaterial {
    /// Present for a single-leaf-certificate bundle (bundle schema v0.3, the
    /// form Fulcio's public-good instance produces for keyless signing).
    #[serde(default)]
    pub certificate: Option<X509Certificate>,
    #[serde(default, rename = "x509CertificateChain")]
    pub x509_certificate_chain: Option<X509CertificateChain>,
    /// Not parsed by this module. Rekor inclusion verification is exactly the
    /// gap documented at module level.
    #[serde(default, rename = "tlogEntries")]
    pub tlog_entries: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct X509Certificate {
    #[serde(rename = "rawBytes")]
    pub raw_bytes: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct X509CertificateChain {
    pub certificates: Vec<X509Certificate>,
}

impl SigstoreBundle {
    /// The leaf certificate, from whichever of the two `oneof` forms the
    /// bundle used. The leaf is always first when a chain is present (proto
    /// spec: "the first member MUST be a leaf certificate").
    pub fn leaf_certificate_der(&self) -> Result<Vec<u8>> {
        let base64_value = self
            .verification_material
            .certificate
            .as_ref()
            .map(|certificate| certificate.raw_bytes.as_str())
            .or_else(|| {
                self.verification_material
                    .x509_certificate_chain
                    .as_ref()
                    .and_then(|chain| chain.certificates.first())
                    .map(|certificate| certificate.raw_bytes.as_str())
            })
            .ok_or_else(|| {
                AnnpackError::InvalidFormat(
                    "bundle verification material carries no certificate".into(),
                )
            })?;
        b64_decode(base64_value, MAX_CERTIFICATE_BYTES)
    }
}

pub fn parse_bundle(bytes: &[u8]) -> Result<SigstoreBundle> {
    if bytes.len() > MAX_BUNDLE_BYTES {
        return Err(AnnpackError::InvalidFormat(
            "sigstore bundle exceeds size limit".into(),
        ));
    }
    let bundle: SigstoreBundle = serde_json::from_slice(bytes)?;
    if !bundle
        .media_type
        .starts_with("application/vnd.dev.sigstore.bundle")
    {
        return Err(AnnpackError::Unsupported(format!(
            "unrecognised bundle media type {:?}",
            bundle.media_type
        )));
    }
    Ok(bundle)
}

fn b64_decode(value: &str, max: usize) -> Result<Vec<u8>> {
    use base64::Engine;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|_| AnnpackError::InvalidFormat("bundle field is not valid base64".into()))?;
    if decoded.len() > max {
        return Err(AnnpackError::InvalidFormat(
            "decoded bundle field exceeds size limit".into(),
        ));
    }
    Ok(decoded)
}

// ---------------------------------------------------------------------------
// Certificate claims
// ---------------------------------------------------------------------------

/// GitHub Actions OIDC claims read from a Fulcio-issued certificate's
/// extensions. Every field is `None` when the certificate did not carry that
/// extension -- never defaulted or inferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GitHubCertificateClaims {
    /// The RFC 5280 Subject Alternative Name, as a URI `GeneralName`. This is
    /// what most Sigstore tooling calls "certificate identity" -- for a GitHub
    /// Actions run, a URI of the form
    /// `https://github.com/OWNER/REPO/.github/workflows/FILE.yml@REF`.
    pub subject_alternative_name: Option<String>,
    pub issuer: Option<String>,
    pub build_signer_uri: Option<String>,
    pub build_signer_digest: Option<String>,
    pub runner_environment: Option<String>,
    pub source_repository_uri: Option<String>,
    pub source_repository_digest: Option<String>,
    pub source_repository_ref: Option<String>,
    pub source_repository_owner_uri: Option<String>,
}

#[cfg(feature = "github-attestation")]
pub fn extract_certificate_claims(der_bytes: &[u8]) -> Result<GitHubCertificateClaims> {
    use der::Decode;
    use x509_cert::Certificate;
    use x509_cert::ext::pkix::SubjectAltName;
    use x509_cert::ext::pkix::name::GeneralName;

    let certificate = Certificate::from_der(der_bytes)
        .map_err(|error| AnnpackError::InvalidFormat(format!("malformed certificate: {error}")))?;

    let mut claims = GitHubCertificateClaims::default();
    let Some(extensions) = &certificate.tbs_certificate.extensions else {
        return Ok(claims);
    };

    for extension in extensions {
        let extension_oid = extension.extn_id.to_string();
        match extension_oid.as_str() {
            oid::ISSUER_V2 => {
                claims.issuer = decode_utf8_string_extension(extension.extn_value.as_bytes())?
            }
            oid::BUILD_SIGNER_URI => {
                claims.build_signer_uri =
                    decode_utf8_string_extension(extension.extn_value.as_bytes())?
            }
            oid::BUILD_SIGNER_DIGEST => {
                claims.build_signer_digest =
                    decode_utf8_string_extension(extension.extn_value.as_bytes())?
            }
            oid::RUNNER_ENVIRONMENT => {
                claims.runner_environment =
                    decode_utf8_string_extension(extension.extn_value.as_bytes())?
            }
            oid::SOURCE_REPOSITORY_URI => {
                claims.source_repository_uri =
                    decode_utf8_string_extension(extension.extn_value.as_bytes())?
            }
            oid::SOURCE_REPOSITORY_DIGEST => {
                claims.source_repository_digest =
                    decode_utf8_string_extension(extension.extn_value.as_bytes())?
            }
            oid::SOURCE_REPOSITORY_REF => {
                claims.source_repository_ref =
                    decode_utf8_string_extension(extension.extn_value.as_bytes())?
            }
            oid::SOURCE_REPOSITORY_OWNER_URI => {
                claims.source_repository_owner_uri =
                    decode_utf8_string_extension(extension.extn_value.as_bytes())?
            }
            // The standard RFC 5280 SAN extension (2.5.29.17), not a Sigstore
            // OID. `x509-cert` exposes it as an OctetString wrapping the DER
            // `SubjectAltName` sequence.
            id if id == const_oid::db::rfc5280::ID_CE_SUBJECT_ALT_NAME.to_string() => {
                if let Ok(san) = SubjectAltName::from_der(extension.extn_value.as_bytes()) {
                    claims.subject_alternative_name =
                        san.0.into_iter().find_map(|name| match name {
                            GeneralName::UniformResourceIdentifier(uri) => Some(uri.to_string()),
                            _ => None,
                        });
                }
            }
            _ => {}
        }
    }

    Ok(claims)
}

#[cfg(not(feature = "github-attestation"))]
pub fn extract_certificate_claims(_der_bytes: &[u8]) -> Result<GitHubCertificateClaims> {
    Err(AnnpackError::Unsupported(
        "built without the github-attestation feature".into(),
    ))
}

/// Extension values `1.3.6.1.4.1.57264.1.8` through `.24` are DER-encoded
/// UTF8Strings (Fulcio OID registry). Returns `None` for an extension present
/// but empty, `Err` for bytes that are not a valid UTF8String -- a
/// certificate carrying a malformed claim should not be silently read as
/// carrying no claim.
#[cfg(feature = "github-attestation")]
fn decode_utf8_string_extension(der_bytes: &[u8]) -> Result<Option<String>> {
    use der::Decode;
    use der::asn1::Utf8StringRef;

    if der_bytes.is_empty() {
        return Ok(None);
    }
    let value = Utf8StringRef::from_der(der_bytes).map_err(|error| {
        AnnpackError::InvalidFormat(format!(
            "certificate extension is not a UTF8String: {error}"
        ))
    })?;
    Ok(Some(value.as_str().to_string()))
}

// ---------------------------------------------------------------------------
// Builder policy
// ---------------------------------------------------------------------------

/// Which GitHub Actions identities are trusted as builders.
///
/// Every list is an allowlist; an empty list matches nothing, which is the
/// safe default for a policy nobody configured, not "match anything." A valid
/// keyless signature proves Fulcio issued a certificate for the identified
/// workflow context -- it does not by itself mean that context should be
/// trusted, any more than a valid Ed25519 signature from an unlisted key
/// means that key should be ([`crate::provenance`]'s own
/// `an_artifact_signing_key_is_not_automatically_a_trusted_builder`).
#[derive(Debug, Clone, Default)]
pub struct BuilderPolicy {
    /// e.g. `https://token.actions.githubusercontent.com`.
    pub allowed_issuers: Vec<String>,
    /// e.g. `https://github.com/Arjun2729/ANNPACK`.
    pub allowed_repositories: Vec<String>,
    /// Exact match against `build_signer_uri`, e.g.
    /// `https://github.com/Arjun2729/ANNPACK/.github/workflows/release.yml@refs/tags/v0.7.0-rc1`.
    /// Callers who want to allow any ref of one workflow file should list
    /// every ref explicitly or match on `subject_alternative_name` themselves;
    /// this policy does not implement prefix or glob matching, which would
    /// make the allowlist's actual scope harder to read from the list itself.
    pub allowed_workflow_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyVerdict {
    Trusted,
    Untrusted,
    /// A claim the policy needs to decide was absent from the certificate.
    Incomplete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub verdict: PolicyVerdict,
    pub issues: Vec<String>,
}

pub fn evaluate_builder_policy(
    claims: &GitHubCertificateClaims,
    policy: &BuilderPolicy,
) -> PolicyDecision {
    let mut issues = Vec::new();

    let Some(issuer) = &claims.issuer else {
        return PolicyDecision {
            verdict: PolicyVerdict::Incomplete,
            issues: vec!["certificate carries no issuer extension".into()],
        };
    };
    let Some(repository) = &claims.source_repository_uri else {
        return PolicyDecision {
            verdict: PolicyVerdict::Incomplete,
            issues: vec!["certificate carries no source repository extension".into()],
        };
    };
    let Some(workflow_ref) = &claims.build_signer_uri else {
        return PolicyDecision {
            verdict: PolicyVerdict::Incomplete,
            issues: vec!["certificate carries no build signer URI extension".into()],
        };
    };

    if !policy
        .allowed_issuers
        .iter()
        .any(|allowed| allowed == issuer)
    {
        issues.push(format!(
            "issuer {issuer:?} is not in the trusted-issuer list"
        ));
    }
    if !policy
        .allowed_repositories
        .iter()
        .any(|allowed| allowed == repository)
    {
        issues.push(format!(
            "repository {repository:?} is not in the trusted-repository list"
        ));
    }
    if !policy
        .allowed_workflow_refs
        .iter()
        .any(|allowed| allowed == workflow_ref)
    {
        issues.push(format!(
            "workflow {workflow_ref:?} is not in the trusted-workflow list"
        ));
    }

    PolicyDecision {
        verdict: if issues.is_empty() {
            PolicyVerdict::Trusted
        } else {
            PolicyVerdict::Untrusted
        },
        issues,
    }
}

// ---------------------------------------------------------------------------
// Composed verification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationState {
    Verified,
    Authenticated,
    Invalid,
    Missing,
    Unsupported,
    Untrusted,
    Incomplete,
    NotEvaluated,
}

/// A claim read from the certificate, compared against an independent source
/// where one exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimAgreement {
    /// The predicate's carried claim and the certificate's claim are equal.
    Agree,
    Disagree,
    /// One or both sides had nothing to compare.
    Incomparable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubAttestationReport {
    pub bundle_structure: VerificationState,
    pub trusted_root: VerificationState,
    pub signing_time: VerificationState,
    pub certificate_chain: VerificationState,
    pub certificate_validity: VerificationState,
    pub certificate_transparency: VerificationState,
    pub artifact_signature: VerificationState,
    pub rekor_checkpoint: VerificationState,
    pub rekor_inclusion: VerificationState,
    pub rekor_entry_consistency: VerificationState,
    pub timestamp_evidence: VerificationState,
    pub workload_claims: VerificationState,
    pub builder_policy: PolicyVerdict,
    pub predicate_type: VerificationState,
    pub predicate_bindings: VerificationState,
    pub certificate_claims: GitHubCertificateClaims,
    pub repository_claim_agreement: ClaimAgreement,
    pub revision_claim_agreement: ClaimAgreement,
    pub artifact_integrity: BindingStatus,
    pub subject_binding: BindingStatus,
    pub artifact_root_binding: BindingStatus,
    pub source_digest_binding: SourceDigestBinding,
    pub trusted_root_sha256: String,
    pub trusted_root_media_type: String,
    pub selected_fulcio_authority: Option<String>,
    pub selected_rekor_log_ids: Vec<String>,
    pub fully_offline: bool,
    pub verified: bool,
    pub policy_issues: Vec<String>,
    pub issues: Vec<String>,
}

/// Extract the predicate from a bundle's DSSE payload without verifying the
/// envelope's signature. Callers MUST NOT treat the returned predicate as
/// authenticated; it exists so binding checks (§ below) can be described
/// precisely as "the predicate says X" versus "X is true," which is exactly
/// the distinction this whole module exists to keep visible.
fn extract_predicate(bundle: &SigstoreBundle) -> Result<BuildPredicate> {
    use base64::Engine;
    let payload = base64::engine::general_purpose::STANDARD
        .decode(&bundle.dsse_envelope.payload)
        .map_err(|_| AnnpackError::InvalidFormat("DSSE payload is not valid base64".into()))?;
    let statement: crate::provenance::Statement = serde_json::from_slice(&payload)?;
    Ok(statement.predicate)
}

fn set_successful_crypto(report: &mut GitHubAttestationReport) {
    report.signing_time = VerificationState::Verified;
    report.certificate_chain = VerificationState::Verified;
    report.certificate_validity = VerificationState::Verified;
    report.certificate_transparency = VerificationState::Verified;
    report.artifact_signature = VerificationState::Verified;
    report.rekor_checkpoint = VerificationState::Verified;
    report.rekor_inclusion = VerificationState::Verified;
    report.rekor_entry_consistency = VerificationState::Verified;
    report.timestamp_evidence = VerificationState::Verified;
}

fn calculate_overall(report: &GitHubAttestationReport) -> bool {
    report.bundle_structure == VerificationState::Verified
        && report.trusted_root == VerificationState::Verified
        && report.signing_time == VerificationState::Verified
        && report.certificate_chain == VerificationState::Verified
        && report.certificate_validity == VerificationState::Verified
        && report.certificate_transparency == VerificationState::Verified
        && report.artifact_signature == VerificationState::Verified
        && report.rekor_checkpoint == VerificationState::Verified
        && report.rekor_inclusion == VerificationState::Verified
        && report.rekor_entry_consistency == VerificationState::Verified
        && report.timestamp_evidence == VerificationState::Verified
        && report.workload_claims == VerificationState::Authenticated
        && report.builder_policy == PolicyVerdict::Trusted
        && report.predicate_type == VerificationState::Verified
        && report.predicate_bindings == VerificationState::Verified
        && report.repository_claim_agreement == ClaimAgreement::Agree
        && report.revision_claim_agreement == ClaimAgreement::Agree
        && report.artifact_integrity == BindingStatus::Verified
        && report.subject_binding == BindingStatus::Verified
        && report.artifact_root_binding == BindingStatus::Verified
        && report.source_digest_binding == SourceDigestBinding::Authenticated
}

/// Classify the verifier's fail-closed error into the first failed stage. Later
/// stages remain `not_evaluated`; stages that necessarily ran earlier are
/// marked verified. `sigstore-verify` intentionally stops on the first failure.
fn classify_crypto_failure(report: &mut GitHubAttestationReport, message: &str) {
    let lower = message.to_ascii_lowercase();
    let fail = if lower.contains("validation time")
        || lower.contains("trustworthy time")
        || lower.contains("timestamp")
    {
        &mut report.signing_time
    } else if lower.contains("certificate has expired")
        || lower.contains("certificate not yet valid")
        || (lower.contains("certificate chain validation failed")
            && (lower.contains("expired") || lower.contains("not valid yet")))
    {
        report.signing_time = VerificationState::Verified;
        report.certificate_chain = VerificationState::Verified;
        &mut report.certificate_validity
    } else if lower.contains("certificate chain")
        || lower.contains("unknownissuer")
        || lower.contains("unknown issuer")
        || lower.contains("certificate authority")
        || lower.contains("fulcio cert")
    {
        report.signing_time = VerificationState::Verified;
        &mut report.certificate_chain
    } else if lower.contains("sct") || lower.contains("certificate transparency") {
        report.signing_time = VerificationState::Verified;
        report.certificate_chain = VerificationState::Verified;
        report.certificate_validity = VerificationState::Verified;
        &mut report.certificate_transparency
    } else if lower.contains("integrated time") {
        report.signing_time = VerificationState::Verified;
        report.certificate_chain = VerificationState::Verified;
        report.certificate_validity = VerificationState::Verified;
        report.certificate_transparency = VerificationState::Verified;
        &mut report.timestamp_evidence
    } else if lower.contains("checkpoint") {
        report.signing_time = VerificationState::Verified;
        report.certificate_chain = VerificationState::Verified;
        report.certificate_validity = VerificationState::Verified;
        report.certificate_transparency = VerificationState::Verified;
        report.rekor_inclusion = VerificationState::Verified;
        &mut report.rekor_checkpoint
    } else if lower.contains("inclusion") || lower.contains("merkle") {
        report.signing_time = VerificationState::Verified;
        report.certificate_chain = VerificationState::Verified;
        report.certificate_validity = VerificationState::Verified;
        report.certificate_transparency = VerificationState::Verified;
        &mut report.rekor_inclusion
    } else if lower.contains("rekor")
        || lower.contains("transparency log")
        || lower.contains("payload hash mismatch")
        || lower.contains("signature or verifier mismatch")
    {
        report.signing_time = VerificationState::Verified;
        report.certificate_chain = VerificationState::Verified;
        report.certificate_validity = VerificationState::Verified;
        report.certificate_transparency = VerificationState::Verified;
        report.rekor_checkpoint = VerificationState::Verified;
        report.rekor_inclusion = VerificationState::Verified;
        report.artifact_signature = VerificationState::Verified;
        &mut report.rekor_entry_consistency
    } else if lower.contains("signature") || lower.contains("artifact hash") {
        report.signing_time = VerificationState::Verified;
        report.certificate_chain = VerificationState::Verified;
        report.certificate_validity = VerificationState::Verified;
        report.certificate_transparency = VerificationState::Verified;
        report.rekor_checkpoint = VerificationState::Verified;
        report.rekor_inclusion = VerificationState::Verified;
        &mut report.artifact_signature
    } else {
        &mut report.artifact_signature
    };
    *fail = VerificationState::Invalid;
    if report.signing_time == VerificationState::Verified
        && report.timestamp_evidence == VerificationState::NotEvaluated
    {
        report.timestamp_evidence = VerificationState::Verified;
    }
    report.issues.push(message.to_string());
}

fn agreement(
    predicate: &BuildPredicate,
    claims: &GitHubCertificateClaims,
) -> (ClaimAgreement, ClaimAgreement) {
    let repository = match &claims.source_repository_uri {
        Some(certificate_repository)
            if predicate
                .source
                .repository
                .trim_start_matches("github.com/")
                == certificate_repository
                    .trim_start_matches("https://")
                    .trim_start_matches("github.com/") =>
        {
            ClaimAgreement::Agree
        }
        Some(_) => ClaimAgreement::Disagree,
        None => ClaimAgreement::Incomparable,
    };
    let revision = match &claims.source_repository_digest {
        Some(certificate_revision)
            if predicate.source.revision.trim_start_matches("git:") == certificate_revision =>
        {
            ClaimAgreement::Agree
        }
        Some(_) => ClaimAgreement::Disagree,
        None => ClaimAgreement::Incomparable,
    };
    (repository, revision)
}

/// Verify a bundle without any network access. Claims and policy are evaluated
/// only after every cryptographic check in `sigstore-verify` succeeds.
#[cfg(feature = "github-attestation")]
pub fn verify_github_attestation(
    bundle_bytes: &[u8],
    trusted_root_bytes: &[u8],
    artifact_path: &std::path::Path,
    policy: &BuilderPolicy,
) -> Result<GitHubAttestationReport> {
    use sha2::{Digest, Sha256};
    use sigstore_verify::bundle::{ValidationOptions, validate_bundle_with_options};
    use sigstore_verify::trust_root::TrustedRoot;
    use sigstore_verify::types::Bundle;

    // Bundle shape is deliberately established before any trust material is
    // consulted, matching the Sigstore client verification order.
    let bundle_json = std::str::from_utf8(bundle_bytes)
        .map_err(|_| AnnpackError::InvalidFormat("Sigstore bundle is not UTF-8 JSON".into()))?;
    let crypto_bundle = Bundle::from_json(bundle_json).map_err(|error| {
        AnnpackError::InvalidFormat(format!("malformed Sigstore bundle: {error}"))
    })?;
    crypto_bundle.version().map_err(|error| {
        AnnpackError::Unsupported(format!("unsupported Sigstore bundle version: {error}"))
    })?;
    validate_bundle_with_options(
        &crypto_bundle,
        &ValidationOptions {
            require_inclusion_proof: true,
            require_timestamp: false,
        },
    )
    .map_err(|error| {
        AnnpackError::InvalidFormat(format!("invalid Sigstore bundle structure: {error}"))
    })?;

    let root_json = std::str::from_utf8(trusted_root_bytes)
        .map_err(|_| AnnpackError::InvalidFormat("trusted root is not UTF-8 JSON".into()))?;
    let trusted_root = TrustedRoot::from_json(root_json).map_err(|error| {
        AnnpackError::InvalidFormat(format!("malformed Sigstore trusted root: {error}"))
    })?;
    if trusted_root.media_type != "application/vnd.dev.sigstore.trustedroot+json;version=0.1" {
        return Err(AnnpackError::Unsupported(format!(
            "unsupported Sigstore trusted-root media type {:?}",
            trusted_root.media_type
        )));
    }
    if trusted_root.certificate_authorities.is_empty() || trusted_root.tlogs.is_empty() {
        return Err(AnnpackError::InvalidFormat(
            "Sigstore trusted root must contain a Fulcio authority and a Rekor log".into(),
        ));
    }

    // Keep ANNPack's envelope parser alongside the library's typed parser,
    // but do not parse or act on its claims until cryptography authenticates it.
    let annpack_bundle = parse_bundle(bundle_bytes)?;
    let root_digest = format!("{:x}", Sha256::digest(trusted_root_bytes));
    let selected_rekor_log_ids = crypto_bundle
        .verification_material
        .tlog_entries
        .iter()
        .map(|entry| entry.log_id.key_id.to_string())
        .collect::<Vec<_>>();

    let mut report = GitHubAttestationReport {
        bundle_structure: VerificationState::Verified,
        trusted_root: VerificationState::Verified,
        signing_time: VerificationState::NotEvaluated,
        certificate_chain: VerificationState::NotEvaluated,
        certificate_validity: VerificationState::NotEvaluated,
        certificate_transparency: VerificationState::NotEvaluated,
        artifact_signature: VerificationState::NotEvaluated,
        rekor_checkpoint: VerificationState::NotEvaluated,
        rekor_inclusion: VerificationState::NotEvaluated,
        rekor_entry_consistency: VerificationState::NotEvaluated,
        timestamp_evidence: VerificationState::NotEvaluated,
        workload_claims: VerificationState::NotEvaluated,
        builder_policy: PolicyVerdict::Incomplete,
        predicate_type: VerificationState::NotEvaluated,
        predicate_bindings: VerificationState::NotEvaluated,
        certificate_claims: GitHubCertificateClaims::default(),
        repository_claim_agreement: ClaimAgreement::Incomparable,
        revision_claim_agreement: ClaimAgreement::Incomparable,
        artifact_integrity: BindingStatus::Unsupported,
        subject_binding: BindingStatus::Unsupported,
        artifact_root_binding: BindingStatus::Unsupported,
        source_digest_binding: SourceDigestBinding::Missing,
        trusted_root_sha256: root_digest,
        trusted_root_media_type: trusted_root.media_type.clone(),
        selected_fulcio_authority: None,
        selected_rekor_log_ids,
        fully_offline: true,
        verified: false,
        policy_issues: Vec::new(),
        issues: Vec::new(),
    };

    let artifact = std::fs::read(artifact_path)?;
    let crypto = sigstore_verify::verify(
        artifact.as_slice(),
        &crypto_bundle,
        &sigstore_verify::VerificationPolicy::default(),
        &trusted_root,
    );
    if let Err(error) = crypto {
        classify_crypto_failure(&mut report, &error.to_string());
        return Ok(report);
    }
    set_successful_crypto(&mut report);

    // Identify the Fulcio authority that independently permits the complete
    // verification. This is report metadata, never a substitute for the full
    // verification above.
    for ca in &trusted_root.certificate_authorities {
        let mut candidate = trusted_root.clone();
        candidate.certificate_authorities = vec![ca.clone()];
        if sigstore_verify::verify(
            artifact.as_slice(),
            &crypto_bundle,
            &sigstore_verify::VerificationPolicy::default(),
            &candidate,
        )
        .is_ok()
        {
            report.selected_fulcio_authority = Some(ca.uri.clone());
            break;
        }
    }

    let der = annpack_bundle.leaf_certificate_der()?;
    let claims = extract_certificate_claims(&der)?;
    report.workload_claims = VerificationState::Authenticated;
    let policy_decision = evaluate_builder_policy(&claims, policy);
    report.builder_policy = policy_decision.verdict;
    report.policy_issues = policy_decision.issues;
    let predicate = extract_predicate(&annpack_bundle)?;
    report.predicate_type =
        if annpack_bundle.dsse_envelope.payload_type == crate::provenance::DSSE_PAYLOAD_TYPE {
            VerificationState::Verified
        } else {
            VerificationState::Unsupported
        };
    let (repository, revision) = agreement(&predicate, &claims);
    report.repository_claim_agreement = repository;
    report.revision_claim_agreement = revision;
    report.certificate_claims = claims;

    // Reuse the existing artifact/predicate implementation. Passing no local
    // builder keys deliberately leaves its Ed25519-only fields unused; the
    // Sigstore signature was already verified above.
    let bindings = crate::provenance::verify_build_provenance(
        &annpack_bundle.dsse_envelope,
        artifact_path,
        &[],
        None,
    )?;
    report.artifact_integrity = bindings.artifact_integrity;
    report.subject_binding = bindings.distributed_file_digest;
    report.artifact_root_binding = bindings.artifact_root_binding;
    report.source_digest_binding = bindings.source_digest_binding;
    let bindings_ok = bindings.predicate_type_supported
        && bindings.subject_valid
        && bindings.artifact_integrity == BindingStatus::Verified
        && bindings.distributed_file_digest == BindingStatus::Verified
        && bindings.artifact_root_binding == BindingStatus::Verified
        && bindings.logical_root_binding != BindingStatus::Mismatched
        && bindings.source_digest_binding == SourceDigestBinding::Authenticated;
    report.predicate_bindings = if bindings_ok {
        VerificationState::Verified
    } else {
        VerificationState::Invalid
    };
    report.issues.extend(
        bindings
            .issues
            .into_iter()
            .filter(|issue| !issue.contains("trusted builder key")),
    );
    if repository == ClaimAgreement::Disagree {
        report
            .issues
            .push("predicate repository claim disagrees with authenticated workload claim".into());
    }
    if revision == ClaimAgreement::Disagree {
        report
            .issues
            .push("predicate revision claim disagrees with authenticated workload claim".into());
    }

    report.verified = calculate_overall(&report);
    Ok(report)
}

#[cfg(all(test, feature = "github-attestation"))]
mod tests {
    use super::*;

    fn utf8_extension_der(value: &str) -> Vec<u8> {
        use der::Encode;
        use der::asn1::Utf8StringRef;
        Utf8StringRef::new(value).unwrap().to_der().unwrap()
    }

    /// A certificate carrying the real Fulcio OIDs plus a standard SAN, built
    /// with `rcgen` so the DER bytes are genuine, not hand-assembled.
    fn certificate_with_claims() -> Vec<u8> {
        use rcgen::{CertificateParams, CustomExtension, KeyPair, SanType};

        let mut params = CertificateParams::new(Vec::new()).unwrap();
        params.subject_alt_names = vec![SanType::URI(
            "https://github.com/example/repo/.github/workflows/release.yml@refs/tags/v1.2.3"
                .try_into()
                .unwrap(),
        )];
        params.custom_extensions = vec![
            CustomExtension::from_oid_content(
                &oid_components(oid::ISSUER_V2),
                utf8_extension_der("https://token.actions.githubusercontent.com"),
            ),
            CustomExtension::from_oid_content(
                &oid_components(oid::BUILD_SIGNER_URI),
                utf8_extension_der(
                    "https://github.com/example/repo/.github/workflows/release.yml@refs/tags/v1.2.3",
                ),
            ),
            CustomExtension::from_oid_content(
                &oid_components(oid::SOURCE_REPOSITORY_URI),
                utf8_extension_der("https://github.com/example/repo"),
            ),
            CustomExtension::from_oid_content(
                &oid_components(oid::SOURCE_REPOSITORY_DIGEST),
                utf8_extension_der("abc123def456"),
            ),
            CustomExtension::from_oid_content(
                &oid_components(oid::SOURCE_REPOSITORY_REF),
                utf8_extension_der("refs/tags/v1.2.3"),
            ),
        ];
        let key_pair = KeyPair::generate().unwrap();
        let certificate = params.self_signed(&key_pair).unwrap();
        certificate.der().to_vec()
    }

    fn oid_components(dotted: &str) -> Vec<u64> {
        dotted
            .split('.')
            .map(|part| part.parse().unwrap())
            .collect()
    }

    #[test]
    fn every_configured_claim_is_extracted() {
        let claims = extract_certificate_claims(&certificate_with_claims()).unwrap();
        assert_eq!(
            claims.issuer.as_deref(),
            Some("https://token.actions.githubusercontent.com")
        );
        assert_eq!(
            claims.build_signer_uri.as_deref(),
            Some("https://github.com/example/repo/.github/workflows/release.yml@refs/tags/v1.2.3")
        );
        assert_eq!(
            claims.source_repository_uri.as_deref(),
            Some("https://github.com/example/repo")
        );
        assert_eq!(
            claims.source_repository_digest.as_deref(),
            Some("abc123def456")
        );
        assert_eq!(
            claims.source_repository_ref.as_deref(),
            Some("refs/tags/v1.2.3")
        );
        assert_eq!(
            claims.subject_alternative_name.as_deref(),
            Some("https://github.com/example/repo/.github/workflows/release.yml@refs/tags/v1.2.3")
        );
    }

    #[test]
    fn a_certificate_with_no_extensions_yields_all_none() {
        use rcgen::{CertificateParams, KeyPair};
        let params = CertificateParams::new(Vec::new()).unwrap();
        let key_pair = KeyPair::generate().unwrap();
        let certificate = params.self_signed(&key_pair).unwrap();
        let claims = extract_certificate_claims(certificate.der()).unwrap();
        assert_eq!(claims, GitHubCertificateClaims::default());
    }

    fn matching_policy() -> BuilderPolicy {
        BuilderPolicy {
            allowed_issuers: vec!["https://token.actions.githubusercontent.com".into()],
            allowed_repositories: vec!["https://github.com/example/repo".into()],
            allowed_workflow_refs: vec![
                "https://github.com/example/repo/.github/workflows/release.yml@refs/tags/v1.2.3"
                    .into(),
            ],
        }
    }

    fn complete_report() -> GitHubAttestationReport {
        GitHubAttestationReport {
            bundle_structure: VerificationState::Verified,
            trusted_root: VerificationState::Verified,
            signing_time: VerificationState::Verified,
            certificate_chain: VerificationState::Verified,
            certificate_validity: VerificationState::Verified,
            certificate_transparency: VerificationState::Verified,
            artifact_signature: VerificationState::Verified,
            rekor_checkpoint: VerificationState::Verified,
            rekor_inclusion: VerificationState::Verified,
            rekor_entry_consistency: VerificationState::Verified,
            timestamp_evidence: VerificationState::Verified,
            workload_claims: VerificationState::Authenticated,
            builder_policy: PolicyVerdict::Trusted,
            predicate_type: VerificationState::Verified,
            predicate_bindings: VerificationState::Verified,
            certificate_claims: GitHubCertificateClaims::default(),
            repository_claim_agreement: ClaimAgreement::Agree,
            revision_claim_agreement: ClaimAgreement::Agree,
            artifact_integrity: BindingStatus::Verified,
            subject_binding: BindingStatus::Verified,
            artifact_root_binding: BindingStatus::Verified,
            source_digest_binding: SourceDigestBinding::Authenticated,
            trusted_root_sha256: "00".repeat(32),
            trusted_root_media_type: "application/vnd.dev.sigstore.trustedroot+json;version=0.1"
                .into(),
            selected_fulcio_authority: Some("https://fulcio.example".into()),
            selected_rekor_log_ids: vec!["log".into()],
            fully_offline: true,
            verified: false,
            policy_issues: Vec::new(),
            issues: Vec::new(),
        }
    }

    #[test]
    fn overall_result_is_the_complete_security_conjunction() {
        let complete = complete_report();
        assert!(calculate_overall(&complete));

        macro_rules! rejected_when {
            ($field:ident, $value:expr) => {{
                let mut report = complete.clone();
                report.$field = $value;
                assert!(!calculate_overall(&report), stringify!($field));
            }};
        }
        rejected_when!(trusted_root, VerificationState::Untrusted);
        rejected_when!(signing_time, VerificationState::Invalid);
        rejected_when!(certificate_chain, VerificationState::Invalid);
        rejected_when!(certificate_validity, VerificationState::Invalid);
        rejected_when!(certificate_transparency, VerificationState::Invalid);
        rejected_when!(artifact_signature, VerificationState::Invalid);
        rejected_when!(rekor_checkpoint, VerificationState::Invalid);
        rejected_when!(rekor_inclusion, VerificationState::Invalid);
        rejected_when!(rekor_entry_consistency, VerificationState::Invalid);
        rejected_when!(timestamp_evidence, VerificationState::Invalid);
        rejected_when!(workload_claims, VerificationState::NotEvaluated);
        rejected_when!(builder_policy, PolicyVerdict::Untrusted);
        rejected_when!(predicate_bindings, VerificationState::Invalid);
        rejected_when!(repository_claim_agreement, ClaimAgreement::Disagree);
        rejected_when!(revision_claim_agreement, ClaimAgreement::Disagree);
        rejected_when!(subject_binding, BindingStatus::Mismatched);
        rejected_when!(artifact_root_binding, BindingStatus::Mismatched);
        rejected_when!(source_digest_binding, SourceDigestBinding::Mismatched);
    }

    #[test]
    fn a_fully_matching_certificate_is_trusted_by_policy() {
        let claims = extract_certificate_claims(&certificate_with_claims()).unwrap();
        let decision = evaluate_builder_policy(&claims, &matching_policy());
        assert_eq!(
            decision.verdict,
            PolicyVerdict::Trusted,
            "{:?}",
            decision.issues
        );
    }

    #[test]
    fn a_repository_outside_the_allowlist_is_untrusted() {
        let claims = extract_certificate_claims(&certificate_with_claims()).unwrap();
        let mut policy = matching_policy();
        policy.allowed_repositories = vec!["https://github.com/other/repo".into()];
        let decision = evaluate_builder_policy(&claims, &policy);
        assert_eq!(decision.verdict, PolicyVerdict::Untrusted);
        assert!(
            decision
                .issues
                .iter()
                .any(|issue| issue.contains("repository"))
        );
    }

    #[test]
    fn correct_repository_but_wrong_workflow_ref_is_untrusted() {
        let claims = extract_certificate_claims(&certificate_with_claims()).unwrap();
        let mut policy = matching_policy();
        policy.allowed_workflow_refs = vec![
            "https://github.com/example/repo/.github/workflows/release.yml@refs/tags/v9.9.9".into(),
        ];
        let decision = evaluate_builder_policy(&claims, &policy);
        assert_eq!(decision.verdict, PolicyVerdict::Untrusted);
        assert!(
            decision
                .issues
                .iter()
                .any(|issue| issue.contains("workflow"))
        );
    }

    #[test]
    fn an_empty_policy_trusts_nothing() {
        // The safe default: a policy nobody configured must not silently
        // accept every certificate, the way an empty trusted-key list for
        // local Ed25519 signing must not either.
        let claims = extract_certificate_claims(&certificate_with_claims()).unwrap();
        let decision = evaluate_builder_policy(&claims, &BuilderPolicy::default());
        assert_eq!(decision.verdict, PolicyVerdict::Untrusted);
    }

    #[test]
    fn a_certificate_missing_a_required_claim_is_incomplete_not_untrusted() {
        use rcgen::{CertificateParams, CustomExtension, KeyPair};
        let mut params = CertificateParams::new(Vec::new()).unwrap();
        // Issuer present; repository and workflow claims absent.
        params.custom_extensions = vec![CustomExtension::from_oid_content(
            &oid_components(oid::ISSUER_V2),
            utf8_extension_der("https://token.actions.githubusercontent.com"),
        )];
        let key_pair = KeyPair::generate().unwrap();
        let certificate = params.self_signed(&key_pair).unwrap();
        let claims = extract_certificate_claims(certificate.der()).unwrap();
        let decision = evaluate_builder_policy(&claims, &matching_policy());
        assert_eq!(decision.verdict, PolicyVerdict::Incomplete);
    }

    #[test]
    fn policy_matching_is_separate_from_cryptographic_verification() {
        // Policy matching is a pure operation. The composed verifier controls
        // when it may run and only supplies cryptographically authenticated
        // claims.
        let bundle = SigstoreBundle {
            media_type: "application/vnd.dev.sigstore.bundle.v0.3+json".into(),
            verification_material: VerificationMaterial {
                certificate: Some(X509Certificate {
                    raw_bytes: {
                        use base64::Engine;
                        base64::engine::general_purpose::STANDARD.encode(certificate_with_claims())
                    },
                }),
                x509_certificate_chain: None,
                tlog_entries: Vec::new(),
            },
            dsse_envelope: Envelope {
                payload: {
                    use base64::Engine;
                    base64::engine::general_purpose::STANDARD.encode(b"{}")
                },
                payload_type: crate::provenance::DSSE_PAYLOAD_TYPE.to_string(),
                signatures: Vec::new(),
            },
        };
        // Extraction of the (malformed, empty-object) predicate will fail
        // here, which is fine: this test only needs evaluate_builder_policy's
        // half of the pipeline, exercised directly.
        let claims = extract_certificate_claims(&bundle.leaf_certificate_der().unwrap()).unwrap();
        let decision = evaluate_builder_policy(&claims, &matching_policy());
        assert_eq!(decision.verdict, PolicyVerdict::Trusted);
        // This result alone is not an overall verification result.
    }

    #[test]
    fn an_unrecognised_bundle_media_type_is_refused() {
        let bytes = br#"{"mediaType":"application/vnd.other.thing+json","verificationMaterial":{},"dsseEnvelope":{"payload":"e30=","payloadType":"x","signatures":[]}}"#;
        assert!(parse_bundle(bytes).is_err());
    }

    #[test]
    fn a_bundle_with_no_certificate_is_refused() {
        let bytes = br#"{"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json","verificationMaterial":{},"dsseEnvelope":{"payload":"e30=","payloadType":"x","signatures":[]}}"#;
        let bundle = parse_bundle(bytes).unwrap();
        assert!(bundle.leaf_certificate_der().is_err());
    }
}
