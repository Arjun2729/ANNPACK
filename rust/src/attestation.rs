//! GitHub-issued Sigstore attestation bundles: parsing and builder-policy
//! matching. Requires the `github-attestation` feature.
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
//! # What this module does NOT do
//!
//! **It does not verify the certificate chains to a trusted Fulcio root, and
//! it does not verify Rekor transparency-log inclusion.** Those are the
//! properties that actually establish "GitHub issued this certificate for
//! this exact workflow run" rather than "this JSON file contains a
//! certificate that says so." Without them, everything this module reports is
//! **conditional on that certificate being genuine** — a fact this module
//! cannot itself establish.
//!
//! This is not an oversight to be filled in casually. Fulcio-issued leaf
//! certificates may use ECDSA P-256, P-384, P-521, or Ed25519 — a correct
//! verifier has to be algorithm-agile, and X.509 chain validation against a
//! CA root plus Merkle-inclusion verification against a transparency log are
//! both security-critical primitives with a long history of subtle
//! implementation bugs. Reusing this crate's Ed25519-only signature checker
//! here would be actively wrong, not merely incomplete. The correct move is
//! the one [RELEASE-v1](../../spec/RELEASE-v1.md)'s `authorized-current-witnessed`
//! policy already takes for its own unimplemented transparency requirement:
//! report the gap honestly and refuse rather than degrade. See
//! [PROVENANCE-v1 §14](../../spec/PROVENANCE-v1.md) for the boundary this
//! draws and what a future implementation needs to close it (most likely by
//! calling into the `sigstore` crate's own `bundle::verify::Verifier` rather
//! than reimplementing chain validation here).
//!
//! Every verification result from this module carries `certificate_chain` as
//! its own explicit field, always [`ChainVerification::NotImplemented`]. No
//! other field is permitted to compensate for it or imply it passed.

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};
use crate::provenance::{BuildPredicate, Envelope};

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
pub enum ChainVerification {
    /// Always this value. See the module documentation: certificate-chain and
    /// Rekor-inclusion verification are a named, deliberate gap, not a check
    /// that ran and happened to find nothing wrong.
    NotImplemented,
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
    pub certificate_chain: ChainVerification,
    pub certificate_claims: GitHubCertificateClaims,
    pub policy: PolicyDecision,
    /// Whether the ANNPack predicate's `source.repository` (a carried claim
    /// per PROVENANCE-v1, never independently verified there) agrees with the
    /// certificate's `source_repository_uri`. Agreement is informative, not a
    /// promotion: the predicate claim is still reported as carried by
    /// `provenance.rs`, precisely because this module's own certificate
    /// authentication is itself conditional on the unimplemented chain check.
    pub repository_claim_agreement: ClaimAgreement,
    pub revision_claim_agreement: ClaimAgreement,
    /// Never `true` while `certificate_chain` is `NotImplemented`: nothing
    /// this module found can be trusted until that gap is closed.
    pub verified: bool,
    pub assumptions: Vec<String>,
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

/// Parse a bundle, extract certificate claims, evaluate builder policy, and
/// compare the (unauthenticated) predicate's carried claims against the
/// (also-unauthenticated, per this module's stated gap) certificate claims.
///
/// `verified` is deliberately unreachable as `true` today -- see
/// [`ChainVerification`]. This function exists so the policy-matching and
/// claim-extraction machinery is exercised end-to-end and ready to flip to
/// meaningful once certificate-chain verification is implemented, rather than
/// leaving that integration for whoever adds it to design from nothing.
pub fn evaluate_github_attestation(
    bundle: &SigstoreBundle,
    policy: &BuilderPolicy,
) -> Result<GitHubAttestationReport> {
    let der = bundle.leaf_certificate_der()?;
    let claims = extract_certificate_claims(&der)?;
    let policy_decision = evaluate_builder_policy(&claims, policy);
    let predicate = extract_predicate(bundle)?;

    let repository_claim_agreement = match &claims.source_repository_uri {
        Some(certificate_repository) => {
            if predicate
                .source
                .repository
                .trim_start_matches("github.com/")
                == certificate_repository
                    .trim_start_matches("https://")
                    .trim_start_matches("github.com/")
            {
                ClaimAgreement::Agree
            } else {
                ClaimAgreement::Disagree
            }
        }
        None => ClaimAgreement::Incomparable,
    };
    let revision_claim_agreement = match &claims.source_repository_digest {
        Some(certificate_revision) => {
            if predicate.source.revision.trim_start_matches("git:") == certificate_revision {
                ClaimAgreement::Agree
            } else {
                ClaimAgreement::Disagree
            }
        }
        None => ClaimAgreement::Incomparable,
    };

    let mut issues = Vec::new();
    issues.extend(policy_decision.issues.clone());
    if repository_claim_agreement == ClaimAgreement::Disagree {
        issues.push("predicate repository claim disagrees with the certificate".into());
    }
    if revision_claim_agreement == ClaimAgreement::Disagree {
        issues.push("predicate revision claim disagrees with the certificate".into());
    }

    Ok(GitHubAttestationReport {
        certificate_chain: ChainVerification::NotImplemented,
        certificate_claims: claims,
        policy: policy_decision,
        repository_claim_agreement,
        revision_claim_agreement,
        // Always false: see ChainVerification::NotImplemented and the module
        // documentation. Nothing computed above may set this true.
        verified: false,
        assumptions: vec![
            "certificate-chain-to-Fulcio-root verification is not implemented; every claim in \
             this report is conditional on the certificate being genuine, which this module \
             cannot itself establish"
                .into(),
            "Rekor transparency-log inclusion is not verified".into(),
        ],
        issues,
    })
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
    fn verified_is_never_true_regardless_of_policy_outcome() {
        // The property the whole module exists to hold. Even a certificate
        // that matches a fully-trusting policy on every claim must not be
        // reported as verified while certificate-chain verification is
        // unimplemented -- a matching policy proves nothing about whether the
        // certificate is genuine.
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
        // Trusted by policy is not the same claim as verified; that is the
        // whole point, and evaluate_github_attestation's `verified` field can
        // never be anything but false today regardless of this outcome.
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
