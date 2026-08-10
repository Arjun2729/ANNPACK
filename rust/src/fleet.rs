//! Fleet policy: what an organization requires, independent of what a
//! publisher offers.
//!
//! A [`crate::trust::TrustRoot`] answers who may publish. A
//! [`crate::policy::TrustPolicy`] answers what one verification call checked.
//! Neither answers a third question: does an organization's fleet of
//! verifiers agree on what to require. `FleetPolicy` is a signed, versioned
//! document an organization issues for that purpose, and
//! [`evaluate_compliance`] is how a verifier checks its local configuration
//! against it.
//!
//! ANNPack does not distribute fleet policy. Step 10a is the object and its
//! own verification; fetching a policy from a control plane is later work.
//! `evaluate_compliance` takes both the locally configured policy and the
//! required one as arguments, from wherever the caller obtained them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};
use crate::policy::TrustPolicy;
use crate::trust::KeyDescriptor;

pub const FLEET_POLICY_SCHEMA_V1: &str = "annpack-fleet-policy-v1";

const FLEET_POLICY_CONTEXT: &[u8] = b"ANNPACK3-FLEET-POLICY\0";

const MAX_KEYS: usize = 128;
const MAX_SIGNATURES: usize = 128;
const MAX_ALLOWED_PUBLISHERS: usize = 4096;
const MAX_ALLOWED_SCOPES: usize = 4096;
const MAX_DENY_INCIDENT_KINDS: usize = 32;

pub const MAX_FLEET_POLICY_FILE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScopeRule {
    pub corpus: String,
    pub channel: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FleetPolicySignature {
    pub key_id: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FleetPolicy {
    pub schema: String,
    /// Organization or security-domain identifier. Not a publisher; this is
    /// the consumer side.
    pub domain: String,
    /// Strictly increasing across rotations, like `TrustRoot.version`.
    pub revision: u64,
    pub issued_at: String,
    pub valid_until: String,
    pub threshold: u32,
    pub keys: BTreeMap<String, KeyDescriptor>,
    pub allowed_publishers: Vec<String>,
    pub allowed_scopes: Vec<ScopeRule>,
    pub required_verification_policy: TrustPolicy,
    /// Digest of the exact Sigsum trust-policy text a verifier must use for
    /// `authorized-current-witnessed`. `None` means the fleet does not pin a
    /// specific log/witness configuration beyond what `required_verification_policy`
    /// already implies.
    #[serde(default)]
    pub required_transparency_policy_digest: Option<String>,
    /// Digest of the workload-trust configuration required for run
    /// attestation verification. `None` means not pinned.
    #[serde(default)]
    pub required_workload_trust_digest: Option<String>,
    #[serde(default)]
    pub max_statement_validity_seconds: Option<u64>,
    /// `monitor::IncidentKind` values (as their serialized names) that must
    /// deny fleet use. Stored as strings, not the enum itself, so a fleet
    /// policy document remains readable by a verifier that predates a given
    /// incident kind rather than failing to parse.
    #[serde(default)]
    pub deny_on_incident_kinds: Vec<String>,
    #[serde(default)]
    pub signatures: Vec<FleetPolicySignature>,
}

#[derive(Serialize)]
struct SignedPayload<'a> {
    schema: &'a str,
    domain: &'a str,
    revision: u64,
    issued_at: &'a str,
    valid_until: &'a str,
    threshold: u32,
    keys: &'a BTreeMap<String, KeyDescriptor>,
    allowed_publishers: &'a [String],
    allowed_scopes: &'a [ScopeRule],
    required_verification_policy: TrustPolicy,
    required_transparency_policy_digest: &'a Option<String>,
    required_workload_trust_digest: &'a Option<String>,
    max_statement_validity_seconds: Option<u64>,
    deny_on_incident_kinds: &'a [String],
}

fn signed_payload_bytes(policy: &FleetPolicy) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&SignedPayload {
        schema: &policy.schema,
        domain: &policy.domain,
        revision: policy.revision,
        issued_at: &policy.issued_at,
        valid_until: &policy.valid_until,
        threshold: policy.threshold,
        keys: &policy.keys,
        allowed_publishers: &policy.allowed_publishers,
        allowed_scopes: &policy.allowed_scopes,
        required_verification_policy: policy.required_verification_policy,
        required_transparency_policy_digest: &policy.required_transparency_policy_digest,
        required_workload_trust_digest: &policy.required_workload_trust_digest,
        max_statement_validity_seconds: policy.max_statement_validity_seconds,
        deny_on_incident_kinds: &policy.deny_on_incident_kinds,
    })?)
}

pub fn fleet_policy_signing_message(policy: &FleetPolicy) -> Result<Vec<u8>> {
    let mut message = FLEET_POLICY_CONTEXT.to_vec();
    message.extend_from_slice(&signed_payload_bytes(policy)?);
    Ok(message)
}

pub fn fleet_policy_digest(policy: &FleetPolicy) -> Result<String> {
    Ok(blake3::hash(&signed_payload_bytes(policy)?)
        .to_hex()
        .to_string())
}

#[cfg(feature = "signing")]
pub fn sign_fleet_policy(policy: &mut FleetPolicy, secret_key: &[u8; 32]) -> Result<String> {
    use ed25519_dalek::{Signer, SigningKey};

    let signing_key = SigningKey::from_bytes(secret_key);
    let (key_id, _) = crate::trust::key_identity(secret_key);
    let message = fleet_policy_signing_message(policy)?;
    let signature = hex::encode(signing_key.sign(&message).to_bytes());
    policy.signatures.retain(|entry| entry.key_id != key_id);
    policy.signatures.push(FleetPolicySignature {
        key_id: key_id.clone(),
        signature,
    });
    Ok(key_id)
}

fn structural_issues(policy: &FleetPolicy) -> Vec<String> {
    let mut issues = Vec::new();
    if policy.keys.len() > MAX_KEYS {
        issues.push(format!("fleet policy declares more than {MAX_KEYS} keys"));
    }
    if policy.signatures.len() > MAX_SIGNATURES {
        issues.push(format!(
            "fleet policy carries more than {MAX_SIGNATURES} signatures"
        ));
    }
    if policy.allowed_publishers.len() > MAX_ALLOWED_PUBLISHERS {
        issues.push(format!(
            "fleet policy lists more than {MAX_ALLOWED_PUBLISHERS} allowed publishers"
        ));
    }
    if policy.allowed_scopes.len() > MAX_ALLOWED_SCOPES {
        issues.push(format!(
            "fleet policy lists more than {MAX_ALLOWED_SCOPES} allowed scopes"
        ));
    }
    if policy.deny_on_incident_kinds.len() > MAX_DENY_INCIDENT_KINDS {
        issues.push(format!(
            "fleet policy lists more than {MAX_DENY_INCIDENT_KINDS} deny-on incident kinds"
        ));
    }
    if policy.revision == 0 {
        issues.push("fleet policy revision must be at least 1".into());
    }
    if policy.domain.trim().is_empty() {
        issues.push("fleet policy names no domain".into());
    }
    if policy.threshold == 0 {
        issues.push("fleet policy has a zero threshold".into());
    }
    if policy.threshold as usize > policy.keys.len() {
        issues.push(format!(
            "fleet policy requires {} signatures but lists {} keys",
            policy.threshold,
            policy.keys.len()
        ));
    }
    issues
}

fn key_ids_match(policy: &FleetPolicy, issues: &mut Vec<String>) -> bool {
    let mut matched = true;
    for (key_id, key) in &policy.keys {
        let Ok(bytes) = hex::decode(&key.public_key) else {
            issues.push(format!("key {key_id} public key is not valid hex"));
            matched = false;
            continue;
        };
        if blake3::hash(&bytes).to_hex().to_string() != *key_id {
            issues.push(format!(
                "key {key_id} is filed under an id that is not its own digest"
            ));
            matched = false;
        }
        if key.algorithm != "Ed25519" {
            issues.push(format!(
                "key {key_id} uses unsupported algorithm {:?}",
                key.algorithm
            ));
            matched = false;
        }
    }
    matched
}

#[cfg(feature = "signing")]
fn valid_signers(policy: &FleetPolicy, authority: &FleetPolicy, message: &[u8]) -> Vec<String> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    use std::collections::BTreeSet;

    let mut signers = BTreeSet::new();
    for entry in &policy.signatures {
        let Some(key) = authority.keys.get(&entry.key_id) else {
            continue;
        };
        let (Ok(public_bytes), Ok(signature_bytes)) =
            (hex::decode(&key.public_key), hex::decode(&entry.signature))
        else {
            continue;
        };
        let (Ok(public_bytes), Ok(signature_bytes)): (
            std::result::Result<[u8; 32], _>,
            std::result::Result<[u8; 64], _>,
        ) = (public_bytes.try_into(), signature_bytes.try_into()) else {
            continue;
        };
        let Ok(verifying_key) = VerifyingKey::from_bytes(&public_bytes) else {
            continue;
        };
        if verifying_key
            .verify(message, &Signature::from_bytes(&signature_bytes))
            .is_ok()
        {
            signers.insert(entry.key_id.clone());
        }
    }
    signers.into_iter().collect()
}

#[cfg(not(feature = "signing"))]
fn valid_signers(_policy: &FleetPolicy, _authority: &FleetPolicy, _message: &[u8]) -> Vec<String> {
    Vec::new()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetPolicyVerification {
    pub domain: String,
    pub revision: u64,
    pub policy_digest: String,
    pub schema_supported: bool,
    pub structurally_valid: bool,
    pub key_ids_match_keys: bool,
    pub self_signed: bool,
    /// `None` on first contact.
    pub signed_by_prior: Option<bool>,
    /// `None` on first contact.
    pub revision_advanced: Option<bool>,
    /// `None` when no trusted clock was supplied.
    pub within_validity: Option<bool>,
    pub first_contact: bool,
    pub verified: bool,
    pub assumptions: Vec<String>,
    pub issues: Vec<String>,
}

/// Verify a fleet policy, optionally as a rotation from a prior trusted one.
/// Rotation requires a threshold of the successor's own keys and a threshold
/// of the prior policy's keys, exactly as `trust::verify_trust_root`
/// requires for the root role, and for the same reason: prior-only lets a
/// compromised old key install keys nobody controls, self-only lets anyone
/// mint a policy and present it.
pub fn verify_fleet_policy(
    policy: &FleetPolicy,
    prior: Option<&FleetPolicy>,
    now: Option<&str>,
) -> Result<FleetPolicyVerification> {
    let mut issues = Vec::new();
    let mut assumptions = Vec::new();

    let schema_supported = policy.schema == FLEET_POLICY_SCHEMA_V1;
    if !schema_supported {
        issues.push(format!(
            "fleet policy schema {:?}; this verifier supports {FLEET_POLICY_SCHEMA_V1}",
            policy.schema
        ));
    }

    let policy_digest = fleet_policy_digest(policy)?;
    let structural = structural_issues(policy);
    let structurally_valid = structural.is_empty();
    issues.extend(structural);

    let key_ids_match_keys = key_ids_match(policy, &mut issues);

    let message = fleet_policy_signing_message(policy)?;
    let evaluate_signatures = schema_supported && structurally_valid && key_ids_match_keys;

    let self_signed = if evaluate_signatures {
        let signers = valid_signers(policy, policy, &message);
        let met = signers.len() >= policy.threshold as usize;
        if !met {
            issues.push(format!(
                "fleet policy has {} valid signatures, needs {}",
                signers.len(),
                policy.threshold
            ));
        }
        met
    } else {
        false
    };

    let mut domain_matches = true;
    let (signed_by_prior, revision_advanced) = match prior {
        None => {
            assumptions.push(
                "no prior fleet policy supplied: accepted on first contact, indistinguishable \
                 from an attacker's policy"
                    .into(),
            );
            (None, None)
        }
        Some(prior) => {
            if prior.domain != policy.domain {
                domain_matches = false;
                issues.push(format!(
                    "rotation changes domain from {:?} to {:?}",
                    prior.domain, policy.domain
                ));
            }
            let advanced = policy.revision > prior.revision;
            if !advanced {
                issues.push(format!(
                    "fleet policy revision {} does not advance past the trusted revision {}",
                    policy.revision, prior.revision
                ));
            }
            let signed_by_prior = if evaluate_signatures {
                let signers = valid_signers(policy, prior, &message);
                let met = signers.len() >= prior.threshold as usize;
                if !met {
                    issues.push(format!(
                        "successor has {} signatures from the prior policy's keys, needs {}",
                        signers.len(),
                        prior.threshold
                    ));
                }
                met
            } else {
                false
            };
            (Some(signed_by_prior), Some(advanced))
        }
    };

    let within_validity = match now {
        None => {
            assumptions.push("no trusted clock supplied: expiry was not evaluated".into());
            None
        }
        Some(now) => {
            let now = crate::trust::parse_utc_timestamp(now)?;
            let issued = crate::trust::parse_utc_timestamp(&policy.issued_at)?;
            let expires = crate::trust::parse_utc_timestamp(&policy.valid_until)?;
            if expires <= issued {
                issues.push("fleet policy expires no later than it was issued".into());
            }
            let valid = now >= issued && now < expires;
            if !valid {
                issues.push(format!(
                    "fleet policy validity {}..{} does not contain the supplied time",
                    policy.issued_at, policy.valid_until
                ));
            }
            Some(valid)
        }
    };

    let verified = schema_supported
        && structurally_valid
        && key_ids_match_keys
        && self_signed
        && domain_matches
        && signed_by_prior.unwrap_or(true)
        && revision_advanced.unwrap_or(true)
        && within_validity.unwrap_or(false);

    Ok(FleetPolicyVerification {
        domain: policy.domain.clone(),
        revision: policy.revision,
        policy_digest,
        schema_supported,
        structurally_valid,
        key_ids_match_keys,
        self_signed,
        signed_by_prior,
        revision_advanced,
        within_validity,
        first_contact: prior.is_none(),
        verified,
        assumptions,
        issues,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplianceStatus {
    Compliant,
    Drifted,
    /// Either input did not verify, or was not supplied. Never treated as
    /// compliant -- an absent or broken input says nothing about compliance.
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetComplianceReport {
    pub domain: String,
    pub local_revision: Option<u64>,
    pub required_revision: Option<u64>,
    pub local_policy_digest: Option<String>,
    pub required_policy_digest: Option<String>,
    pub status: ComplianceStatus,
    pub issues: Vec<String>,
}

/// Compare a locally configured, already-verified fleet policy against the
/// policy that should be in effect. Both are re-verified here rather than
/// trusted from a caller-supplied bool, so a caller cannot short-circuit
/// compliance by passing an unverified document with `verified: true` typed
/// into a report by hand.
pub fn evaluate_compliance(
    local: Option<&FleetPolicy>,
    required: Option<&FleetPolicy>,
    now: Option<&str>,
) -> Result<FleetComplianceReport> {
    let local_verification = local
        .map(|policy| verify_fleet_policy(policy, None, now))
        .transpose()?;
    let required_verification = required
        .map(|policy| verify_fleet_policy(policy, None, now))
        .transpose()?;

    let mut issues = Vec::new();
    let domain = required
        .map(|policy| policy.domain.clone())
        .or_else(|| local.map(|policy| policy.domain.clone()))
        .unwrap_or_default();

    let (Some(local), Some(local_verification)) = (local, &local_verification) else {
        issues.push("no locally configured fleet policy supplied".into());
        return Ok(FleetComplianceReport {
            domain,
            local_revision: None,
            required_revision: required.map(|p| p.revision),
            local_policy_digest: None,
            required_policy_digest: required_verification
                .as_ref()
                .map(|v| v.policy_digest.clone()),
            status: ComplianceStatus::Unavailable,
            issues,
        });
    };
    let (Some(required), Some(required_verification)) = (required, &required_verification) else {
        issues.push("no required fleet policy supplied".into());
        return Ok(FleetComplianceReport {
            domain,
            local_revision: Some(local.revision),
            required_revision: None,
            local_policy_digest: Some(local_verification.policy_digest.clone()),
            required_policy_digest: None,
            status: ComplianceStatus::Unavailable,
            issues,
        });
    };

    if local.domain != required.domain {
        return Err(AnnpackError::InvalidInput(format!(
            "local fleet policy domain {:?} does not match required fleet policy domain {:?}",
            local.domain, required.domain
        )));
    }

    if !local_verification.verified {
        issues.push("locally configured fleet policy did not verify".into());
    }
    if !required_verification.verified {
        issues.push("required fleet policy did not verify".into());
    }

    let status = if !local_verification.verified || !required_verification.verified {
        ComplianceStatus::Unavailable
    } else if local.revision == required.revision
        && local_verification.policy_digest == required_verification.policy_digest
    {
        ComplianceStatus::Compliant
    } else {
        ComplianceStatus::Drifted
    };

    Ok(FleetComplianceReport {
        domain,
        local_revision: Some(local.revision),
        required_revision: Some(required.revision),
        local_policy_digest: Some(local_verification.policy_digest.clone()),
        required_policy_digest: Some(required_verification.policy_digest.clone()),
        status,
        issues,
    })
}

#[cfg(all(test, feature = "signing"))]
mod tests {
    use super::*;

    fn keypair(seed: u8) -> ([u8; 32], [u8; 32]) {
        use ed25519_dalek::SigningKey;
        let secret = [seed; 32];
        let key = SigningKey::from_bytes(&secret);
        (key.verifying_key().to_bytes(), secret)
    }

    fn base_policy(domain: &str, revision: u64, public: &[u8; 32]) -> FleetPolicy {
        let mut keys = BTreeMap::new();
        // Note: `key_identity` takes a *secret* key and derives the public
        // key from it; `public` here already is the public key, so the id is
        // computed directly rather than by (mis)calling `key_identity(public)`.
        let id = blake3::hash(public).to_hex().to_string();
        keys.insert(
            id,
            KeyDescriptor {
                algorithm: "Ed25519".into(),
                public_key: hex::encode(public),
            },
        );
        FleetPolicy {
            schema: FLEET_POLICY_SCHEMA_V1.into(),
            domain: domain.into(),
            revision,
            issued_at: "2026-08-09T00:00:00Z".into(),
            valid_until: "2099-01-01T00:00:00Z".into(),
            threshold: 1,
            keys,
            allowed_publishers: vec!["example.com".into()],
            allowed_scopes: vec![ScopeRule {
                corpus: "support-manual".into(),
                channel: "production".into(),
            }],
            required_verification_policy: TrustPolicy::AuthorizedCurrent,
            required_transparency_policy_digest: None,
            required_workload_trust_digest: None,
            max_statement_validity_seconds: None,
            deny_on_incident_kinds: vec!["equivocation".into()],
            signatures: Vec::new(),
        }
    }

    #[test]
    fn a_self_signed_policy_verifies_on_first_contact() {
        let (public, secret) = keypair(1);
        let mut policy = base_policy("acme.example", 1, &public);
        sign_fleet_policy(&mut policy, &secret).unwrap();

        let report = verify_fleet_policy(&policy, None, Some("2026-08-09T01:00:00Z")).unwrap();
        assert!(report.verified, "{:?}", report.issues);
        assert!(report.first_contact);
        assert_eq!(report.signed_by_prior, None);
    }

    #[test]
    fn an_unsigned_policy_does_not_verify() {
        let (public, _) = keypair(1);
        let policy = base_policy("acme.example", 1, &public);
        let report = verify_fleet_policy(&policy, None, Some("2026-08-09T01:00:00Z")).unwrap();
        assert!(!report.verified);
        assert!(!report.self_signed);
    }

    #[test]
    fn rotation_requires_both_self_and_prior_signatures() {
        let (old_public, old_secret) = keypair(1);
        let mut prior = base_policy("acme.example", 1, &old_public);
        sign_fleet_policy(&mut prior, &old_secret).unwrap();

        let (new_public, new_secret) = keypair(2);
        let mut successor = base_policy("acme.example", 2, &new_public);
        sign_fleet_policy(&mut successor, &new_secret).unwrap();
        // Self-signed only: no signature from the prior policy's keys.
        let report =
            verify_fleet_policy(&successor, Some(&prior), Some("2026-08-09T01:00:00Z")).unwrap();
        assert!(!report.verified, "{:?}", report.issues);
        assert_eq!(report.signed_by_prior, Some(false));

        // Add the prior key's signature over the successor -- both thresholds met.
        sign_fleet_policy(&mut successor, &old_secret).unwrap();
        let report =
            verify_fleet_policy(&successor, Some(&prior), Some("2026-08-09T01:00:00Z")).unwrap();
        assert!(report.verified, "{:?}", report.issues);
        assert_eq!(report.signed_by_prior, Some(true));
    }

    #[test]
    fn a_non_advancing_revision_does_not_verify() {
        let (public, secret) = keypair(1);
        let mut prior = base_policy("acme.example", 2, &public);
        sign_fleet_policy(&mut prior, &secret).unwrap();

        let mut same_revision = base_policy("acme.example", 2, &public);
        sign_fleet_policy(&mut same_revision, &secret).unwrap();

        let report =
            verify_fleet_policy(&same_revision, Some(&prior), Some("2026-08-09T01:00:00Z"))
                .unwrap();
        assert!(!report.verified);
        assert_eq!(report.revision_advanced, Some(false));
    }

    #[test]
    fn rotation_across_a_different_domain_is_flagged() {
        let (public, secret) = keypair(1);
        let mut prior = base_policy("acme.example", 1, &public);
        sign_fleet_policy(&mut prior, &secret).unwrap();

        let mut successor = base_policy("other.example", 2, &public);
        sign_fleet_policy(&mut successor, &secret).unwrap();

        let report =
            verify_fleet_policy(&successor, Some(&prior), Some("2026-08-09T01:00:00Z")).unwrap();
        assert!(!report.verified, "{:?}", report.issues);
        assert!(report.issues.iter().any(|issue| issue.contains("domain")));
    }

    #[test]
    fn no_clock_means_unknown_validity_and_no_verification() {
        let (public, secret) = keypair(1);
        let mut policy = base_policy("acme.example", 1, &public);
        sign_fleet_policy(&mut policy, &secret).unwrap();

        let report = verify_fleet_policy(&policy, None, None).unwrap();
        assert_eq!(report.within_validity, None);
        assert!(!report.verified);
    }

    fn verified_policy(domain: &str, revision: u64) -> (FleetPolicy, [u8; 32]) {
        let (public, secret) = keypair(revision as u8 + 10);
        let mut policy = base_policy(domain, revision, &public);
        sign_fleet_policy(&mut policy, &secret).unwrap();
        (policy, secret)
    }

    #[test]
    fn matching_revision_and_digest_is_compliant() {
        let (local, _) = verified_policy("acme.example", 3);
        let required = local.clone();
        let report =
            evaluate_compliance(Some(&local), Some(&required), Some("2026-08-09T01:00:00Z"))
                .unwrap();
        assert_eq!(
            report.status,
            ComplianceStatus::Compliant,
            "{:?}",
            report.issues
        );
    }

    #[test]
    fn a_lower_local_revision_is_drifted() {
        let (local, _) = verified_policy("acme.example", 2);
        let (required, _) = verified_policy("acme.example", 3);
        let report =
            evaluate_compliance(Some(&local), Some(&required), Some("2026-08-09T01:00:00Z"))
                .unwrap();
        assert_eq!(report.status, ComplianceStatus::Drifted);
        assert_eq!(report.local_revision, Some(2));
        assert_eq!(report.required_revision, Some(3));
    }

    #[test]
    fn same_revision_different_content_is_drifted_not_compliant() {
        // Same revision number, but the documents genuinely differ -- must
        // not be treated as compliant just because the revision matches.
        let (public, secret) = keypair(20);
        let mut local = base_policy("acme.example", 5, &public);
        sign_fleet_policy(&mut local, &secret).unwrap();
        let mut required = base_policy("acme.example", 5, &public);
        required.required_verification_policy = TrustPolicy::AuthorizedCurrentWitnessed;
        sign_fleet_policy(&mut required, &secret).unwrap();

        let report =
            evaluate_compliance(Some(&local), Some(&required), Some("2026-08-09T01:00:00Z"))
                .unwrap();
        assert_eq!(report.status, ComplianceStatus::Drifted);
    }

    #[test]
    fn a_missing_local_policy_is_unavailable_not_compliant() {
        let (required, _) = verified_policy("acme.example", 1);
        let report =
            evaluate_compliance(None, Some(&required), Some("2026-08-09T01:00:00Z")).unwrap();
        assert_eq!(report.status, ComplianceStatus::Unavailable);
        assert_eq!(report.local_revision, None);
    }

    #[test]
    fn a_missing_required_policy_is_unavailable_not_compliant() {
        let (local, _) = verified_policy("acme.example", 1);
        let report = evaluate_compliance(Some(&local), None, Some("2026-08-09T01:00:00Z")).unwrap();
        assert_eq!(report.status, ComplianceStatus::Unavailable);
        assert_eq!(report.required_revision, None);
    }

    #[test]
    fn an_unverified_local_policy_is_unavailable_not_compliant() {
        let (public, _) = keypair(1);
        let unsigned_local = base_policy("acme.example", 1, &public);
        let (required, _) = verified_policy("acme.example", 1);
        let report = evaluate_compliance(
            Some(&unsigned_local),
            Some(&required),
            Some("2026-08-09T01:00:00Z"),
        )
        .unwrap();
        assert_eq!(report.status, ComplianceStatus::Unavailable);
    }

    #[test]
    fn mismatched_domains_are_a_hard_error_not_a_drift_verdict() {
        let (local, _) = verified_policy("acme.example", 1);
        let (required, _) = verified_policy("other.example", 1);
        let error =
            evaluate_compliance(Some(&local), Some(&required), Some("2026-08-09T01:00:00Z"))
                .unwrap_err();
        assert!(error.to_string().contains("domain"));
    }
}
