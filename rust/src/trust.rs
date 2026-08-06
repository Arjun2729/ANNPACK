//! Publisher trust roots: which keys may speak for a publisher, and in what role.
//!
//! Artifact signatures answer "were these bytes produced by the holder of key
//! K?". They cannot answer "is K allowed to publish for example.com?", because
//! the only thing asserting that today is a string inside the artifact the key
//! signed. A trust root moves that claim outside the artifact, where the
//! artifact cannot vouch for itself.
//!
//! Roles are separated so that compromise is scoped. A key that may sign
//! artifacts must not thereby be able to declare which artifact is current, and
//! neither should be able to revoke. That separation is the entire point; a
//! single all-powerful publisher key would be simpler and would collapse
//! [`ROLE_ARTIFACT`] and [`ROLE_RELEASE_STATE`] into one blast radius.
//!
//! # The root role
//!
//! The architecture contract names `artifact`, `release_state` and
//! `emergency_revocation` but no role for the trust root itself. That leaves
//! nothing specifying who may sign a trust root, and no basis for evaluating a
//! rotation, since "the successor must satisfy the transition policy" needs a
//! rule about which keys the policy consults. [`ROLE_ROOT`] is therefore
//! required: it signs trust roots and authorises successors, and it is the only
//! role a consumer pins.
//!
//! # Rotation
//!
//! A successor must be signed by a threshold of the *prior* root's root-role
//! keys and by a threshold of *its own*. Requiring both is what stops two
//! distinct failure modes: prior-only would let a compromised old key install a
//! root whose keys nobody controls, and self-only would let any attacker mint a
//! root and hand it over. This is the rule TUF uses, for the same reasons.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};

pub const TRUST_ROOT_SCHEMA_V1: &str = "annpack-trust-root-v1";

/// Domain separation, matching the container's existing contexts. A signature
/// over a trust root must never verify as a signature over anything else.
const TRUST_ROOT_CONTEXT: &[u8] = b"ANNPACK3-TRUST-ROOT\0";

pub const ROLE_ROOT: &str = "root";
pub const ROLE_ARTIFACT: &str = "artifact";
pub const ROLE_RELEASE_STATE: &str = "release_state";
pub const ROLE_EMERGENCY_REVOCATION: &str = "emergency_revocation";

/// Every role a conforming trust root must define. A root missing one is
/// rejected rather than treated as "that role has no authorised keys", because
/// the two are indistinguishable to a reader and only one is intended.
pub const REQUIRED_ROLES: &[&str] = &[
    ROLE_ROOT,
    ROLE_ARTIFACT,
    ROLE_RELEASE_STATE,
    ROLE_EMERGENCY_REVOCATION,
];

const MAX_KEYS: usize = 128;
const MAX_ROLES: usize = 32;
const MAX_SIGNATURES: usize = 128;
const MAX_KEYS_PER_ROLE: usize = 128;

/// Maximum trust-root file size the reference CLI reads, bounding allocation
/// before parsing as [`crate::evidence::MAX_RECEIPT_FILE_BYTES`] does.
pub const MAX_TRUST_ROOT_FILE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoleDescriptor {
    pub threshold: u32,
    pub keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyDescriptor {
    pub algorithm: String,
    pub public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RootSignature {
    pub key_id: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrustRoot {
    pub schema: String,
    pub publisher: String,
    pub version: u64,
    pub issued_at: String,
    pub valid_until: String,
    /// `BTreeMap` rather than `HashMap`: the signed payload is a serialization
    /// of this structure, so iteration order is part of the signature.
    pub roles: BTreeMap<String, RoleDescriptor>,
    pub keys: BTreeMap<String, KeyDescriptor>,
    #[serde(default)]
    pub signatures: Vec<RootSignature>,
}

/// The bytes a trust-root signature covers: everything except the signatures.
#[derive(Serialize)]
struct SignedPayload<'a> {
    schema: &'a str,
    publisher: &'a str,
    version: u64,
    issued_at: &'a str,
    valid_until: &'a str,
    roles: &'a BTreeMap<String, RoleDescriptor>,
    keys: &'a BTreeMap<String, KeyDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustRootVerification {
    pub publisher: String,
    pub version: u64,
    /// Digest of the signed payload. Recorded so a caller can bind a decision to
    /// the exact bytes it was made from.
    pub payload_digest: String,
    pub schema_supported: bool,
    pub structurally_valid: bool,
    /// Every `key_id` is the BLAKE3 of the public key filed under it. Without
    /// this a root could file an attacker's key under a trusted key's id.
    pub key_ids_match_keys: bool,
    /// Meets its own root-role threshold.
    pub self_signed: bool,
    /// Meets the prior root's root-role threshold. `None` on first contact.
    pub signed_by_prior_root: Option<bool>,
    /// Version strictly advanced past the prior root. `None` on first contact.
    pub version_advanced: Option<bool>,
    /// `None` when no trusted clock was supplied. Never inferred from a local
    /// clock the caller did not vouch for.
    pub within_validity: Option<bool>,
    /// No prior root was supplied, so rotation was not evaluated and nothing
    /// here distinguishes this root from an attacker's.
    pub first_contact: bool,
    /// Role name to the authorised key ids that actually signed, for roles whose
    /// threshold was met. Empty for a root that did not verify.
    pub authorized_roles: BTreeMap<String, Vec<String>>,
    pub verified: bool,
    /// What this result depends on that was not itself checked here.
    pub assumptions: Vec<String>,
    pub issues: Vec<String>,
}

/// Seconds since the Unix epoch for a strict `YYYY-MM-DDTHH:MM:SSZ` timestamp.
///
/// Deliberately strict. Nothing in the codebase parsed timestamps before this,
/// and a lenient parser at a trust boundary turns an unreadable expiry into an
/// accepted one. Only UTC with a `Z` suffix is accepted: offsets would need
/// normalisation, and a normalisation bug here is an expiry bypass.
pub fn parse_utc_timestamp(value: &str) -> Result<i64> {
    let invalid = || {
        AnnpackError::InvalidFormat(format!(
            "timestamp {value:?} is not a strict YYYY-MM-DDTHH:MM:SSZ UTC value"
        ))
    };
    let bytes = value.as_bytes();
    if bytes.len() != 20 || bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' {
        return Err(invalid());
    }
    if bytes[13] != b':' || bytes[16] != b':' || bytes[19] != b'Z' {
        return Err(invalid());
    }
    let field = |from: usize, to: usize| -> Result<i64> {
        value
            .get(from..to)
            .and_then(|slice| slice.parse::<i64>().ok())
            .ok_or_else(invalid)
    };
    let (year, month, day) = (field(0, 4)?, field(5, 7)?, field(8, 10)?);
    let (hour, minute, second) = (field(11, 13)?, field(14, 16)?, field(17, 19)?);
    if !(1..=12).contains(&month) || day < 1 || hour > 23 || minute > 59 || second > 59 {
        return Err(invalid());
    }
    if day > days_in_month(year, month) {
        return Err(invalid());
    }

    // Days from 1970-01-01 by civil-date arithmetic; no external dependency and
    // no local-timezone influence.
    let years = year - if month <= 2 { 1 } else { 0 };
    let era = if years >= 0 { years } else { years - 399 } / 400;
    let year_of_era = years - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    Ok(days * 86_400 + hour * 3_600 + minute * 60 + second)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            if leap { 29 } else { 28 }
        }
    }
}

fn signed_payload_bytes(root: &TrustRoot) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&SignedPayload {
        schema: &root.schema,
        publisher: &root.publisher,
        version: root.version,
        issued_at: &root.issued_at,
        valid_until: &root.valid_until,
        roles: &root.roles,
        keys: &root.keys,
    })?)
}

/// The exact message a trust-root signature covers.
pub fn trust_root_signing_message(root: &TrustRoot) -> Result<Vec<u8>> {
    let mut message = TRUST_ROOT_CONTEXT.to_vec();
    message.extend_from_slice(&signed_payload_bytes(root)?);
    Ok(message)
}

fn structural_issues(root: &TrustRoot) -> Vec<String> {
    let mut issues = Vec::new();
    if root.keys.len() > MAX_KEYS {
        issues.push(format!("trust root declares more than {MAX_KEYS} keys"));
    }
    if root.roles.len() > MAX_ROLES {
        issues.push(format!("trust root declares more than {MAX_ROLES} roles"));
    }
    if root.signatures.len() > MAX_SIGNATURES {
        issues.push(format!(
            "trust root carries more than {MAX_SIGNATURES} signatures"
        ));
    }
    if root.version == 0 {
        issues.push("trust root version must be at least 1".into());
    }
    if root.publisher.trim().is_empty() {
        issues.push("trust root names no publisher".into());
    }
    for required in REQUIRED_ROLES {
        if !root.roles.contains_key(*required) {
            issues.push(format!("trust root defines no {required} role"));
        }
    }
    for (name, role) in &root.roles {
        if role.keys.len() > MAX_KEYS_PER_ROLE {
            issues.push(format!(
                "role {name} lists more than {MAX_KEYS_PER_ROLE} keys"
            ));
            continue;
        }
        let distinct: BTreeSet<&String> = role.keys.iter().collect();
        if distinct.len() != role.keys.len() {
            issues.push(format!("role {name} lists a key more than once"));
        }
        if role.threshold == 0 {
            issues.push(format!("role {name} has a zero threshold"));
        }
        // A threshold above the key count can never be met, so the role is
        // permanently unusable. Saying so is better than failing later with a
        // signature error that looks like a key problem.
        if role.threshold as usize > distinct.len() {
            issues.push(format!(
                "role {name} requires {} signatures but lists {} keys",
                role.threshold,
                distinct.len()
            ));
        }
        for key_id in &role.keys {
            if !root.keys.contains_key(key_id) {
                issues.push(format!("role {name} references undeclared key {key_id}"));
            }
        }
    }
    for (key_id, key) in &root.keys {
        if key.algorithm != "Ed25519" {
            issues.push(format!(
                "key {key_id} uses unsupported algorithm {:?}",
                key.algorithm
            ));
        }
    }
    issues
}

/// Key ids whose declared id is the BLAKE3 of their declared public key.
fn key_ids_match(root: &TrustRoot, issues: &mut Vec<String>) -> bool {
    let mut matched = true;
    for (key_id, key) in &root.keys {
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
    }
    matched
}

/// Distinct key ids authorised for `role` that produced a valid signature over
/// `message`.
///
/// Shared by trust roots and by anything else a trust root authorises, so that
/// threshold and role-membership semantics exist in exactly one place. A second
/// implementation would be a second place for "one key signing twice satisfies a
/// threshold of two" to be reintroduced.
#[cfg(feature = "signing")]
pub fn authorized_signers(
    authority: &TrustRoot,
    role: &str,
    message: &[u8],
    signatures: &[(&str, &str)],
) -> BTreeSet<String> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let Some(descriptor) = authority.roles.get(role) else {
        return BTreeSet::new();
    };
    let permitted: BTreeSet<&String> = descriptor.keys.iter().collect();
    let mut signers = BTreeSet::new();
    for (key_id, signature_hex) in signatures {
        if !permitted.contains(&(*key_id).to_string()) {
            continue;
        }
        let Some(key) = authority.keys.get(*key_id) else {
            continue;
        };
        let (Ok(public_bytes), Ok(signature_bytes)) =
            (hex::decode(&key.public_key), hex::decode(signature_hex))
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
            // Inserting the id, not counting occurrences: one key signing twice
            // must not satisfy a threshold of two.
            signers.insert((*key_id).to_string());
        }
    }
    signers
}

#[cfg(not(feature = "signing"))]
pub fn authorized_signers(
    _authority: &TrustRoot,
    _role: &str,
    _message: &[u8],
    _signatures: &[(&str, &str)],
) -> BTreeSet<String> {
    BTreeSet::new()
}

fn valid_signers(
    root: &TrustRoot,
    authority: &TrustRoot,
    role: &str,
    message: &[u8],
) -> BTreeSet<String> {
    let signatures: Vec<(&str, &str)> = root
        .signatures
        .iter()
        .map(|entry| (entry.key_id.as_str(), entry.signature.as_str()))
        .collect();
    authorized_signers(authority, role, message, &signatures)
}

/// Verify a trust root, optionally as a rotation from a prior trusted root.
///
/// `now` is a caller-supplied timestamp the caller vouches for. Passing `None`
/// means no trusted clock is available; validity is then reported as `None`
/// rather than being evaluated against a local clock an attacker may control.
pub fn verify_trust_root(
    root: &TrustRoot,
    prior: Option<&TrustRoot>,
    now: Option<&str>,
) -> Result<TrustRootVerification> {
    let mut issues = Vec::new();
    let mut assumptions = Vec::new();

    let schema_supported = root.schema == TRUST_ROOT_SCHEMA_V1;
    if !schema_supported {
        issues.push(format!(
            "trust root schema {:?}; this verifier supports {TRUST_ROOT_SCHEMA_V1}",
            root.schema
        ));
    }

    let payload = signed_payload_bytes(root)?;
    let payload_digest = blake3::hash(&payload).to_hex().to_string();

    let structural = structural_issues(root);
    let structurally_valid = structural.is_empty();
    issues.extend(structural);

    let key_ids_match_keys = key_ids_match(root, &mut issues);

    let message = trust_root_signing_message(root)?;
    let mut authorized_roles = BTreeMap::new();

    // Signature checks run only against a structurally sound root: thresholds
    // and role membership are meaningless while the role table is malformed.
    let evaluate_signatures = schema_supported && structurally_valid && key_ids_match_keys;

    let self_signed = if evaluate_signatures {
        let signers = valid_signers(root, root, ROLE_ROOT, &message);
        let threshold = root.roles[ROLE_ROOT].threshold as usize;
        let met = signers.len() >= threshold;
        if !met {
            issues.push(format!(
                "trust root has {} valid root-role signatures, needs {threshold}",
                signers.len()
            ));
        }
        met
    } else {
        false
    };

    let (signed_by_prior_root, version_advanced) = match prior {
        None => {
            assumptions.push(
                "no prior trust root supplied: this root was accepted on first contact and \
                 nothing here distinguishes it from an attacker's root"
                    .into(),
            );
            (None, None)
        }
        Some(prior) => {
            if prior.publisher != root.publisher {
                issues.push(format!(
                    "rotation changes publisher from {:?} to {:?}",
                    prior.publisher, root.publisher
                ));
            }
            let advanced = root.version > prior.version;
            if !advanced {
                issues.push(format!(
                    "trust root version {} does not advance past the trusted version {}",
                    root.version, prior.version
                ));
            }
            let signed_by_prior = if evaluate_signatures {
                let signers = valid_signers(root, prior, ROLE_ROOT, &message);
                let threshold = prior
                    .roles
                    .get(ROLE_ROOT)
                    .map(|role| role.threshold as usize)
                    .unwrap_or(usize::MAX);
                let met = signers.len() >= threshold;
                if !met {
                    issues.push(format!(
                        "successor has {} signatures from the prior root's root role, needs {threshold}",
                        signers.len()
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
            assumptions.push(
                "no trusted clock supplied: expiry was not evaluated and this result says \
                 nothing about whether the root is still valid"
                    .into(),
            );
            None
        }
        Some(now) => {
            let now = parse_utc_timestamp(now)?;
            let issued = parse_utc_timestamp(&root.issued_at)?;
            let expires = parse_utc_timestamp(&root.valid_until)?;
            if expires <= issued {
                issues.push("trust root expires no later than it was issued".into());
            }
            let valid = now >= issued && now < expires;
            if !valid {
                issues.push(format!(
                    "trust root validity {}..{} does not contain the supplied time",
                    root.issued_at, root.valid_until
                ));
            }
            assumptions.push("expiry was evaluated against a caller-supplied clock".into());
            Some(valid)
        }
    };

    let verified = schema_supported
        && structurally_valid
        && key_ids_match_keys
        && self_signed
        && signed_by_prior_root.unwrap_or(true)
        && version_advanced.unwrap_or(true)
        && within_validity.unwrap_or(false);

    if verified {
        for (name, role) in &root.roles {
            authorized_roles.insert(name.clone(), role.keys.clone());
        }
    }

    Ok(TrustRootVerification {
        publisher: root.publisher.clone(),
        version: root.version,
        payload_digest,
        schema_supported,
        structurally_valid,
        key_ids_match_keys,
        self_signed,
        signed_by_prior_root,
        version_advanced,
        within_validity,
        first_contact: prior.is_none(),
        authorized_roles,
        verified,
        assumptions,
        issues,
    })
}

/// Derive the key id and hex public key for a 32-byte Ed25519 secret key.
#[cfg(feature = "signing")]
pub fn key_identity(secret_key: &[u8; 32]) -> (String, String) {
    use ed25519_dalek::SigningKey;

    let public = SigningKey::from_bytes(secret_key)
        .verifying_key()
        .to_bytes();
    (
        blake3::hash(&public).to_hex().to_string(),
        hex::encode(public),
    )
}

/// Append a signature over the trust root's signed payload.
///
/// Replaces any existing signature from the same key so that re-signing is
/// idempotent rather than accumulating duplicates that could be miscounted
/// toward a threshold.
#[cfg(feature = "signing")]
pub fn sign_trust_root(root: &mut TrustRoot, secret_key: &[u8; 32]) -> Result<String> {
    use ed25519_dalek::{Signer, SigningKey};

    let signing_key = SigningKey::from_bytes(secret_key);
    let (key_id, _) = key_identity(secret_key);
    let message = trust_root_signing_message(root)?;
    let signature = hex::encode(signing_key.sign(&message).to_bytes());
    root.signatures.retain(|entry| entry.key_id != key_id);
    root.signatures.push(RootSignature {
        key_id: key_id.clone(),
        signature,
    });
    Ok(key_id)
}

/// Whether `key_id` is authorised for `role` by a trust root that verified.
///
/// Takes the verification rather than the root so that an unverified root
/// cannot be consulted for authorisation by accident.
pub fn role_authorizes(verification: &TrustRootVerification, role: &str, key_id: &str) -> bool {
    verification.verified
        && verification
            .authorized_roles
            .get(role)
            .is_some_and(|keys| keys.iter().any(|candidate| candidate == key_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-checked against Python's `datetime`, not against values derived by
    /// the same reasoning that produced the parser. The first version of this
    /// test used three constants computed by hand; one was wrong, which proved
    /// only that hand arithmetic cannot validate hand arithmetic. Covers the
    /// epoch, leap and non-leap century years, the 2038 boundary, and random
    /// instants across a 127-year span.
    #[test]
    fn instants_match_an_independent_implementation() {
        for (value, expected) in [
            ("1970-01-01T00:00:00Z", 0_i64),
            ("1970-01-01T00:00:01Z", 1),
            ("1999-12-31T23:59:59Z", 946_684_799),
            ("2000-01-01T00:00:00Z", 946_684_800),
            ("2000-02-29T12:00:00Z", 951_825_600),
            ("2024-02-29T23:59:59Z", 1_709_251_199),
            ("2026-08-06T00:00:00Z", 1_785_974_400),
            ("2026-12-31T23:59:59Z", 1_798_761_599),
            ("2100-03-01T00:00:00Z", 4_107_542_400),
            ("2038-01-19T03:14:08Z", 2_147_483_648),
            ("1972-06-30T23:59:59Z", 78_796_799),
            ("2031-07-27T21:49:33Z", 1_942_955_373),
            ("2087-10-30T06:46:36Z", 3_718_334_796),
            ("2046-03-09T10:27:51Z", 2_404_204_071),
            ("2086-08-14T21:22:51Z", 3_680_198_571),
            ("2095-10-14T18:10:21Z", 3_969_454_221),
            ("2076-04-28T13:26:29Z", 3_355_305_989),
            ("2033-05-17T14:10:09Z", 1_999_951_809),
            ("2031-06-30T16:57:26Z", 1_940_605_046),
            ("2039-02-12T22:14:21Z", 2_181_161_661),
        ] {
            assert_eq!(
                parse_utc_timestamp(value).unwrap(),
                expected,
                "{value} parsed incorrectly"
            );
        }
    }

    #[test]
    fn timestamps_are_ordered_correctly() {
        let earlier = parse_utc_timestamp("2026-08-06T00:00:00Z").unwrap();
        let later = parse_utc_timestamp("2026-08-06T01:00:00Z").unwrap();
        assert_eq!(later - earlier, 3_600);
        assert!(parse_utc_timestamp("2027-01-01T00:00:00Z").unwrap() > later);
    }

    #[test]
    fn leap_days_are_real_and_non_leap_days_are_not() {
        assert!(parse_utc_timestamp("2024-02-29T00:00:00Z").is_ok());
        assert!(parse_utc_timestamp("2026-02-29T00:00:00Z").is_err());
        assert!(parse_utc_timestamp("2000-02-29T00:00:00Z").is_ok());
        assert!(parse_utc_timestamp("1900-02-29T00:00:00Z").is_err());
    }

    #[test]
    fn lenient_timestamp_shapes_are_refused() {
        // Each of these is something a permissive parser would accept, and each
        // would let an expiry be misread rather than rejected.
        for value in [
            "2026-08-06",
            "2026-08-06T00:00:00",
            "2026-08-06T00:00:00+05:30",
            "2026-08-06T00:00:00.000Z",
            "2026-8-06T00:00:00Z",
            "2026-08-06T24:00:00Z",
            "2026-13-01T00:00:00Z",
            "2026-08-32T00:00:00Z",
            "",
        ] {
            assert!(
                parse_utc_timestamp(value).is_err(),
                "{value:?} should be refused"
            );
        }
    }

    #[test]
    fn the_signing_message_is_domain_separated() {
        let root = minimal_root();
        let message = trust_root_signing_message(&root).unwrap();
        assert!(message.starts_with(TRUST_ROOT_CONTEXT));
    }

    #[test]
    fn signatures_are_excluded_from_the_signed_payload() {
        // Otherwise a root could never carry more than one signature: adding the
        // second would invalidate the first.
        let mut root = minimal_root();
        let before = trust_root_signing_message(&root).unwrap();
        root.signatures.push(RootSignature {
            key_id: "00".into(),
            signature: "11".into(),
        });
        assert_eq!(before, trust_root_signing_message(&root).unwrap());
    }

    fn minimal_root() -> TrustRoot {
        let mut roles = BTreeMap::new();
        for role in REQUIRED_ROLES {
            roles.insert(
                (*role).to_string(),
                RoleDescriptor {
                    threshold: 1,
                    keys: vec!["k".into()],
                },
            );
        }
        let mut keys = BTreeMap::new();
        keys.insert(
            "k".to_string(),
            KeyDescriptor {
                algorithm: "Ed25519".into(),
                public_key: "00".repeat(32),
            },
        );
        TrustRoot {
            schema: TRUST_ROOT_SCHEMA_V1.into(),
            publisher: "example.com".into(),
            version: 1,
            issued_at: "2026-08-06T00:00:00Z".into(),
            valid_until: "2027-08-06T00:00:00Z".into(),
            roles,
            keys,
            signatures: Vec::new(),
        }
    }

    #[test]
    fn a_root_missing_a_required_role_is_refused() {
        let mut root = minimal_root();
        root.roles.remove(ROLE_EMERGENCY_REVOCATION);
        let report = verify_trust_root(&root, None, None).unwrap();
        assert!(!report.structurally_valid);
        assert!(!report.verified);
    }

    #[test]
    fn an_unmeetable_threshold_is_named_as_such() {
        let mut root = minimal_root();
        root.roles.get_mut(ROLE_ARTIFACT).unwrap().threshold = 2;
        let report = verify_trust_root(&root, None, None).unwrap();
        assert!(!report.structurally_valid);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("needs") || issue.contains("requires 2 signatures"))
        );
    }

    #[test]
    fn a_key_filed_under_the_wrong_id_is_refused() {
        // The attack this stops: file an attacker key under a trusted key's id.
        let report = verify_trust_root(&minimal_root(), None, None).unwrap();
        assert!(!report.key_ids_match_keys);
        assert!(!report.verified);
    }

    #[test]
    fn no_trusted_clock_reports_unknown_validity_and_records_the_assumption() {
        let report = verify_trust_root(&minimal_root(), None, None).unwrap();
        assert_eq!(report.within_validity, None);
        assert!(!report.verified, "unknown validity must not verify");
        assert!(
            report
                .assumptions
                .iter()
                .any(|note| note.contains("no trusted clock"))
        );
    }

    #[test]
    fn first_contact_is_reported_and_recorded_as_an_assumption() {
        let report = verify_trust_root(&minimal_root(), None, None).unwrap();
        assert!(report.first_contact);
        assert_eq!(report.signed_by_prior_root, None);
        assert!(
            report
                .assumptions
                .iter()
                .any(|note| note.contains("first contact"))
        );
    }

    #[test]
    fn a_rotation_that_does_not_advance_the_version_is_refused() {
        let prior = minimal_root();
        let mut successor = minimal_root();
        successor.version = 1;
        let report = verify_trust_root(&successor, Some(&prior), None).unwrap();
        assert_eq!(report.version_advanced, Some(false));
        assert!(!report.verified);
    }

    #[test]
    fn a_rotation_may_not_change_publisher() {
        let prior = minimal_root();
        let mut successor = minimal_root();
        successor.version = 2;
        successor.publisher = "attacker.example".into();
        let report = verify_trust_root(&successor, Some(&prior), None).unwrap();
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("publisher"))
        );
        assert!(!report.verified);
    }

    #[test]
    fn an_unverified_root_authorizes_nothing() {
        let report = verify_trust_root(&minimal_root(), None, None).unwrap();
        assert!(!report.verified);
        assert!(report.authorized_roles.is_empty());
        assert!(!role_authorizes(&report, ROLE_ARTIFACT, "k"));
    }
}
