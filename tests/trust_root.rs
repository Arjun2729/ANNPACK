//! Trust roots exercised with real Ed25519 keys.
//!
//! The unit tests in `trust.rs` cover rejection paths using a placeholder key,
//! which means they never reach signature verification at all — every one of
//! them would still pass if the signature check were deleted. These tests build
//! correctly signed roots so that the accepting path is exercised, and then
//! break one thing at a time so each rejection is attributable to that thing
//! rather than to a root that was never valid.

use std::collections::BTreeMap;

use annpack::trust::{
    KeyDescriptor, ROLE_ARTIFACT, ROLE_EMERGENCY_REVOCATION, ROLE_RELEASE_STATE, ROLE_ROOT,
    RoleDescriptor, TRUST_ROOT_SCHEMA_V1, TrustRoot, key_identity, role_authorizes,
    sign_trust_root, verify_trust_root,
};

const NOW: &str = "2026-08-06T12:00:00Z";

fn secret(seed: u8) -> [u8; 32] {
    [seed; 32]
}

/// A root whose four roles are held by the given keys, unsigned.
fn root_with(version: u64, role_keys: &[(&str, Vec<[u8; 32]>, u32)]) -> TrustRoot {
    let mut roles = BTreeMap::new();
    let mut keys = BTreeMap::new();
    for (role, secrets, threshold) in role_keys {
        let mut ids = Vec::new();
        for secret in secrets {
            let (key_id, public_key) = key_identity(secret);
            keys.insert(
                key_id.clone(),
                KeyDescriptor {
                    algorithm: "Ed25519".into(),
                    public_key,
                },
            );
            ids.push(key_id);
        }
        roles.insert(
            (*role).to_string(),
            RoleDescriptor {
                threshold: *threshold,
                keys: ids,
            },
        );
    }
    TrustRoot {
        schema: TRUST_ROOT_SCHEMA_V1.into(),
        publisher: "example.com".into(),
        version,
        issued_at: "2026-08-01T00:00:00Z".into(),
        valid_until: "2027-08-01T00:00:00Z".into(),
        roles,
        keys,
        signatures: Vec::new(),
    }
}

/// One key per role, threshold one, signed by the root key.
fn simple_root(version: u64) -> TrustRoot {
    let mut root = root_with(
        version,
        &[
            (ROLE_ROOT, vec![secret(1)], 1),
            (ROLE_ARTIFACT, vec![secret(2)], 1),
            (ROLE_RELEASE_STATE, vec![secret(3)], 1),
            (ROLE_EMERGENCY_REVOCATION, vec![secret(4)], 1),
        ],
    );
    sign_trust_root(&mut root, &secret(1)).unwrap();
    root
}

#[test]
fn a_correctly_signed_root_verifies_and_authorizes_its_roles() {
    let root = simple_root(1);
    let report = verify_trust_root(&root, None, Some(NOW)).unwrap();
    assert!(report.verified, "issues: {:?}", report.issues);
    assert!(report.self_signed);
    assert_eq!(report.within_validity, Some(true));
    assert!(report.first_contact);

    let (artifact_key, _) = key_identity(&secret(2));
    let (release_key, _) = key_identity(&secret(3));
    assert!(role_authorizes(&report, ROLE_ARTIFACT, &artifact_key));
    assert!(role_authorizes(&report, ROLE_RELEASE_STATE, &release_key));
    // Role separation is the point: the artifact key must not be able to speak
    // for release state.
    assert!(!role_authorizes(&report, ROLE_RELEASE_STATE, &artifact_key));
    assert!(!role_authorizes(&report, ROLE_ARTIFACT, &release_key));
}

#[test]
fn a_root_signed_by_a_non_root_role_does_not_verify() {
    // The artifact key is in the root's key table and is perfectly valid -- it
    // just is not authorised to sign trust roots.
    let mut root = simple_root(1);
    root.signatures.clear();
    sign_trust_root(&mut root, &secret(2)).unwrap();
    let report = verify_trust_root(&root, None, Some(NOW)).unwrap();
    assert!(!report.self_signed);
    assert!(!report.verified);
}

#[test]
fn tampering_with_the_payload_invalidates_the_signature() {
    let mut root = simple_root(1);
    assert!(verify_trust_root(&root, None, Some(NOW)).unwrap().verified);
    // Grant the artifact key release-state authority without re-signing.
    let (artifact_key, _) = key_identity(&secret(2));
    root.roles
        .get_mut(ROLE_RELEASE_STATE)
        .unwrap()
        .keys
        .push(artifact_key);
    let report = verify_trust_root(&root, None, Some(NOW)).unwrap();
    assert!(
        !report.self_signed,
        "a role change must break the signature"
    );
    assert!(!report.verified);
}

#[test]
fn one_key_cannot_satisfy_a_threshold_of_two() {
    // The concrete attack: sign twice with the same key and hope the verifier
    // counts signatures rather than distinct authorised signers.
    let mut root = root_with(
        1,
        &[
            (ROLE_ROOT, vec![secret(1), secret(9)], 2),
            (ROLE_ARTIFACT, vec![secret(2)], 1),
            (ROLE_RELEASE_STATE, vec![secret(3)], 1),
            (ROLE_EMERGENCY_REVOCATION, vec![secret(4)], 1),
        ],
    );
    sign_trust_root(&mut root, &secret(1)).unwrap();
    let duplicate = root.signatures[0].clone();
    root.signatures.push(duplicate);
    assert_eq!(root.signatures.len(), 2);

    let report = verify_trust_root(&root, None, Some(NOW)).unwrap();
    assert!(
        !report.self_signed,
        "duplicate signatures met the threshold"
    );
    assert!(!report.verified);

    // The same root with two genuinely distinct signers does verify, which is
    // what shows the rejection above was about distinctness and not about
    // thresholds being broken generally.
    root.signatures.clear();
    sign_trust_root(&mut root, &secret(1)).unwrap();
    sign_trust_root(&mut root, &secret(9)).unwrap();
    let report = verify_trust_root(&root, None, Some(NOW)).unwrap();
    assert!(report.verified, "issues: {:?}", report.issues);
}

#[test]
fn rotation_requires_signatures_from_both_the_old_and_new_root_roles() {
    let prior = simple_root(1);

    // Successor moves the root role to a new key.
    let mut successor = root_with(
        2,
        &[
            (ROLE_ROOT, vec![secret(5)], 1),
            (ROLE_ARTIFACT, vec![secret(2)], 1),
            (ROLE_RELEASE_STATE, vec![secret(3)], 1),
            (ROLE_EMERGENCY_REVOCATION, vec![secret(4)], 1),
        ],
    );

    // Signed only by itself: an attacker who mints a root and presents it.
    sign_trust_root(&mut successor, &secret(5)).unwrap();
    let report = verify_trust_root(&successor, Some(&prior), Some(NOW)).unwrap();
    assert!(report.self_signed);
    assert_eq!(report.signed_by_prior_root, Some(false));
    assert!(!report.verified);

    // Signed only by the old root: an old key installing keys nobody holds.
    successor.signatures.clear();
    sign_trust_root(&mut successor, &secret(1)).unwrap();
    let report = verify_trust_root(&successor, Some(&prior), Some(NOW)).unwrap();
    assert!(!report.self_signed);
    assert_eq!(report.signed_by_prior_root, Some(true));
    assert!(!report.verified);

    // Both: the only accepted rotation.
    sign_trust_root(&mut successor, &secret(5)).unwrap();
    let report = verify_trust_root(&successor, Some(&prior), Some(NOW)).unwrap();
    assert!(report.self_signed);
    assert_eq!(report.signed_by_prior_root, Some(true));
    assert_eq!(report.version_advanced, Some(true));
    assert!(report.verified, "issues: {:?}", report.issues);
}

#[test]
fn a_correctly_signed_rollback_is_still_refused() {
    // Version 1 is genuinely valid and correctly signed. Replaying it after
    // version 2 has been accepted must not succeed.
    let current = simple_root(2);
    let replayed = simple_root(1);
    let report = verify_trust_root(&replayed, Some(&current), Some(NOW)).unwrap();
    assert!(report.self_signed, "the replayed root is genuinely signed");
    assert_eq!(report.version_advanced, Some(false));
    assert!(!report.verified);
}

#[test]
fn a_different_root_at_the_same_version_is_refused() {
    // Found by scripts/check-mutations.py: relaxing the version rule from `>` to
    // `>=` left every test green, because the rollback test only replayed an
    // older version. Equal-version is the publisher-equivocation case at the
    // trust-root layer -- two differently-keyed roots both claiming version 2 --
    // and it has to be refused rather than treated as a no-op update.
    let current = simple_root(2);
    let mut impostor = root_with(
        2,
        &[
            (ROLE_ROOT, vec![secret(1)], 1),
            (ROLE_ARTIFACT, vec![secret(7)], 1),
            (ROLE_RELEASE_STATE, vec![secret(3)], 1),
            (ROLE_EMERGENCY_REVOCATION, vec![secret(4)], 1),
        ],
    );
    sign_trust_root(&mut impostor, &secret(1)).unwrap();

    // Correctly signed by the trusted root key, and still not an upgrade.
    let report = verify_trust_root(&impostor, Some(&current), Some(NOW)).unwrap();
    assert!(report.self_signed);
    assert_eq!(report.signed_by_prior_root, Some(true));
    assert_eq!(report.version_advanced, Some(false));
    assert!(!report.verified);
}

#[test]
fn expiry_is_enforced_when_a_clock_is_supplied() {
    let root = simple_root(1);
    assert!(verify_trust_root(&root, None, Some(NOW)).unwrap().verified);

    for outside in ["2026-07-01T00:00:00Z", "2028-01-01T00:00:00Z"] {
        let report = verify_trust_root(&root, None, Some(outside)).unwrap();
        assert_eq!(report.within_validity, Some(false), "at {outside}");
        assert!(!report.verified);
    }
}

#[test]
fn without_a_clock_a_valid_root_still_does_not_verify() {
    // Everything else about this root is correct. Currency is the one claim
    // that cannot be made without a clock, and it must not be assumed.
    let root = simple_root(1);
    let report = verify_trust_root(&root, None, None).unwrap();
    assert!(report.self_signed);
    assert!(report.structurally_valid);
    assert_eq!(report.within_validity, None);
    assert!(!report.verified);
    assert!(
        report
            .assumptions
            .iter()
            .any(|note| note.contains("no trusted clock"))
    );
}
