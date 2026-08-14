//! Channel-state statements exercised with real Ed25519 keys and a real trust root.
//!
//! The unit tests in `release.rs` cover digest and sequence logic without ever
//! reaching signature verification. These build a properly signed trust root and
//! statements authorised by it, so the accepting path runs, and then break one
//! property at a time.

#![cfg(feature = "signing")]

use std::collections::BTreeMap;

use adyar::release::{
    CHANNEL_STATE_SCHEMA_V1, ChannelState, Currency, CurrentRelease, RetainedState, Revocation,
    SequenceVerdict, SigningAuthority, Supersession, currency_for_root, load_retained_state,
    persist_retained_state, sign_channel_state, state_to_retain, statement_digest,
    verify_channel_state,
};
use adyar::trust::{
    KeyDescriptor, ROLE_ARTIFACT, ROLE_EMERGENCY_REVOCATION, ROLE_RELEASE_STATE, ROLE_ROOT,
    RoleDescriptor, TRUST_ROOT_SCHEMA_V1, TrustRoot, TrustRootVerification, key_identity,
    sign_trust_root, verify_trust_root,
};

const NOW: &str = "2026-08-06T00:30:00Z";
const ROOT_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const ROOT_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const ROOT_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

const KEY_ROOT: [u8; 32] = [1; 32];
const KEY_ARTIFACT: [u8; 32] = [2; 32];
const KEY_RELEASE: [u8; 32] = [3; 32];
const KEY_REVOCATION: [u8; 32] = [4; 32];

fn trust() -> (TrustRoot, TrustRootVerification) {
    let mut roles = BTreeMap::new();
    let mut keys = BTreeMap::new();
    for (role, secret) in [
        (ROLE_ROOT, KEY_ROOT),
        (ROLE_ARTIFACT, KEY_ARTIFACT),
        (ROLE_RELEASE_STATE, KEY_RELEASE),
        (ROLE_EMERGENCY_REVOCATION, KEY_REVOCATION),
    ] {
        let (key_id, public_key) = key_identity(&secret);
        keys.insert(
            key_id.clone(),
            KeyDescriptor {
                algorithm: "Ed25519".into(),
                public_key,
            },
        );
        roles.insert(
            role.to_string(),
            RoleDescriptor {
                threshold: 1,
                keys: vec![key_id],
            },
        );
    }
    let mut root = TrustRoot {
        schema: TRUST_ROOT_SCHEMA_V1.into(),
        publisher: "example.com".into(),
        version: 1,
        issued_at: "2026-08-01T00:00:00Z".into(),
        valid_until: "2027-08-01T00:00:00Z".into(),
        roles,
        keys,
        signatures: Vec::new(),
    };
    sign_trust_root(&mut root, &KEY_ROOT).unwrap();
    let verification = verify_trust_root(&root, None, Some(NOW)).unwrap();
    assert!(verification.verified, "{:?}", verification.issues);
    (root, verification)
}

fn statement(sequence: u64, current: &str) -> ChannelState {
    ChannelState {
        schema: CHANNEL_STATE_SCHEMA_V1.into(),
        publisher: "example.com".into(),
        corpus: "support-manual".into(),
        channel: "production".into(),
        sequence,
        issued_at: "2026-08-06T00:00:00Z".into(),
        valid_until: "2026-08-06T01:00:00Z".into(),
        current: CurrentRelease {
            version: "4.3.0".into(),
            artifact_root: current.into(),
        },
        superseded: Vec::new(),
        revoked: Vec::new(),
        signatures: Vec::new(),
    }
}

fn scope() -> (&'static str, &'static str, &'static str) {
    ("example.com", "support-manual", "production")
}

#[test]
fn a_signed_statement_verifies_and_names_the_current_root() {
    let (root, trust_verification) = trust();
    let mut input = statement(4, ROOT_A);
    sign_channel_state(&mut input, &KEY_RELEASE).unwrap();

    let report =
        verify_channel_state(&input, &root, &trust_verification, None, Some(NOW), scope()).unwrap();
    assert!(report.verified, "{:?}", report.issues);
    assert_eq!(report.authority, SigningAuthority::Full);
    assert_eq!(report.sequence_verdict, SequenceVerdict::FirstContact);
    assert_eq!(
        currency_for_root(&input, &report, ROOT_A),
        Currency::Current
    );
    // A root the statement never mentions is unknown, never current.
    assert_eq!(
        currency_for_root(&input, &report, ROOT_C),
        Currency::Unknown
    );
}

#[test]
fn the_artifact_key_cannot_speak_for_release_state() {
    // The whole point of role separation: a key that may sign artifacts must not
    // be able to declare which artifact is current.
    let (root, trust_verification) = trust();
    let mut input = statement(4, ROOT_A);
    sign_channel_state(&mut input, &KEY_ARTIFACT).unwrap();

    let report =
        verify_channel_state(&input, &root, &trust_verification, None, Some(NOW), scope()).unwrap();
    assert_eq!(report.authority, SigningAuthority::None);
    assert!(!report.verified);
    assert_eq!(
        currency_for_root(&input, &report, ROOT_A),
        Currency::Unknown
    );
}

#[test]
fn the_revocation_key_can_withdraw_but_cannot_promote() {
    // Signed only by emergency_revocation. Honouring the whole statement would
    // give the revocation key release authority, collapsing the separation.
    let (root, trust_verification) = trust();
    let mut input = statement(5, ROOT_B);
    input.revoked.push(Revocation {
        artifact_root: ROOT_A.into(),
        at: "2026-08-06T00:10:00Z".into(),
        reason: "incorrect-content".into(),
    });
    sign_channel_state(&mut input, &KEY_REVOCATION).unwrap();

    let report =
        verify_channel_state(&input, &root, &trust_verification, None, Some(NOW), scope()).unwrap();
    assert_eq!(report.authority, SigningAuthority::RevocationOnly);
    assert!(report.verified);
    // The revocation is acted on.
    assert_eq!(
        currency_for_root(&input, &report, ROOT_A),
        Currency::Revoked
    );
    // The promotion is not.
    assert_eq!(
        currency_for_root(&input, &report, ROOT_B),
        Currency::Unknown
    );

    // The same statement signed by release_state does promote.
    let mut promoted = input.clone();
    promoted.signatures.clear();
    sign_channel_state(&mut promoted, &KEY_RELEASE).unwrap();
    let report = verify_channel_state(
        &promoted,
        &root,
        &trust_verification,
        None,
        Some(NOW),
        scope(),
    )
    .unwrap();
    assert_eq!(report.authority, SigningAuthority::Full);
    assert_eq!(
        currency_for_root(&promoted, &report, ROOT_B),
        Currency::Current
    );
    assert_eq!(
        currency_for_root(&promoted, &report, ROOT_A),
        Currency::Revoked
    );
}

#[test]
fn revocation_outranks_supersession() {
    let (root, trust_verification) = trust();
    let mut input = statement(6, ROOT_C);
    input.superseded.push(Supersession {
        artifact_root: ROOT_A.into(),
        by: ROOT_C.into(),
        at: "2026-08-06T00:05:00Z".into(),
    });
    input.revoked.push(Revocation {
        artifact_root: ROOT_A.into(),
        at: "2026-08-06T00:10:00Z".into(),
        reason: "incorrect-content".into(),
    });
    sign_channel_state(&mut input, &KEY_RELEASE).unwrap();
    let report =
        verify_channel_state(&input, &root, &trust_verification, None, Some(NOW), scope()).unwrap();
    // Superseded is policy; revoked is a security event. The stronger verdict
    // must win or a withdrawn artifact reads as merely out of date.
    assert_eq!(
        currency_for_root(&input, &report, ROOT_A),
        Currency::Revoked
    );
}

#[test]
fn tampering_with_the_payload_invalidates_the_signature() {
    let (root, trust_verification) = trust();
    let mut input = statement(4, ROOT_A);
    sign_channel_state(&mut input, &KEY_RELEASE).unwrap();
    input.current.artifact_root = ROOT_B.into();

    let report =
        verify_channel_state(&input, &root, &trust_verification, None, Some(NOW), scope()).unwrap();
    assert_eq!(report.authority, SigningAuthority::None);
    assert!(!report.verified);
}

#[test]
fn a_correctly_signed_rollback_is_refused() {
    let (root, trust_verification) = trust();
    let mut old = statement(3, ROOT_A);
    sign_channel_state(&mut old, &KEY_RELEASE).unwrap();

    let retained = RetainedState {
        publisher: "example.com".into(),
        corpus: "support-manual".into(),
        channel: "production".into(),
        highest_sequence: 7,
        statement_digest: "irrelevant".into(),
        artifact_root: ROOT_B.into(),
        accepted_at: "2026-08-06T00:20:00Z".into(),
    };
    let report = verify_channel_state(
        &old,
        &root,
        &trust_verification,
        Some(&retained),
        Some(NOW),
        scope(),
    )
    .unwrap();
    assert_eq!(
        report.authority,
        SigningAuthority::Full,
        "signature is genuine"
    );
    assert_eq!(report.sequence_verdict, SequenceVerdict::Rollback);
    assert!(!report.verified);
    assert!(state_to_retain(&old, &report, scope(), NOW).is_none());
}

#[test]
fn two_statements_at_one_sequence_are_equivocation() {
    let (root, trust_verification) = trust();
    let mut first = statement(9, ROOT_A);
    sign_channel_state(&mut first, &KEY_RELEASE).unwrap();
    let mut second = statement(9, ROOT_B);
    sign_channel_state(&mut second, &KEY_RELEASE).unwrap();

    let retained = RetainedState {
        publisher: "example.com".into(),
        corpus: "support-manual".into(),
        channel: "production".into(),
        highest_sequence: 9,
        statement_digest: statement_digest(&first).unwrap(),
        artifact_root: ROOT_A.into(),
        accepted_at: "2026-08-06T00:20:00Z".into(),
    };

    // Re-presenting the same statement is idempotent, not an attack.
    let report = verify_channel_state(
        &first,
        &root,
        &trust_verification,
        Some(&retained),
        Some(NOW),
        scope(),
    )
    .unwrap();
    assert_eq!(report.sequence_verdict, SequenceVerdict::Idempotent);
    assert!(report.verified);

    // A different, equally well-signed statement at the same sequence is not.
    let report = verify_channel_state(
        &second,
        &root,
        &trust_verification,
        Some(&retained),
        Some(NOW),
        scope(),
    )
    .unwrap();
    assert_eq!(report.authority, SigningAuthority::Full);
    assert_eq!(report.sequence_verdict, SequenceVerdict::Equivocation);
    assert!(!report.verified);
}

#[test]
fn scope_and_expiry_are_enforced() {
    let (root, trust_verification) = trust();
    let mut input = statement(4, ROOT_A);
    sign_channel_state(&mut input, &KEY_RELEASE).unwrap();

    let wrong_scope = verify_channel_state(
        &input,
        &root,
        &trust_verification,
        None,
        Some(NOW),
        ("example.com", "support-manual", "staging"),
    )
    .unwrap();
    assert!(!wrong_scope.scope_matches);
    // Retained state must not be consulted for a statement scoped elsewhere.
    assert_eq!(wrong_scope.sequence_verdict, SequenceVerdict::NotEvaluated);
    assert!(!wrong_scope.verified);

    let expired = verify_channel_state(
        &input,
        &root,
        &trust_verification,
        None,
        Some("2026-08-06T02:00:00Z"),
        scope(),
    )
    .unwrap();
    assert_eq!(expired.within_validity, Some(false));
    assert!(!expired.verified);
}

#[test]
fn without_a_clock_nothing_is_current() {
    let (root, trust_verification) = trust();
    let mut input = statement(4, ROOT_A);
    sign_channel_state(&mut input, &KEY_RELEASE).unwrap();

    let report =
        verify_channel_state(&input, &root, &trust_verification, None, None, scope()).unwrap();
    assert_eq!(
        report.authority,
        SigningAuthority::Full,
        "signature is genuine"
    );
    assert_eq!(report.within_validity, None);
    assert!(!report.verified);
    assert_eq!(
        currency_for_root(&input, &report, ROOT_A),
        Currency::Unknown
    );
}

#[test]
fn an_unverified_trust_root_authorizes_no_statement() {
    let (root, _) = trust();
    // A trust root verified without a clock does not verify, so nothing it
    // nominally authorises may be honoured.
    let unverified = verify_trust_root(&root, None, None).unwrap();
    assert!(!unverified.verified);

    let mut input = statement(4, ROOT_A);
    sign_channel_state(&mut input, &KEY_RELEASE).unwrap();
    let report =
        verify_channel_state(&input, &root, &unverified, None, Some(NOW), scope()).unwrap();
    assert!(!report.trust_root_verified);
    assert!(!report.verified);
    assert_eq!(
        currency_for_root(&input, &report, ROOT_A),
        Currency::Unknown
    );
}

#[test]
fn retained_state_survives_a_round_trip_and_is_written_atomically() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nested/state.json");
    assert!(load_retained_state(&path).unwrap().is_none());

    let (root, trust_verification) = trust();
    let mut input = statement(4, ROOT_A);
    sign_channel_state(&mut input, &KEY_RELEASE).unwrap();
    let report =
        verify_channel_state(&input, &root, &trust_verification, None, Some(NOW), scope()).unwrap();

    let state = state_to_retain(&input, &report, scope(), NOW).unwrap();
    persist_retained_state(&path, &state).unwrap();
    assert_eq!(load_retained_state(&path).unwrap().unwrap(), state);

    // The temporary file must not survive, or a later reader could find a
    // half-written sibling and treat it as state.
    assert!(!path.with_extension("tmp").exists());
}
