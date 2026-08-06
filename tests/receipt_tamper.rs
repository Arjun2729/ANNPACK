//! Security regression tests for standalone evidence receipts.
//!
//! Each externally visible field is mutated independently. Resource and codec
//! defenses live in `evidence.rs` unit tests because they exercise private
//! directory-decoding helpers directly.

use annpack::build::{BuildOptions, build_pack};
use annpack::evidence::verify_receipt;
use annpack::model::AccessClass;
use annpack::search::SearchEngine;
use annpack::signing::{generate_keypair, sign_pack};

fn signed_engine() -> (SearchEngine, String, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir(&src).unwrap();
    std::fs::write(
        src.join("guide.md"),
        "---\ntitle: Acme Security Guide\nurl: https://acme.example/docs/security\n---\n\
         # Rotating keys\n\nCall `rotateKey()` every 90 days for one grace period.\n",
    )
    .unwrap();

    let pack = dir.path().join("pack.annpack");
    build_pack(&BuildOptions {
        input: src,
        output: pack.clone(),
        name: "acme-security".into(),
        version: "1.0.0".into(),
        description: None,
        source_revision: Some("git:REAL-abc123".into()),
        base_url: None,
        created_at: None,
        license: None,
        access: AccessClass::Public,
        redistributable: None,
        policy_expires_at: None,
        policy_url: None,
        policy_override: None,
        vector_input: None,
        expansion_input: None,
        splade_input: None,
        target_chars: 1_200,
        max_chars: 2_400,
        input_format: annpack::ingest::InputFormat::Auto,
    })
    .unwrap();

    let secret = dir.path().join("secret.key");
    let public = dir.path().join("public.key");
    let (_, publisher_hex) = generate_keypair(&secret, Some(&public)).unwrap();
    let signed = dir.path().join("signed.annpack");
    sign_pack(
        &pack,
        &signed,
        &secret,
        Some("human:kliu@acme".into()),
        None,
    )
    .unwrap();

    (
        SearchEngine::open_path(&signed).unwrap(),
        publisher_hex,
        dir,
    )
}

fn only_passage_id(engine: &SearchEngine) -> String {
    engine
        .search("grace", &annpack::search::SearchOptions::default())
        .unwrap()
        .results[0]
        .passage_id
        .clone()
}

#[test]
fn honest_receipt_verifies_and_binds_every_field() {
    let (engine, publisher, _dir) = signed_engine();
    let receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    assert_eq!(receipt.schema, "annpack-receipt-v2");
    assert!(receipt.canonical_url.is_some());
    assert!(receipt.documents_bytes_b64.is_some());

    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(report.verified, "issues: {:?}", report.issues);
    assert!(report.canonical_url_matches);
    assert!(report.source_revision_matches);
    assert!(report.passage_metadata_matches);
    assert!(report.pack_matches);
    assert!(report.identity_trusted);
}

#[test]
fn forged_canonical_url_fails() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    receipt.canonical_url = Some("https://evil.example/backdoor#rotating-keys".into());
    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(!report.verified);
    assert!(!report.canonical_url_matches);
}

#[test]
fn dropping_documents_cannot_downgrade_a_url_claim() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    receipt.canonical_url = Some("https://evil.example/x".into());
    receipt.documents_bytes_b64 = None;
    receipt.documents_section_id = None;
    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(!report.verified);
    assert!(!report.canonical_url_matches);
}

#[test]
fn forged_source_revision_fails() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    receipt.source_revision = Some("git:FORGED-deadbeef".into());
    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(!report.verified);
    assert!(!report.source_revision_matches);
}

#[test]
fn forged_pack_identity_fails() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    receipt.pack = "trusted-vendor@9.9.9".into();
    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(!report.verified);
    assert!(!report.pack_matches);
}

#[test]
fn forged_passage_id_alone_fails() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    receipt.passage_id = "0".repeat(64);
    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(!report.verified);
    assert!(!report.passage_metadata_matches);
}

#[test]
fn forged_passage_ordinal_alone_fails() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    receipt.passage_ordinal = 999;
    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(!report.verified);
    assert!(!report.passage_metadata_matches);
}

#[test]
fn forged_passage_hash_alone_fails() {
    // Found by scripts/check-mutations.py: forcing `passage_hash_matches = true`
    // left every test in this file green, because tampering with the *record*
    // also breaks the inclusion proof, and that was the only path exercised.
    // Changing the declared hash alone leaves the proof intact and isolates the
    // one check that was going unverified.
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    let honest = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(honest.verified);

    receipt.passage_hash = "0".repeat(64);
    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(!report.verified);
    assert!(
        !report.passage_hash_matches,
        "the declared passage hash no longer matches the record"
    );
}

#[test]
fn unknown_receipt_schema_is_rejected() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    receipt.schema = "annpack-receipt-v999".into();
    let error = verify_receipt(&receipt, Some(&publisher)).unwrap_err();
    assert!(error.to_string().contains("receipt schema"));
}

#[test]
fn oversized_proof_is_rejected_before_replay() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    let step =
        receipt
            .inclusion_proof
            .first()
            .cloned()
            .unwrap_or_else(|| annpack::evidence::ProofStep {
                sibling: "0".repeat(64),
                sibling_is_left: false,
            });
    receipt.inclusion_proof = vec![step; 65];
    let error = verify_receipt(&receipt, Some(&publisher)).unwrap_err();
    assert!(error.to_string().contains("64 steps"));
}
