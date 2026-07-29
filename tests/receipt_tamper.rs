//! A signed receipt authenticates the passage record *and* every descriptive
//! field a consumer reads off it. Before v2 the integrity chain covered only the
//! passage record, so `source_revision`, `pack`, `passage_id`/`passage_ordinal`,
//! and `canonical_url` could be rewritten while the receipt still reported
//! `verified: true` under a trusted publisher key. Each case below rewrites one
//! such field on an otherwise-valid receipt and asserts verification now fails.

use annpack::build::{BuildOptions, build_pack};
use annpack::evidence::verify_receipt;
use annpack::model::AccessClass;
use annpack::search::SearchEngine;
use annpack::signing::{generate_keypair, sign_pack};

/// Build a one-document pack, sign it, and return `(engine, publisher_hex)`.
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
        dependencies: Vec::new(),
        policy_override: None,
        vector_input: None,
        expansion_input: None,
        splade_input: None,
        anchors_input: None,
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
    let response = engine
        .search("grace", &annpack::search::SearchOptions::default())
        .unwrap();
    response.results[0].passage_id.clone()
}

#[test]
fn honest_receipt_verifies_and_binds_every_field() {
    let (engine, publisher, _dir) = signed_engine();
    let receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    // The reference issuer now emits v2 receipts that carry the Documents section.
    assert_eq!(receipt.schema, "annpack-receipt-v2");
    assert!(receipt.canonical_url.is_some());
    assert!(receipt.documents_bytes_b64.is_some());

    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(report.verified, "issues: {:?}", report.issues);
    assert!(report.canonical_url_matches);
    assert!(report.source_revision_matches);
    assert!(report.passage_metadata_matches);
    assert!(report.pack_matches);
    assert!(report.identity_trusted, "trusted key must establish trust");
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
fn dropping_the_documents_section_cannot_downgrade_a_url_claim() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    // Attacker keeps a (forged) URL but strips the bytes that would authenticate it.
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
fn forged_passage_metadata_fails() {
    let (engine, publisher, _dir) = signed_engine();
    let mut receipt = engine
        .receipt_for_passage(&only_passage_id(&engine))
        .unwrap();
    receipt.passage_id = "0".repeat(64);
    receipt.passage_ordinal = 999;

    let report = verify_receipt(&receipt, Some(&publisher)).unwrap();
    assert!(!report.verified);
    assert!(!report.passage_metadata_matches);
}
