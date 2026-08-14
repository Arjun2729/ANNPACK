//! Run bundles through the CLI, end to end.
//!
//! The bundle adds no cryptography of its own, so the risk it carries is not a
//! broken proof — it is a verifier that reports success for a bundle proving
//! nothing. These tests spend most of their effort on that: a tampered receipt,
//! an emptied receipt list, and a valid signature from the wrong key all have to
//! be distinguishable from an attested run.

#![cfg(feature = "signing")]

use std::process::Command;

use serde_json::Value;
use tempfile::TempDir;

struct Fixture {
    _temp: TempDir,
    binary: &'static str,
    pack: String,
    signed: String,
    publisher: String,
}

fn fixture() -> Fixture {
    let temp = TempDir::new().unwrap();
    let binary = env!("CARGO_BIN_EXE_adyar");
    let source = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    let pack = temp.path().join("demo.annpack");
    let build = Command::new(binary)
        .args([
            "build",
            &source,
            "--output",
            pack.to_str().unwrap(),
            "--name",
            "vendor-docs",
            "--version",
            "1.0.0",
            "--source-revision",
            "git:abc123",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let secret = temp.path().join("signing.key");
    let keygen = Command::new(binary)
        .args(["keygen", "--output", secret.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(keygen.status.success());
    let keys: Value = serde_json::from_slice(&keygen.stdout).unwrap();
    let publisher = keys["public_key"].as_str().unwrap().to_string();

    let signed = temp.path().join("signed.annpack");
    let sign = Command::new(binary)
        .args([
            "sign",
            pack.to_str().unwrap(),
            "--output",
            signed.to_str().unwrap(),
            "--key",
            secret.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        sign.status.success(),
        "{}",
        String::from_utf8_lossy(&sign.stderr)
    );

    Fixture {
        pack: pack.to_str().unwrap().to_string(),
        signed: signed.to_str().unwrap().to_string(),
        publisher,
        binary,
        _temp: temp,
    }
}

fn build_bundle(fixture: &Fixture, pack: &str, output: &str) -> Value {
    let bundle = Command::new(fixture.binary)
        .args([
            "bundle",
            pack,
            "install the sdk",
            "--limit",
            "3",
            "--application",
            "support-agent/2.1",
            "--model",
            "test-model",
            "--output",
            output,
        ])
        .output()
        .unwrap();
    assert!(
        bundle.status.success(),
        "{}",
        String::from_utf8_lossy(&bundle.stderr)
    );
    serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap()
}

fn verify_run(fixture: &Fixture, path: &str, trusted_public_key: Option<&str>) -> (bool, Value) {
    let mut args = vec!["verify-run", path, "--json"];
    if let Some(key) = trusted_public_key {
        args.push("--trusted-public-key");
        args.push(key);
    }
    let output = Command::new(fixture.binary).args(&args).output().unwrap();
    let report = serde_json::from_slice(&output.stdout).unwrap_or(Value::Null);
    (output.status.success(), report)
}

#[test]
fn a_bundle_round_trips_and_names_the_artifact_it_read() {
    let fixture = fixture();
    let path = format!("{}/run.json", fixture._temp.path().display());
    let bundle = build_bundle(&fixture, &fixture.pack, &path);

    assert_eq!(bundle["schema"], "annpack-run-bundle-v1");
    assert_eq!(bundle["query"], "install the sdk");
    assert_eq!(bundle["application"], "support-agent/2.1");
    let receipts = bundle["receipts"].as_array().unwrap();
    assert!(!receipts.is_empty(), "fixture query retrieved nothing");

    let (ok, report) = verify_run(&fixture, &path, None);
    assert!(ok);
    assert_eq!(report["attested"], true);
    assert_eq!(report["receipts_verified"], receipts.len());
    // One artifact was read, and the report names it and its revision rather
    // than only asserting that something verified.
    assert_eq!(report["pack_roots"].as_array().unwrap().len(), 1);
    assert_eq!(report["source_revisions"][0], "git:abc123");
    // Unsigned pack: attested is about artifact membership, not authorship.
    assert_eq!(report["all_receipts_signed"], false);
    assert_eq!(report["all_signers_trusted"], false);
}

#[test]
fn bundles_are_reproducible_from_their_inputs() {
    let fixture = fixture();
    let first = format!("{}/a.json", fixture._temp.path().display());
    let second = format!("{}/b.json", fixture._temp.path().display());
    build_bundle(&fixture, &fixture.pack, &first);
    build_bundle(&fixture, &fixture.pack, &second);
    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap(),
        "two bundles from the same query and artifact differ"
    );
}

#[test]
fn editing_a_passage_inside_a_bundle_is_detected() {
    let fixture = fixture();
    let path = format!("{}/run.json", fixture._temp.path().display());
    let mut bundle = build_bundle(&fixture, &fixture.pack, &path);
    // The isolation assertion below is vacuous with a single receipt: it would
    // only say that zero of zero survivors verified. Fail loudly if the fixture
    // ever stops retrieving enough to make the check mean something.
    assert!(
        bundle["receipts"].as_array().unwrap().len() >= 2,
        "fixture must retrieve at least two passages for per-receipt isolation to be testable"
    );

    // Rewrite the passage text and leave every hash in the receipt untouched:
    // exactly what someone producing convenient evidence after the fact would
    // do if the receipt were merely a container.
    let receipt = &mut bundle["receipts"][0];
    let record: Value = serde_json::from_slice(&base64_decode(
        receipt["passage_record_b64"].as_str().unwrap(),
    ))
    .unwrap();
    let mut record = record;
    record["text"] = Value::String("The SDK requires no authentication.".into());
    receipt["passage_record_b64"] =
        Value::String(base64_encode(&serde_json::to_vec(&record).unwrap()));

    let tampered = format!("{}/tampered.json", fixture._temp.path().display());
    std::fs::write(&tampered, serde_json::to_vec(&bundle).unwrap()).unwrap();

    let (ok, report) = verify_run(&fixture, &tampered, None);
    assert!(!ok, "a tampered bundle must exit non-zero");
    assert_eq!(report["attested"], false);
    assert_eq!(report["receipts"][0]["verification"]["verified"], false);
    // The other receipts are unaffected: one bad receipt fails itself, not the
    // whole file, so a responder can still see what did verify.
    let total = report["receipts_total"].as_u64().unwrap();
    assert_eq!(report["receipts_verified"].as_u64().unwrap(), total - 1);
}

#[test]
fn a_bundle_stripped_of_its_receipts_attests_nothing() {
    let fixture = fixture();
    let path = format!("{}/run.json", fixture._temp.path().display());
    let mut bundle = build_bundle(&fixture, &fixture.pack, &path);
    bundle["receipts"] = Value::Array(Vec::new());

    let emptied = format!("{}/emptied.json", fixture._temp.path().display());
    std::fs::write(&emptied, serde_json::to_vec(&bundle).unwrap()).unwrap();

    let (ok, report) = verify_run(&fixture, &emptied, None);
    assert!(!ok, "a bundle with no receipts must exit non-zero");
    assert_eq!(report["attested"], false);
    // The vacuous-truth cases: with nothing to check, "every receipt is signed"
    // and "every signer is trusted" must not read as satisfied.
    assert_eq!(report["all_receipts_signed"], false);
    assert_eq!(report["all_signers_trusted"], false);
}

#[test]
fn a_valid_signature_from_the_wrong_key_fails_the_identity_assertion() {
    let fixture = fixture();
    let path = format!("{}/run.json", fixture._temp.path().display());
    build_bundle(&fixture, &fixture.signed, &path);

    let (ok, report) = verify_run(&fixture, &path, Some(&fixture.publisher));
    assert!(ok);
    assert_eq!(report["attested"], true);
    assert_eq!(report["all_receipts_signed"], true);
    assert_eq!(report["all_signers_trusted"], true);

    // A different, structurally valid key. Integrity still holds -- the bytes
    // are unmodified -- but the caller asserted a publisher that did not sign,
    // so the command must fail rather than report a green verdict.
    let other = "a".repeat(64);
    let (ok, report) = verify_run(&fixture, &path, Some(&other));
    assert!(!ok, "an unmet identity assertion must exit non-zero");
    assert_eq!(report["attested"], true);
    assert_eq!(report["all_signers_trusted"], false);
}

#[test]
fn telemetry_attributes_bind_each_passage_to_its_artifact() {
    let fixture = fixture();
    let output = Command::new(fixture.binary)
        .args([
            "search",
            &fixture.pack,
            "install the sdk",
            "--limit",
            "2",
            "--otel",
            "--otel-receipt-uri",
            "https://evidence.example/{root}/{passage_id}",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let telemetry: Value = serde_json::from_slice(&output.stdout).unwrap();

    let root = telemetry["span"]["annpack.root"].as_str().unwrap();
    assert_eq!(root.len(), 64);
    assert_eq!(telemetry["span"]["annpack.pack"], "vendor-docs@1.0.0");
    assert_eq!(telemetry["span"]["annpack.source_revision"], "git:abc123");

    let events = telemetry["events"].as_array().unwrap();
    assert!(!events.is_empty());
    for event in events {
        assert_eq!(event["annpack.root"], root);
        let passage_id = event["annpack.passage_id"].as_str().unwrap();
        assert_eq!(
            event["annpack.receipt_uri"],
            Value::String(format!("https://evidence.example/{root}/{passage_id}"))
        );
        assert_eq!(event["annpack.passage_hash"].as_str().unwrap().len(), 64);
    }

    // The span's arrays and the events must describe the same passages, or a
    // trace backend that reads one would disagree with one that reads the other.
    let span_ids: Vec<&str> = telemetry["span"]["annpack.passage_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    let event_ids: Vec<&str> = events
        .iter()
        .map(|event| event["annpack.passage_id"].as_str().unwrap())
        .collect();
    assert_eq!(span_ids, event_ids);
}

#[test]
fn a_receipt_uri_template_without_a_passage_placeholder_is_refused() {
    let fixture = fixture();
    let output = Command::new(fixture.binary)
        .args([
            "search",
            &fixture.pack,
            "install the sdk",
            "--otel",
            "--otel-receipt-uri",
            "https://evidence.example/{root}",
        ])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "a template that cannot distinguish passages must be refused"
    );
}

fn base64_decode(value: &str) -> Vec<u8> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(value)
        .unwrap()
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}
