//! What supplying `--trusted-public-key` means at the command line.
//!
//! The library keeps three claims separate — internal receipt integrity, a
//! cryptographically valid signature, and a trusted publisher identity — and
//! `ReceiptVerification.verified` reports only the first. That separation is
//! deliberate and these tests do not change it.
//!
//! The command line is a different contract. Passing a trusted public key is an
//! explicit assertion that this publisher signed the receipt, so `annpack
//! verify-evidence` must exit non-zero when no valid signature from that exact
//! key is present, even though the integrity chain verified.

use std::path::Path;
use std::process::{Command, Output};

use serde_json::Value;
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_annpack")
}

fn run(args: &[&str]) -> Output {
    Command::new(binary()).args(args).output().unwrap()
}

fn succeed(args: &[&str]) -> String {
    let output = run(args);
    assert!(
        output.status.success(),
        "{args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn build_pack(temp: &TempDir, name: &str) -> String {
    let pack = temp.path().join(format!("{name}.annpack"));
    let fixture = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    succeed(&[
        "build",
        &fixture,
        "--output",
        pack.to_str().unwrap(),
        "--name",
        name,
        "--version",
        "1.0.0",
        "--source-revision",
        "git:trusted-key-test",
    ]);
    pack.to_str().unwrap().to_string()
}

fn first_passage_id(pack: &str) -> String {
    let response: Value = serde_json::from_str(&succeed(&[
        "search", pack, "AP-104", "--mode", "lexical", "--json",
    ]))
    .unwrap();
    response["results"][0]["passage_id"]
        .as_str()
        .unwrap()
        .to_string()
}

fn issue_receipt(pack: &str, output: &Path) {
    let passage = first_passage_id(pack);
    succeed(&[
        "receipt",
        pack,
        &passage,
        "--output",
        output.to_str().unwrap(),
    ]);
}

/// Creates a keypair and returns `(secret path, public key hex)`.
#[cfg(feature = "signing")]
fn keygen(temp: &TempDir, name: &str) -> (String, String) {
    let secret = temp.path().join(format!("{name}.key"));
    let report: Value = serde_json::from_str(&succeed(&[
        "keygen",
        "--output",
        secret.to_str().unwrap(),
        "--json",
    ]))
    .unwrap();
    (
        secret.to_str().unwrap().to_string(),
        report["public_key"].as_str().unwrap().to_string(),
    )
}

#[test]
fn an_unsigned_receipt_verifies_when_no_trusted_key_is_supplied() {
    let temp = TempDir::new().unwrap();
    let pack = build_pack(&temp, "unsigned-docs");
    let receipt = temp.path().join("receipt.json");
    issue_receipt(&pack, &receipt);

    let output = run(&["verify-evidence", receipt.to_str().unwrap(), "--json"]);
    assert!(
        output.status.success(),
        "integrity-only verification must succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // And it must say plainly that neither signature nor identity was
    // established, rather than implying them by exiting zero.
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["verified"], true);
    assert_eq!(report["signature_valid"], false);
    assert_eq!(report["identity_trusted"], false);
}

#[cfg(feature = "signing")]
#[test]
fn an_unsigned_receipt_fails_when_a_trusted_key_is_supplied() {
    let temp = TempDir::new().unwrap();
    let pack = build_pack(&temp, "unsigned-docs");
    let receipt = temp.path().join("receipt.json");
    issue_receipt(&pack, &receipt);
    let (_, public_hex) = keygen(&temp, "publisher");

    let output = run(&[
        "verify-evidence",
        receipt.to_str().unwrap(),
        "--trusted-public-key",
        &public_hex,
    ]);
    assert!(
        !output.status.success(),
        "an unsigned receipt must not satisfy an explicit trusted-key assertion"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("trusted public key"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A pack signed by `publisher`, plus a receipt issued from it.
#[cfg(feature = "signing")]
fn signed_receipt(temp: &TempDir) -> (String, std::path::PathBuf) {
    let pack = build_pack(temp, "signed-docs");
    let (secret, public_hex) = keygen(temp, "publisher");
    let signed = temp.path().join("signed.annpack");
    succeed(&[
        "sign",
        &pack,
        "--output",
        signed.to_str().unwrap(),
        "--key",
        &secret,
        "--identity",
        "vendor.example",
    ]);
    let receipt = temp.path().join("signed-receipt.json");
    issue_receipt(signed.to_str().unwrap(), &receipt);
    (public_hex, receipt)
}

#[cfg(feature = "signing")]
#[test]
fn a_correct_signature_with_the_correct_trusted_key_verifies() {
    let temp = TempDir::new().unwrap();
    let (public_hex, receipt) = signed_receipt(&temp);

    let output = run(&[
        "verify-evidence",
        receipt.to_str().unwrap(),
        "--trusted-public-key",
        &public_hex,
        "--json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["verified"], true);
    assert_eq!(report["signature_valid"], true);
    assert_eq!(report["identity_trusted"], true);
}

#[cfg(feature = "signing")]
#[test]
fn a_correct_signature_with_the_wrong_trusted_key_fails() {
    let temp = TempDir::new().unwrap();
    let (_, receipt) = signed_receipt(&temp);
    let (_, other_hex) = keygen(&temp, "someone-else");

    let output = run(&[
        "verify-evidence",
        receipt.to_str().unwrap(),
        "--trusted-public-key",
        &other_hex,
    ]);
    assert!(
        !output.status.success(),
        "a signature from a different key must not satisfy the assertion"
    );
}

#[cfg(feature = "signing")]
#[test]
fn a_tampered_signature_fails_with_and_without_a_trusted_key() {
    let temp = TempDir::new().unwrap();
    let (public_hex, receipt) = signed_receipt(&temp);

    let mut parsed: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    let signature = parsed["signature"]["signature"].as_str().unwrap();
    // Flip one hex digit: still well-formed, no longer a valid signature.
    let mut bytes = signature.as_bytes().to_vec();
    bytes[0] = if bytes[0] == b'a' { b'b' } else { b'a' };
    parsed["signature"]["signature"] = Value::String(String::from_utf8(bytes).unwrap());
    let tampered = temp.path().join("tampered-receipt.json");
    std::fs::write(&tampered, serde_json::to_vec(&parsed).unwrap()).unwrap();

    // With a trusted key the command must fail outright.
    assert!(
        !run(&[
            "verify-evidence",
            tampered.to_str().unwrap(),
            "--trusted-public-key",
            &public_hex,
        ])
        .status
        .success()
    );

    // Without one, the integrity chain still holds — that is the separation the
    // library maintains — but the report must not claim a valid signature.
    let output = run(&["verify-evidence", tampered.to_str().unwrap(), "--json"]);
    assert!(output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["signature_valid"], false);
    assert_eq!(report["identity_trusted"], false);
    assert!(
        report["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue.as_str().unwrap().contains("signature")),
        "{report:?}"
    );
}

#[cfg(feature = "signing")]
#[test]
fn pack_verification_already_enforces_an_explicit_trusted_key() {
    // `annpack verify --public-key` has the same contract and already fails
    // closed. Pinned here so the two commands cannot drift apart.
    let temp = TempDir::new().unwrap();
    let pack = build_pack(&temp, "unsigned-docs");
    let (_, public_hex) = keygen(&temp, "publisher");
    let public_path = temp.path().join("publisher.pub");
    std::fs::write(&public_path, format!("{public_hex}\n")).unwrap();

    assert!(run(&["verify", &pack]).status.success());
    assert!(
        !run(&[
            "verify",
            &pack,
            "--public-key",
            public_path.to_str().unwrap()
        ])
        .status
        .success(),
        "an unsigned pack must not satisfy an explicit trusted-key assertion"
    );
}
