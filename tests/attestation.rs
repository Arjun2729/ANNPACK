//! GitHub attestation parsing and fail-closed CLI integration.

#![cfg(feature = "github-attestation")]

use adyar::attestation::parse_bundle;
use serde_json::json;

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn oid_components(dotted: &str) -> Vec<u64> {
    dotted
        .split('.')
        .map(|part| part.parse().unwrap())
        .collect()
}

fn utf8_extension_der(value: &str) -> Vec<u8> {
    use der::Encode;
    use der::asn1::Utf8StringRef;
    Utf8StringRef::new(value).unwrap().to_der().unwrap()
}

const ISSUER: &str = "https://token.actions.githubusercontent.com";
/// What the *certificate* claims: Fulcio's source-repository extension is
/// always a fully-qualified URL (the OID registry: "SHOULD be a fully
/// qualified URL when available").
const CERTIFICATE_REPOSITORY: &str = "https://github.com/example/repo";
/// What the *predicate* carries: `release.yml` passes
/// `--repository "github.com/${{ github.repository }}"` -- no scheme. The two
/// conventions differ on purpose, and `attestation.rs`'s comparison strips
/// both `https://` and `github.com/` from the certificate side specifically
/// to reconcile them. An earlier version of this test used the same
/// (schemed) string for both sides and their disagreement was a test bug, not
/// an attestation.rs bug.
const PREDICATE_REPOSITORY: &str = "github.com/example/repo";
const WORKFLOW_REF: &str =
    "https://github.com/example/repo/.github/workflows/release.yml@refs/tags/v1.2.3";
const REVISION: &str = "deadbeefcafe";

/// A genuine DER certificate carrying the real Fulcio OIDs, built with
/// `rcgen` so the bytes are actual ASN.1, not approximations of it.
fn github_certificate() -> Vec<u8> {
    use rcgen::{CertificateParams, CustomExtension, KeyPair};

    let mut params = CertificateParams::new(Vec::new()).unwrap();
    params.custom_extensions = vec![
        CustomExtension::from_oid_content(
            &oid_components("1.3.6.1.4.1.57264.1.8"),
            utf8_extension_der(ISSUER),
        ),
        CustomExtension::from_oid_content(
            &oid_components("1.3.6.1.4.1.57264.1.9"),
            utf8_extension_der(WORKFLOW_REF),
        ),
        CustomExtension::from_oid_content(
            &oid_components("1.3.6.1.4.1.57264.1.12"),
            utf8_extension_der(CERTIFICATE_REPOSITORY),
        ),
        CustomExtension::from_oid_content(
            &oid_components("1.3.6.1.4.1.57264.1.13"),
            utf8_extension_der(REVISION),
        ),
        CustomExtension::from_oid_content(
            &oid_components("1.3.6.1.4.1.57264.1.14"),
            utf8_extension_der("refs/tags/v1.2.3"),
        ),
    ];
    let key_pair = KeyPair::generate().unwrap();
    params.self_signed(&key_pair).unwrap().der().to_vec()
}

/// A genuine `application/vnd.dev.sigstore.bundle.v0.3+json` document, shaped
/// exactly per the real protobuf spec field names (`mediaType`,
/// `verificationMaterial.certificate.rawBytes`, `dsseEnvelope`), carrying a
/// real ANNPack build-provenance statement as its DSSE payload.
fn bundle_json(repository: &str, revision: &str) -> Vec<u8> {
    let statement = json!({
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": "annpack-x86_64-unknown-linux-gnu.tar.gz", "digest": {"sha256": "a".repeat(64)}}],
        "predicateType": "https://annpack.dev/attestations/build/v1",
        "predicate": {
            "builder": {"id": "github-actions:release:1", "annpack_version": "0.7.0-rc1", "annpack_binary_sha256": null},
            "source": {
                "repository": repository,
                "revision": format!("git:{revision}"),
                "tree_digest": "b".repeat(64),
                "tree_digest_algorithm": "blake3",
                "format": "markdown"
            },
            "build": {
                "invocation_id": "1", "started_at": "2026-08-06T00:00:00Z", "finished_at": "2026-08-06T00:01:00Z",
                "parameters": {}, "environment": {}, "platform": null, "locked": null
            },
            "annpack": {
                "artifact_root": "c".repeat(64), "logical_content_root": null,
                "manifest_format_version": 4, "source_binding": "authenticated"
            }
        }
    });
    let payload = serde_json::to_vec(&statement).unwrap();

    let bundle = json!({
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {
            "certificate": { "rawBytes": base64_encode(&github_certificate()) },
            "tlogEntries": [{
                "logIndex": "44238954",
                "logId": {"keyId": "wNI9atQGlz+VWfO6LRygH4QUfY/8W4RFwiT5i5WRgB0="},
                "kindVersion": {"kind": "dsse", "version": "0.0.1"},
                "integratedTime": "1738060096",
                "inclusionPromise": {"signedEntryTimestamp": "AA=="},
                "inclusionProof": {
                    "logIndex": "44238954",
                    "rootHash": "TiowMOu0x46fW4pXrRyW7TeVb6f1/VDnDZWcP1xL/HU=",
                    "treeSize": "44238955",
                    "hashes": [],
                    "checkpoint": {"envelope": "rekor.sigstore.dev - 1193050959916656506\n44238955\nTiowMOu0x46fW4pXrRyW7TeVb6f1/VDnDZWcP1xL/HU=\n\n— rekor.sigstore.dev wNI9ajBEAiBF3lyT0Jg0paKCvqJQ0t97+hcneAqZHeiRuLinOba/YQIgG65ZKAhE+byLy+VQ4/14FwvJG0FMhq4CNoDONpzvOMc=\n"}
                },
                "canonicalizedBody": "e30="
            }]
        },
        "dsseEnvelope": {
            "payload": base64_encode(&payload),
            "payloadType": "application/vnd.in-toto+json",
            "signatures": [{"sig": base64_encode(b"not-a-real-signature"), "keyid": ""}]
        }
    });
    serde_json::to_vec(&bundle).unwrap()
}

#[test]
fn a_genuine_bundle_parses_and_its_claims_are_extracted() {
    let bytes = bundle_json(PREDICATE_REPOSITORY, REVISION);
    let bundle = parse_bundle(&bytes).unwrap();
    let der = bundle.leaf_certificate_der().unwrap();
    let claims = adyar::attestation::extract_certificate_claims(&der).unwrap();
    assert_eq!(claims.issuer.as_deref(), Some(ISSUER));
    assert_eq!(
        claims.source_repository_uri.as_deref(),
        Some(CERTIFICATE_REPOSITORY)
    );
    assert_eq!(claims.build_signer_uri.as_deref(), Some(WORKFLOW_REF));
}

#[test]
fn cli_rejects_a_malformed_explicit_trusted_root_with_one_json_object() {
    let temp = tempfile::TempDir::new().unwrap();
    let artifact_path = temp.path().join("artifact.annpack");
    let bundle_path = temp.path().join("bundle.json");
    let root_path = temp.path().join("trusted-root.json");
    std::fs::write(&artifact_path, b"artifact").unwrap();
    std::fs::write(&bundle_path, bundle_json(PREDICATE_REPOSITORY, REVISION)).unwrap();
    std::fs::write(&root_path, b"not json").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_adyar"))
        .args([
            "provenance",
            "verify-github",
            artifact_path.to_str().unwrap(),
            bundle_path.to_str().unwrap(),
            "--trusted-root",
            root_path.to_str().unwrap(),
            "--allowed-issuer",
            ISSUER,
            "--allowed-repository",
            CERTIFICATE_REPOSITORY,
            "--allowed-workflow-ref",
            WORKFLOW_REF,
            "--json",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["error"]["kind"], "malformed_trusted_root");
}

#[test]
fn cli_distinguishes_a_malformed_bundle_from_a_malformed_root() {
    let temp = tempfile::TempDir::new().unwrap();
    let artifact_path = temp.path().join("artifact.annpack");
    let bundle_path = temp.path().join("bundle.json");
    let root_path = temp.path().join("trusted-root.json");
    std::fs::write(&artifact_path, b"artifact").unwrap();
    std::fs::write(&bundle_path, b"not json").unwrap();
    std::fs::write(&root_path, b"also not json").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_adyar"))
        .args([
            "provenance",
            "verify-github",
            artifact_path.to_str().unwrap(),
            bundle_path.to_str().unwrap(),
            "--trusted-root",
            root_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["error"]["kind"], "malformed_bundle");
    assert_eq!(envelope["stage"], "bundle_structure");
}

#[test]
fn local_ed25519_provenance_verification_is_unaffected() {
    // Confirms the new command lives beside, not inside, the existing path:
    // the original CLI subcommand and its behavior are untouched.
    let temp = tempfile::TempDir::new().unwrap();
    let source = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    let artifact = temp.path().join("a.annpack");
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_adyar"))
        .args([
            "build",
            &source,
            "--output",
            artifact.to_str().unwrap(),
            "--name",
            "docs",
            "--version",
            "1.0.0",
            "--json",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let key = temp.path().join("builder.key");
    let keygen = std::process::Command::new(env!("CARGO_BIN_EXE_adyar"))
        .args(["keygen", "--output", key.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(keygen.status.success());
    let key_report: serde_json::Value = serde_json::from_slice(&keygen.stdout).unwrap();
    let public_key = key_report["public_key"].as_str().unwrap();

    let provenance = temp.path().join("prov.json");
    let create = std::process::Command::new(env!("CARGO_BIN_EXE_adyar"))
        .args([
            "provenance",
            "create",
            artifact.to_str().unwrap(),
            "--output",
            provenance.to_str().unwrap(),
            "--repository",
            "r",
            "--revision",
            "v",
            "--builder-id",
            "id",
            "--system-clock",
        ])
        .output()
        .unwrap();
    assert!(
        create.status.success(),
        "{}",
        String::from_utf8_lossy(&create.stderr)
    );

    let sign = std::process::Command::new(env!("CARGO_BIN_EXE_adyar"))
        .args([
            "provenance",
            "sign",
            provenance.to_str().unwrap(),
            "--key",
            key.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(sign.status.success());

    let verify = std::process::Command::new(env!("CARGO_BIN_EXE_adyar"))
        .args([
            "provenance",
            "verify",
            artifact.to_str().unwrap(),
            provenance.to_str().unwrap(),
            "--trusted-builder-key",
            public_key,
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&verify.stdout).unwrap();
    assert_eq!(report["verified"], true);
    assert_eq!(report["completeness"], "complete");
}
