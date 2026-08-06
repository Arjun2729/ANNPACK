//! GitHub attestation bundle parsing and policy matching, exercised through
//! genuine bundle JSON and the CLI, not just the library's own unit tests.
//!
//! Every test in this file operates under one governing fact, stated here
//! once rather than in each assertion: certificate-chain-to-Fulcio-root
//! verification is not implemented, so `verified` must be `false` in every
//! case below, including the ones where every other check passes cleanly.
//! `verified_is_never_true_even_when_everything_else_matches` makes this the
//! subject of its own test rather than an incidental assertion, because it is
//! the single property this whole module exists to hold.

#![cfg(feature = "github-attestation")]

use annpack::attestation::{ChainVerification, ClaimAgreement, PolicyVerdict, parse_bundle};
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
            "certificate": { "rawBytes": base64_encode(&github_certificate()) }
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
    let claims = annpack::attestation::extract_certificate_claims(&der).unwrap();
    assert_eq!(claims.issuer.as_deref(), Some(ISSUER));
    assert_eq!(
        claims.source_repository_uri.as_deref(),
        Some(CERTIFICATE_REPOSITORY)
    );
    assert_eq!(claims.build_signer_uri.as_deref(), Some(WORKFLOW_REF));
}

#[test]
fn verified_is_never_true_even_when_everything_else_matches() {
    // The predicate's repository/revision agree with the certificate, and the
    // policy matches every claim. If any implementation change ever makes
    // `verified` true under these conditions, it did so without adding
    // certificate-chain verification -- the one thing that would actually
    // justify it -- so this must keep failing until that specific gap closes.
    let bytes = bundle_json(PREDICATE_REPOSITORY, REVISION);
    let bundle = parse_bundle(&bytes).unwrap();
    let policy = annpack::attestation::BuilderPolicy {
        allowed_issuers: vec![ISSUER.to_string()],
        allowed_repositories: vec![CERTIFICATE_REPOSITORY.to_string()],
        allowed_workflow_refs: vec![WORKFLOW_REF.to_string()],
    };
    let report = annpack::attestation::evaluate_github_attestation(&bundle, &policy).unwrap();

    assert_eq!(
        report.policy.verdict,
        PolicyVerdict::Trusted,
        "{:?}",
        report.policy.issues
    );
    assert_eq!(report.repository_claim_agreement, ClaimAgreement::Agree);
    assert_eq!(report.revision_claim_agreement, ClaimAgreement::Agree);
    assert_eq!(report.certificate_chain, ChainVerification::NotImplemented);
    assert!(
        !report.verified,
        "verified must stay false without chain verification"
    );
}

#[test]
fn a_predicate_repository_disagreeing_with_the_certificate_is_flagged() {
    // The predicate claims one repository; the certificate says another. Both
    // are merely asserted -- this is not fraud detection -- but the
    // disagreement itself must be visible rather than silently dropped.
    let bytes = bundle_json("https://github.com/attacker/evil", REVISION);
    let bundle = parse_bundle(&bytes).unwrap();
    let policy = annpack::attestation::BuilderPolicy {
        allowed_issuers: vec![ISSUER.to_string()],
        allowed_repositories: vec![CERTIFICATE_REPOSITORY.to_string()],
        allowed_workflow_refs: vec![WORKFLOW_REF.to_string()],
    };
    let report = annpack::attestation::evaluate_github_attestation(&bundle, &policy).unwrap();
    assert_eq!(report.repository_claim_agreement, ClaimAgreement::Disagree);
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.contains("repository claim disagrees"))
    );
}

#[test]
fn a_generic_slsa_predicate_type_does_not_satisfy_annpack_policy() {
    // A bundle carrying GitHub's own generic SLSA predicate, not ANNPack's,
    // must not be silently reinterpreted as one.
    let statement = json!({
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": "x", "digest": {"sha256": "a".repeat(64)}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {"buildDefinition": {}}
    });
    let payload = serde_json::to_vec(&statement).unwrap();
    let bundle = json!({
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {"certificate": {"rawBytes": base64_encode(&github_certificate())}},
        "dsseEnvelope": {
            "payload": base64_encode(&payload),
            "payloadType": "application/vnd.in-toto+json",
            "signatures": [{"sig": base64_encode(b"x"), "keyid": ""}]
        }
    });
    let bytes = serde_json::to_vec(&bundle).unwrap();
    let parsed = parse_bundle(&bytes).unwrap();
    let policy = annpack::attestation::BuilderPolicy::default();
    // The predicate does not deserialize as an ANNPack BuildPredicate (no
    // `builder`/`source`/`build`/`annpack` fields), so evaluation fails at
    // extraction rather than silently proceeding with an empty predicate.
    let result = annpack::attestation::evaluate_github_attestation(&parsed, &policy);
    assert!(
        result.is_err(),
        "a generic SLSA predicate was accepted as an ANNPack one"
    );
}

#[test]
fn cli_reports_certificate_chain_not_implemented_and_exits_nonzero() {
    let temp = tempfile::TempDir::new().unwrap();
    let bundle_path = temp.path().join("bundle.json");
    std::fs::write(&bundle_path, bundle_json(PREDICATE_REPOSITORY, REVISION)).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args([
            "provenance",
            "verify-github",
            bundle_path.to_str().unwrap(),
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

    assert!(
        !output.status.success(),
        "verify-github must not exit 0 while unverified"
    );
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        envelope["error"]["kind"],
        "certificate_chain_not_implemented"
    );
    assert_eq!(envelope["details"]["policy"]["verdict"], "trusted");
    assert_eq!(envelope["details"]["verified"], false);
}

#[test]
fn cli_reports_untrusted_policy_when_repository_is_not_allowlisted() {
    let temp = tempfile::TempDir::new().unwrap();
    let bundle_path = temp.path().join("bundle.json");
    std::fs::write(&bundle_path, bundle_json(PREDICATE_REPOSITORY, REVISION)).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args([
            "provenance",
            "verify-github",
            bundle_path.to_str().unwrap(),
            "--allowed-issuer",
            ISSUER,
            "--allowed-repository",
            "https://github.com/other/repo",
            "--allowed-workflow-ref",
            WORKFLOW_REF,
            "--json",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["details"]["policy"]["verdict"], "untrusted");
}

#[test]
fn local_ed25519_provenance_verification_is_unaffected() {
    // Confirms the new command lives beside, not inside, the existing path:
    // the original CLI subcommand and its behavior are untouched.
    let temp = tempfile::TempDir::new().unwrap();
    let source = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    let artifact = temp.path().join("a.annpack");
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_annpack"))
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
    let keygen = std::process::Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args(["keygen", "--output", key.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(keygen.status.success());
    let key_report: serde_json::Value = serde_json::from_slice(&keygen.stdout).unwrap();
    let public_key = key_report["public_key"].as_str().unwrap();

    let provenance = temp.path().join("prov.json");
    let create = std::process::Command::new(env!("CARGO_BIN_EXE_annpack"))
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

    let sign = std::process::Command::new(env!("CARGO_BIN_EXE_annpack"))
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

    let verify = std::process::Command::new(env!("CARGO_BIN_EXE_annpack"))
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
