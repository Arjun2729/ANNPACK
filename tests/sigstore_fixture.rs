//! Repository-owned GitHub/Sigstore fixture and controlled failure matrix.

#![cfg(feature = "github-attestation")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use annpack::attestation::{
    BuilderPolicy, ClaimAgreement, PolicyVerdict, VerificationState, verify_github_attestation,
};
use base64::Engine;
use serde_json::Value;
use sha2::{Digest, Sha256};

const ISSUER: &str = "https://token.actions.githubusercontent.com";
const REPOSITORY: &str = "https://github.com/Arjun2729/ANNPACK";
const WORKFLOW_REF: &str = "https://github.com/Arjun2729/ANNPACK/.github/workflows/sigstore-verification-fixture.yml@refs/heads/codex/sigstore-verification-fixture";
const REVISION: &str = "9cdaf8ae36659bfa7cc825ec4aacc3e86a586df0";
const REKOR_LOG_ID: &str = "wNI9atQGlz+VWfO6LRygH4QUfY/8W4RFwiT5i5WRgB0=";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/sigstore-v1")
        .join(name)
}

fn policy(repository: &str, workflow_ref: &str) -> BuilderPolicy {
    BuilderPolicy {
        allowed_issuers: vec![ISSUER.into()],
        allowed_repositories: vec![repository.into()],
        allowed_workflow_refs: vec![workflow_ref.into()],
    }
}

fn verify(
    bundle: &[u8],
    root: &[u8],
    artifact: &Path,
    policy: &BuilderPolicy,
) -> annpack::attestation::GitHubAttestationReport {
    verify_github_attestation(bundle, root, artifact, policy).unwrap()
}

fn original() -> (Vec<u8>, Vec<u8>, PathBuf) {
    (
        fs::read(fixture("sigstore-fixture.bundle.json")).unwrap(),
        fs::read(fixture("sigstore-fixture.trusted-root.json")).unwrap(),
        fixture("sigstore-fixture.annpack"),
    )
}

fn flip_base64(value: &mut Value) {
    let text = value.as_str().unwrap();
    let first = if text.starts_with('A') { "B" } else { "A" };
    *value = Value::String(format!("{first}{}", &text[1..]));
}

#[test]
fn official_keyless_fixture_reaches_the_pinned_verified_report() {
    let (bundle, root, artifact) = original();
    let report = verify(&bundle, &root, &artifact, &policy(REPOSITORY, WORKFLOW_REF));
    let expected: Value = serde_json::from_slice(
        &fs::read(fixture("sigstore-fixture.expected-report.json")).unwrap(),
    )
    .unwrap();

    assert!(report.verified);
    assert!(report.fully_offline);
    assert_eq!(serde_json::to_value(report).unwrap(), expected);
}

#[test]
fn fixture_verification_succeeds_with_every_network_proxy_unreachable() {
    let output = Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args([
            "provenance",
            "verify-github",
            fixture("sigstore-fixture.annpack").to_str().unwrap(),
            fixture("sigstore-fixture.bundle.json").to_str().unwrap(),
            "--trusted-root",
            fixture("sigstore-fixture.trusted-root.json")
                .to_str()
                .unwrap(),
            "--allowed-issuer",
            ISSUER,
            "--allowed-repository",
            REPOSITORY,
            "--allowed-workflow-ref",
            WORKFLOW_REF,
            "--json",
        ])
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .env("HTTPS_PROXY", "http://127.0.0.1:9")
        .env("ALL_PROXY", "http://127.0.0.1:9")
        .env("NO_PROXY", "")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["fully_offline"], true);
    assert_eq!(report["verified"], true);
}

#[test]
fn wrong_trusted_root_fails_closed() {
    let (bundle, root, artifact) = original();
    let mut wrong: Value = serde_json::from_slice(&root).unwrap();
    wrong["certificateAuthorities"]
        .as_array_mut()
        .unwrap()
        .truncate(1);
    let report = verify(
        &bundle,
        &serde_json::to_vec(&wrong).unwrap(),
        &artifact,
        &policy(REPOSITORY, WORKFLOW_REF),
    );
    assert!(!report.verified);
    assert_ne!(report.certificate_chain, VerificationState::Verified);
}

#[test]
fn trusted_root_requires_both_fulcio_and_rekor_material() {
    let (bundle, root, artifact) = original();
    for missing in ["certificateAuthorities", "tlogs"] {
        let mut incomplete: Value = serde_json::from_slice(&root).unwrap();
        incomplete[missing] = Value::Array(Vec::new());
        let error = verify_github_attestation(
            &bundle,
            &serde_json::to_vec(&incomplete).unwrap(),
            &artifact,
            &policy(REPOSITORY, WORKFLOW_REF),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must contain a Fulcio authority and a Rekor log"),
            "{missing}: {error}"
        );
    }
}

#[test]
fn artifact_mutation_fails_closed() {
    let (bundle, root, artifact) = original();
    let temp = tempfile::TempDir::new().unwrap();
    let changed = temp.path().join("changed.annpack");
    let mut bytes = fs::read(artifact).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&changed, bytes).unwrap();
    let report = verify(&bundle, &root, &changed, &policy(REPOSITORY, WORKFLOW_REF));
    assert!(!report.verified);
    assert_ne!(report.artifact_signature, VerificationState::Verified);
}

#[test]
fn bundle_signature_mutation_fails_closed() {
    let (bundle, root, artifact) = original();
    let mut changed: Value = serde_json::from_slice(&bundle).unwrap();
    flip_base64(&mut changed["dsseEnvelope"]["signatures"][0]["sig"]);
    let report = verify(
        &serde_json::to_vec(&changed).unwrap(),
        &root,
        &artifact,
        &policy(REPOSITORY, WORKFLOW_REF),
    );
    assert!(!report.verified);
    assert_ne!(report.artifact_signature, VerificationState::Verified);
}

#[test]
fn authenticated_predicate_mutation_fails_closed() {
    let (bundle, root, artifact) = original();
    let mut changed: Value = serde_json::from_slice(&bundle).unwrap();
    let payload = changed["dsseEnvelope"]["payload"].as_str().unwrap();
    let mut statement: Value = serde_json::from_slice(
        &base64::engine::general_purpose::STANDARD
            .decode(payload)
            .unwrap(),
    )
    .unwrap();
    statement["predicate"]["builder"]["id"] = Value::String("mutated".into());
    changed["dsseEnvelope"]["payload"] = Value::String(
        base64::engine::general_purpose::STANDARD.encode(serde_json::to_vec(&statement).unwrap()),
    );
    let report = verify(
        &serde_json::to_vec(&changed).unwrap(),
        &root,
        &artifact,
        &policy(REPOSITORY, WORKFLOW_REF),
    );
    assert!(!report.verified);
    assert_ne!(report.artifact_signature, VerificationState::Verified);
}

#[test]
fn rekor_entry_mutation_fails_closed() {
    let (bundle, root, artifact) = original();
    let mut changed: Value = serde_json::from_slice(&bundle).unwrap();
    flip_base64(&mut changed["verificationMaterial"]["tlogEntries"][0]["canonicalizedBody"]);
    let report = verify(
        &serde_json::to_vec(&changed).unwrap(),
        &root,
        &artifact,
        &policy(REPOSITORY, WORKFLOW_REF),
    );
    assert!(!report.verified);
    assert_ne!(report.rekor_entry_consistency, VerificationState::Verified);
}

#[test]
fn repository_policy_mismatch_preserves_crypto_but_denies_trust() {
    let (bundle, root, artifact) = original();
    let report = verify(
        &bundle,
        &root,
        &artifact,
        &policy("https://github.com/example/wrong", WORKFLOW_REF),
    );
    assert_eq!(report.artifact_signature, VerificationState::Verified);
    assert_eq!(report.certificate_chain, VerificationState::Verified);
    assert_eq!(report.rekor_inclusion, VerificationState::Verified);
    assert_eq!(report.builder_policy, PolicyVerdict::Untrusted);
    assert!(!report.verified);
}

#[test]
fn workflow_policy_mismatch_preserves_crypto_but_denies_trust() {
    let (bundle, root, artifact) = original();
    let report = verify(
        &bundle,
        &root,
        &artifact,
        &policy(
            REPOSITORY,
            "https://github.com/example/wrong.yml@refs/heads/main",
        ),
    );
    assert_eq!(report.artifact_signature, VerificationState::Verified);
    assert_eq!(report.certificate_chain, VerificationState::Verified);
    assert_eq!(report.rekor_inclusion, VerificationState::Verified);
    assert_eq!(report.builder_policy, PolicyVerdict::Untrusted);
    assert!(!report.verified);
}

#[test]
fn subject_rekor_certificate_and_predicate_identities_are_mutually_bound() {
    let (bundle, root, artifact) = original();
    let bundle_json: Value = serde_json::from_slice(&bundle).unwrap();
    let report = verify(&bundle, &root, &artifact, &policy(REPOSITORY, WORKFLOW_REF));

    assert_eq!(
        report
            .certificate_claims
            .subject_alternative_name
            .as_deref(),
        Some(WORKFLOW_REF)
    );
    assert_eq!(
        report.certificate_claims.build_signer_uri.as_deref(),
        Some(WORKFLOW_REF)
    );
    assert_eq!(
        report.certificate_claims.source_repository_uri.as_deref(),
        Some(REPOSITORY)
    );
    assert_eq!(
        report
            .certificate_claims
            .source_repository_digest
            .as_deref(),
        Some(REVISION)
    );
    assert_eq!(report.repository_claim_agreement, ClaimAgreement::Agree);
    assert_eq!(report.revision_claim_agreement, ClaimAgreement::Agree);
    assert_eq!(report.selected_rekor_log_ids, vec![REKOR_LOG_ID]);
    assert_eq!(
        bundle_json["verificationMaterial"]["tlogEntries"][0]["logId"]["keyId"],
        REKOR_LOG_ID
    );
    assert_eq!(
        report.subject_binding,
        annpack::provenance::BindingStatus::Verified
    );
    assert_eq!(report.rekor_entry_consistency, VerificationState::Verified);
}

#[test]
fn every_fixture_digest_is_pinned_and_current() {
    let pins = fs::read_to_string(fixture("sigstore-fixture.sha256")).unwrap();
    let expected = pins
        .lines()
        .map(|line| {
            let (digest, path) = line.split_once("  ").unwrap();
            (path.rsplit('/').next().unwrap(), digest)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    for name in [
        "sigstore-fixture.annpack",
        "sigstore-fixture.bundle.json",
        "sigstore-fixture.predicate.json",
        "sigstore-fixture.trusted-root.json",
        "sigstore-fixture.expected-report.json",
    ] {
        let digest = format!("{:x}", Sha256::digest(fs::read(fixture(name)).unwrap()));
        assert_eq!(expected.get(name), Some(&digest.as_str()), "{name}");
    }
}
