//! Build provenance, exercised with real Ed25519 keys and real artifacts.
//!
//! Each test breaks one binding at a time so a failure is attributable to that
//! binding rather than to a statement that was never valid. The distinction
//! this module exists to hold -- carried claims are never reported as
//! verified, and legacy source binding is never reported as full binding -- is
//! asserted directly rather than inferred from "it didn't crash".

use std::path::{Path, PathBuf};

use annpack::provenance::{
    BindingStatus, BuildProvenanceInput, BuilderIdentity, Completeness, DsseSignature, Envelope,
    EnvelopeSignature, SourceDigestBinding, create_build_provenance,
    create_legacy_build_provenance, sign_provenance, verify_build_provenance,
};
use annpack::trust::key_identity;
use tempfile::TempDir;

const BUILDER_KEY: [u8; 32] = [7; 32];
const OTHER_KEY: [u8; 32] = [8; 32];

fn builder_pub() -> String {
    key_identity(&BUILDER_KEY).1
}
fn other_pub() -> String {
    key_identity(&OTHER_KEY).1
}

struct Fixture {
    _temp: TempDir,
    artifact: PathBuf,
    binary: PathBuf,
}

fn fixture() -> Fixture {
    fixture_from("fixtures/docs-v1", "vendor-docs")
}

/// A build is deterministic in its source and name, so two fixtures built from
/// the same corpus under the same name are byte-identical -- which made an
/// early version of `a_provenance_subject_naming_the_wrong_file_is_detected`
/// pass for the wrong reason: the "other" file was accidentally identical to
/// the first. Genuinely distinct fixtures need either a different corpus or a
/// different declared name.
fn fixture_from(corpus: &str, name: &str) -> Fixture {
    let temp = TempDir::new().unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(corpus);
    let artifact = temp.path().join("pack.annpack");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args([
            "build",
            source.to_str().unwrap(),
            "--output",
            artifact.to_str().unwrap(),
            "--name",
            name,
            "--version",
            "1.0.0",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Fixture {
        artifact,
        binary: PathBuf::from(env!("CARGO_BIN_EXE_annpack")),
        _temp: temp,
    }
}

fn input(fixture: &Fixture, with_binary: bool) -> BuildProvenanceInput<'_> {
    BuildProvenanceInput {
        artifact_path: &fixture.artifact,
        repository: "github.com/example/docs".into(),
        revision: "git:deadbeef".into(),
        builder_id: "test-workflow".into(),
        builder_binary_path: with_binary.then_some(fixture.binary.as_path()),
        invocation_id: "invocation-1".into(),
        started_at: "2026-08-06T00:00:00Z".into(),
        finished_at: "2026-08-06T00:01:00Z".into(),
        parameters: Default::default(),
        environment: Default::default(),
        platform: Some("x86_64-unknown-linux-gnu".into()),
        locked: Some(true),
    }
}

#[test]
fn a_correctly_signed_statement_verifies_completely() {
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, true)).unwrap();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    let report = verify_build_provenance(
        &envelope,
        &fixture.artifact,
        &[builder_pub()],
        Some(&fixture.binary),
    )
    .unwrap();

    assert!(report.verified, "{:?}", report.issues);
    assert_eq!(report.completeness, Completeness::Complete);
    assert_eq!(report.envelope_signature, EnvelopeSignature::Valid);
    assert_eq!(report.builder_identity, BuilderIdentity::Trusted);
    assert_eq!(report.artifact_integrity, BindingStatus::Verified);
    assert_eq!(report.distributed_file_digest, BindingStatus::Verified);
    assert_eq!(report.artifact_root_binding, BindingStatus::Verified);
    assert_eq!(report.builder_binary_binding, BindingStatus::Verified);
    assert_eq!(report.builder_version_binding, BindingStatus::Verified);
    assert_eq!(
        report.source_digest_binding,
        SourceDigestBinding::Authenticated
    );
    // repository/revision are asserted, never proven -- this is the one
    // distinction the whole module exists to hold.
    assert_eq!(
        report.repository_claim,
        annpack::provenance::ClaimStatus::Carried
    );
    assert_eq!(
        report.revision_claim,
        annpack::provenance::ClaimStatus::Carried
    );
}

#[test]
fn without_a_builder_binary_the_two_builder_claims_are_carried_not_verified() {
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, false)).unwrap();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();
    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();

    // Signature, artifact and source bindings are unaffected by the absence of
    // a builder binary; only the two claims that binary would corroborate stay
    // unverified.
    assert!(report.verified);
    assert_eq!(report.builder_binary_binding, BindingStatus::Unsupported);
    assert_eq!(report.builder_version_binding, BindingStatus::Unsupported);
    assert!(
        report
            .assumptions
            .iter()
            .any(|note| note.contains("carried claims"))
    );
}

#[test]
fn the_distributed_file_being_modified_is_detected() {
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, false)).unwrap();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    // Append one byte after signing. The artifact root is unaffected -- the
    // reader only ever reads the referenced sections -- so this specifically
    // isolates the *distributed file* digest binding from artifact integrity.
    let mut bytes = std::fs::read(&fixture.artifact).unwrap();
    bytes.push(0);
    std::fs::write(&fixture.artifact, &bytes).unwrap();

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.distributed_file_digest, BindingStatus::Mismatched);
    assert_eq!(report.completeness, Completeness::Invalid);
}

#[test]
fn a_provenance_subject_naming_the_wrong_file_is_detected() {
    let signed_for = fixture();
    let other = fixture_from("fixtures/docs-v2", "other-vendor-docs");
    let statement = create_build_provenance(input(&signed_for, false)).unwrap();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    // Verify against a different, independently built artifact. This is the
    // "platform provenance attached to another platform asset" case: same
    // schema, same signer, wrong file.
    let report =
        verify_build_provenance(&envelope, &other.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.distributed_file_digest, BindingStatus::Mismatched);
}

#[test]
fn a_tampered_artifact_root_claim_is_detected() {
    let fixture = fixture();
    let mut statement = create_build_provenance(input(&fixture, false)).unwrap();
    statement.predicate.annpack.artifact_root = "0".repeat(64);
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.artifact_root_binding, BindingStatus::Mismatched);
}

#[test]
fn a_tampered_logical_root_claim_is_detected() {
    let fixture = fixture();
    let mut statement = create_build_provenance(input(&fixture, false)).unwrap();
    statement.predicate.annpack.logical_content_root = Some("1".repeat(64));
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.logical_root_binding, BindingStatus::Mismatched);
}

#[test]
fn a_tampered_source_digest_claim_is_detected() {
    let fixture = fixture();
    let mut statement = create_build_provenance(input(&fixture, false)).unwrap();
    statement.predicate.source.tree_digest = "2".repeat(64);
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(
        report.source_digest_binding,
        SourceDigestBinding::Mismatched
    );
}

#[test]
fn repository_and_revision_can_change_freely_without_invalidating_the_signature() {
    // Deliberately demonstrating the limitation the module documents: nothing
    // binds these fields to external truth, only to what the signer wrote. If
    // this test ever started failing because tampering the fields broke the
    // signature, the module's central claim -- that these are carried, not
    // verified -- would be false and every other test would be reporting a
    // false sense of strength.
    let fixture = fixture();
    let mut statement = create_build_provenance(input(&fixture, false)).unwrap();
    statement.predicate.source.repository = "github.com/attacker/evil".into();
    statement.predicate.source.revision = "git:0000000".into();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(
        report.verified,
        "changing carried fields before signing must not itself break verification"
    );
    assert_eq!(
        report.repository_claim,
        annpack::provenance::ClaimStatus::Carried
    );
}

#[test]
fn changing_the_repository_claim_after_signing_invalidates_the_signature() {
    // The complementary case: tampering *after* signing, without re-signing,
    // is what the signature actually defends against.
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, false)).unwrap();
    let mut envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();
    let payload_json = base64_decode(&envelope.payload);
    let mut tampered: serde_json::Value = serde_json::from_slice(&payload_json).unwrap();
    tampered["predicate"]["source"]["repository"] = "github.com/attacker/evil".into();
    envelope.payload = base64_encode(&serde_json::to_vec(&tampered).unwrap());

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.envelope_signature, EnvelopeSignature::Invalid);
}

#[test]
fn changing_the_revision_claim_after_signing_invalidates_the_signature() {
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, false)).unwrap();
    let mut envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();
    let payload_json = base64_decode(&envelope.payload);
    let mut tampered: serde_json::Value = serde_json::from_slice(&payload_json).unwrap();
    tampered["predicate"]["source"]["revision"] = "git:0000000".into();
    envelope.payload = base64_encode(&serde_json::to_vec(&tampered).unwrap());

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.envelope_signature, EnvelopeSignature::Invalid);
}

#[test]
fn a_valid_signature_from_an_untrusted_builder_is_not_trusted() {
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, false)).unwrap();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    // Signed by BUILDER_KEY, but the caller only trusts OTHER_KEY. The
    // signature is genuinely valid; it must still fail verification.
    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[other_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.builder_identity, BuilderIdentity::Untrusted);
    assert_eq!(report.envelope_signature, EnvelopeSignature::Invalid);
}

#[test]
fn an_artifact_signing_key_is_not_automatically_a_trusted_builder() {
    // Simulates "artifact-signing key used as the builder key without explicit
    // builder trust": the artifact's own signing key signs provenance, but the
    // verifier's trusted-builder list -- built from separate policy -- never
    // named it.
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, false)).unwrap();
    let artifact_signing_key = [42_u8; 32];
    let envelope = sign_provenance(&statement, &artifact_signing_key).unwrap();

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.builder_identity, BuilderIdentity::Untrusted);
}

#[test]
fn no_trusted_keys_supplied_reports_unknown_not_untrusted() {
    // Distinct from the untrusted case: here trust was never evaluated at all,
    // which is a different fact than "evaluated and rejected".
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, false)).unwrap();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    let report = verify_build_provenance(&envelope, &fixture.artifact, &[], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.builder_identity, BuilderIdentity::Unknown);
    assert!(
        report
            .assumptions
            .iter()
            .any(|note| note.contains("builder identity is unknown"))
    );
}

#[test]
fn an_unsupported_predicate_type_is_refused() {
    let fixture = fixture();
    let mut statement = create_build_provenance(input(&fixture, false)).unwrap();
    statement.predicate_type = "https://example.com/some-other-predicate/v1".into();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert!(!report.predicate_type_supported);
}

#[test]
fn an_unsupported_statement_type_is_refused() {
    let fixture = fixture();
    let mut statement = create_build_provenance(input(&fixture, false)).unwrap();
    statement.statement_type = "https://in-toto.io/Statement/v0".into();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert!(!report.predicate_type_supported);
}

#[test]
fn duplicate_or_missing_subjects_are_refused() {
    let fixture = fixture();
    let base = create_build_provenance(input(&fixture, false)).unwrap();

    let mut duplicated = base.clone();
    duplicated.subject.push(duplicated.subject[0].clone());
    let envelope = sign_provenance(&duplicated, &BUILDER_KEY).unwrap();
    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert!(!report.subject_valid);

    let mut empty = base;
    empty.subject.clear();
    let envelope = sign_provenance(&empty, &BUILDER_KEY).unwrap();
    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert!(!report.subject_valid);
    assert_eq!(report.distributed_file_digest, BindingStatus::Missing);
}

#[test]
fn a_malformed_dsse_payload_is_refused_before_interpretation() {
    let envelope = Envelope {
        payload: base64_encode(b"{not json"),
        payload_type: annpack::provenance::DSSE_PAYLOAD_TYPE.into(),
        signatures: vec![DsseSignature {
            keyid: builder_pub(),
            sig: "00".repeat(64),
        }],
    };
    let fixture = fixture();
    let error = verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None);
    assert!(
        error.is_err(),
        "malformed payload must not parse into a report"
    );
}

#[test]
fn an_unsupported_dsse_payload_type_is_flagged() {
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, false)).unwrap();
    let mut envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();
    envelope.payload_type = "application/octet-stream".into();

    let report =
        verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    // Changing payload_type also changes what PAE was computed over at
    // signing time, so the signature itself no longer validates -- this
    // doubles as the "signature valid over a different payload" case.
    assert_eq!(report.envelope_signature, EnvelopeSignature::Invalid);
}

#[test]
fn a_signature_copied_onto_a_different_payload_does_not_verify() {
    let fixture = fixture();
    let first = create_build_provenance(input(&fixture, false)).unwrap();
    let first_envelope = sign_provenance(&first, &BUILDER_KEY).unwrap();

    let mut second = first;
    second.predicate.build.invocation_id = "different-invocation".into();
    let mut spliced = sign_provenance(&second, &BUILDER_KEY).unwrap();
    // Replace the genuine signature over `second` with the one computed over
    // `first`. PAE includes payload length and bytes, so this is exactly the
    // forgery DSSE's construction is meant to prevent.
    spliced.signatures = first_envelope.signatures;

    let report =
        verify_build_provenance(&spliced, &fixture.artifact, &[builder_pub()], None).unwrap();
    assert!(!report.verified);
    assert_eq!(report.envelope_signature, EnvelopeSignature::Invalid);
}

#[test]
fn provenance_cannot_be_created_from_an_artifact_that_fails_integrity() {
    let fixture = fixture();
    let mut bytes = std::fs::read(&fixture.artifact).unwrap();
    // Flip a byte inside the header root hash (offset 48..80), leaving every
    // section self-consistent so only the content-root check fires.
    bytes[48] ^= 0xFF;
    std::fs::write(&fixture.artifact, &bytes).unwrap();

    let result = create_build_provenance(input(&fixture, false));
    assert!(
        result.is_err(),
        "provenance was created for an artifact that fails integrity verification"
    );
}

#[test]
fn a_changed_builder_binary_digest_is_detected_when_a_binary_is_supplied() {
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, true)).unwrap();
    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();

    // Verify against a different executable: the shell, standing in for "the
    // wrong binary was packaged".
    let wrong_binary = PathBuf::from("/bin/sh");
    let report = verify_build_provenance(
        &envelope,
        &fixture.artifact,
        &[builder_pub()],
        Some(&wrong_binary),
    )
    .unwrap();
    assert!(!report.verified);
    assert_eq!(report.builder_binary_binding, BindingStatus::Mismatched);
}

#[test]
fn a_legacy_artifact_produces_partial_source_binding_not_full() {
    let legacy = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("spec/test-vectors/compat/manifest-v1-legacy.annpack");
    let statement = create_legacy_build_provenance(
        BuildProvenanceInput {
            artifact_path: &legacy,
            repository: "github.com/example/legacy".into(),
            revision: "git:legacy".into(),
            builder_id: "test-workflow".into(),
            builder_binary_path: None,
            invocation_id: "invocation-legacy".into(),
            started_at: "2026-08-06T00:00:00Z".into(),
            finished_at: "2026-08-06T00:01:00Z".into(),
            parameters: Default::default(),
            environment: Default::default(),
            platform: None,
            locked: None,
        },
        "3".repeat(64),
    )
    .unwrap();
    assert_eq!(
        statement.predicate.annpack.source_binding,
        "absent_legacy_artifact"
    );

    let envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();
    let report = verify_build_provenance(&envelope, &legacy, &[builder_pub()], None).unwrap();

    // The legacy artifact still verifies -- integrity, file digest, artifact
    // root all hold -- but completeness must say so honestly rather than
    // reporting full source-to-artifact binding it cannot support.
    assert!(report.verified, "{:?}", report.issues);
    assert_eq!(
        report.completeness,
        Completeness::PartialLegacySourceBinding
    );
    assert_eq!(
        report.source_digest_binding,
        SourceDigestBinding::AbsentLegacyArtifact
    );
    assert!(
        report
            .assumptions
            .iter()
            .any(|note| note.contains("cannot corroborate"))
    );
}

#[test]
fn create_build_provenance_refuses_a_legacy_artifact() {
    // The common-path function must not silently accept the weaker case; a
    // caller has to name create_legacy_build_provenance explicitly.
    let legacy = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("spec/test-vectors/compat/manifest-v1-legacy.annpack");
    let result = create_build_provenance(BuildProvenanceInput {
        artifact_path: &legacy,
        repository: "r".into(),
        revision: "v".into(),
        builder_id: "b".into(),
        builder_binary_path: None,
        invocation_id: "i".into(),
        started_at: "2026-08-06T00:00:00Z".into(),
        finished_at: "2026-08-06T00:01:00Z".into(),
        parameters: Default::default(),
        environment: Default::default(),
        platform: None,
        locked: None,
    });
    assert!(result.is_err());
}

#[test]
fn create_legacy_build_provenance_refuses_a_format_four_artifact() {
    // The reverse guard: the weaker function must not be usable on an artifact
    // that actually has authenticated binding, which would silently discard it.
    let fixture = fixture();
    let result = create_legacy_build_provenance(input(&fixture, false), "4".repeat(64));
    assert!(result.is_err());
}

#[test]
fn oversized_signature_lists_are_refused_before_verification() {
    let fixture = fixture();
    let statement = create_build_provenance(input(&fixture, false)).unwrap();
    let mut envelope = sign_provenance(&statement, &BUILDER_KEY).unwrap();
    let one = envelope.signatures[0].clone();
    envelope.signatures = std::iter::repeat_n(one, 17).collect();

    let result = verify_build_provenance(&envelope, &fixture.artifact, &[builder_pub()], None);
    assert!(result.is_err());
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
