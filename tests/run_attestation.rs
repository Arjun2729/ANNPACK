#![cfg(feature = "signing")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use annpack::bundle::RunBundle;
use annpack::provenance::Envelope;
use annpack::release::{ChannelState, verify_channel_state};
use annpack::run_attestation::{
    ExternalWorkloadVerification, RunExpectations, VerifyRunAttestationInput,
    verify_run_attestation,
};
use annpack::trust::{TrustRoot, verify_trust_root};

const NOW: &str = "2030-01-02T00:00:00Z";
const START: &str = "2030-01-01T12:00:00Z";
const COMPLETE: &str = "2030-01-01T12:00:01Z";

struct Scenario {
    _temp: tempfile::TempDir,
    binary: &'static str,
    root: PathBuf,
    bundle: PathBuf,
    channel: PathBuf,
    statement: PathBuf,
    envelope: PathBuf,
    output: PathBuf,
    prompt: PathBuf,
    workload_key: PathBuf,
    workload_public: String,
    other_public: String,
    publisher_key: PathBuf,
    builder_key: PathBuf,
    artifact_root: String,
}

impl Scenario {
    fn path(&self, name: &str) -> PathBuf {
        self._temp.path().join(name)
    }

    fn run(&self, args: &[String]) -> Output {
        Command::new(self.binary).args(args).output().unwrap()
    }

    fn ok(&self, args: &[String]) -> Value {
        let output = self.run(args);
        assert!(
            output.status.success(),
            "command failed: {args:?}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap_or(Value::Null)
    }

    fn denied(&self, args: &[String]) -> Value {
        let output = self.run(args);
        assert!(
            !output.status.success(),
            "command unexpectedly succeeded: {args:?}\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
        serde_json::from_slice(&output.stdout).unwrap_or(Value::Null)
    }

    fn verify_args(&self, envelope: &Path, bundle: &Path) -> Vec<String> {
        vec![
            "run-attestation".into(),
            "verify".into(),
            envelope.display().to_string(),
            "--bundle".into(),
            bundle.display().to_string(),
            "--channel-state".into(),
            self.channel.display().to_string(),
            "--trust-root".into(),
            self.root.display().to_string(),
            "--expect-publisher".into(),
            "example.test".into(),
            "--expect-corpus".into(),
            "support".into(),
            "--expect-channel".into(),
            "production".into(),
            "--now".into(),
            NOW.into(),
            "--trusted-workload-key".into(),
            format!("support-agent={}", self.workload_public),
            "--expect-run-id".into(),
            "run-001".into(),
            "--expect-trace-id".into(),
            "trace-001".into(),
            "--expect-model".into(),
            "model-1".into(),
            "--prompt-policy".into(),
            self.prompt.display().to_string(),
            "--output-bytes".into(),
            self.output.display().to_string(),
            "--require-output".into(),
            "--json".into(),
        ]
    }

    fn sign(&self, statement: &Path, key: &Path, output: &Path) {
        self.ok(&[
            "run-attestation".into(),
            "sign".into(),
            statement.display().to_string(),
            "--key".into(),
            key.display().to_string(),
            "--output".into(),
            output.display().to_string(),
            "--json".into(),
        ]);
    }

    fn mutate_statement(&self, name: &str, mutation: impl FnOnce(&mut Value)) -> PathBuf {
        let path = self.path(name);
        let mut value: Value = serde_json::from_slice(&fs::read(&self.statement).unwrap()).unwrap();
        mutation(&mut value);
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        path
    }

    fn mutate_bundle(&self, name: &str, mutation: impl FnOnce(&mut Value)) -> PathBuf {
        let path = self.path(name);
        let mut value: Value = serde_json::from_slice(&fs::read(&self.bundle).unwrap()).unwrap();
        mutation(&mut value);
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        path
    }
}

fn s(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).into()).collect()
}

fn public(path: &Path) -> String {
    fs::read_to_string(path).unwrap().trim().to_string()
}

fn scenario() -> Scenario {
    let temp = tempfile::tempdir().unwrap();
    let binary = env!("CARGO_BIN_EXE_annpack");
    let p = |name: &str| temp.path().join(name);
    let run = |args: &[String]| {
        let output = Command::new(binary).args(args).output().unwrap();
        assert!(
            output.status.success(),
            "setup failed: {args:?}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        output
    };

    for name in [
        "root",
        "publisher",
        "release",
        "revocation",
        "workload",
        "other",
        "builder",
    ] {
        run(&s(&[
            "keygen",
            "--output",
            p(&format!("{name}.key")).to_str().unwrap(),
            "--json",
        ]));
    }
    let root = p("trust.json");
    run(&s(&[
        "trust",
        "init",
        "--output",
        root.to_str().unwrap(),
        "--publisher",
        "example.test",
        "--issued-at",
        "2030-01-01T00:00:00Z",
        "--valid-until",
        "2031-01-01T00:00:00Z",
        "--root-key",
        p("root.pub").to_str().unwrap(),
        "--artifact-key",
        p("publisher.pub").to_str().unwrap(),
        "--release-key",
        p("release.pub").to_str().unwrap(),
        "--revocation-key",
        p("revocation.pub").to_str().unwrap(),
    ]));
    run(&s(&[
        "trust",
        "sign",
        root.to_str().unwrap(),
        "--key",
        p("root.key").to_str().unwrap(),
    ]));

    let unsigned = p("unsigned.annpack");
    let build = run(&s(&[
        "build",
        "fixtures/docs-v1",
        "--output",
        unsigned.to_str().unwrap(),
        "--name",
        "support",
        "--version",
        "1.0.0",
        "--source-revision",
        "git:test",
        "--json",
    ]));
    let artifact_root = serde_json::from_slice::<Value>(&build.stdout).unwrap()["root_hash"]
        .as_str()
        .unwrap()
        .to_string();
    let signed = p("signed.annpack");
    run(&s(&[
        "sign",
        unsigned.to_str().unwrap(),
        "--output",
        signed.to_str().unwrap(),
        "--key",
        p("publisher.key").to_str().unwrap(),
        "--json",
    ]));

    // The same automated lifecycle establishes authenticated source binding
    // and trusted build provenance before it moves into publisher and runtime
    // evidence. The builder key remains distinct from every later role.
    let provenance_statement = p("build-provenance-statement.json");
    run(&s(&[
        "provenance",
        "create",
        signed.to_str().unwrap(),
        "--output",
        provenance_statement.to_str().unwrap(),
        "--repository",
        "https://github.com/example/support",
        "--revision",
        "git:test",
        "--builder-id",
        "annpack-test-builder",
        "--builder-binary",
        binary,
        "--invocation-id",
        "build-001",
        "--started-at",
        START,
        "--finished-at",
        COMPLETE,
    ]));
    let provenance = p("build-provenance.json");
    run(&s(&[
        "provenance",
        "sign",
        provenance_statement.to_str().unwrap(),
        "--key",
        p("builder.key").to_str().unwrap(),
        "--output",
        provenance.to_str().unwrap(),
    ]));
    let provenance_report = run(&s(&[
        "provenance",
        "verify",
        signed.to_str().unwrap(),
        provenance.to_str().unwrap(),
        "--trusted-builder-key",
        public(&p("builder.pub")).as_str(),
        "--builder-binary",
        binary,
        "--json",
    ]));
    assert_eq!(
        serde_json::from_slice::<Value>(&provenance_report.stdout).unwrap()["verified"],
        true
    );
    let bundle = p("run.json");
    run(&s(&[
        "bundle",
        signed.to_str().unwrap(),
        "install the sdk",
        "--output",
        bundle.to_str().unwrap(),
        "--limit",
        "2",
        "--run-id",
        "run-001",
        "--application",
        "support-agent/1.0",
        "--model",
        "model-1",
        "--public-key",
        p("publisher.pub").to_str().unwrap(),
    ]));
    let receipts =
        serde_json::from_slice::<Value>(&fs::read(&bundle).unwrap()).unwrap()["receipts"]
            .as_array()
            .unwrap()
            .len();
    assert!(
        receipts >= 2,
        "fixture must exercise canonical receipt ordering"
    );

    let channel = p("channel-1.json");
    run(&s(&[
        "release",
        "statement",
        "--output",
        channel.to_str().unwrap(),
        "--publisher",
        "example.test",
        "--corpus",
        "support",
        "--channel",
        "production",
        "--sequence",
        "1",
        "--current-root",
        &artifact_root,
        "--current-version",
        "1.0.0",
        "--issued-at",
        "2030-01-01T00:00:00Z",
        "--valid-until",
        "2031-01-01T00:00:00Z",
    ]));
    run(&s(&[
        "release",
        "sign",
        channel.to_str().unwrap(),
        "--key",
        p("release.key").to_str().unwrap(),
    ]));

    let output = p("answer.txt");
    let prompt = p("prompt-policy.txt");
    fs::write(&output, b"deterministic answer\n").unwrap();
    fs::write(&prompt, b"system-policy-v1\n").unwrap();
    let statement = p("run-statement.json");
    run(&s(&[
        "run-attestation",
        "create",
        bundle.to_str().unwrap(),
        "--channel-state",
        channel.to_str().unwrap(),
        "--trust-root",
        root.to_str().unwrap(),
        "--expect-publisher",
        "example.test",
        "--expect-corpus",
        "support",
        "--expect-channel",
        "production",
        "--policy",
        "authorized-current",
        "--now",
        NOW,
        "--output-bytes",
        output.to_str().unwrap(),
        "--prompt-policy",
        prompt.to_str().unwrap(),
        "--output",
        statement.to_str().unwrap(),
        "--run-id",
        "run-001",
        "--trace-id",
        "trace-001",
        "--workload-identity",
        "support-agent",
        "--started-at",
        START,
        "--completed-at",
        COMPLETE,
        "--retrieval-policy-revision",
        "retrieval-v1",
        "--retrieval-mode",
        "lexical",
        "--application-identity",
        "support-agent",
        "--application-version",
        "1.0.0",
        "--model-identifier",
        "model-1",
        "--model-provider",
        "test-provider",
        "--tool-policy-revision",
        "tools-v1",
        "--deployment-identity",
        "test",
        "--json",
    ]));
    let envelope = p("run-attestation.json");
    run(&s(&[
        "run-attestation",
        "sign",
        statement.to_str().unwrap(),
        "--key",
        p("workload.key").to_str().unwrap(),
        "--output",
        envelope.to_str().unwrap(),
        "--json",
    ]));

    let workload_key = p("workload.key");
    let workload_public = public(&p("workload.pub"));
    let other_public = public(&p("other.pub"));
    let publisher_key = p("publisher.key");
    let builder_key = p("builder.key");
    Scenario {
        _temp: temp,
        binary,
        root,
        bundle,
        channel,
        statement,
        envelope,
        output,
        prompt,
        workload_key,
        workload_public,
        other_public,
        publisher_key,
        builder_key,
        artifact_root,
    }
}

#[test]
fn complete_local_run_attestation_and_adversarial_matrix() {
    type StatementMutation = (&'static str, Box<dyn FnOnce(&mut Value)>, &'static str);

    let s = scenario();
    let report = s.ok(&s.verify_args(&s.envelope, &s.bundle));
    assert_eq!(report["overall_occurrence_evidence"], true);
    assert_eq!(report["occurrence_strength"], "workload_attested");
    assert_eq!(report["execution_time"], "carried");
    assert_eq!(report["cryptographic_signing_time"], "unknown");

    let envelope: Envelope = serde_json::from_slice(&fs::read(&s.envelope).unwrap()).unwrap();
    let bundle: RunBundle = serde_json::from_slice(&fs::read(&s.bundle).unwrap()).unwrap();
    let channel: ChannelState = serde_json::from_slice(&fs::read(&s.channel).unwrap()).unwrap();
    let root: TrustRoot = serde_json::from_slice(&fs::read(&s.root).unwrap()).unwrap();
    let trust = verify_trust_root(&root, None, Some(NOW)).unwrap();
    let channel_verification = verify_channel_state(
        &channel,
        &root,
        &trust,
        None,
        Some(NOW),
        ("example.test", "support", "production"),
    )
    .unwrap();
    let external = ExternalWorkloadVerification {
        payload_sha256: report["statement_digest"].as_str().unwrap().into(),
        envelope_signature_verified: true,
        identity: "support-agent".into(),
        trusted: true,
        signer_key_ids: vec!["sigstore-workload".into()],
        trusted_signing_time: Some(COMPLETE.into()),
        externally_anchored: true,
    };
    let prompt_policy_sha256 = hex::encode(Sha256::digest(fs::read(&s.prompt).unwrap()));
    let external_report = verify_run_attestation(VerifyRunAttestationInput {
        envelope: &envelope,
        run_bundle: &bundle,
        bound_channel_state: &channel,
        bound_channel_verification: &channel_verification,
        publisher_trust: &trust,
        workload_keys: &[],
        external_workload: Some(&external),
        expectations: &RunExpectations {
            run_id: "run-001".into(),
            trace_id: Some("trace-001".into()),
            model_identifier: "model-1".into(),
            prompt_policy_sha256,
        },
        output: Some(&fs::read(&s.output).unwrap()),
        require_output: true,
        current_channel_state: None,
    })
    .unwrap();
    assert!(external_report.overall_occurrence_evidence);
    assert_eq!(
        external_report.occurrence_strength,
        annpack::run_attestation::OccurrenceStrength::ExternallyAnchored
    );
    assert_eq!(
        external_report.cryptographic_signing_time,
        annpack::run_attestation::VerificationStatus::Verified
    );

    // Receipt order is not semantic: the canonical digest set still matches.
    let reordered = s.mutate_bundle("reordered.json", |bundle| {
        bundle["receipts"].as_array_mut().unwrap().reverse();
    });
    let ordered_report = s.ok(&s.verify_args(&s.envelope, &reordered));
    assert_eq!(ordered_report["receipt_set_binding"], "verified");

    let removed = s.mutate_bundle("removed.json", |bundle| {
        bundle["receipts"].as_array_mut().unwrap().pop();
    });
    assert_eq!(
        s.denied(&s.verify_args(&s.envelope, &removed))["details"]["receipt_set_binding"],
        "missing"
    );
    let duplicate = s.mutate_bundle("duplicate.json", |bundle| {
        let receipt = bundle["receipts"][0].clone();
        bundle["receipts"].as_array_mut().unwrap().push(receipt);
    });
    assert_eq!(
        s.denied(&s.verify_args(&s.envelope, &duplicate))["error"]["kind"],
        "duplicate_receipt"
    );
    let changed_receipt = s.mutate_bundle("changed-receipt.json", |bundle| {
        bundle["receipts"][0]["passage_hash"] = json!("00".repeat(32));
    });
    let receipt_failure = s.denied(&s.verify_args(&s.envelope, &changed_receipt));
    assert_ne!(
        receipt_failure["details"]["receipt_verification"],
        "verified"
    );
    let changed_proof = s.mutate_bundle("changed-proof.json", |bundle| {
        bundle["receipts"][0]["inclusion_proof"][0]["sibling"] = json!("22".repeat(32));
    });
    assert_ne!(
        s.denied(&s.verify_args(&s.envelope, &changed_proof))["details"]["receipt_verification"],
        "verified"
    );
    let extra = s.mutate_bundle("extra.json", |bundle| {
        let mut receipt = bundle["receipts"][0].clone();
        receipt["passage_id"] = json!("extra-passage");
        bundle["receipts"].as_array_mut().unwrap().push(receipt);
    });
    assert_ne!(
        s.denied(&s.verify_args(&s.envelope, &extra))["details"]["receipt_set_binding"],
        "verified"
    );

    let wrong_output = s.path("wrong-output.txt");
    fs::write(&wrong_output, b"different output").unwrap();
    let mut args = s.verify_args(&s.envelope, &s.bundle);
    let index = args.iter().position(|arg| arg == "--output-bytes").unwrap() + 1;
    args[index] = wrong_output.display().to_string();
    assert_eq!(s.denied(&args)["error"]["kind"], "output_digest_mismatch");
    let mut args = s.verify_args(&s.envelope, &s.bundle);
    let index = args.iter().position(|arg| arg == "--output-bytes").unwrap();
    args.drain(index..=index + 1);
    assert_eq!(
        s.denied(&args)["details"]["output_digest_binding"],
        "missing"
    );

    let mut args = s.verify_args(&s.envelope, &s.bundle);
    let index = args.iter().position(|arg| arg == "--expect-model").unwrap() + 1;
    args[index] = "model-2".into();
    assert_eq!(s.denied(&args)["error"]["kind"], "model_identity_mismatch");

    let wrong_prompt = s.path("wrong-prompt.txt");
    fs::write(&wrong_prompt, b"different prompt policy").unwrap();
    let mut args = s.verify_args(&s.envelope, &s.bundle);
    let index = args
        .iter()
        .position(|arg| arg == "--prompt-policy")
        .unwrap()
        + 1;
    args[index] = wrong_prompt.display().to_string();
    assert_eq!(s.denied(&args)["error"]["kind"], "prompt_policy_mismatch");

    let mut args = s.verify_args(&s.envelope, &s.bundle);
    let index = args
        .iter()
        .position(|arg| arg == "--expect-trace-id")
        .unwrap()
        + 1;
    args[index] = "trace-other".into();
    assert_eq!(s.denied(&args)["error"]["kind"], "run_identity_mismatch");

    // A valid signature from a supplied-but-untrusted workload stays distinct
    // from a malformed signature.
    let mut args = s.verify_args(&s.envelope, &s.bundle);
    let key_index = args
        .iter()
        .position(|arg| arg == "--trusted-workload-key")
        .unwrap();
    args[key_index + 1] = format!("other={}", s.other_public);
    args.splice(
        key_index + 2..key_index + 2,
        [
            "--untrusted-workload-key".into(),
            format!("support-agent={}", s.workload_public),
        ],
    );
    let untrusted = s.denied(&args);
    assert_eq!(untrusted["details"]["envelope_signature"], "verified");
    assert_eq!(untrusted["details"]["workload_identity"], "untrusted");

    for (name, key) in [
        ("publisher-workload", &s.publisher_key),
        ("builder-workload", &s.builder_key),
    ] {
        let envelope = s.path(&format!("{name}.json"));
        s.sign(&s.statement, key, &envelope);
        assert_eq!(
            s.denied(&s.verify_args(&envelope, &s.bundle))["error"]["kind"],
            "invalid_envelope"
        );
    }

    let cases: Vec<StatementMutation> = vec![
        (
            "run-id",
            Box::new(|v| v["predicate"]["execution"]["run_id"] = json!("run-other")),
            "run_identity_mismatch",
        ),
        (
            "model",
            Box::new(|v| v["predicate"]["application"]["model_identifier"] = json!("model-2")),
            "model_identity_mismatch",
        ),
        (
            "query",
            Box::new(|v| {
                v["predicate"]["retrieval"]["query_digest"]["value"] = json!("00".repeat(32))
            }),
            "query_digest_mismatch",
        ),
        (
            "release-digest",
            Box::new(|v| {
                v["predicate"]["knowledge"]["channel_state_digest"]["value"] =
                    json!("00".repeat(32))
            }),
            "channel_state_mismatch",
        ),
        (
            "artifact-root",
            Box::new(|v| v["predicate"]["knowledge"]["artifact_roots"][0] = json!("11".repeat(32))),
            "artifact_root_mismatch",
        ),
        (
            "time",
            Box::new(|v| {
                v["predicate"]["execution"]["started_at"] = json!(COMPLETE);
                v["predicate"]["execution"]["completed_at"] = json!(START);
            }),
            "impossible_time_ordering",
        ),
        (
            "signing-time",
            Box::new(|v| v["predicate"]["execution"]["signing_time"] = json!(START)),
            "impossible_time_ordering",
        ),
        (
            "predicate",
            Box::new(|v| v["predicateType"] = json!("https://example.test/unsupported")),
            "verification_failed",
        ),
        (
            "runtime-policy",
            Box::new(|v| {
                v["predicate"]["knowledge"]["trust_policy"] = json!("authorized_current_witnessed")
            }),
            "runtime_policy_denied",
        ),
    ];
    for (name, mutation, expected_kind) in cases {
        let statement = s.mutate_statement(&format!("{name}-statement.json"), mutation);
        let envelope = s.path(&format!("{name}-envelope.json"));
        s.sign(&statement, &s.workload_key, &envelope);
        assert_eq!(
            s.denied(&s.verify_args(&envelope, &s.bundle))["error"]["kind"],
            expected_kind,
            "wrong failure for {name}"
        );
    }

    let extension_statement = s.mutate_statement("extension-statement.json", |value| {
        value["predicate"]["extensions"] = json!({"x-example.test/context": "opaque"});
    });
    let extension_envelope = s.path("extension-envelope.json");
    s.sign(&extension_statement, &s.workload_key, &extension_envelope);
    assert_eq!(
        s.ok(&s.verify_args(&extension_envelope, &s.bundle))["overall_occurrence_evidence"],
        true
    );
    let invalid_extension = s.mutate_statement("invalid-extension-statement.json", |value| {
        value["predicate"]["extensions"] = json!({"future": true});
    });
    let invalid_extension_envelope = s.path("invalid-extension-envelope.json");
    s.sign(
        &invalid_extension,
        &s.workload_key,
        &invalid_extension_envelope,
    );
    assert_eq!(
        s.denied(&s.verify_args(&invalid_extension_envelope, &s.bundle))["error"]["kind"],
        "verification_failed"
    );

    // Payload mutation and signature transplantation both fail DSSE.
    use base64::Engine;
    let mut tampered: Value = serde_json::from_slice(&fs::read(&s.envelope).unwrap()).unwrap();
    let mut payload = base64::engine::general_purpose::STANDARD
        .decode(tampered["payload"].as_str().unwrap())
        .unwrap();
    // JSON permits trailing whitespace, so parsing still succeeds while the
    // exact DSSE payload bytes (and therefore the signature) change.
    payload.push(b' ');
    tampered["payload"] = json!(base64::engine::general_purpose::STANDARD.encode(payload));
    let tampered_path = s.path("tampered-envelope.json");
    fs::write(
        &tampered_path,
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    assert_eq!(
        s.denied(&s.verify_args(&tampered_path, &s.bundle))["error"]["kind"],
        "invalid_envelope"
    );

    let transplanted_statement = s.mutate_statement("transplanted-statement.json", |statement| {
        statement["predicate"]["execution"]["run_id"] = json!("run-transplanted");
    });
    let transplanted = s.path("transplanted-envelope.json");
    s.sign(&transplanted_statement, &s.workload_key, &transplanted);
    let original: Value = serde_json::from_slice(&fs::read(&s.envelope).unwrap()).unwrap();
    let mut value: Value = serde_json::from_slice(&fs::read(&transplanted).unwrap()).unwrap();
    value["signatures"] = original["signatures"].clone();
    fs::write(&transplanted, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    assert_eq!(
        s.denied(&s.verify_args(&transplanted, &s.bundle))["error"]["kind"],
        "invalid_envelope"
    );

    let wrong_scope = s.path("wrong-scope-channel.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&s.channel).unwrap()).unwrap();
    value["channel"] = json!("staging");
    fs::write(&wrong_scope, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let mut args = s.verify_args(&s.envelope, &s.bundle);
    let index = args
        .iter()
        .position(|arg| arg == "--channel-state")
        .unwrap()
        + 1;
    args[index] = wrong_scope.display().to_string();
    assert_eq!(s.denied(&args)["error"]["kind"], "scope_mismatch");
}

#[test]
fn empty_receipt_set_is_explicit_and_never_complete_publisher_evidence() {
    let s = scenario();
    let empty = s.mutate_bundle("empty.json", |bundle| {
        bundle["receipts"] = json!([]);
    });
    let statement = s.path("empty-statement.json");
    s.ok(&[
        "run-attestation".into(),
        "create".into(),
        empty.display().to_string(),
        "--channel-state".into(),
        s.channel.display().to_string(),
        "--trust-root".into(),
        s.root.display().to_string(),
        "--expect-publisher".into(),
        "example.test".into(),
        "--expect-corpus".into(),
        "support".into(),
        "--expect-channel".into(),
        "production".into(),
        "--policy".into(),
        "integrity-only".into(),
        "--allow-empty-receipts".into(),
        "--now".into(),
        NOW.into(),
        "--output-bytes".into(),
        s.output.display().to_string(),
        "--prompt-policy".into(),
        s.prompt.display().to_string(),
        "--output".into(),
        statement.display().to_string(),
        "--run-id".into(),
        "run-001".into(),
        "--trace-id".into(),
        "trace-001".into(),
        "--workload-identity".into(),
        "support-agent".into(),
        "--started-at".into(),
        START.into(),
        "--completed-at".into(),
        COMPLETE.into(),
        "--retrieval-policy-revision".into(),
        "retrieval-v1".into(),
        "--retrieval-mode".into(),
        "lexical".into(),
        "--application-identity".into(),
        "support-agent".into(),
        "--application-version".into(),
        "1.0.0".into(),
        "--model-identifier".into(),
        "model-1".into(),
        "--model-provider".into(),
        "test-provider".into(),
        "--tool-policy-revision".into(),
        "tools-v1".into(),
        "--deployment-identity".into(),
        "test".into(),
        "--json".into(),
    ]);
    let value: Value = serde_json::from_slice(&fs::read(&statement).unwrap()).unwrap();
    assert_eq!(value["predicate"]["knowledge"]["receipt_count"], 0);
    assert_eq!(
        value["predicate"]["knowledge"]["no_passages_retrieved"],
        true
    );
    let envelope = s.path("empty-envelope.json");
    s.sign(&statement, &s.workload_key, &envelope);
    let report = s.denied(&s.verify_args(&envelope, &empty));
    assert_eq!(report["details"]["publisher_authority"], "untrusted");
    assert_eq!(report["details"]["overall_occurrence_evidence"], false);
}

#[test]
fn historical_occurrence_survives_supersession_and_revocation() {
    let s = scenario();
    for (sequence, action, expected) in
        [(2, "--supersede", "superseded"), (3, "--revoke", "revoked")]
    {
        let current = s.path(&format!("channel-{sequence}.json"));
        let mut args = vec![
            "release".into(),
            "statement".into(),
            "--output".into(),
            current.display().to_string(),
            "--publisher".into(),
            "example.test".into(),
            "--corpus".into(),
            "support".into(),
            "--channel".into(),
            "production".into(),
            "--sequence".into(),
            sequence.to_string(),
            "--current-root".into(),
            "11".repeat(32),
            "--current-version".into(),
            "2.0.0".into(),
            "--issued-at".into(),
            "2030-01-01T01:00:00Z".into(),
            "--valid-until".into(),
            "2031-01-01T00:00:00Z".into(),
            action.into(),
            s.artifact_root.clone(),
        ];
        if action == "--revoke" {
            args.extend(["--revoke-reason".into(), "security-event".into()]);
        }
        s.ok(&args);
        s.ok(&[
            "release".into(),
            "sign".into(),
            current.display().to_string(),
            "--key".into(),
            s.path("release.key").display().to_string(),
        ]);
        let mut verify = s.verify_args(&s.envelope, &s.bundle);
        verify.extend([
            "--current-channel-state".into(),
            current.display().to_string(),
        ]);
        let report = s.ok(&verify);
        assert_eq!(report["overall_occurrence_evidence"], true);
        assert_eq!(report["currency_at_evaluation"], expected);
        assert_eq!(report["present_use_permitted"], false);
    }
}
