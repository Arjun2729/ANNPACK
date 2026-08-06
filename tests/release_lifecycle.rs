//! The publisher-to-consumer lifecycle, driven entirely through the CLI.
//!
//! Unit and integration tests elsewhere check each stage in isolation against
//! hand-built structures. This checks that the stages compose when a real
//! operator runs real commands: keys on disk, a signed trust root, a signed
//! artifact, a signed channel-state statement, and a consumer applying a policy.
//!
//! It is the acceptance scenario from the architecture contract, minus the
//! transparency steps, which are not implemented. Every claim is asserted
//! separately -- no single green status is accepted as evidence that the rest
//! held.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use tempfile::TempDir;

struct Publisher {
    _temp: TempDir,
    dir: PathBuf,
    binary: &'static str,
}

impl Publisher {
    fn new() -> Self {
        let temp = TempDir::new().unwrap();
        let dir = temp.path().to_path_buf();
        let publisher = Self {
            dir,
            binary: env!("CARGO_BIN_EXE_annpack"),
            _temp: temp,
        };
        for role in ["root", "artifact", "release", "revocation"] {
            publisher.ok(&[
                "keygen",
                "--output",
                &publisher.path(&format!("{role}.key")),
            ]);
        }
        publisher
    }

    fn path(&self, name: &str) -> String {
        self.dir.join(name).to_str().unwrap().to_string()
    }

    fn run(&self, args: &[&str]) -> std::process::Output {
        Command::new(self.binary).args(args).output().unwrap()
    }

    fn ok(&self, args: &[&str]) -> String {
        let output = self.run(args);
        assert!(
            output.status.success(),
            "{:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).to_string()
    }

    /// Run a command expected to fail, returning its combined output so the
    /// reason can be asserted rather than merely the failure.
    fn denied(&self, args: &[&str]) -> String {
        let output = self.run(args);
        assert!(
            !output.status.success(),
            "{:?} unexpectedly succeeded: {}",
            args,
            String::from_utf8_lossy(&output.stdout)
        );
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    }

    fn trust_root(&self) {
        self.ok(&[
            "trust",
            "init",
            "--output",
            &self.path("trust.json"),
            "--publisher",
            "example.com",
            "--valid-until",
            "2099-01-01T00:00:00Z",
            "--root-key",
            &self.path("root.pub"),
            "--artifact-key",
            &self.path("artifact.pub"),
            "--release-key",
            &self.path("release.pub"),
            "--revocation-key",
            &self.path("revocation.pub"),
        ]);
        self.ok(&[
            "trust",
            "sign",
            &self.path("trust.json"),
            "--key",
            &self.path("root.key"),
        ]);
    }

    /// Build and sign an artifact, returning its root.
    fn artifact(&self, name: &str, fixture: &str, version: &str) -> String {
        let source = format!("{}/{fixture}", env!("CARGO_MANIFEST_DIR"));
        let unsigned = self.path(&format!("{name}-unsigned.annpack"));
        let signed = self.path(&format!("{name}.annpack"));
        self.ok(&[
            "build",
            &source,
            "--output",
            &unsigned,
            "--name",
            "support-manual",
            "--version",
            version,
            "--json",
        ]);
        self.ok(&[
            "sign",
            &unsigned,
            "--output",
            &signed,
            "--key",
            &self.path("artifact.key"),
        ]);
        let report: Value = serde_json::from_str(&self.ok(&["verify", &signed, "--json"])).unwrap();
        report["root_hash"].as_str().unwrap().to_string()
    }

    fn statement(&self, name: &str, sequence: &str, current: &str, extra: &[&str]) {
        let output = self.path(name);
        let mut args = vec![
            "release",
            "statement",
            "--output",
            &output,
            "--publisher",
            "example.com",
            "--corpus",
            "support-manual",
            "--sequence",
            sequence,
            "--current-root",
            current,
            "--current-version",
            "1.0.0",
            "--valid-until",
            "2099-01-01T00:00:00Z",
        ];
        args.extend_from_slice(extra);
        self.ok(&args);
    }

    fn sign_statement(&self, name: &str, key: &str) {
        self.ok(&[
            "release",
            "sign",
            &self.path(name),
            "--key",
            &self.path(key),
        ]);
    }

    fn verify_under(&self, pack: &str, policy: &str, statement: Option<&str>) -> Vec<String> {
        let pack = self.path(&format!("{pack}.annpack"));
        let trust = self.path("trust.json");
        let mut args = vec![
            "verify",
            &pack,
            "--policy",
            policy,
            "--trust-root",
            &trust,
            "--system-clock",
            "--json",
        ];
        let state;
        if let Some(name) = statement {
            state = self.path(name);
            args.push("--channel-state");
            args.push(&state);
            // Scope is stated by the consumer, never read from the statement.
            args.push("--expect-corpus");
            args.push("support-manual");
            args.push("--expect-channel");
            args.push("production");
        }
        let output = self.run(&args);
        // Exactly one JSON object on stdout, whichever way the decision goes.
        let emitted: Value = serde_json::from_slice(&output.stdout)
            .unwrap_or_else(|e| panic!("stdout was not one JSON object: {e}"));
        if output.status.success() {
            assert_eq!(emitted["policy"]["permitted"], Value::Bool(true));
            return Vec::new();
        }
        assert_eq!(emitted["ok"], Value::Bool(false));
        assert!(
            emitted["error"]["kind"].is_string(),
            "failure envelope carried no error.kind"
        );
        emitted["details"]["policy"]["unmet_requirements"]
            .as_array()
            .expect("failure envelope carried no policy details")
            .iter()
            .map(|reason| reason.as_str().unwrap().to_string())
            .collect()
    }
}

fn mentions(reasons: &[String], needle: &str) -> bool {
    reasons.iter().any(|reason| reason.contains(needle))
}

#[test]
fn the_full_release_lifecycle_holds_each_claim_separately() {
    let publisher = Publisher::new();
    publisher.trust_root();

    // Trust root verifies on its own before anything depends on it.
    publisher.ok(&[
        "trust",
        "verify",
        &publisher.path("trust.json"),
        "--system-clock",
    ]);

    let root_a = publisher.artifact("a", "fixtures/docs-v1", "1.0.0");
    let root_b = publisher.artifact("b", "fixtures/docs-v2", "2.0.0");
    assert_ne!(root_a, root_b);

    // ---- sequence 1: A is current -------------------------------------------
    publisher.statement("s1.json", "1", &root_a, &[]);
    publisher.sign_statement("s1.json", "release.key");

    assert!(
        publisher
            .verify_under("a", "integrity-only", None)
            .is_empty()
    );
    assert!(
        publisher
            .verify_under("a", "authorized-publisher", None)
            .is_empty()
    );
    assert!(
        publisher
            .verify_under("a", "authorized-current", Some("s1.json"))
            .is_empty()
    );

    // The witnessed policy must refuse rather than behave like the one below it.
    let witnessed = publisher.verify_under("a", "authorized-current-witnessed", Some("s1.json"));
    assert!(
        mentions(&witnessed, "transparency"),
        "witnessed policy did not deny for the right reason: {witnessed:?}"
    );

    // Withholding the statement must deny, not fall back to publisher authority.
    let withheld = publisher.verify_under("a", "authorized-current", None);
    assert!(
        mentions(&withheld, "no channel-state statement"),
        "{withheld:?}"
    );

    // ---- sequence 2: B supersedes A -----------------------------------------
    publisher.statement("s2.json", "2", &root_b, &["--supersede", &root_a]);
    publisher.sign_statement("s2.json", "release.key");

    let superseded = publisher.verify_under("a", "authorized-current", Some("s2.json"));
    assert!(mentions(&superseded, "supersedes"), "{superseded:?}");
    assert!(
        publisher
            .verify_under("b", "authorized-current", Some("s2.json"))
            .is_empty()
    );
    // A is superseded, not compromised: the weaker policy still permits it.
    assert!(
        publisher
            .verify_under("a", "integrity-only", None)
            .is_empty()
    );

    // ---- sequence 3: B revoked by the emergency key --------------------------
    publisher.statement(
        "s3.json",
        "3",
        &root_a,
        &["--revoke", &root_b, "--revoke-reason", "incorrect-content"],
    );
    publisher.sign_statement("s3.json", "revocation.key");

    // Revocation is honoured under the weakest policy: a withdrawn artifact is
    // not usable merely because the caller asked a smaller question.
    let revoked = publisher.verify_under("b", "integrity-only", Some("s3.json"));
    assert!(mentions(&revoked, "revoked"), "{revoked:?}");

    // The same statement cannot promote A back, because the revocation role may
    // withdraw and may not put anything into service.
    let promoted = publisher.verify_under("a", "authorized-current", Some("s3.json"));
    assert!(mentions(&promoted, "currency is unknown"), "{promoted:?}");
}

#[test]
fn retained_state_refuses_a_genuinely_signed_rollback() {
    let publisher = Publisher::new();
    publisher.trust_root();
    let root_a = publisher.artifact("a", "fixtures/docs-v1", "1.0.0");
    let root_b = publisher.artifact("b", "fixtures/docs-v2", "2.0.0");

    publisher.statement("s1.json", "1", &root_a, &[]);
    publisher.sign_statement("s1.json", "release.key");
    publisher.statement("s2.json", "2", &root_b, &[]);
    publisher.sign_statement("s2.json", "release.key");

    let state = publisher.path("client-state.json");
    let trust = publisher.path("trust.json");

    // Accept sequence 2 and persist it.
    publisher.ok(&[
        "release",
        "verify",
        &publisher.path("s2.json"),
        "--trust-root",
        &trust,
        "--expect-corpus",
        "support-manual",
        "--expect-channel",
        "production",
        "--system-clock",
        "--retained-state",
        &state,
        "--accept",
    ]);
    let retained: Value =
        serde_json::from_slice(&std::fs::read(Path::new(&state)).unwrap()).unwrap();
    assert_eq!(retained["highest_sequence"], 2);

    // Sequence 1 is genuinely signed and still must not be accepted.
    let output = publisher.denied(&[
        "release",
        "verify",
        &publisher.path("s1.json"),
        "--trust-root",
        &trust,
        "--expect-corpus",
        "support-manual",
        "--expect-channel",
        "production",
        "--system-clock",
        "--retained-state",
        &state,
    ]);
    assert!(
        output.contains("below the accepted sequence"),
        "rollback refused for the wrong reason: {output}"
    );

    // Re-presenting the accepted statement is idempotent, not an attack.
    publisher.ok(&[
        "release",
        "verify",
        &publisher.path("s2.json"),
        "--trust-root",
        &trust,
        "--expect-corpus",
        "support-manual",
        "--expect-channel",
        "production",
        "--system-clock",
        "--retained-state",
        &state,
    ]);
}

#[test]
fn accepting_state_requires_a_stated_clock() {
    // Recording an acceptance time from a clock nobody vouched for would put an
    // unverified value into durable state that later decisions read.
    let publisher = Publisher::new();
    publisher.trust_root();
    let root_a = publisher.artifact("a", "fixtures/docs-v1", "1.0.0");
    publisher.statement("s1.json", "1", &root_a, &[]);
    publisher.sign_statement("s1.json", "release.key");

    let output = publisher.denied(&[
        "release",
        "verify",
        &publisher.path("s1.json"),
        "--trust-root",
        &publisher.path("trust.json"),
        "--expect-corpus",
        "support-manual",
        "--expect-channel",
        "production",
        "--retained-state",
        &publisher.path("client-state.json"),
        "--accept",
    ]);
    assert!(output.contains("requires a stated clock"), "{output}");
}

#[test]
fn an_artifact_signed_by_an_unauthorized_key_fails_publisher_authority() {
    let publisher = Publisher::new();
    publisher.trust_root();

    // A key that exists and signs correctly, but which the trust root never
    // authorised for the artifact role.
    publisher.ok(&["keygen", "--output", &publisher.path("stranger.key")]);
    let source = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    publisher.ok(&[
        "build",
        &source,
        "--output",
        &publisher.path("x-unsigned.annpack"),
        "--name",
        "support-manual",
        "--version",
        "1.0.0",
        "--json",
    ]);
    publisher.ok(&[
        "sign",
        &publisher.path("x-unsigned.annpack"),
        "--output",
        &publisher.path("x.annpack"),
        "--key",
        &publisher.path("stranger.key"),
    ]);

    // Integrity is fine; authority is not.
    assert!(
        publisher
            .verify_under("x", "integrity-only", None)
            .is_empty()
    );
    let reasons = publisher.verify_under("x", "authorized-publisher", None);
    assert!(
        mentions(&reasons, "no authorised artifact key"),
        "{reasons:?}"
    );
}

/// Helpers for asserting the machine contract: one JSON object, a stable
/// `error.kind`, and an exit class a caller can branch on without parsing prose.
impl Publisher {
    fn release_verify(
        &self,
        statement: &str,
        corpus: &str,
        channel: &str,
        extra: &[&str],
    ) -> std::process::Output {
        let stmt = self.path(statement);
        let trust = self.path("trust.json");
        let mut args = vec![
            "release",
            "verify",
            &stmt,
            "--trust-root",
            &trust,
            "--expect-corpus",
            corpus,
            "--expect-channel",
            channel,
            "--system-clock",
            "--json",
        ];
        args.extend_from_slice(extra);
        self.run(&args)
    }
}

fn envelope(output: &std::process::Output) -> Value {
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("stdout was not exactly one JSON object: {e}"))
}

fn assert_failure(output: &std::process::Output, class: i32, kind: &str) {
    assert_eq!(
        output.status.code(),
        Some(class),
        "wrong exit class for {kind}"
    );
    let e = envelope(output);
    assert_eq!(e["ok"], Value::Bool(false));
    assert_eq!(
        e["error"]["kind"],
        Value::String(kind.into()),
        "wrong error.kind"
    );
}

#[test]
fn a_statement_for_another_channel_is_refused() {
    // The defect this replaces: both CLI call sites took the expected scope
    // from the statement being verified, so `scope_matches` was tautological
    // and a staging statement verified cleanly for a production consumer.
    let publisher = Publisher::new();
    publisher.trust_root();
    let root_a = publisher.artifact("a", "fixtures/docs-v1", "1.0.0");

    publisher.statement("staging.json", "1", &root_a, &["--channel", "staging"]);
    publisher.sign_statement("staging.json", "release.key");

    // Correct in its own scope.
    let ok = publisher.release_verify("staging.json", "support-manual", "staging", &[]);
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );

    // Refused when production was asked for.
    let denied = publisher.release_verify("staging.json", "support-manual", "production", &[]);
    assert_failure(&denied, 5, "scope_mismatch");
    let e = envelope(&denied);
    assert_eq!(e["details"]["scope_matches"], Value::Bool(false));
    // Retained state must not have been consulted for a foreign scope.
    assert_eq!(
        e["details"]["sequence_verdict"],
        Value::String("not_evaluated".into())
    );
}

#[test]
fn a_statement_for_another_corpus_is_refused() {
    let publisher = Publisher::new();
    publisher.trust_root();
    let root_a = publisher.artifact("a", "fixtures/docs-v1", "1.0.0");
    publisher.statement("s1.json", "1", &root_a, &[]);
    publisher.sign_statement("s1.json", "release.key");

    let denied = publisher.release_verify("s1.json", "other-corpus", "production", &[]);
    assert_failure(&denied, 5, "scope_mismatch");
}

#[test]
fn a_statement_from_another_publisher_is_refused() {
    // The publisher is taken from the trusted root, so a statement naming a
    // different one cannot select its own expectation.
    let publisher = Publisher::new();
    publisher.trust_root();
    let root_a = publisher.artifact("a", "fixtures/docs-v1", "1.0.0");
    publisher.ok(&[
        "release",
        "statement",
        "--output",
        &publisher.path("foreign.json"),
        "--publisher",
        "attacker.example",
        "--corpus",
        "support-manual",
        "--sequence",
        "1",
        "--current-root",
        &root_a,
        "--current-version",
        "1.0.0",
        "--valid-until",
        "2099-01-01T00:00:00Z",
    ]);
    publisher.sign_statement("foreign.json", "release.key");

    let denied = publisher.release_verify("foreign.json", "support-manual", "production", &[]);
    assert_failure(&denied, 5, "scope_mismatch");
}

#[test]
fn a_scope_mismatch_leaves_retained_state_untouched() {
    let publisher = Publisher::new();
    publisher.trust_root();
    let root_a = publisher.artifact("a", "fixtures/docs-v1", "1.0.0");
    let root_b = publisher.artifact("b", "fixtures/docs-v2", "2.0.0");

    publisher.statement("s5.json", "5", &root_a, &[]);
    publisher.sign_statement("s5.json", "release.key");
    let state = publisher.path("client-state.json");
    publisher.release_verify(
        "s5.json",
        "support-manual",
        "production",
        &["--retained-state", &state, "--accept"],
    );
    let before = std::fs::read(Path::new(&state)).unwrap();

    // A far-higher sequence, correctly signed, but scoped to staging. It must
    // neither be accepted nor overwrite production's state.
    publisher.statement("staging9.json", "9", &root_b, &["--channel", "staging"]);
    publisher.sign_statement("staging9.json", "release.key");
    let denied = publisher.release_verify(
        "staging9.json",
        "support-manual",
        "production",
        &["--retained-state", &state, "--accept"],
    );
    assert_failure(&denied, 5, "scope_mismatch");

    assert_eq!(
        before,
        std::fs::read(Path::new(&state)).unwrap(),
        "a scope-mismatched statement modified retained state"
    );
}

#[test]
fn every_failure_class_is_distinguishable_by_a_machine() {
    let publisher = Publisher::new();
    publisher.trust_root();
    let root_a = publisher.artifact("a", "fixtures/docs-v1", "1.0.0");
    let root_b = publisher.artifact("b", "fixtures/docs-v2", "2.0.0");
    let trust = publisher.path("trust.json");
    let state = publisher.path("client-state.json");

    publisher.statement("s1.json", "1", &root_a, &[]);
    publisher.sign_statement("s1.json", "release.key");
    publisher.statement("s2.json", "2", &root_b, &["--supersede", &root_a]);
    publisher.sign_statement("s2.json", "release.key");
    publisher.statement("s2b.json", "2", &root_a, &[]);
    publisher.sign_statement("s2b.json", "release.key");
    publisher.statement("s3.json", "3", &root_a, &["--revoke", &root_b]);
    publisher.sign_statement("s3.json", "revocation.key");
    publisher.statement("unsigned.json", "1", &root_a, &[]);
    std::fs::write(publisher.path("garbage.json"), "{not json").unwrap();

    publisher.release_verify(
        "s2.json",
        "support-manual",
        "production",
        &["--retained-state", &state, "--accept"],
    );

    // malformed input and unavailable input, both before any report exists
    assert_failure(
        &publisher.release_verify("garbage.json", "support-manual", "production", &[]),
        3,
        "malformed_input",
    );
    assert_failure(
        &publisher.release_verify("absent.json", "support-manual", "production", &[]),
        3,
        "input_unavailable",
    );

    // authority
    assert_failure(
        &publisher.release_verify("unsigned.json", "support-manual", "production", &[]),
        5,
        "unauthorized_role",
    );

    // monotonic-state safety
    assert_failure(
        &publisher.release_verify(
            "s1.json",
            "support-manual",
            "production",
            &["--retained-state", &state],
        ),
        6,
        "rollback",
    );
    assert_failure(
        &publisher.release_verify(
            "s2b.json",
            "support-manual",
            "production",
            &["--retained-state", &state],
        ),
        6,
        "equivocation",
    );

    // status denials, via the policy path
    let pack_b = publisher.path("b.annpack");
    let s3 = publisher.path("s3.json");
    let revoked = publisher.run(&[
        "verify",
        &pack_b,
        "--policy",
        "integrity-only",
        "--trust-root",
        &trust,
        "--channel-state",
        &s3,
        "--expect-corpus",
        "support-manual",
        "--expect-channel",
        "production",
        "--system-clock",
        "--json",
    ]);
    assert_failure(&revoked, 7, "revoked");

    let pack_a = publisher.path("a.annpack");
    let s2 = publisher.path("s2.json");
    let superseded = publisher.run(&[
        "verify",
        &pack_a,
        "--policy",
        "authorized-current",
        "--trust-root",
        &trust,
        "--channel-state",
        &s2,
        "--expect-corpus",
        "support-manual",
        "--expect-channel",
        "production",
        "--system-clock",
        "--json",
    ]);
    assert_failure(&superseded, 7, "superseded");

    let s1 = publisher.path("s1.json");
    let witnessed = publisher.run(&[
        "verify",
        &pack_a,
        "--policy",
        "authorized-current-witnessed",
        "--trust-root",
        &trust,
        "--channel-state",
        &s1,
        "--expect-corpus",
        "support-manual",
        "--expect-channel",
        "production",
        "--system-clock",
        "--json",
    ]);
    assert_failure(&witnessed, 7, "unmet_policy_requirement");

    // usage
    let usage = publisher.run(&[
        "release",
        "verify",
        &s1,
        "--trust-root",
        &trust,
        "--expect-corpus",
        "support-manual",
        "--expect-channel",
        "production",
        "--retained-state",
        &state,
        "--accept",
        "--json",
    ]);
    assert_failure(&usage, 2, "invalid_usage");
}
