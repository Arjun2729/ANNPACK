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
        }
        let output = self.run(&args);
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        let permitted = report["policy"]["permitted"].as_bool().unwrap();
        assert_eq!(
            permitted,
            output.status.success(),
            "exit status disagreed with the reported decision"
        );
        if permitted {
            Vec::new()
        } else {
            report["policy"]["unmet_requirements"]
                .as_array()
                .unwrap()
                .iter()
                .map(|reason| reason.as_str().unwrap().to_string())
                .collect()
        }
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
