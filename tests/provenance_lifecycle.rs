//! The §14 acceptance scenario: build, provenance, and release authorization
//! composed through the CLI, with every claim reported separately.
//!
//! This is the test that answers the stop condition in the brief: run it and
//! confirm the report never conflates builder identity with publisher
//! identity, source revision with authenticated source bytes, artifact
//! integrity with build provenance, or build provenance with release
//! authorization.

#![cfg(feature = "signing")]

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;
use tempfile::TempDir;

struct Env {
    _temp: TempDir,
    dir: PathBuf,
    binary: &'static str,
}

impl Env {
    fn new() -> Self {
        let temp = TempDir::new().unwrap();
        Self {
            dir: temp.path().to_path_buf(),
            binary: env!("CARGO_BIN_EXE_adyar"),
            _temp: temp,
        }
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
}

#[test]
fn build_provenance_composes_with_publisher_authority_and_release_currency() {
    let env = Env::new();

    // 1. Keys: artifact, release, revocation, root, and a builder key that is
    //    deliberately not any of them.
    for role in ["root", "artifact", "release", "revocation", "builder"] {
        env.ok(&["keygen", "--output", &env.path(&format!("{role}.key"))]);
    }

    // 2. Trust root, publisher = example.com.
    env.ok(&[
        "trust",
        "init",
        "--output",
        &env.path("trust.json"),
        "--publisher",
        "example.com",
        "--valid-until",
        "2099-01-01T00:00:00Z",
        "--root-key",
        &env.path("root.pub"),
        "--artifact-key",
        &env.path("artifact.pub"),
        "--release-key",
        &env.path("release.pub"),
        "--revocation-key",
        &env.path("revocation.pub"),
    ]);
    env.ok(&[
        "trust",
        "sign",
        &env.path("trust.json"),
        "--key",
        &env.path("root.key"),
    ]);

    // 3-4. Build a format-4 artifact (authenticated source digest by
    //      construction) and sign it with the artifact role.
    let source = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    env.ok(&[
        "build",
        &source,
        "--output",
        &env.path("unsigned.annpack"),
        "--name",
        "support-manual",
        "--version",
        "1.0.0",
        "--json",
    ]);
    env.ok(&[
        "sign",
        &env.path("unsigned.annpack"),
        "--output",
        &env.path("pack.annpack"),
        "--key",
        &env.path("artifact.key"),
    ]);
    let root_a: Value =
        serde_json::from_str(&env.ok(&["verify", &env.path("pack.annpack"), "--json"])).unwrap();
    let root_a = root_a["root_hash"].as_str().unwrap().to_string();

    // 5-6. Create and sign provenance with a builder key unrelated to any
    //      publisher role.
    env.ok(&[
        "provenance",
        "create",
        &env.path("pack.annpack"),
        "--output",
        &env.path("prov.json"),
        "--repository",
        "github.com/example/support-manual",
        "--revision",
        "git:abc123",
        "--builder-id",
        "workflow:release",
        "--builder-binary",
        env.binary,
        "--system-clock",
    ]);
    env.ok(&[
        "provenance",
        "sign",
        &env.path("prov.json"),
        "--key",
        &env.path("builder.key"),
    ]);
    let builder_pub = std::fs::read_to_string(env.path("builder.pub")).unwrap();
    let builder_pub = builder_pub.trim();

    // 7-10. Verify: envelope, file digest, artifact root, source binding.
    let report: Value = serde_json::from_str(&env.ok(&[
        "provenance",
        "verify",
        &env.path("pack.annpack"),
        &env.path("prov.json"),
        "--trusted-builder-key",
        builder_pub,
        "--builder-binary",
        env.binary,
        "--json",
    ]))
    .unwrap();
    assert_eq!(report["envelope_signature"], "valid");
    assert_eq!(report["builder_identity"], "trusted");
    assert_eq!(report["distributed_file_digest"], "verified");
    assert_eq!(report["artifact_root_binding"], "verified");
    assert_eq!(report["source_digest_binding"], "authenticated");
    assert_eq!(report["completeness"], "complete");
    // 11. Repository and revision are named as carried, never as proven.
    assert_eq!(report["repository_claim"], "carried");
    assert_eq!(report["revision_claim"], "carried");

    // 12. A source byte changes -> a rebuilt artifact does not match this
    //     provenance's recorded source digest.
    let mutated_source = env.dir.join("mutated-src");
    std::fs::create_dir_all(&mutated_source).unwrap();
    std::fs::write(
        mutated_source.join("guide.md"),
        "---\ntitle: Guide\n---\n\n# Guide\n\nA changed sentence.\n",
    )
    .unwrap();
    env.ok(&[
        "build",
        mutated_source.to_str().unwrap(),
        "--output",
        &env.path("mutated.annpack"),
        "--name",
        "support-manual",
        "--version",
        "1.0.0",
        "--json",
    ]);
    env.ok(&[
        "sign",
        &env.path("mutated.annpack"),
        "--output",
        &env.path("mutated-signed.annpack"),
        "--key",
        &env.path("artifact.key"),
    ]);
    let mutated_result = env.run(&[
        "provenance",
        "verify",
        &env.path("mutated-signed.annpack"),
        &env.path("prov.json"),
        "--trusted-builder-key",
        builder_pub,
        "--json",
    ]);
    assert!(
        !mutated_result.status.success(),
        "provenance for one artifact verified against a different one"
    );

    // 13. Directly modifying the distributed artifact after provenance was
    //     issued for it must fail.
    let mut bytes = std::fs::read(env.path("pack.annpack")).unwrap();
    bytes.push(0);
    std::fs::write(env.path("pack.annpack"), &bytes).unwrap();
    let tampered_result = env.run(&[
        "provenance",
        "verify",
        &env.path("pack.annpack"),
        &env.path("prov.json"),
        "--trusted-builder-key",
        builder_pub,
        "--json",
    ]);
    assert!(!tampered_result.status.success());
    let tampered: Value = serde_json::from_slice(&tampered_result.stdout).unwrap();
    assert_eq!(tampered["details"]["distributed_file_digest"], "mismatched");
    // Restore for the remaining steps.
    bytes.pop();
    std::fs::write(env.path("pack.annpack"), &bytes).unwrap();

    // 14. Re-signing the same statement with an unauthorized builder must fail
    //     trust even though the signature is cryptographically genuine.
    env.ok(&["keygen", "--output", &env.path("stranger.key")]);
    let unsigned_path = env.path("prov-unsigned.json");
    env.ok(&[
        "provenance",
        "create",
        &env.path("pack.annpack"),
        "--output",
        &unsigned_path,
        "--repository",
        "github.com/example/support-manual",
        "--revision",
        "git:abc123",
        "--builder-id",
        "workflow:release",
        "--system-clock",
    ]);
    env.ok(&[
        "provenance",
        "sign",
        &unsigned_path,
        "--key",
        &env.path("stranger.key"),
    ]);
    let unauthorized_result = env.run(&[
        "provenance",
        "verify",
        &env.path("pack.annpack"),
        &unsigned_path,
        "--trusted-builder-key",
        builder_pub,
        "--json",
    ]);
    assert!(!unauthorized_result.status.success());
    let unauthorized: Value = serde_json::from_slice(&unauthorized_result.stdout).unwrap();
    assert_eq!(unauthorized["details"]["builder_identity"], "untrusted");

    // 16. Compose with publisher authority and current channel state.
    env.ok(&[
        "release",
        "statement",
        "--output",
        &env.path("state.json"),
        "--publisher",
        "example.com",
        "--corpus",
        "support-manual",
        "--channel",
        "production",
        "--sequence",
        "1",
        "--current-root",
        &root_a,
        "--current-version",
        "1.0.0",
        "--valid-until",
        "2099-01-01T00:00:00Z",
    ]);
    env.ok(&[
        "release",
        "sign",
        &env.path("state.json"),
        "--key",
        &env.path("release.key"),
    ]);
    let policy: Value = serde_json::from_str(&env.ok(&[
        "verify",
        &env.path("pack.annpack"),
        "--policy",
        "authorized-current",
        "--trust-root",
        &env.path("trust.json"),
        "--channel-state",
        &env.path("state.json"),
        "--expect-corpus",
        "support-manual",
        "--expect-channel",
        "production",
        "--system-clock",
        "--json",
    ]))
    .unwrap();

    // 17. Every claim reported separately -- provenance's report and the
    //     policy's report are two objects with no field merged between them.
    assert_eq!(policy["policy"]["permitted"], true);
    assert_eq!(policy["policy"]["artifact_integrity"], "valid");
    assert_eq!(policy["policy"]["publisher_authority"], "authorized");
    assert_eq!(policy["policy"]["currency"], "current");
    // Confirming the stop conditions directly: neither report contains a field
    // from the other's vocabulary.
    assert!(
        policy.get("builder_identity").is_none(),
        "policy report absorbed a provenance field"
    );
    assert!(
        policy.get("envelope_signature").is_none(),
        "policy report absorbed a provenance field"
    );
    assert!(
        report.get("publisher_authority").is_none(),
        "provenance report absorbed a policy field"
    );
    assert!(
        report.get("currency").is_none(),
        "provenance report absorbed a policy field"
    );
}
