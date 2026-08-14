//! Fleet policy lifecycle, driven entirely through the CLI: init, sign,
//! rotate, verify, and compliance evaluation.

#![cfg(feature = "signing")]

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;
use tempfile::TempDir;

struct Operator {
    _temp: TempDir,
    dir: PathBuf,
    binary: &'static str,
}

impl Operator {
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

    fn keygen(&self, name: &str) {
        self.ok(&["keygen", "--output", &self.path(&format!("{name}.key"))]);
    }

    fn init_policy(&self, output: &str, domain: &str, revision: &str, key: &str) {
        self.ok(&[
            "fleet",
            "policy",
            "init",
            "--output",
            &self.path(output),
            "--domain",
            domain,
            "--revision",
            revision,
            "--valid-until",
            "2099-01-01T00:00:00Z",
            "--key",
            &self.path(&format!("{key}.pub")),
            "--threshold",
            "1",
            "--allow-publisher",
            "example.com",
            "--allow-scope",
            "support-manual:production",
            "--required-policy",
            "authorized-current",
            "--deny-on-incident",
            "equivocation",
        ]);
    }

    fn sign(&self, policy: &str, key: &str) {
        self.ok(&[
            "fleet",
            "policy",
            "sign",
            &self.path(policy),
            "--key",
            &self.path(&format!("{key}.key")),
        ]);
    }

    fn verify(&self, policy: &str, prior: Option<&str>) -> (bool, Value) {
        let path = self.path(policy);
        let mut args = vec![
            "fleet",
            "policy",
            "verify",
            &path,
            "--system-clock",
            "--json",
        ];
        let prior_path;
        if let Some(name) = prior {
            prior_path = self.path(name);
            args.push("--prior");
            args.push(&prior_path);
        }
        let output = self.run(&args);
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        (output.status.success(), value)
    }

    fn evaluate(&self, local: &str, required: &str) -> (bool, Value) {
        let local_path = self.path(local);
        let required_path = self.path(required);
        let output = self.run(&[
            "fleet",
            "policy",
            "evaluate",
            "--local",
            &local_path,
            "--required",
            &required_path,
            "--system-clock",
            "--json",
        ]);
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        (output.status.success(), value)
    }
}

#[test]
fn init_sign_and_verify_on_first_contact() {
    let operator = Operator::new();
    operator.keygen("k1");
    operator.init_policy("p1.json", "acme.example", "1", "k1");
    operator.sign("p1.json", "k1");

    let (ok, report) = operator.verify("p1.json", None);
    assert!(ok, "{report}");
    assert_eq!(report["verified"], Value::Bool(true));
    assert_eq!(report["first_contact"], Value::Bool(true));
}

#[test]
fn an_unsigned_policy_fails_verification() {
    let operator = Operator::new();
    operator.keygen("k1");
    operator.init_policy("p1.json", "acme.example", "1", "k1");

    let (ok, report) = operator.verify("p1.json", None);
    assert!(!ok);
    assert_eq!(report["ok"], Value::Bool(false));
    assert_eq!(report["error"]["kind"], "unauthorized_role");
}

#[test]
fn rotation_requires_both_keys_and_a_higher_revision() {
    let operator = Operator::new();
    operator.keygen("k1");
    operator.keygen("k2");
    operator.init_policy("p1.json", "acme.example", "1", "k1");
    operator.sign("p1.json", "k1");

    operator.init_policy("p2.json", "acme.example", "2", "k2");
    operator.sign("p2.json", "k2");

    // Self-signed only: missing the prior key's signature.
    let (ok, report) = operator.verify("p2.json", Some("p1.json"));
    assert!(!ok, "{report}");
    assert_eq!(report["details"]["signed_by_prior"], Value::Bool(false));

    operator.sign("p2.json", "k1");
    let (ok, report) = operator.verify("p2.json", Some("p1.json"));
    assert!(ok, "{report}");
    assert_eq!(report["signed_by_prior"], Value::Bool(true));
    assert_eq!(report["revision_advanced"], Value::Bool(true));
}

#[test]
fn matching_local_and_required_policy_is_compliant() {
    let operator = Operator::new();
    operator.keygen("k1");
    operator.init_policy("p1.json", "acme.example", "1", "k1");
    operator.sign("p1.json", "k1");

    let (ok, report) = operator.evaluate("p1.json", "p1.json");
    assert!(ok, "{report}");
    assert_eq!(report["status"], "compliant");
}

#[test]
fn a_lower_local_revision_is_reported_as_drifted() {
    let operator = Operator::new();
    operator.keygen("k1");
    operator.keygen("k2");
    operator.init_policy("p1.json", "acme.example", "1", "k1");
    operator.sign("p1.json", "k1");
    operator.init_policy("p2.json", "acme.example", "2", "k2");
    operator.sign("p2.json", "k2");
    operator.sign("p2.json", "k1");

    let (ok, report) = operator.evaluate("p1.json", "p2.json");
    assert!(!ok);
    assert_eq!(report["error"]["kind"], "fleet_policy_drifted");
    assert_eq!(report["details"]["status"], "drifted");
    assert_eq!(report["details"]["local_revision"], 1);
    assert_eq!(report["details"]["required_revision"], 2);
}

#[test]
fn an_unverified_required_policy_is_unavailable_not_compliant() {
    let operator = Operator::new();
    operator.keygen("k1");
    operator.init_policy("p1.json", "acme.example", "1", "k1");
    operator.sign("p1.json", "k1");
    // Required policy exists but was never signed.
    operator.init_policy("p2.json", "acme.example", "2", "k1");

    let (ok, report) = operator.evaluate("p1.json", "p2.json");
    assert!(!ok);
    assert_eq!(report["error"]["kind"], "fleet_policy_unavailable");
    assert_eq!(report["details"]["status"], "unavailable");
}

#[test]
fn exactly_one_json_object_on_stdout_on_every_path() {
    // Regression coverage for the two-JSON-objects bug class fixed earlier
    // this session in other commands: a failing evaluate must not print a
    // report and then a separate error envelope.
    let operator = Operator::new();
    operator.keygen("k1");
    operator.init_policy("p1.json", "acme.example", "1", "k1");
    operator.sign("p1.json", "k1");
    operator.init_policy("p2.json", "acme.example", "2", "k1");

    let output = operator.run(&[
        "fleet",
        "policy",
        "evaluate",
        "--local",
        &operator.path("p1.json"),
        "--required",
        &operator.path("p2.json"),
        "--system-clock",
        "--json",
    ]);
    let parsed: Result<Value, _> = serde_json::from_slice(&output.stdout);
    assert!(
        parsed.is_ok(),
        "stdout was not exactly one JSON object: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}
