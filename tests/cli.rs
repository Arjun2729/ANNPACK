use std::process::Command;

use serde_json::Value;
use tempfile::TempDir;

#[test]
fn cli_build_verify_search_workflow() {
    let temp = TempDir::new().unwrap();
    let binary = env!("CARGO_BIN_EXE_annpack");
    let fixture = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    let pack = temp.path().join("demo.annpack");
    let build = Command::new(binary)
        .args([
            "build",
            &fixture,
            "--output",
            pack.to_str().unwrap(),
            "--name",
            "vendor-docs",
            "--version",
            "1.0.0",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let verify = Command::new(binary)
        .args(["verify", pack.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(verify.status.success());
    let verified: Value = serde_json::from_slice(&verify.stdout).unwrap();
    assert_eq!(verified["integrity_verified"], true);
    assert_eq!(verified["publisher_identity_trusted"], false);
    assert_eq!(verified["conformance"]["core_conformant"], true);

    let search = Command::new(binary)
        .args([
            "search",
            pack.to_str().unwrap(),
            "AP-104",
            "--mode",
            "lexical",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(search.status.success());
    let response: Value = serde_json::from_slice(&search.stdout).unwrap();
    assert!(
        response["results"][0]["text"]
            .as_str()
            .unwrap()
            .contains("API key has expired")
    );
    assert_eq!(
        response["results"][0]["evidence"]["schema"],
        "annpack-evidence-v1"
    );

    let passages = temp.path().join("passages.json");
    let export = Command::new(binary)
        .args([
            "export-passages",
            pack.to_str().unwrap(),
            "--output",
            passages.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(export.status.success());
    let passages: Value = serde_json::from_slice(&std::fs::read(passages).unwrap()).unwrap();
    assert_eq!(passages.as_array().unwrap().len(), 7);
}

#[test]
fn cli_inspect_defaults_to_json_and_human_opts_out() {
    let temp = TempDir::new().unwrap();
    let binary = env!("CARGO_BIN_EXE_annpack");
    let fixture = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    let pack = temp.path().join("inspect.annpack");
    let build = Command::new(binary)
        .args([
            "build",
            &fixture,
            "--output",
            pack.to_str().unwrap(),
            "--name",
            "inspect-docs",
            "--version",
            "1.0.0",
        ])
        .output()
        .unwrap();
    assert!(build.status.success());

    // JSON is the default, with or without the explicit flag. `--json` used to
    // be accepted and ignored; it now states the default rather than doing
    // nothing, and existing callers are unaffected either way.
    let mut root = None;
    for args in [
        vec!["inspect", pack.to_str().unwrap()],
        vec!["inspect", pack.to_str().unwrap(), "--json"],
    ] {
        let output = Command::new(binary).args(&args).output().unwrap();
        assert!(output.status.success(), "{args:?}");
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["conformance"]["core_conformant"], true);
        let hash = report["root_hash"].as_str().unwrap().to_string();
        assert_eq!(hash.len(), 64);
        root = Some(root.map_or(hash.clone(), |first: String| {
            assert_eq!(first, hash, "both spellings must produce the same report");
            first
        }));
    }

    let summary = Command::new(binary)
        .args(["inspect", pack.to_str().unwrap(), "--human"])
        .output()
        .unwrap();
    assert!(summary.status.success());
    let text = String::from_utf8(summary.stdout).unwrap();
    assert!(
        serde_json::from_str::<Value>(&text).is_err(),
        "--human must not print JSON: {text}"
    );
    assert!(text.contains("inspect-docs@1.0.0"), "{text}");
    assert!(text.contains(&root.unwrap()), "{text}");
    assert!(text.contains("core conformant: true"), "{text}");

    // The two formats are mutually exclusive rather than silently ranked.
    let conflict = Command::new(binary)
        .args(["inspect", pack.to_str().unwrap(), "--json", "--human"])
        .output()
        .unwrap();
    assert!(!conflict.status.success());
}

#[test]
fn cli_configures_a_verified_gemini_mcp_integration() {
    let temp = TempDir::new().unwrap();
    let binary = env!("CARGO_BIN_EXE_annpack");
    let fixture = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    let pack = temp.path().join("gemini.annpack");
    let build = Command::new(binary)
        .args([
            "build",
            &fixture,
            "--output",
            pack.to_str().unwrap(),
            "--name",
            "gemini-docs",
            "--version",
            "1.0.0",
        ])
        .output()
        .unwrap();
    assert!(build.status.success());

    let settings = temp.path().join(".gemini/settings.json");
    let integration = Command::new(binary)
        .args([
            "integrate",
            "gemini",
            pack.to_str().unwrap(),
            "--output",
            settings.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        integration.status.success(),
        "{}",
        String::from_utf8_lossy(&integration.stderr)
    );
    let report: Value = serde_json::from_slice(&integration.stdout).unwrap();
    assert_eq!(report["integration"], "gemini-cli-mcp");
    assert_eq!(report["verified_before_configuration"], true);
    assert_eq!(report["root_hash"].as_str().unwrap().len(), 64);

    let configured: Value = serde_json::from_slice(&std::fs::read(&settings).unwrap()).unwrap();
    assert_eq!(configured["mcpServers"]["annpack"]["args"][0], "mcp");
    assert_eq!(configured["mcpServers"]["annpack"]["trust"], true);
    assert_eq!(
        configured["mcpServers"]["annpack"]["includeTools"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        configured["mcpServers"]["annpack"]["args"][1],
        std::fs::canonicalize(&pack).unwrap().to_str().unwrap()
    );

    let duplicate = Command::new(binary)
        .args([
            "integrate",
            "gemini",
            pack.to_str().unwrap(),
            "--output",
            settings.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("already exists"));
}
