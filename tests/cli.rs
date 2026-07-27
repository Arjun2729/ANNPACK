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
