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

/// Sets up a project directory holding a copy of the docs fixture, so that
/// relative paths in `annpack.toml` resolve the way a real project's would.
fn project_with_config(config: &str) -> TempDir {
    let temp = TempDir::new().unwrap();
    let fixture = format!("{}/fixtures/docs-v1", env!("CARGO_MANIFEST_DIR"));
    let docs = temp.path().join("docs");
    std::fs::create_dir(&docs).unwrap();
    for entry in std::fs::read_dir(&fixture).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), docs.join(entry.file_name())).unwrap();
        }
    }
    std::fs::write(temp.path().join("annpack.toml"), config).unwrap();
    temp
}

fn manifest_of(pack: &std::path::Path) -> Value {
    let binary = env!("CARGO_BIN_EXE_annpack");
    let inspect = Command::new(binary)
        .args(["inspect", pack.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        inspect.status.success(),
        "{}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    serde_json::from_slice(&inspect.stdout).unwrap()
}

/// The shortest build a configured project can run: no arguments at all.
#[test]
fn cli_build_reads_stable_fields_from_project_configuration() {
    let temp = project_with_config(
        "[build]\nname = \"vendor-docs\"\nversion = \"1.0.0\"\nsource = \"docs\"\noutput = \"knowledge.annpack\"\n",
    );
    let build = Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args(["build", "--json"])
        .current_dir(temp.path())
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let manifest = manifest_of(&temp.path().join("knowledge.annpack"));
    assert_eq!(manifest["manifest"]["name"], "vendor-docs");
    assert_eq!(manifest["manifest"]["version"], "1.0.0");
}

/// Configuration is a default, not an override: a build script that passes a
/// version explicitly must still get the version it asked for.
#[test]
fn cli_build_prefers_explicit_arguments_over_configuration() {
    let temp = project_with_config(
        "[build]\nname = \"vendor-docs\"\nversion = \"1.0.0\"\nsource = \"docs\"\noutput = \"from-config.annpack\"\n",
    );
    let pack = temp.path().join("explicit.annpack");
    let build = Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args([
            "build",
            "docs",
            "--output",
            pack.to_str().unwrap(),
            "--name",
            "release-notes",
            "--version",
            "2.5.0",
            "--json",
        ])
        .current_dir(temp.path())
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(
        !temp.path().join("from-config.annpack").exists(),
        "the configured output path was written despite an explicit --output"
    );

    let manifest = manifest_of(&pack);
    assert_eq!(manifest["manifest"]["name"], "release-notes");
    assert_eq!(manifest["manifest"]["version"], "2.5.0");
}

/// Configuration must be a way of typing the same values, not a second code
/// path: identical inputs have to produce identical bytes whichever route they
/// arrive by, or the artifact root would depend on how the build was invoked.
#[test]
fn cli_build_from_configuration_is_byte_identical_to_explicit_arguments() {
    let binary = env!("CARGO_BIN_EXE_annpack");
    let temp = project_with_config(
        "[build]\nname = \"vendor-docs\"\nversion = \"1.0.0\"\nsource = \"docs\"\noutput = \"configured.annpack\"\n",
    );

    let configured = Command::new(binary)
        .args(["build", "--created-at", "2026-01-01T00:00:00Z", "--json"])
        .current_dir(temp.path())
        .output()
        .unwrap();
    assert!(
        configured.status.success(),
        "{}",
        String::from_utf8_lossy(&configured.stderr)
    );

    let explicit = Command::new(binary)
        .args([
            "build",
            "docs",
            "--output",
            "explicit.annpack",
            "--name",
            "vendor-docs",
            "--version",
            "1.0.0",
            "--created-at",
            "2026-01-01T00:00:00Z",
            "--json",
        ])
        .current_dir(temp.path())
        .output()
        .unwrap();
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );

    assert_eq!(
        std::fs::read(temp.path().join("configured.annpack")).unwrap(),
        std::fs::read(temp.path().join("explicit.annpack")).unwrap(),
        "a configured build and an explicit build produced different bytes"
    );
}

/// A build run with neither source of a required value has to say both ways of
/// supplying it, since the answer is different for a one-off and a project.
#[test]
fn cli_build_without_a_required_field_names_both_ways_to_supply_it() {
    let temp = project_with_config("[build]\nsource = \"docs\"\noutput = \"out.annpack\"\n");
    let build = Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args(["build"])
        .current_dir(temp.path())
        .output()
        .unwrap();
    assert!(!build.status.success());
    let message = String::from_utf8_lossy(&build.stderr);
    assert!(message.contains("--name"), "{message}");
    assert!(message.contains("annpack.toml"), "{message}");
}

/// A typo in configuration must not silently build an artifact whose identity
/// differs from what the project wrote down.
#[test]
fn cli_build_refuses_malformed_project_configuration() {
    let temp = project_with_config("[build]\nnmae = \"typo\"\n");
    let build = Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args([
            "build",
            "docs",
            "--output",
            "out.annpack",
            "--name",
            "vendor-docs",
            "--version",
            "1.0.0",
        ])
        .current_dir(temp.path())
        .output()
        .unwrap();
    assert!(!build.status.success());
    assert!(
        String::from_utf8_lossy(&build.stderr).contains("annpack.toml"),
        "the error did not name the file it came from"
    );
}
