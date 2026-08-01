//! Documented input bounds on the two surfaces that read attacker-supplied
//! bytes before anything else validates them: the MCP JSON-RPC line reader and
//! the receipt file the CLI verifier opens.
//!
//! These are bounds on a single input, not a general defence against memory
//! exhaustion. They stop one specific failure: growing a buffer in proportion
//! to input that has not been checked yet.

use std::io::{BufReader, Cursor};
use std::path::PathBuf;
use std::process::Command;

use annpack::build::{BuildOptions, build_pack};
use annpack::evidence::MAX_RECEIPT_FILE_BYTES;
use annpack::mcp::{MAX_REQUEST_LINE_BYTES, McpServer};
use annpack::model::AccessClass;
use annpack::search::SearchEngine;
use serde_json::Value;
use tempfile::TempDir;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/docs-v1")
}

fn build(temp: &TempDir, name: &str) -> PathBuf {
    let output = temp.path().join(format!("{name}.annpack"));
    build_pack(&BuildOptions {
        input: fixture(),
        output: output.clone(),
        name: name.into(),
        version: "1.0.0".into(),
        description: None,
        source_revision: Some("git:limits".into()),
        base_url: None,
        created_at: None,
        license: None,
        access: AccessClass::Public,
        redistributable: None,
        policy_expires_at: None,
        policy_url: None,
        dependencies: Vec::new(),
        policy_override: None,
        vector_input: None,
        expansion_input: None,
        splade_input: None,
        anchors_input: None,
        target_chars: 1_200,
        max_chars: 2_400,
        input_format: annpack::ingest::InputFormat::Auto,
    })
    .unwrap();
    output
}

/// Runs the MCP server over a fixed script and returns one response per line.
fn serve(pack: &PathBuf, script: &str) -> Vec<Value> {
    let engine = SearchEngine::open_path(pack).unwrap();
    let mut output = Vec::new();
    McpServer::new(engine)
        .run(
            BufReader::new(Cursor::new(script.as_bytes().to_vec())),
            &mut output,
        )
        .unwrap();
    String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

const PING: &str = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;

#[test]
fn a_request_just_under_the_line_limit_is_still_served() {
    let temp = TempDir::new().unwrap();
    let pack = build(&temp, "limits");

    // A well-formed request padded with whitespace to just under the bound.
    // JSON-RPC tolerates the padding, so this exercises the size check alone.
    let padding = " ".repeat(MAX_REQUEST_LINE_BYTES - PING.len() - 1);
    let script = format!("{PING}{padding}\n");
    assert_eq!(script.len(), MAX_REQUEST_LINE_BYTES);

    let responses = serve(&pack, &script);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0]["id"], 1);
    assert!(responses[0].get("error").is_none(), "{:?}", responses[0]);
}

#[test]
fn a_request_over_the_line_limit_is_refused_and_framing_survives() {
    let temp = TempDir::new().unwrap();
    let pack = build(&temp, "limits");

    // One byte past the bound, then a normal request on the next line. The
    // second request must still be answered: the server skips the oversized
    // line rather than losing framing or closing the connection.
    let oversized = "x".repeat(MAX_REQUEST_LINE_BYTES + 1);
    let script = format!("{oversized}\n{PING}\n");

    let responses = serve(&pack, &script);
    assert_eq!(responses.len(), 2, "{responses:?}");
    assert_eq!(responses[0]["error"]["code"], -32600);
    assert!(
        responses[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("line limit"),
        "{:?}",
        responses[0]
    );
    assert_eq!(responses[1]["id"], 1);
    assert!(responses[1].get("error").is_none(), "{:?}", responses[1]);
}

#[test]
fn an_oversized_receipt_file_is_refused_before_it_is_read() {
    let temp = TempDir::new().unwrap();
    let receipt = temp.path().join("huge-receipt.json");

    // A sparse file one byte past the limit. Its contents are never parsed, so
    // this asserts the size check runs first.
    let file = std::fs::File::create(&receipt).unwrap();
    file.set_len(MAX_RECEIPT_FILE_BYTES + 1).unwrap();
    drop(file);

    let output = Command::new(env!("CARGO_BIN_EXE_annpack"))
        .args(["verify-evidence", receipt.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("byte limit"), "{stderr}");
}

#[test]
fn an_honest_receipt_stays_well_inside_the_file_limit() {
    let temp = TempDir::new().unwrap();
    let pack = build(&temp, "receipt-size");
    let engine = SearchEngine::open_path(&pack).unwrap();
    let passage = engine.passages().unwrap()[0].id.clone();
    let receipt = engine.receipt_for_passage(&passage).unwrap();
    let encoded = serde_json::to_vec(&receipt).unwrap();
    assert!(
        (encoded.len() as u64) < MAX_RECEIPT_FILE_BYTES,
        "a receipt for the standard fixture is {} bytes",
        encoded.len()
    );
}
