use std::fs;

use annpack::build::{BuildOptions, build_pack};
use annpack::format::{PackReader, SectionType};
use annpack::ingest::InputFormat;
use annpack::model::{AccessClass, Document};
use annpack::search::{SearchEngine, SearchMode, SearchOptions};
use tempfile::TempDir;

fn options(input: std::path::PathBuf, output: std::path::PathBuf) -> BuildOptions {
    BuildOptions {
        input,
        output,
        name: "okf-ledger".into(),
        version: "0.1.0".into(),
        description: Some("OKF compiler conformance fixture".into()),
        source_revision: Some("okf:test-vector-v01".into()),
        base_url: None,
        created_at: None,
        license: Some("CC0-1.0".into()),
        access: AccessClass::Public,
        redistributable: Some(true),
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
        input_format: InputFormat::Auto,
    }
}

#[test]
fn auto_detects_compiles_and_searches_okf() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let temp = TempDir::new().unwrap();
    let first = options(
        root.join("fixtures/okf-v01"),
        temp.path().join("first.annpack"),
    );
    let second = options(
        root.join("fixtures/okf-v01"),
        temp.path().join("second.annpack"),
    );
    let report = build_pack(&first).unwrap();
    build_pack(&second).unwrap();

    assert_eq!(report.input_format, "okf");
    assert_eq!(report.input_format_version.as_deref(), Some("0.1"));
    assert_eq!(report.source_digest.len(), 64);
    assert!(report.capabilities.contains(&"source-okf".to_string()));
    assert_eq!(
        fs::read(&first.output).unwrap(),
        fs::read(&second.output).unwrap()
    );

    let reader = PackReader::open_path(&first.output).unwrap();
    let manifest = reader.manifest().unwrap();
    let source = manifest.source.unwrap();
    assert_eq!(source.format, "okf");
    assert_eq!(source.version.as_deref(), Some("0.1"));
    assert_eq!(source.digest, report.source_digest);

    let documents_entry = reader.first_entry(SectionType::Documents).unwrap();
    let documents: Vec<Document> =
        serde_json::from_slice(&reader.read_section(documents_entry.section_id).unwrap()).unwrap();
    let ledger = documents
        .iter()
        .find(|document| document.source_path == "datasets/ledger.md")
        .unwrap();
    assert_eq!(ledger.metadata["type"], "BigQuery Dataset");
    assert_eq!(
        ledger.metadata["tags"],
        "[\"accounting\",\"deterministic\"]"
    );
    assert_eq!(
        ledger.metadata["producer_extension"],
        "{\"owner\":\"finance-platform\",\"tier\":\"critical\"}"
    );

    let engine = SearchEngine::open_path(&first.output).unwrap();
    let response = engine
        .search(
            "What does AP-104 mean?",
            &SearchOptions {
                mode: SearchMode::Lexical,
                ..SearchOptions::default()
            },
        )
        .unwrap();
    assert!(
        response.results[0]
            .text
            .contains("obsolete ledger revision")
    );
    assert_eq!(
        response.results[0].url.as_deref(),
        Some("https://example.test/bigquery/ledger/transactions#operational-knowledge")
    );
}

#[test]
fn explicit_okf_rejects_a_concept_without_type() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("index.md"), "# Concepts\n").unwrap();
    fs::write(
        temp.path().join("broken.md"),
        "---\ntitle: Broken\n---\nMissing the required OKF type.\n",
    )
    .unwrap();
    let mut build = options(
        temp.path().to_path_buf(),
        temp.path().join("broken.annpack"),
    );
    build.input_format = InputFormat::Okf;
    let error = build_pack(&build).unwrap_err().to_string();
    assert!(error.contains("missing required non-empty type"), "{error}");
}
