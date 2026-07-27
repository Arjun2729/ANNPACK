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

/// OKF v0.2 added the provenance, trust and lifecycle frontmatter families.
/// The build must accept them, preserve them losslessly, and must not invent a
/// version the bundle never declared.
#[test]
fn compiles_okf_v02_and_preserves_the_trust_families() {
    let temp = TempDir::new().unwrap();
    let output = temp.path().join("okf-v02.annpack");
    let mut options = options(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/okf-v02"),
        output.clone(),
    );
    options.input_format = InputFormat::Okf;
    let report = build_pack(&options).unwrap();
    assert_eq!(report.input_format, "okf");
    assert_eq!(report.input_format_version.as_deref(), Some("0.2"));

    let engine = SearchEngine::open_path(&output).unwrap();
    let documents_section = engine
        .reader()
        .first_entry(SectionType::Documents)
        .unwrap()
        .section_id;
    let documents: Vec<Document> =
        serde_json::from_slice(&engine.reader().read_section(documents_section).unwrap()).unwrap();
    let metric = documents
        .iter()
        .find(|d| d.source_path == "metrics/gross-margin.md")
        .expect("the v0.2 concept must be ingested");

    // Provenance, trust and lifecycle families survive verbatim. Structured
    // values are stored as compact JSON, which round-trips losslessly.
    assert_eq!(metric.metadata["status"], "stable");
    assert_eq!(metric.metadata["stale_after"], "2026-12-31");
    assert_eq!(
        metric.metadata["generated"],
        r#"{"by":"human:jsmith@acme","at":"2026-02-01T10:00:00Z"}"#
    );
    assert_eq!(
        metric.metadata["verified"],
        r#"[{"by":"human:kliu@acme","at":"2026-06-25T09:00:00Z"}]"#
    );
    assert_eq!(metric.metadata["okf.version"], "0.2");
}

/// Regression: neither OKF v0.1 nor v0.2 prohibits frontmatter in `log.md`.
/// v0.1's "Index files contain no frontmatter" governs `index.md` only, and
/// v0.2 §9 constrains just the body. We invented the rule and it made the
/// upstream `acme_retail` v0.2 bundle unbuildable.
#[test]
fn a_log_file_may_carry_frontmatter() {
    let temp = TempDir::new().unwrap();
    let mut options = options(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/okf-v02"),
        temp.path().join("log-frontmatter.annpack"),
    );
    options.input_format = InputFormat::Okf;
    build_pack(&options).expect("a log.md with frontmatter is conformant");
}

/// OKF §12 makes `okf_version` optional. Absent means undeclared, not 0.1;
/// claiming 0.1 mislabels a v0.2 bundle that simply omits the key.
#[test]
fn an_undeclared_okf_version_is_not_reported_as_0_1() {
    let temp = TempDir::new().unwrap();
    let bundle = temp.path().join("undeclared");
    std::fs::create_dir_all(bundle.join("metrics")).unwrap();
    std::fs::write(
        bundle.join("index.md"),
        "# Subdirectories\n\n* [metrics](metrics/index.md)\n",
    )
    .unwrap();
    std::fs::write(
        bundle.join("metrics/index.md"),
        "# Metrics\n\n* [m](m.md)\n",
    )
    .unwrap();
    std::fs::write(
        bundle.join("metrics/m.md"),
        "---\ntype: Metric\ntitle: M\n---\n\n# Body\n\nText.\n",
    )
    .unwrap();
    let mut options = options(bundle, temp.path().join("undeclared.annpack"));
    options.input_format = InputFormat::Okf;
    let report = build_pack(&options).unwrap();
    assert_eq!(report.input_format_version, None);
}
