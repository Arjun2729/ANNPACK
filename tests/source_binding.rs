//! Manifest format 4: every artifact commits to the bytes it was built from.
//!
//! The digest was always computed for every input format; only OKF artifacts
//! committed to it, so provenance for a Markdown artifact was a builder claim
//! the artifact could not corroborate. These tests hold the new property and,
//! equally, hold the compatibility boundary: old artifacts keep their roots,
//! remain readable, and their missing descriptor stays legitimate history rather
//! than being reported as corruption (ADR-0005).

use std::path::{Path, PathBuf};

use adyar::conformance::{SourceBinding, inspect_conformance};
use adyar::format::{MANIFEST_FORMAT_VERSION, PackReader};
use serde_json::Value;
use tempfile::TempDir;

const BINARY: &str = env!("CARGO_BIN_EXE_adyar");

struct Corpus {
    _temp: TempDir,
    source: PathBuf,
    out: PathBuf,
}

impl Corpus {
    /// A corpus of the given format. `files` are (relative path, contents).
    fn new(files: &[(&str, &str)]) -> Self {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("src");
        std::fs::create_dir_all(&source).unwrap();
        let corpus = Self {
            source,
            out: temp.path().join("out.annpack"),
            _temp: temp,
        };
        for (name, body) in files {
            corpus.write(name, body);
        }
        corpus
    }

    fn write(&self, name: &str, body: &str) {
        let path = self.source.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    /// Build, returning (artifact root, source digest reported by build --json).
    fn build(&self) -> (String, String) {
        let output = std::process::Command::new(BINARY)
            .args([
                "build",
                self.source.to_str().unwrap(),
                "--output",
                self.out.to_str().unwrap(),
                "--name",
                "fixture",
                "--version",
                "1.0.0",
                "--json",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        (
            report["root_hash"].as_str().unwrap().to_string(),
            report["source_digest"].as_str().unwrap().to_string(),
        )
    }

    /// The source descriptor the artifact itself commits to.
    fn authenticated_source(&self) -> Value {
        let reader = PackReader::open_path(&self.out).unwrap();
        reader.verify_all().unwrap();
        serde_json::to_value(reader.manifest().unwrap().source).unwrap()
    }
}

const MARKDOWN: &[(&str, &str)] = &[(
    "guide.md",
    "---\ntitle: Guide\n---\n\n# Guide\n\nRotate the signing key every ninety days.\n",
)];

const MDX: &[(&str, &str)] = &[(
    "guide.mdx",
    "---\ntitle: Guide\n---\n\n# Guide\n\nRotate the signing key every ninety days.\n",
)];

#[test]
fn a_markdown_artifact_authenticates_its_source_digest() {
    let corpus = Corpus::new(MARKDOWN);
    let (_, reported) = corpus.build();
    let source = corpus.authenticated_source();
    assert_eq!(source["format"], "markdown");
    assert_eq!(source["digest_algorithm"], "blake3");
    assert_eq!(source["digest"], Value::String(reported));
}

#[test]
fn an_mdx_artifact_authenticates_its_source_digest() {
    let corpus = Corpus::new(MDX);
    let (_, reported) = corpus.build();
    let source = corpus.authenticated_source();
    // MDX is ingested through the Markdown path, so it reports that format.
    assert_eq!(source["format"], "markdown");
    assert_eq!(source["digest"], Value::String(reported));
}

#[test]
fn an_okf_artifact_still_authenticates_its_source_digest() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/okf-v02");
    let temp = TempDir::new().unwrap();
    let out = temp.path().join("okf.annpack");
    let output = std::process::Command::new(BINARY)
        .args([
            "build",
            fixture.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--name",
            "okf",
            "--version",
            "1.0.0",
            "--source-format",
            "okf",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    let reader = PackReader::open_path(&out).unwrap();
    let source = serde_json::to_value(reader.manifest().unwrap().source).unwrap();
    assert_eq!(source["format"], "okf");
    assert_eq!(source["digest"], report["source_digest"]);
}

#[test]
fn the_reported_digest_and_the_authenticated_digest_are_one_value() {
    // One computation read twice. Two would be two things to keep in agreement.
    for files in [MARKDOWN, MDX] {
        let corpus = Corpus::new(files);
        let (_, reported) = corpus.build();
        assert_eq!(corpus.authenticated_source()["digest"], reported);
    }
}

#[test]
fn changing_one_consumed_byte_changes_the_digest_and_the_root() {
    let corpus = Corpus::new(MARKDOWN);
    let (root_before, digest_before) = corpus.build();

    corpus.write(
        "guide.md",
        "---\ntitle: Guide\n---\n\n# Guide\n\nRotate the signing key every sixty days.\n",
    );
    let (root_after, digest_after) = corpus.build();

    assert_ne!(digest_before, digest_after, "source digest did not move");
    assert_ne!(root_before, root_after, "artifact root did not move");
}

#[test]
fn changing_a_consumed_path_changes_the_digest_and_the_root() {
    // Same bytes, different filename. The digest covers paths as well as
    // contents, or two corpora with rearranged files would be indistinguishable.
    let corpus = Corpus::new(MARKDOWN);
    let (root_before, digest_before) = corpus.build();

    std::fs::remove_file(corpus.source.join("guide.md")).unwrap();
    corpus.write(
        "handbook.md",
        "---\ntitle: Guide\n---\n\n# Guide\n\nRotate the signing key every ninety days.\n",
    );
    let (root_after, digest_after) = corpus.build();

    assert_ne!(digest_before, digest_after);
    assert_ne!(root_before, root_after);
}

#[test]
fn a_file_ingestion_ignores_changes_neither_digest_nor_root() {
    let corpus = Corpus::new(MARKDOWN);
    let (root_before, digest_before) = corpus.build();

    // Not a `.md`/`.mdx` file, so ingestion never reads it.
    corpus.write(
        "notes.txt",
        "scratch notes that are not part of the corpus\n",
    );
    corpus.write("build.log", "irrelevant\n");
    let (root_after, digest_after) = corpus.build();

    assert_eq!(
        digest_before, digest_after,
        "an unread file moved the digest"
    );
    assert_eq!(root_before, root_after, "an unread file moved the root");
}

#[test]
fn repeated_builds_stay_byte_identical() {
    let corpus = Corpus::new(MARKDOWN);
    let (first_root, first_digest) = corpus.build();
    let first = std::fs::read(&corpus.out).unwrap();
    let (second_root, second_digest) = corpus.build();
    let second = std::fs::read(&corpus.out).unwrap();

    assert_eq!(first_root, second_root);
    assert_eq!(first_digest, second_digest);
    assert_eq!(first, second, "the builder is no longer deterministic");
}

#[test]
fn new_readers_open_old_artifacts_whose_manifest_predates_the_requirement() {
    let legacy = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("spec/test-vectors/compat/manifest-v1-legacy.annpack");
    let reader = PackReader::open_path(&legacy).unwrap();
    reader.verify_all().unwrap();

    let manifest = reader.manifest().unwrap();
    assert!(manifest.source.is_none(), "fixture is not actually legacy");

    // Absence here is history, not corruption, and must not be reported as a
    // Core defect.
    let report = inspect_conformance(&reader).unwrap();
    assert_eq!(report.source_binding, SourceBinding::AbsentLegacyArtifact);
    assert!(
        report.core_conformant,
        "a legacy artifact was called non-conformant: {:?}",
        report.core_issues
    );
}

#[test]
fn an_old_artifacts_root_is_still_computed_identically() {
    // The format change alters what a builder emits. It must not alter what a
    // reader computes for something already published.
    let legacy = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("spec/test-vectors/compat/manifest-v1-legacy.annpack");
    let reader = PackReader::open_path(&legacy).unwrap();
    assert_eq!(
        reader.root_hex(),
        "7fb855794ac5bbe4049947fd2421c44acd51ba6495e2db95b18995ac36db119b"
    );
}

#[test]
fn a_current_artifact_declares_the_new_manifest_format() {
    let corpus = Corpus::new(MARKDOWN);
    corpus.build();
    let reader = PackReader::open_path(&corpus.out).unwrap();
    let entry = reader.entry(reader.header.manifest_section_id).unwrap();
    assert_eq!(entry.format_version, MANIFEST_FORMAT_VERSION);
    assert_eq!(MANIFEST_FORMAT_VERSION, 4);

    let report = inspect_conformance(&reader).unwrap();
    assert_eq!(report.source_binding, SourceBinding::Authenticated);
    assert!(report.core_conformant);
}

#[test]
fn a_format_four_artifact_missing_its_source_descriptor_is_refused() {
    // Constructed directly: the builder cannot emit this, and the reader must
    // still refuse it rather than trusting that no producer would.
    use adyar::format::manifest_source_digest_issue;
    use adyar::model::Manifest;

    let corpus = Corpus::new(MARKDOWN);
    corpus.build();
    let reader = PackReader::open_path(&corpus.out).unwrap();
    let mut manifest: Manifest = reader.manifest().unwrap();
    assert!(manifest.source.is_some());

    manifest.source = None;
    assert!(
        manifest_source_digest_issue(&manifest, 4).is_some(),
        "format 4 accepted a manifest with no source descriptor"
    );
    // The same manifest is fine under the formats that predate the rule.
    for version in [1, 2, 3] {
        assert!(manifest_source_digest_issue(&manifest, version).is_none());
    }
}

#[test]
fn a_malformed_source_descriptor_is_refused_under_format_four() {
    use adyar::format::manifest_source_digest_issue;
    use adyar::model::{Manifest, SourceDescriptor};

    let corpus = Corpus::new(MARKDOWN);
    corpus.build();
    let reader = PackReader::open_path(&corpus.out).unwrap();
    let base: Manifest = reader.manifest().unwrap();

    let cases = [
        ("wrong algorithm", "sha256", "aa".repeat(32), "markdown"),
        ("short digest", "blake3", "aa".repeat(16), "markdown"),
        ("uppercase digest", "blake3", "AA".repeat(32), "markdown"),
        // `auto` is a request, not a resolved format: recording it would leave a
        // verifier unable to tell which ingestion rules produced the digest.
        ("unresolved format", "blake3", "aa".repeat(32), "auto"),
        ("empty format", "blake3", "aa".repeat(32), ""),
    ];
    for (label, algorithm, digest, format) in cases {
        let mut manifest = base.clone();
        manifest.source = Some(SourceDescriptor {
            format: format.into(),
            version: None,
            digest_algorithm: algorithm.into(),
            digest,
        });
        assert!(
            manifest_source_digest_issue(&manifest, 4).is_some(),
            "format 4 accepted a descriptor with a {label}"
        );
    }
}

// Forward compatibility for unknown manifest versions is covered by
// `tests/compatibility.rs::an_unknown_manifest_format_version_is_refused_at_the_container_boundary`,
// which repairs the content root after patching so the refusal is attributable
// to the version rather than to the root check firing first.

#[test]
fn every_format_uses_one_digest_algorithm() {
    let markdown = Corpus::new(MARKDOWN);
    markdown.build();
    assert_eq!(
        markdown.authenticated_source()["digest_algorithm"],
        "blake3"
    );

    let mdx = Corpus::new(MDX);
    mdx.build();
    assert_eq!(mdx.authenticated_source()["digest_algorithm"], "blake3");
}
