//! Explicit manifest-schema compatibility boundary.
//!
//! v0.3.1 removed a required manifest field without changing the wire version,
//! the manifest section format version, or the media type. New readers tolerated
//! old packs (unknown JSON fields are ignored) but old readers could not open
//! new packs, and the failure surfaced as a bare `missing field` deserialization
//! error rather than a version refusal. v0.4.0 makes the boundary explicit by
//! bumping the manifest section format version to 2.
//!
//! v0.5.0 crosses the same boundary again, to manifest format 3, by *removing*
//! `dependencies` and the ANN-5 policy descriptors. A v2 reader requires
//! `dependencies`, so the bump is what makes it decline rather than fail
//! mid-deserialization.
//!
//! These tests pin every direction of both boundaries so neither can regress.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use annpack::error::AnnpackError;
use annpack::format::{
    DIRECTORY_ENTRY_SIZE, HEADER_SIZE, MANIFEST_FORMAT_VERSION, PackReader,
    SUPPORTED_MANIFEST_FORMAT_VERSIONS,
};
use annpack::reader::MemoryReader;
use annpack::search::SearchEngine;

/// A pack written by v0.3.0: manifest section format 1, a `builder` field the
/// current `Manifest` struct no longer declares, and no `passage_merkle_root`.
fn legacy_pack() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("spec/test-vectors/compat/manifest-v1-legacy.annpack")
}

/// The current golden artifact, whatever the current manifest format is.
fn current_pack() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("spec/test-vectors/minimal-v3.annpack")
}

#[test]
fn new_reader_opens_an_old_manifest_v1_pack() {
    let reader = PackReader::open_path(legacy_pack()).unwrap();
    let manifest_entry = reader.entry(reader.header.manifest_section_id).unwrap();
    assert_eq!(
        manifest_entry.format_version, 1,
        "fixture must exercise the v1 manifest schema"
    );
    reader.verify_all().unwrap();

    // The dropped `builder` field must be ignored, not fatal.
    let manifest = reader.manifest().unwrap();
    assert_eq!(manifest.name, "golden-docs");
    assert!(
        manifest.passage_merkle_root.is_none(),
        "a v1 manifest commits no logical content root"
    );

    // And it must remain fully searchable, not merely parseable.
    let engine = SearchEngine::open_path(legacy_pack()).unwrap();
    let response = engine
        .search("AP-104", &annpack::search::SearchOptions::default())
        .unwrap();
    assert!(!response.results.is_empty());
}

#[test]
fn reading_an_old_pack_does_not_change_its_root() {
    // The v0.3.1 root reset changed which bytes a *builder* emits. It must not
    // change what a *reader* computes for an already-published artifact.
    let reader = PackReader::open_path(legacy_pack()).unwrap();
    assert_eq!(
        reader.root_hex(),
        "7fb855794ac5bbe4049947fd2421c44acd51ba6495e2db95b18995ac36db119b",
        "a previously published artifact must keep its published identity"
    );
}

#[test]
fn new_reader_opens_a_current_manifest_pack() {
    let reader = PackReader::open_path(current_pack()).unwrap();
    let manifest_entry = reader.entry(reader.header.manifest_section_id).unwrap();
    assert_eq!(manifest_entry.format_version, MANIFEST_FORMAT_VERSION);
    reader.verify_all().unwrap();
    let manifest = reader.manifest().unwrap();
    assert!(
        manifest.passage_merkle_root.is_some(),
        "manifest format 2 and later must commit a logical content root"
    );
}

#[test]
fn an_unknown_manifest_format_version_is_refused_at_the_container_boundary() {
    // Forward compatibility: a future manifest schema must be declined with an
    // explicit version error, not accepted and then mis-parsed. This is exactly
    // the failure mode v0.3.1 shipped, where an old reader hit a bare
    // `missing field \`builder\`` JSON error instead of a version refusal.
    let mut bytes = std::fs::read(current_pack()).unwrap();
    let unsupported = SUPPORTED_MANIFEST_FORMAT_VERSIONS.iter().max().unwrap() + 1;
    patch_manifest_format_version(&mut bytes, unsupported);

    let Err(error) = PackReader::open(Arc::new(MemoryReader::new(bytes))) else {
        panic!("an unsupported manifest format version must be refused");
    };
    match error {
        AnnpackError::Unsupported(message) => {
            assert!(
                message.contains("manifest section format version"),
                "expected a version refusal, got: {message}"
            );
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

/// Rewrite the manifest entry's section-format version and repair the content
/// root, so the artifact is structurally valid and fails *only* on the version.
fn patch_manifest_format_version(bytes: &mut [u8], version: u16) {
    let directory_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
    let directory_length = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
    let manifest_section_id = u32::from_le_bytes(bytes[40..44].try_into().unwrap());

    let mut patched = false;
    for index in 0..directory_length / DIRECTORY_ENTRY_SIZE {
        let entry = directory_offset + index * DIRECTORY_ENTRY_SIZE;
        if u32::from_le_bytes(bytes[entry..entry + 4].try_into().unwrap()) == manifest_section_id {
            bytes[entry + 6..entry + 8].copy_from_slice(&version.to_le_bytes());
            patched = true;
        }
    }
    assert!(patched, "fixture must contain a manifest directory entry");

    // Recompute BLAKE3("ANNPACK3-CONTENT-ROOT\0" || non-signature entries).
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ANNPACK3-CONTENT-ROOT\0");
    for index in 0..directory_length / DIRECTORY_ENTRY_SIZE {
        let entry = directory_offset + index * DIRECTORY_ENTRY_SIZE;
        let section_type = u16::from_le_bytes(bytes[entry + 4..entry + 6].try_into().unwrap());
        if section_type != 10 {
            hasher.update(&bytes[entry..entry + DIRECTORY_ENTRY_SIZE]);
        }
    }
    let root = *hasher.finalize().as_bytes();
    bytes[48..80].copy_from_slice(&root);
    assert_eq!(HEADER_SIZE, 128);
}

#[test]
fn a_legacy_pack_cannot_issue_a_standalone_receipt() {
    // A v1 manifest commits no logical content root, so a receipt's chain
    // cannot close. Refusing is correct; emitting an unverifiable receipt is not.
    let engine = SearchEngine::open_path(legacy_pack()).unwrap();
    let passages = engine.passages().unwrap();
    let error = engine
        .receipt_for_passage(&passages[0].id)
        .expect_err("a manifest-v1 pack must not issue receipts");
    assert!(matches!(error, AnnpackError::Unsupported(_)), "{error:?}");
}

/// A pack from the previous generation (manifest format 2, lexical index format
/// 2) must still open and search. This is the artifact a v0.4.x publisher
/// produced, and breaking it silently is the failure this whole file exists to
/// catch.
#[test]
fn new_reader_opens_a_previous_generation_pack() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("spec/test-vectors/compat/manifest-v2-lexical-v2.annpack");
    let reader = PackReader::open_path(&path).unwrap();
    let manifest_entry = reader.entry(reader.header.manifest_section_id).unwrap();
    assert_eq!(manifest_entry.format_version, 2);
    reader.verify_all().unwrap();

    let engine = annpack::search::SearchEngine::open_path(&path).unwrap();
    let hits = engine
        .search(
            "AP-104",
            &annpack::search::SearchOptions {
                limit: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        hits.results[0].text.contains("API key has expired"),
        "a previous-generation pack must still return the right passage"
    );
}
