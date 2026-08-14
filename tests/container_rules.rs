//! Container-level acceptance rules that the specification states normatively
//! and that no other suite pins directly.
//!
//! * FORMAT-v3 §2 — "An unknown required section or required codec MUST be
//!   rejected. Unknown optional sections, derived or not, MUST be ignored
//!   safely." Whether an unknown section carries the derived flag is not a
//!   reason to refuse it; only *required* is.
//! * FORMAT-v3 §3.1 — the artifact root is not a whole-file hash.
//! * FORMAT-v3 §8.1 — the signature covers the artifact root, so the rest of a
//!   signature envelope is unauthenticated metadata.
//! * FORMAT-v3 §4.1 — manifest section format 2 and later MUST carry a
//!   `passage_merkle_root` of exactly 64 lowercase hexadecimal characters.
//!   Format 1 predates the field and stays readable without it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use adyar::conformance::inspect_conformance_with_manifest;
use adyar::error::AdyarError;
use adyar::format::{
    Codec, FLAG_DERIVED, FLAG_REQUIRED, MANIFEST_FORMAT_VERSION, PackReader, PackWriter,
    SectionData, SectionType,
};
use adyar::reader::MemoryReader;
use adyar::search::SearchEngine;

/// A section type this reader does not know and never will: high enough to stay
/// outside any plausible future v3 assignment.
const UNKNOWN_SECTION_TYPE: u16 = 9_000;

fn golden_pack() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("spec/test-vectors/minimal-v3.annpack")
}

// --- Unknown sections --------------------------------------------------------

/// The golden artifact with one extra section of an unrecognized type, carrying
/// the requested flags. Everything else about the container stays valid, so a
/// rejection can only come from the unknown section itself.
fn golden_with_unknown_section(flags: u16) -> Vec<u8> {
    let reader = PackReader::open_path(golden_pack()).unwrap();
    let mut sections = reader.all_section_data(true).unwrap();
    let section_id = sections
        .iter()
        .map(|section| section.section_id)
        .max()
        .unwrap()
        + 1;
    let bytes = b"opaque payload from a future extension".to_vec();
    sections.push(SectionData {
        section_id,
        section_type: SectionType::Other(UNKNOWN_SECTION_TYPE),
        format_version: 1,
        codec: Codec::None,
        flags,
        item_count: 1,
        logical_length: bytes.len() as u64,
        bytes,
    });
    let mut writer = PackWriter::new().with_flags(reader.header.flags);
    for section in sections {
        writer.push(section).unwrap();
    }
    writer.build_bytes().unwrap()
}

fn open_unknown(flags: u16) -> adyar::error::Result<PackReader> {
    PackReader::open(Arc::new(MemoryReader::new(golden_with_unknown_section(
        flags,
    ))))
}

/// `PackReader` is not `Debug`, so unwrap the error side explicitly.
fn expect_unknown_rejected(flags: u16, why: &str) -> AdyarError {
    match open_unknown(flags) {
        Ok(_) => panic!("{why}"),
        Err(error) => error,
    }
}

#[test]
fn an_unknown_optional_section_is_accepted_and_ignored() {
    let reader = open_unknown(0).expect("an unknown optional section must be ignored safely");
    assert!(
        reader
            .entries
            .iter()
            .any(|entry| entry.section_type == SectionType::Other(UNKNOWN_SECTION_TYPE)),
        "the unknown section must still appear in the directory"
    );
    // "Ignored safely" means the rest of the artifact keeps working, including
    // whole-artifact verification and Core conformance.
    reader.verify_all().unwrap();
    let manifest = reader.manifest().unwrap();
    assert!(inspect_conformance_with_manifest(&reader, &manifest).core_conformant);
}

#[test]
fn an_unknown_optional_derived_section_is_accepted_and_ignored() {
    // The derived flag on an unknown type carries no meaning for a reader that
    // never interprets the section. Refusing it would break forward
    // compatibility for exactly the extensions the flag exists to describe.
    let reader =
        open_unknown(FLAG_DERIVED).expect("an unknown optional derived section must be ignored");
    let entry = reader
        .entries
        .iter()
        .find(|entry| entry.section_type == SectionType::Other(UNKNOWN_SECTION_TYPE))
        .expect("the unknown section must still appear in the directory");
    assert!(entry.derived());
    assert!(!entry.required());
    reader.verify_all().unwrap();
    let manifest = reader.manifest().unwrap();
    assert!(inspect_conformance_with_manifest(&reader, &manifest).core_conformant);
}

#[test]
fn an_unknown_required_section_is_rejected() {
    let error =
        expect_unknown_rejected(FLAG_REQUIRED, "an unknown required section must be refused");
    assert!(
        matches!(error, AdyarError::Unsupported(ref message)
            if message.contains("required section type")),
        "{error:?}"
    );
}

#[test]
fn an_unknown_required_derived_section_is_rejected() {
    // Required-and-derived is contradictory on its own terms: a derived section
    // is matching-only, so a reader cannot be obliged to understand it.
    let error = expect_unknown_rejected(
        FLAG_REQUIRED | FLAG_DERIVED,
        "a required derived section must be refused",
    );
    assert!(matches!(error, AdyarError::Unsupported(_)), "{error:?}");
}

// --- What the artifact root does not cover ----------------------------------

#[test]
fn appending_unreferenced_trailing_bytes_does_not_change_the_artifact_root() {
    // The artifact root is BLAKE3 over the non-signature directory entries and,
    // through the per-section hashes those entries carry, the stored section
    // bytes they reference. It is deliberately not a whole-file hash, so bytes
    // that no directory entry references are outside its coverage. This test
    // exists so the limitation is explicit rather than incidental.
    let original = std::fs::read(golden_pack()).unwrap();
    let reader = PackReader::open(Arc::new(MemoryReader::new(original.clone()))).unwrap();
    let root = reader.root_hex();

    let mut appended = original;
    appended.extend_from_slice(b"unreferenced trailing bytes");
    let tampered = PackReader::open(Arc::new(MemoryReader::new(appended))).unwrap();

    assert_eq!(
        tampered.root_hex(),
        root,
        "trailing bytes are outside the artifact root's coverage"
    );
    // Everything the root *does* cover still verifies, which is the point: the
    // limitation is a scope boundary, not a broken check.
    tampered.verify_all().unwrap();
}

// --- What the signature authenticates ---------------------------------------

#[cfg(feature = "signing")]
#[test]
fn unauthenticated_signature_metadata_does_not_affect_verification() {
    // FORMAT-v3 §8.1: the signature covers the artifact root, and signature
    // sections are excluded from that root. The envelope's descriptive fields
    // are therefore bound by nothing. Rewriting them must leave both the root
    // and the signature verdict unchanged — which is precisely why no runtime
    // decision may read them.
    use adyar::model::SignatureEnvelope;
    use adyar::signing::{generate_keypair, sign_pack, verify_signatures};

    let temp = tempfile::TempDir::new().unwrap();
    let secret = temp.path().join("publisher.key");
    generate_keypair(&secret, None).unwrap();
    let signed = temp.path().join("signed.annpack");
    sign_pack(
        &golden_pack(),
        &signed,
        &secret,
        Some("vendor.example".into()),
        Some("2026-01-01".into()),
    )
    .unwrap();

    let reader = PackReader::open_path(&signed).unwrap();
    let original_root = reader.root_hex();
    assert_eq!(verify_signatures(&reader, None).unwrap().len(), 1);

    // Rewrite every unauthenticated field in the envelope.
    let signature_entry = reader
        .first_entry(SectionType::Signature)
        .expect("the signed fixture must carry a signature section");
    let signature_id = signature_entry.section_id;
    let mut envelope: SignatureEnvelope =
        serde_json::from_slice(&reader.read_section(signature_id).unwrap()).unwrap();
    envelope.identity = Some("attacker.example".into());
    envelope.expires_at = Some("1999-01-01".into());
    envelope.transparency_log_url = Some("https://attacker.example/log".into());
    envelope.revocation_url = Some("https://attacker.example/revoked".into());
    envelope.build_attestation = Some("forged".into());

    let mut writer = PackWriter::new().with_flags(reader.header.flags);
    for section in reader.all_section_data(true).unwrap() {
        if section.section_id == signature_id {
            writer
                .push(SectionData::optional(
                    signature_id,
                    SectionType::Signature,
                    1,
                    serde_json::to_vec(&envelope).unwrap(),
                ))
                .unwrap();
        } else {
            writer.push(section).unwrap();
        }
    }
    let rewritten =
        PackReader::open(Arc::new(MemoryReader::new(writer.build_bytes().unwrap()))).unwrap();

    assert_eq!(
        rewritten.root_hex(),
        original_root,
        "signature sections are excluded from the artifact root"
    );
    let reports = verify_signatures(&rewritten, None).unwrap();
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].cryptographically_valid,
        "the signature still verifies over the unchanged root"
    );
    assert!(
        !reports[0].identity_trusted,
        "a rewritten identity string must never confer trust"
    );
    assert_eq!(reports[0].identity.as_deref(), Some("attacker.example"));
}

// --- Manifest format 2 logical content root ---------------------------------

const MANIFEST_WITHOUT_ROOT: &str = r#"{"name":"container-rules","version":"1","description":null,"source_revision":null,"base_url":null,"created_at":null,"document_count":0,"passage_count":0,"capabilities":[],"embedding_profiles":[],"policy":{"license":null,"access":"public","redistributable":null,"expires_at":null,"policy_url":null},"dependencies":[]}"#;

fn manifest_only_pack(format_version: u16, manifest_json: &str) -> PackReader {
    let mut writer = PackWriter::new();
    writer
        .push(SectionData::required_versioned(
            1,
            SectionType::Manifest,
            1,
            format_version,
            manifest_json.as_bytes().to_vec(),
        ))
        .unwrap();
    PackReader::open(Arc::new(MemoryReader::new(writer.build_bytes().unwrap()))).unwrap()
}

fn manifest_with_root(root: &str) -> String {
    MANIFEST_WITHOUT_ROOT.replace(
        r#""dependencies":[]"#,
        &format!(r#""dependencies":[],"passage_merkle_root":"{root}""#),
    )
}

/// Version 2 is deliberate rather than `MANIFEST_FORMAT_VERSION`. These tests
/// cover the logical-root rule, which begins at format 2; pinning them to
/// whatever the current version happens to be coupled them to unrelated later
/// requirements, and format 4's source-descriptor rule started firing first and
/// masking the rule under test.
const LOGICAL_ROOT_FORMAT: u16 = 2;

#[test]
fn a_format_2_manifest_without_a_logical_root_is_rejected() {
    let reader = manifest_only_pack(LOGICAL_ROOT_FORMAT, MANIFEST_WITHOUT_ROOT);
    let error = reader
        .manifest()
        .expect_err("manifest format 2 requires passage_merkle_root");
    assert!(
        matches!(error, AdyarError::InvalidFormat(ref message)
            if message.contains("passage_merkle_root")),
        "{error:?}"
    );
}

#[test]
fn a_format_2_logical_root_must_be_64_lowercase_hex_characters() {
    for root in [
        "",
        "abc",
        // 64 characters, but uppercase.
        "AB12CD34EF56AB12CD34EF56AB12CD34EF56AB12CD34EF56AB12CD34EF56AB12",
        // 64 characters, but not hexadecimal.
        "zz12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12",
        // 63 characters.
        "ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab1",
        // 65 characters.
        "ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab123",
    ] {
        let reader = manifest_only_pack(LOGICAL_ROOT_FORMAT, &manifest_with_root(root));
        assert!(
            reader.manifest().is_err(),
            "passage_merkle_root {root:?} must be refused"
        );
    }

    let valid = "ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12";
    let reader = manifest_only_pack(LOGICAL_ROOT_FORMAT, &manifest_with_root(valid));
    assert_eq!(
        reader.manifest().unwrap().passage_merkle_root.as_deref(),
        Some(valid)
    );
}

#[test]
fn a_format_1_manifest_stays_readable_without_a_logical_root() {
    // The compatibility policy keeps v0.3.x artifacts readable. They simply
    // cannot issue standalone receipts, which tests/compatibility.rs pins.
    let reader = manifest_only_pack(1, MANIFEST_WITHOUT_ROOT);
    assert!(reader.manifest().unwrap().passage_merkle_root.is_none());
}

#[test]
fn a_format_2_pack_without_a_logical_root_is_not_core_conformant() {
    // Reported at the conformance boundary as well as refused at the container
    // boundary: a caller that assembles a Manifest by other means must still not
    // be told a rootless format-2 pack is Core-conformant.
    let reader = PackReader::open_path(golden_pack()).unwrap();
    let mut manifest = reader.manifest().unwrap();
    assert_eq!(
        reader
            .entry(reader.header.manifest_section_id)
            .unwrap()
            .format_version,
        MANIFEST_FORMAT_VERSION
    );

    assert!(inspect_conformance_with_manifest(&reader, &manifest).core_conformant);

    manifest.passage_merkle_root = None;
    let report = inspect_conformance_with_manifest(&reader, &manifest);
    assert!(!report.core_conformant);
    assert!(
        report
            .core_issues
            .iter()
            .any(|issue| issue.contains("passage_merkle_root")),
        "{:?}",
        report.core_issues
    );

    manifest.passage_merkle_root = Some("NOT-HEX".into());
    assert!(!inspect_conformance_with_manifest(&reader, &manifest).core_conformant);
}

#[test]
fn the_golden_artifact_still_opens_and_searches() {
    // Guards the two changes above against over-reach: the normal path must be
    // untouched.
    let engine = SearchEngine::open_path(golden_pack()).unwrap();
    assert!(engine.manifest().passage_merkle_root.is_some());
}
