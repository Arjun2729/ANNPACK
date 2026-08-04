//! Conformance tests for ANN-7 (expansion), ANN-8 (splade), and ANN-10 (fat
//! packs). These assert the five task invariants directly:
//! Core is unchanged, builds are deterministic, derived text is never citable,
//! degradation is graceful, and every new failure mode is an explicit error.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use annpack::build::{BuildOptions, build_pack_bytes};
use annpack::derive::{
    OverlaySidecar, RawCandidate, RawExpansion, RawExpansionPassage, RawSplade, RawSpladePassage,
    generate_expansion, generate_splade,
};
use annpack::format::{PackReader, SectionData, SectionType};
use annpack::model::{AccessClass, OverlayVocabulary, TermOverlaySection};
use annpack::reader::{MemoryReader, ReadAt};
use annpack::search::{ProfileRequest, ProfileSelection, SearchEngine, SearchMode, SearchOptions};
use annpack::{AnnpackError, Result};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/docs-v1")
}

fn base_options() -> BuildOptions {
    BuildOptions {
        input: fixtures(),
        output: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/unused-extensions.annpack"),
        name: "extensions-demo".into(),
        version: "1.0.0".into(),
        description: None,
        source_revision: Some("git:test".into()),
        base_url: Some("https://example.test".into()),
        created_at: None,
        license: None,
        access: AccessClass::Public,
        redistributable: None,
        policy_expires_at: None,
        policy_url: None,
        policy_override: None,
        vector_input: None,
        expansion_input: None,
        splade_input: None,
        target_chars: 1_200,
        max_chars: 2_400,
        input_format: annpack::ingest::InputFormat::Auto,
    }
}

fn passage_ids(pack_bytes: &[u8]) -> Vec<String> {
    let engine =
        SearchEngine::open_source(Arc::new(MemoryReader::new(pack_bytes.to_vec()))).unwrap();
    engine
        .passages()
        .unwrap()
        .into_iter()
        .map(|passage| passage.id)
        .collect()
}

fn write_sidecar(value: &impl serde::Serialize) -> tempfile::NamedTempFile {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), serde_json::to_vec_pretty(value).unwrap()).unwrap();
    file
}

/// Build a core pack, then an expansion sidecar that adds a distinctive term
/// ("chartreuse", which never appears in the corpus) to the first passage.
fn expansion_sidecar(ids: &[String]) -> OverlaySidecar {
    let raw = RawExpansion {
        generator: "docTTTTTquery-ref".into(),
        model: "t5-test".into(),
        revision: "rev1".into(),
        passages: vec![RawExpansionPassage {
            passage_id: ids[0].clone(),
            candidates: vec![
                RawCandidate {
                    text: "what is chartreuse".into(),
                    score: 0.9,
                },
                RawCandidate {
                    text: "irrelevant hallucination zzz".into(),
                    score: 0.05,
                },
            ],
        }],
    };
    generate_expansion(&raw, 0.5).unwrap()
}

// --------------------------------------------------------------------------
// Invariant 2: determinism is preserved.
// --------------------------------------------------------------------------

#[test]
fn expansion_build_is_byte_identical() {
    let core = build_pack_bytes(&base_options()).unwrap();
    let ids = passage_ids(&core);
    let sidecar = write_sidecar(&expansion_sidecar(&ids));
    let mut options = base_options();
    options.expansion_input = Some(sidecar.path().to_path_buf());
    let first = build_pack_bytes(&options).unwrap();
    let second = build_pack_bytes(&options).unwrap();
    assert_eq!(first, second, "second build must be byte-identical");
}

#[test]
fn build_records_sidecar_digest_in_provenance() {
    let core = build_pack_bytes(&base_options()).unwrap();
    let ids = passage_ids(&core);
    let sidecar_value = expansion_sidecar(&ids);
    let sidecar = write_sidecar(&sidecar_value);
    let expected = annpack::derive::sidecar_digest(&std::fs::read(sidecar.path()).unwrap());
    let mut options = base_options();
    options.expansion_input = Some(sidecar.path().to_path_buf());
    let bytes = build_pack_bytes(&options).unwrap();
    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes))).unwrap();
    let manifest = reader.manifest().unwrap();
    assert_eq!(manifest.derived_inputs.len(), 1);
    assert_eq!(manifest.derived_inputs[0].sidecar_digest, expected);
    assert_eq!(manifest.derived_inputs[0].kind, "expansion-v1");
}

// --------------------------------------------------------------------------
// Invariant 4: graceful degradation. Core reader ignores extensions; default
// weights reproduce Core results exactly.
// --------------------------------------------------------------------------

#[test]
fn core_verify_opens_extension_pack() {
    let core = build_pack_bytes(&base_options()).unwrap();
    let ids = passage_ids(&core);
    let sidecar = write_sidecar(&expansion_sidecar(&ids));
    let mut options = base_options();
    options.expansion_input = Some(sidecar.path().to_path_buf());
    let bytes = build_pack_bytes(&options).unwrap();
    // A Core reader that only verifies container integrity opens the pack and
    // ignores the unknown optional derived section.
    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes))).unwrap();
    reader.verify_all().unwrap();
}

#[test]
fn default_weight_reproduces_core_ranking() {
    let core_bytes = build_pack_bytes(&base_options()).unwrap();
    let ids = passage_ids(&core_bytes);
    let sidecar = write_sidecar(&expansion_sidecar(&ids));
    let mut options = base_options();
    options.expansion_input = Some(sidecar.path().to_path_buf());
    let exp_bytes = build_pack_bytes(&options).unwrap();

    let core = SearchEngine::open_source(Arc::new(MemoryReader::new(core_bytes))).unwrap();
    let exp = SearchEngine::open_source(Arc::new(MemoryReader::new(exp_bytes))).unwrap();
    let query = "cache";
    let core_hits: Vec<_> = core
        .search(
            query,
            &SearchOptions {
                mode: SearchMode::Lexical,
                ..Default::default()
            },
        )
        .unwrap()
        .results
        .into_iter()
        .map(|hit| (hit.passage_id, hit.score))
        .collect();
    let exp_hits: Vec<_> = exp
        .search(
            query,
            &SearchOptions {
                mode: SearchMode::Lexical,
                ..Default::default()
            },
        )
        .unwrap()
        .results
        .into_iter()
        .map(|hit| (hit.passage_id, hit.score))
        .collect();
    assert_eq!(core_hits, exp_hits, "default weight must reproduce Core");

    // A positive weight surfaces the passage via its generated term "chartreuse"
    // even though that word never appears in the passage text.
    let surfaced = exp
        .search(
            "chartreuse",
            &SearchOptions {
                mode: SearchMode::Lexical,
                expansion_weight: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        surfaced.results.first().map(|hit| hit.passage_id.clone()),
        Some(ids[0].clone())
    );
    // ...and Core lexical search for the same word finds nothing.
    assert!(
        exp.search(
            "chartreuse",
            &SearchOptions {
                mode: SearchMode::Lexical,
                ..Default::default()
            }
        )
        .unwrap()
        .results
        .is_empty()
    );
}

// --------------------------------------------------------------------------
// Invariant 3: evidence integrity. No evidence envelope may reference
// expansion-derived text.
// --------------------------------------------------------------------------

#[test]
fn evidence_never_references_derived_text() {
    let core_bytes = build_pack_bytes(&base_options()).unwrap();
    let ids = passage_ids(&core_bytes);
    let sidecar = write_sidecar(&expansion_sidecar(&ids));
    let mut options = base_options();
    options.expansion_input = Some(sidecar.path().to_path_buf());
    let exp_bytes = build_pack_bytes(&options).unwrap();

    let core = SearchEngine::open_source(Arc::new(MemoryReader::new(core_bytes))).unwrap();
    let exp = SearchEngine::open_source(Arc::new(MemoryReader::new(exp_bytes))).unwrap();

    // The passage surfaced only by the generated term "chartreuse".
    let hit = exp
        .search(
            "chartreuse",
            &SearchOptions {
                mode: SearchMode::Lexical,
                expansion_weight: 5.0,
                ..Default::default()
            },
        )
        .unwrap()
        .results
        .into_iter()
        .next()
        .unwrap();

    // The generated term must never appear in the returned passage text or in
    // any evidence/citation field.
    assert!(!hit.text.to_lowercase().contains("chartreuse"));
    assert!(
        !serde_json::to_string(&hit.evidence)
            .unwrap()
            .contains("chartreuse")
    );
    assert!(
        !serde_json::to_string(&hit.citation)
            .unwrap()
            .contains("chartreuse")
    );

    // The evidence passage_hash is identical to the Core pack's hash for the
    // same passage: expansion changed ranking, not the cited record.
    let core_passage = core.get_passage(&hit.passage_id).unwrap();
    let core_evidence = core.evidence_for_passage(&core_passage).unwrap();
    assert_eq!(core_evidence.passage_hash, hit.evidence.passage_hash);
}

// --------------------------------------------------------------------------
// Invariant 5: strict rejection. Every new failure mode is an explicit error.
// --------------------------------------------------------------------------

/// Rebuild a pack, replacing its term-overlay section bytes with a hand-crafted
/// malicious overlay. The container hashes/root are recomputed, so integrity
/// holds but the overlay semantics are attacker-controlled.
fn pack_with_tampered_overlay(overlay: &TermOverlaySection) -> Vec<u8> {
    let core = build_pack_bytes(&base_options()).unwrap();
    let ids = passage_ids(&core);
    let sidecar = write_sidecar(&expansion_sidecar(&ids));
    let mut options = base_options();
    options.expansion_input = Some(sidecar.path().to_path_buf());
    let bytes = build_pack_bytes(&options).unwrap();

    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes))).unwrap();
    let mut writer = annpack::format::PackWriter::new();
    for section in reader.all_section_data(true).unwrap() {
        if section.section_type == SectionType::TermOverlay {
            writer
                .push(SectionData::derived_deflate(
                    section.section_id,
                    SectionType::TermOverlay,
                    overlay.terms.len() as u64,
                    serde_json::to_vec(overlay).unwrap(),
                ))
                .unwrap();
        } else {
            writer.push(section).unwrap();
        }
    }
    writer.build_bytes().unwrap()
}

/// Rebuild a pack with a mutated manifest, preserving every other section and
/// the manifest's section-format version, so only the descriptor changes.
fn rewrite_manifest(bytes: &[u8], mutate: impl FnOnce(&mut annpack::model::Manifest)) -> Vec<u8> {
    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes.to_vec()))).unwrap();
    let mut manifest = reader.manifest().unwrap();
    mutate(&mut manifest);
    let manifest_id = reader.header.manifest_section_id;
    let manifest_version = reader.entry(manifest_id).unwrap().format_version;

    let mut writer = annpack::format::PackWriter::new();
    for section in reader.all_section_data(true).unwrap() {
        if section.section_id == manifest_id {
            writer
                .push(SectionData::required_versioned(
                    manifest_id,
                    SectionType::Manifest,
                    1,
                    manifest_version,
                    serde_json::to_vec(&manifest).unwrap(),
                ))
                .unwrap();
        } else {
            writer.push(section).unwrap();
        }
    }
    writer.build_bytes().unwrap()
}

fn search_with_expansion(bytes: Vec<u8>) -> Result<()> {
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes)))?;
    engine.search(
        "cache",
        &SearchOptions {
            mode: SearchMode::Lexical,
            expansion_weight: 1.0,
            ..Default::default()
        },
    )?;
    Ok(())
}

#[test]
fn rejects_overlay_ordinal_out_of_range() {
    let mut terms = std::collections::BTreeMap::new();
    terms.insert("cache".to_string(), vec![(9_999_999_u32, 1_u32)]);
    let overlay = TermOverlaySection {
        kind: "expansion-v1".into(),
        generator: "x".into(),
        model: "x".into(),
        revision: "x".into(),
        threshold: Some(0.5),
        vocabulary: None,
        terms,
    };
    let error = search_with_expansion(pack_with_tampered_overlay(&overlay)).unwrap_err();
    assert!(matches!(error, AnnpackError::InvalidFormat(_)));
}

#[test]
fn rejects_overlay_non_increasing_ordinals() {
    let mut terms = std::collections::BTreeMap::new();
    terms.insert("cache".to_string(), vec![(2_u32, 1_u32), (2_u32, 1_u32)]);
    let overlay = TermOverlaySection {
        kind: "expansion-v1".into(),
        generator: "x".into(),
        model: "x".into(),
        revision: "x".into(),
        threshold: Some(0.5),
        vocabulary: None,
        terms,
    };
    let error = search_with_expansion(pack_with_tampered_overlay(&overlay)).unwrap_err();
    assert!(matches!(error, AnnpackError::InvalidFormat(_)));
}

#[test]
fn rejects_overlay_zero_weight() {
    let mut terms = std::collections::BTreeMap::new();
    terms.insert("cache".to_string(), vec![(0_u32, 0_u32)]);
    let overlay = TermOverlaySection {
        kind: "expansion-v1".into(),
        generator: "x".into(),
        model: "x".into(),
        revision: "x".into(),
        threshold: Some(0.5),
        vocabulary: None,
        terms,
    };
    let error = search_with_expansion(pack_with_tampered_overlay(&overlay)).unwrap_err();
    assert!(matches!(error, AnnpackError::InvalidFormat(_)));
}

#[test]
fn rejects_unknown_overlay_kind() {
    let mut terms = std::collections::BTreeMap::new();
    terms.insert("cache".to_string(), vec![(0_u32, 1_u32)]);
    let overlay = TermOverlaySection {
        kind: "mystery-v1".into(),
        generator: "x".into(),
        model: "x".into(),
        revision: "x".into(),
        threshold: None,
        vocabulary: None,
        terms,
    };
    let error = search_with_expansion(pack_with_tampered_overlay(&overlay)).unwrap_err();
    assert!(matches!(error, AnnpackError::InvalidFormat(_)));
}

#[test]
fn rejects_splade_without_vocabulary() {
    let mut terms = std::collections::BTreeMap::new();
    terms.insert("cache".to_string(), vec![(0_u32, 1_u32)]);
    let overlay = TermOverlaySection {
        kind: "splade-v1".into(),
        generator: "x".into(),
        model: "x".into(),
        revision: "x".into(),
        threshold: None,
        vocabulary: None,
        terms,
    };
    let error = search_with_expansion(pack_with_tampered_overlay(&overlay)).unwrap_err();
    assert!(matches!(error, AnnpackError::InvalidFormat(_)));
}

/// A derived section marked required must be rejected at the container level.
#[test]
fn rejects_required_derived_section() {
    use annpack::format::{FLAG_DERIVED, FLAG_REQUIRED};
    let core = build_pack_bytes(&base_options()).unwrap();
    let reader = PackReader::open(Arc::new(MemoryReader::new(core))).unwrap();
    let mut writer = annpack::format::PackWriter::new();
    for section in reader.all_section_data(true).unwrap() {
        writer.push(section).unwrap();
    }
    // A hand-built section that is both derived and required.
    let mut bad = SectionData::derived_deflate(99, SectionType::TermOverlay, 0, b"{}".to_vec());
    bad.flags = FLAG_DERIVED | FLAG_REQUIRED;
    writer.push(bad).unwrap();
    let bytes = writer.build_bytes().unwrap();
    match PackReader::open(Arc::new(MemoryReader::new(bytes))) {
        Err(AnnpackError::InvalidFormat(_)) => {}
        other => panic!("expected InvalidFormat, got {:?}", other.map(|_| "ok")),
    }
}

// --------------------------------------------------------------------------
// Invariant 1 / ANN-10: unused profiles are never fetched under range serving.
// --------------------------------------------------------------------------

/// A ReadAt that records every byte range it is asked to read.
struct RecordingReader {
    inner: MemoryReader,
    reads: Mutex<Vec<(u64, u64)>>,
}

impl ReadAt for RecordingReader {
    fn len(&self) -> Result<u64> {
        self.inner.len()
    }
    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        self.reads
            .lock()
            .unwrap()
            .push((offset, buffer.len() as u64));
        self.inner.read_exact_at(offset, buffer)
    }
}

fn build_fat_pack() -> Vec<u8> {
    let core = build_pack_bytes(&base_options()).unwrap();
    let ids = passage_ids(&core);

    let expansion = write_sidecar(&expansion_sidecar(&ids));

    let splade_raw = RawSplade {
        generator: "splade-ref".into(),
        model: "splade-test".into(),
        revision: "rev1".into(),
        vocabulary: OverlayVocabulary {
            id: "bert-base-uncased-wordpiece".into(),
            size: 30522,
            quantization: "linear-u16".into(),
            scale: 0.001,
        },
        passages: ids
            .iter()
            .map(|id| RawSpladePassage {
                passage_id: id.clone(),
                weights: [("cache".to_string(), 0.8_f64)].into_iter().collect(),
            })
            .collect(),
    };
    let splade = write_sidecar(&generate_splade(&splade_raw).unwrap());

    let mut options = base_options();
    options.expansion_input = Some(expansion.path().to_path_buf());
    options.splade_input = Some(splade.path().to_path_buf());
    build_pack_bytes(&options).unwrap()
}

#[test]
fn lexical_search_never_fetches_unused_profiles() {
    let bytes = build_fat_pack();

    // Byte ranges of every optional-profile section.
    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes.clone()))).unwrap();
    let profile_ranges: Vec<(u64, u64)> = reader
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.section_type,
                SectionType::TermOverlay
                    | SectionType::VectorProfile
                    | SectionType::VectorData
                    | SectionType::VectorIndex
            )
        })
        .map(|entry| (entry.offset, entry.offset + entry.stored_length))
        .collect();
    assert!(
        !profile_ranges.is_empty(),
        "fat pack must carry profile sections"
    );

    let recorder = Arc::new(RecordingReader {
        inner: MemoryReader::new(bytes),
        reads: Mutex::new(Vec::new()),
    });
    let engine = SearchEngine::open_source(recorder.clone()).unwrap();
    // A default (lexical, zero-weight) search.
    engine
        .search(
            "cache",
            &SearchOptions {
                mode: SearchMode::Lexical,
                ..Default::default()
            },
        )
        .unwrap();

    let reads = recorder.reads.lock().unwrap();
    for (read_start, read_len) in reads.iter() {
        let read_end = read_start + read_len;
        for (section_start, section_end) in &profile_ranges {
            let overlaps = read_start < section_end && read_end > *section_start;
            assert!(
                !overlaps,
                "lexical search fetched bytes {read_start}..{read_end} inside unused profile \
                 range {section_start}..{section_end}",
            );
        }
    }
}

fn fat_pack_selection(profile: ProfileRequest) -> ProfileSelection {
    let bytes = build_fat_pack();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    engine
        .search(
            "cache",
            &SearchOptions {
                profile,
                ..Default::default()
            },
        )
        .unwrap()
        .profile_selection
}

#[test]
fn default_profile_selects_core_lexical() {
    let sel = fat_pack_selection(ProfileRequest::Lexical);
    assert_eq!(sel.selected.as_deref(), Some("lexical"));
    assert_eq!(sel.selected_kind.as_deref(), Some("lexical"));
    assert_eq!(sel.effective_mode, SearchMode::Lexical);
    assert_eq!(sel.effective_expansion_weight, 0.0);
    assert_eq!(sel.effective_splade_weight, 0.0);
}

#[test]
fn auto_selects_first_supported_profile() {
    // build_fat_pack advertises [splade, expansion, anchors, lexical]; splade is
    // the first the runtime can execute.
    let sel = fat_pack_selection(ProfileRequest::Auto);
    assert_eq!(sel.selected.as_deref(), Some("splade"));
    assert_eq!(sel.effective_splade_weight, 1.0);
    assert_eq!(sel.effective_expansion_weight, 0.0);
    assert_eq!(sel.effective_mode, SearchMode::Lexical);
}

#[test]
fn named_supported_profile_is_activated() {
    let sel = fat_pack_selection(ProfileRequest::Named("expansion".into()));
    assert_eq!(sel.selected.as_deref(), Some("expansion"));
    assert_eq!(sel.effective_expansion_weight, 1.0);
    assert_eq!(sel.effective_splade_weight, 0.0);
}

#[test]
fn named_absent_profile_falls_back_to_lexical() {
    let sel = fat_pack_selection(ProfileRequest::Named("does-not-exist".into()));
    assert_eq!(sel.selected.as_deref(), Some("lexical"));
    assert!(sel.reason.contains("absent"), "reason: {}", sel.reason);
}

#[test]
fn non_fat_pack_reports_no_profile() {
    // A pack with no retrieval_profiles: selection is a no-op, raw options apply.
    let bytes = build_pack_bytes(&base_options()).unwrap();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    let resp = engine
        .search(
            "cache",
            &SearchOptions {
                mode: SearchMode::Lexical,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(resp.profile_selection.selected, None);
    assert_eq!(resp.profile_selection.effective_mode, SearchMode::Lexical);
}

#[test]
fn default_fat_pack_search_matches_core_lexical_ranking() {
    // Selecting lexical (the default) on a fat pack must reproduce Core exactly —
    // no derived profile silently activates.
    let core = build_pack_bytes(&base_options()).unwrap();
    let core_engine = SearchEngine::open_source(Arc::new(MemoryReader::new(core))).unwrap();
    let core_hits: Vec<String> = core_engine
        .search(
            "cache",
            &SearchOptions {
                mode: SearchMode::Lexical,
                ..Default::default()
            },
        )
        .unwrap()
        .results
        .into_iter()
        .map(|h| h.passage_id)
        .collect();

    let fat_engine =
        SearchEngine::open_source(Arc::new(MemoryReader::new(build_fat_pack()))).unwrap();
    let fat_hits: Vec<String> = fat_engine
        .search("cache", &SearchOptions::default()) // default profile == lexical
        .unwrap()
        .results
        .into_iter()
        .map(|h| h.passage_id)
        .collect();

    assert_eq!(
        core_hits, fat_hits,
        "default fat-pack search must equal the Core lexical ranking"
    );
}

#[test]
fn profile_referencing_wrong_section_type_is_flagged() {
    use annpack::conformance::inspect_conformance_with_manifest;
    // A fat pack whose splade profile is tampered to reference a lexical section
    // must be flagged: kind/section-type mismatch, not just section existence.
    let reader = PackReader::open(Arc::new(MemoryReader::new(build_fat_pack()))).unwrap();
    let mut manifest = reader.manifest().unwrap();
    let lexical_section = reader
        .entries
        .iter()
        .find(|e| e.section_type == SectionType::PassageIndex)
        .map(|e| e.section_id)
        .expect("fat pack has a passage index section");
    let splade = manifest
        .retrieval_profiles
        .iter_mut()
        .find(|p| p.kind == "splade")
        .expect("fat pack advertises a splade profile");
    splade.section_ids = vec![lexical_section];

    let report = inspect_conformance_with_manifest(&reader, &manifest);
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.contains("incompatible type")),
        "expected a kind/section-type mismatch issue, got {:?}",
        report.issues
    );
}

#[test]
fn rejects_non_finite_or_negative_weights() {
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(
        build_pack_bytes(&base_options()).unwrap(),
    )))
    .unwrap();
    for opts in [
        SearchOptions {
            expansion_weight: f64::INFINITY,
            ..Default::default()
        },
        SearchOptions {
            splade_weight: f64::NAN,
            ..Default::default()
        },
        SearchOptions {
            expansion_weight: -1.0,
            ..Default::default()
        },
        SearchOptions {
            lexical_weight: f64::NEG_INFINITY,
            ..Default::default()
        },
    ] {
        assert!(
            matches!(
                engine.search("cache", &opts),
                Err(AnnpackError::InvalidInput(_))
            ),
            "expected InvalidInput for weights {:?}/{:?}/{:?}/{:?}",
            opts.lexical_weight,
            opts.vector_weight,
            opts.expansion_weight,
            opts.splade_weight
        );
    }
}

#[test]
fn conformance_flags_profiles_not_ending_at_lexical() {
    use annpack::conformance::inspect_conformance_with_manifest;
    let reader = PackReader::open(Arc::new(MemoryReader::new(build_fat_pack()))).unwrap();
    let mut manifest = reader.manifest().unwrap();
    // Drop the terminal lexical profile so the fallback order no longer ends at it.
    manifest.retrieval_profiles.retain(|p| p.kind != "lexical");
    let report = inspect_conformance_with_manifest(&reader, &manifest);
    assert!(
        report
            .issues
            .iter()
            .any(|i| i.contains("must end at the Core lexical profile")),
        "issues: {:?}",
        report.issues
    );
}

#[test]
fn conformance_flags_duplicate_profile_ids() {
    use annpack::conformance::inspect_conformance_with_manifest;
    let reader = PackReader::open(Arc::new(MemoryReader::new(build_fat_pack()))).unwrap();
    let mut manifest = reader.manifest().unwrap();
    // Duplicate the first profile's id onto a second entry.
    let dup = manifest.retrieval_profiles[0].clone();
    manifest.retrieval_profiles.insert(1, dup);
    let report = inspect_conformance_with_manifest(&reader, &manifest);
    assert!(
        report
            .issues
            .iter()
            .any(|i| i.contains("duplicate retrieval profile id")),
        "issues: {:?}",
        report.issues
    );
}

/// The ANN-10 spec claims a client that selects one profile fetches only that
/// profile's ranges. Before v0.4.0 the loader read every term overlay, so
/// selecting `expansion` also pulled the SPLADE section: the claim only held for
/// Core lexical. These tests pin the real property, profile to profile.
/// `(reads overlapping the expansion overlay, reads overlapping the splade overlay)`.
type OverlayReads = (Vec<(u64, u64)>, Vec<(u64, u64)>);

fn ranges_touched_by_named_profile(profile: &str) -> OverlayReads {
    let bytes = build_fat_pack();
    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes.clone()))).unwrap();

    // Map each overlay section to its byte range by kind.
    let mut expansion_range = None;
    let mut splade_range = None;
    for entry in reader.entries_of_type(SectionType::TermOverlay) {
        let section: annpack::model::TermOverlaySection =
            serde_json::from_slice(&reader.read_section(entry.section_id).unwrap()).unwrap();
        let range = (entry.offset, entry.offset + entry.stored_length);
        match section.kind.as_str() {
            "expansion-v1" => expansion_range = Some(range),
            "splade-v1" => splade_range = Some(range),
            other => panic!("unexpected overlay kind {other}"),
        }
    }
    let expansion_range = expansion_range.expect("fat pack must carry an expansion overlay");
    let splade_range = splade_range.expect("fat pack must carry a splade overlay");

    let recorder = Arc::new(RecordingReader {
        inner: MemoryReader::new(bytes),
        reads: Mutex::new(Vec::new()),
    });
    let engine = SearchEngine::open_source(recorder.clone()).unwrap();
    engine
        .search(
            "cache",
            &SearchOptions {
                profile: ProfileRequest::Named(profile.into()),
                ..Default::default()
            },
        )
        .unwrap();
    let reads = recorder.reads.lock().unwrap().clone();

    let touched = |range: (u64, u64)| -> Vec<(u64, u64)> {
        reads
            .iter()
            .copied()
            .filter(|(start, length)| *start < range.1 && start + length > range.0)
            .collect()
    };
    (touched(expansion_range), touched(splade_range))
}

#[test]
fn selecting_expansion_never_fetches_the_splade_ranges() {
    let (expansion_reads, splade_reads) = ranges_touched_by_named_profile("expansion");
    assert!(
        !expansion_reads.is_empty(),
        "the selected expansion overlay must actually be read"
    );
    assert!(
        splade_reads.is_empty(),
        "selecting expansion fetched unused splade bytes: {splade_reads:?}"
    );
}

#[test]
fn selecting_splade_never_fetches_the_expansion_ranges() {
    let (expansion_reads, splade_reads) = ranges_touched_by_named_profile("splade");
    assert!(
        !splade_reads.is_empty(),
        "the selected splade overlay must actually be read"
    );
    assert!(
        expansion_reads.is_empty(),
        "selecting splade fetched unused expansion bytes: {expansion_reads:?}"
    );
}

#[test]
fn a_malformed_profile_descriptor_cannot_reach_the_default_lexical_path() {
    // Strip the terminal Core lexical profile so the ANN-10 descriptor is
    // invalid. Default lexical retrieval must still work and must be byte-for-
    // byte Core, while any profile-enabled request is refused outright.
    let bytes = build_fat_pack();
    let core = build_pack_bytes(&base_options()).unwrap();
    let broken = rewrite_manifest(&bytes, |manifest| {
        manifest.retrieval_profiles.retain(|p| p.kind != "lexical");
    });

    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(broken))).unwrap();
    assert!(
        !engine.conformance().extensions_conformant,
        "a descriptor without its terminal lexical profile must fail extension conformance"
    );
    assert!(
        engine.conformance().core_conformant,
        "an extension defect must never invalidate Core"
    );

    // Default search still serves Core lexical.
    let response = engine
        .search("cache", &SearchOptions::default())
        .expect("default lexical must remain available");
    assert!(!response.results.is_empty());
    assert_eq!(response.profile_selection.effective_expansion_weight, 0.0);
    assert_eq!(response.profile_selection.effective_splade_weight, 0.0);

    let core_engine = SearchEngine::open_source(Arc::new(MemoryReader::new(core))).unwrap();
    let core_response = core_engine
        .search("cache", &SearchOptions::default())
        .unwrap();
    let ranked: Vec<&str> = response
        .results
        .iter()
        .map(|hit| hit.passage_id.as_str())
        .collect();
    let core_ranked: Vec<&str> = core_response
        .results
        .iter()
        .map(|hit| hit.passage_id.as_str())
        .collect();
    assert_eq!(
        ranked, core_ranked,
        "a malformed descriptor must not perturb Core ranking"
    );

    // And a profile request is refused rather than silently downgraded.
    for request in [
        ProfileRequest::Auto,
        ProfileRequest::Named("expansion".into()),
    ] {
        let error = engine
            .search(
                "cache",
                &SearchOptions {
                    profile: request.clone(),
                    ..Default::default()
                },
            )
            .expect_err("profile-enabled search must be refused on an invalid descriptor");
        assert!(
            error.to_string().contains("extension metadata is invalid"),
            "unexpected error for {request:?}: {error}"
        );
    }
}

/// A pack carrying ANN-1 vectors and an ANN-8 overlay, so the fat-pack
/// descriptor advertises two optional profiles plus the terminal Core lexical
/// one. Both non-Core retrieval paths are genuinely present in the artifact.
fn build_vector_and_overlay_pack() -> Vec<u8> {
    let core = build_pack_bytes(&base_options()).unwrap();
    let ids = passage_ids(&core);
    let splade_raw = RawSplade {
        generator: "splade-ref".into(),
        model: "splade-test".into(),
        revision: "rev1".into(),
        vocabulary: OverlayVocabulary {
            id: "bert-base-uncased-wordpiece".into(),
            size: 30522,
            quantization: "linear-u16".into(),
            scale: 0.001,
        },
        passages: ids
            .iter()
            .map(|id| RawSpladePassage {
                passage_id: id.clone(),
                weights: [("cache".to_string(), 0.8_f64)].into_iter().collect(),
            })
            .collect(),
    };
    let splade = write_sidecar(&generate_splade(&splade_raw).unwrap());

    let mut options = base_options();
    options.vector_input =
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/vectors-v1.json"));
    options.splade_input = Some(splade.path().to_path_buf());
    build_pack_bytes(&options).unwrap()
}

#[test]
fn invalid_extension_metadata_cannot_activate_any_non_core_retrieval() {
    // The adversarial case the profile-request guard alone did not cover: keep
    // the default (lexical) profile, but reach for optional retrieval through
    // the search mode or an overlay weight instead. Core must stay Core.
    let broken = rewrite_manifest(&build_vector_and_overlay_pack(), |manifest| {
        // An unrecognized required capability makes the ANN-10 descriptor
        // invalid without touching any Core section.
        manifest.retrieval_profiles[0]
            .requires
            .push("made-up-capability".into());
    });
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(broken))).unwrap();
    assert!(engine.conformance().core_conformant);
    assert!(
        !engine.conformance().extensions_conformant,
        "the doctored descriptor must fail extension conformance"
    );

    let query_vector = Some(vec![0.0_f32, 0.0, 1.0]);
    for (label, options) in [
        (
            "vector mode",
            SearchOptions {
                mode: SearchMode::Vector,
                query_vector: query_vector.clone(),
                ..Default::default()
            },
        ),
        (
            "hybrid mode",
            SearchOptions {
                mode: SearchMode::Hybrid,
                query_vector: query_vector.clone(),
                ..Default::default()
            },
        ),
        (
            "expansion overlay weight",
            SearchOptions {
                expansion_weight: 1.0,
                ..Default::default()
            },
        ),
        (
            "splade overlay weight",
            SearchOptions {
                splade_weight: 1.0,
                ..Default::default()
            },
        ),
    ] {
        // Every one of these selects the default lexical profile.
        assert_eq!(options.profile, ProfileRequest::Lexical);
        let Err(error) = engine.search("cache", &options) else {
            panic!("{label} must be refused on an invalid extension descriptor");
        };
        assert!(
            error.to_string().contains("extension metadata is invalid"),
            "unexpected error for {label}: {error}"
        );
    }

    // Core lexical itself is untouched and identical to the Core-only pack.
    let response = engine
        .search("cache", &SearchOptions::default())
        .expect("Core lexical must remain available");
    let core_engine = SearchEngine::open_source(Arc::new(MemoryReader::new(
        build_pack_bytes(&base_options()).unwrap(),
    )))
    .unwrap();
    let core_response = core_engine
        .search("cache", &SearchOptions::default())
        .unwrap();
    assert_eq!(
        response
            .results
            .iter()
            .map(|hit| hit.passage_id.as_str())
            .collect::<Vec<_>>(),
        core_response
            .results
            .iter()
            .map(|hit| hit.passage_id.as_str())
            .collect::<Vec<_>>(),
        "invalid extension metadata must not perturb Core lexical ranking"
    );
    assert_eq!(response.effective_mode, SearchMode::Lexical);
    assert_eq!(response.profile_selection.effective_expansion_weight, 0.0);
    assert_eq!(response.profile_selection.effective_splade_weight, 0.0);

    // Hybrid without a query vector reaches no optional section, so it is Core
    // lexical and must stay available rather than being refused as collateral.
    engine
        .search(
            "cache",
            &SearchOptions {
                mode: SearchMode::Hybrid,
                query_vector: None,
                ..Default::default()
            },
        )
        .expect("hybrid without a query vector is Core lexical");
}

#[test]
fn a_valid_extension_descriptor_still_permits_non_core_retrieval() {
    // The guard above must not become a blanket refusal: with a conformant
    // descriptor, vector mode and overlay weights stay available.
    let engine =
        SearchEngine::open_source(Arc::new(MemoryReader::new(build_vector_and_overlay_pack())))
            .unwrap();
    assert!(engine.conformance().extensions_conformant);
    engine
        .search(
            "cache",
            &SearchOptions {
                mode: SearchMode::Vector,
                query_vector: Some(vec![0.0_f32, 0.0, 1.0]),
                ..Default::default()
            },
        )
        .expect("vector mode must work on a conformant pack");
    engine
        .search(
            "cache",
            &SearchOptions {
                splade_weight: 1.0,
                ..Default::default()
            },
        )
        .expect("overlay weights must work on a conformant pack");
}
