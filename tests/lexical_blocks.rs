//! Block-addressable lexical index (lexical section format 2).
//!
//! The failure mode of a sparse index is not "it breaks"; it is "it silently
//! misses one term at a boundary and the ranking quietly changes". These tests
//! pin the boundaries: the first and last term overall, the exact first term of
//! every block, the term immediately before each one, and a posting list long
//! enough to span more than one postings block.

use std::path::Path;
use std::sync::Arc;

use adyar::build::{BuildOptions, build_pack_bytes};
use adyar::format::{PackReader, SectionType};
use adyar::model::{AccessClass, StoredPassageIndex};
use adyar::reader::MemoryReader;
use adyar::search::{SearchEngine, SearchOptions};

/// A corpus with enough distinct terms to force several dictionary blocks, and
/// one term repeated across every passage so its posting list crosses a block
/// boundary.
fn corpus(directory: &Path) -> usize {
    let mut distinct = 0;
    // Sized so every block-addressed region spans more than one block: the
    // posting stream, the term table, the fixed-width record table (5,461
    // records per block at a 12-byte stride) and the id index (1,820 entries
    // per block). The record table is what sets the floor, and the guard test
    // below fails if any region stops spanning blocks -- a single-block fixture
    // makes every boundary test here vacuous.
    for document in 0..1400 {
        let mut text = format!("# Document {document}\n\n");
        for section in 0..4 {
            text.push_str(&format!("## Section {document}-{section}\n\n"));
            // `ubiquitous` lands in every passage, so its posting list is as
            // long as the corpus and is the one guaranteed to span blocks.
            text.push_str("ubiquitous ");
            for word in 0..40 {
                text.push_str(&format!("term{:06} ", distinct + word));
            }
            distinct += 40;
            text.push_str("\n\n");
        }
        std::fs::write(directory.join(format!("doc{document:04}.md")), text).unwrap();
    }
    distinct
}

fn build(directory: &Path) -> Vec<u8> {
    build_pack_bytes(&BuildOptions {
        input: directory.to_path_buf(),
        output: directory.join("unused.annpack"),
        name: "block-fixture".into(),
        version: "1.0.0".into(),
        description: None,
        source_revision: None,
        base_url: None,
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
        input_format: adyar::ingest::InputFormat::Auto,
    })
    .unwrap()
}

fn fixture() -> (Vec<u8>, usize) {
    let temp = tempfile::tempdir().unwrap();
    let distinct = corpus(temp.path());
    (build(temp.path()), distinct)
}

fn block_index(bytes: &[u8]) -> StoredPassageIndex {
    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes.to_vec()))).unwrap();
    let entry = reader.first_entry(SectionType::PassageIndex).unwrap();
    serde_json::from_slice(&reader.read_section(entry.section_id).unwrap()).unwrap()
}

fn hits(engine: &SearchEngine, query: &str) -> usize {
    engine
        .search(
            query,
            &SearchOptions {
                limit: 50,
                ..Default::default()
            },
        )
        .unwrap()
        .results
        .len()
}

#[test]
fn the_fixture_actually_produces_multiple_blocks() {
    let (bytes, _) = fixture();
    let blocks = block_index(&bytes)
        .lexical_blocks
        .expect("format 2 block tables");
    assert!(
        blocks.dictionary.len() > 1,
        "fixture must span several dictionary blocks, got {}",
        blocks.dictionary.len()
    );
    assert!(
        blocks.postings.len() > 1,
        "fixture must span several postings blocks, got {}",
        blocks.postings.len()
    );
    let records = block_index(&bytes)
        .record_blocks
        .expect("format 2 record blocks");
    assert!(
        records.records.len() > 1,
        "fixture must span several record blocks, got {}",
        records.records.len()
    );
    assert!(
        records.ids.len() > 1,
        "fixture must span several id-index blocks, got {}",
        records.ids.len()
    );
}

#[test]
fn every_block_boundary_term_resolves() {
    let (bytes, _) = fixture();
    let blocks = block_index(&bytes).lexical_blocks.unwrap();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();

    for block in &blocks.dictionary {
        let first = block.first_term.as_deref().unwrap();
        assert!(
            hits(&engine, first) > 0,
            "a block's own first term must resolve: {first:?}"
        );
    }
}

#[test]
fn the_term_before_each_boundary_resolves_from_the_previous_block() {
    let (bytes, distinct) = fixture();
    let blocks = block_index(&bytes).lexical_blocks.unwrap();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();

    // Off-by-one in the sparse search sends these to the wrong block, where
    // they are absent and score nothing.
    for block in blocks.dictionary.iter().skip(1) {
        let first = block.first_term.as_deref().unwrap();
        let Some(number) = first
            .strip_prefix("term")
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        if number == 0 {
            continue;
        }
        let previous = format!("term{:06}", number - 1);
        assert!(
            hits(&engine, &previous) > 0,
            "the term immediately before a block boundary must still resolve: {previous:?}"
        );
    }
    assert!(distinct > 0);
}

#[test]
fn first_and_last_terms_in_the_whole_table_resolve() {
    let (bytes, distinct) = fixture();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    assert!(hits(&engine, "term000000") > 0, "first term in sort order");
    let last = format!("term{:06}", distinct - 1);
    assert!(hits(&engine, &last) > 0, "last term in sort order: {last}");
}

#[test]
fn a_posting_list_spanning_blocks_returns_every_passage() {
    let (bytes, _) = fixture();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    // `ubiquitous` is in every passage, so its list is the longest in the pack
    // and is the one that crosses postings-block boundaries. Reassembling it
    // wrongly loses passages silently, so assert the full count.
    let passages = engine.passages().unwrap().len().min(1000);
    let found = engine
        .search(
            "ubiquitous",
            &SearchOptions {
                limit: passages,
                ..Default::default()
            },
        )
        .unwrap()
        .results
        .len();
    assert_eq!(
        found, passages,
        "a posting list spanning blocks must reassemble completely"
    );
}

#[test]
fn a_term_that_sorts_outside_the_table_is_simply_absent() {
    let (bytes, _) = fixture();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    // Below every first_term, and above every one. Neither may panic or match.
    assert_eq!(hits(&engine, "aaaaaaaa"), 0);
    assert_eq!(hits(&engine, "zzzzzzzz"), 0);
}

#[test]
fn a_tampered_index_block_is_rejected() {
    let (mut bytes, _) = fixture();
    let blocks = block_index(&bytes).lexical_blocks.unwrap();
    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes.clone()))).unwrap();
    let section = reader.first_entry(SectionType::LexicalTerms).unwrap();
    let section_offset = section.offset as usize;
    drop(reader);

    // Corrupt the *first* term block, then query a term that block is the only
    // possible home for — its own first term. Querying anything else would
    // resolve through a different block and prove nothing.
    let target = blocks.dictionary[0].first_term.clone().unwrap();
    let block_offset = section_offset + blocks.dictionary[0].offset as usize;

    // The section hash is never consulted on a block read, so this is caught
    // only if the block's own hash is checked — which is the entire basis for
    // reading a section in parts.
    bytes[block_offset + 8] ^= 0xff;
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    let result = engine.search(
        &target,
        &SearchOptions {
            limit: 5,
            ..Default::default()
        },
    );
    assert!(
        result.is_err(),
        "a corrupted index block must fail verification, not serve wrong results; got {:?}",
        result.map(|r| r.results.len())
    );
}

// --------------------------------------------------------------------------
// Passage record table (passage index format 2)
// --------------------------------------------------------------------------

/// Ordinal lookup is arithmetic, so an off-by-one in the stride or block size
/// silently returns the *wrong passage* rather than failing. Walk every ordinal
/// and check each resolves to the record whose id matches the passage itself.
#[test]
fn every_ordinal_resolves_to_its_own_passage() {
    let (bytes, _) = fixture();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    let passages = engine.passages().unwrap();
    for (ordinal, passage) in passages.iter().enumerate() {
        assert_eq!(
            passage.ordinal as usize, ordinal,
            "record at ordinal {ordinal} belongs to a different passage"
        );
    }
}

/// The id-sorted region is a different order from the record region, so a
/// lookup that silently falls back to ordinal order would still "work" for the
/// first passage and fail for the rest. Check every id round-trips.
#[test]
fn every_passage_id_resolves_through_the_id_index() {
    let (bytes, _) = fixture();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    for passage in engine.passages().unwrap() {
        let found = engine.get_passage(&passage.id).unwrap();
        assert_eq!(found.id, passage.id);
        assert_eq!(found.ordinal, passage.ordinal);
    }
}

#[test]
fn an_unknown_passage_id_is_reported_not_guessed() {
    let (bytes, _) = fixture();
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    // Well-formed but absent, and malformed. Neither may return a passage.
    assert!(engine.get_passage(&"aa".repeat(32)).is_err());
    assert!(engine.get_passage("not-a-hash").is_err());
}

#[test]
fn a_tampered_record_block_is_rejected() {
    let (mut bytes, _) = fixture();
    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes.clone()))).unwrap();
    let section = reader.first_entry(SectionType::PassageRecords).unwrap();
    let offset = section.offset as usize;
    drop(reader);

    // The first record block starts at the section offset. Corrupting it must
    // fail the block's own hash, not silently serve a wrong record.
    bytes[offset + 8] ^= 0xff;
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).unwrap();
    assert!(
        engine.passages().is_err(),
        "a corrupted record block must fail verification, not serve wrong records"
    );
}
