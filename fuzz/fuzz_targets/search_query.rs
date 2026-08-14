#![no_main]

use std::sync::Arc;

use adyar::reader::MemoryReader;
use adyar::search::{SearchEngine, SearchOptions};
use libfuzzer_sys::fuzz_target;

/// Query-side fuzzing against a valid artifact.
///
/// The container targets mutate bytes; this one holds the artifact fixed and
/// makes the *query* arbitrary, which is the other half of the attack surface
/// and the one a deployed reader is most exposed to: a pack is usually pinned
/// and trusted, while queries arrive from users and agents.
///
/// It exercises normative tokenization (NFKC, the technical punctuation set,
/// edge trimming), BM25 scoring, the sparse dictionary-block search, posting
/// reassembly across block boundaries, ordinal arithmetic into the record table,
/// and passage decoding. None of that is reachable from a byte-mutation target,
/// because those all sit behind the content-root check.
const GOLDEN: &[u8] = include_bytes!("../../spec/test-vectors/minimal-v3.annpack");

fuzz_target!(|data: &[u8]| {
    // Arbitrary bytes, not just valid UTF-8: the tokenizer normalizes and
    // trims, and lossy conversion still produces the surrogate and combining
    // sequences worth exercising.
    let query = String::from_utf8_lossy(data);
    if query.is_empty() {
        return;
    }
    let Ok(engine) = SearchEngine::open_source(Arc::new(MemoryReader::new(GOLDEN.to_vec()))) else {
        return;
    };
    let _ = engine.search(
        &query,
        &SearchOptions {
            limit: 10,
            ..Default::default()
        },
    );
    // Identifier lookup takes a different path through the record table than
    // ordinal access: the id-sorted region and its binary search.
    let _ = engine.get_passage(&query);
});
