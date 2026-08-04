//! Pin the reference implementation to the published conformance vectors.
//!
//! The vectors in `spec/conformance/vectors/` are the contract handed to an
//! independent implementer. They must not silently follow the reference: if the
//! reference changes behaviour, these tests fail and the change has to be an
//! explicit, reviewed edit to the vectors and the specification.
//!
//! This direction matters. The specification is normative; the reference
//! implementation is what changes when they disagree.

use std::path::{Path, PathBuf};

use annpack::search::{SearchEngine, SearchMode, SearchOptions, tokenize};
use serde_json::Value;

fn packet() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("spec/conformance")
}

fn vectors(name: &str) -> Value {
    let path = packet().join("vectors").join(name);
    serde_json::from_slice(
        &std::fs::read(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display())),
    )
    .unwrap()
}

#[test]
fn reference_tokenizer_matches_the_published_vectors() {
    let vectors = vectors("tokenizer.json");
    let cases = vectors["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let input = case["input"].as_str().unwrap();
        let expected: Vec<&str> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert_eq!(
            tokenize(input),
            expected,
            "tokenizer vector disagrees for {input:?} ({})",
            case["why"].as_str().unwrap_or("")
        );
    }
}

#[test]
fn published_technical_punctuation_set_is_the_one_the_boost_uses() {
    let vectors = vectors("tokenizer.json");
    let published: Vec<&str> = vectors["technical_punctuation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert_eq!(published, ["_", "-", ".", ":", "/", "@", "#"]);
    // Each member must survive tokenization inside a token, which is what makes
    // it "technical punctuation" rather than a separator.
    for character in &published {
        let token = format!("a{character}b");
        assert_eq!(
            tokenize(&token),
            vec![token.clone()],
            "{character:?} must be interior-preserving"
        );
    }
}

#[test]
fn reference_scoring_matches_the_published_vectors_exactly() {
    let vectors = vectors("scoring.json");
    let pack = packet().join("artifacts/conformance-v2.annpack");
    let engine = SearchEngine::open_path(&pack).unwrap();
    assert_eq!(
        engine.reader().root_hex(),
        vectors["pack_root"].as_str().unwrap(),
        "conformance artifact root drifted from the vectors"
    );

    for entry in vectors["queries"].as_array().unwrap() {
        let query = entry["query"].as_str().unwrap();
        let response = engine
            .search(
                query,
                &SearchOptions {
                    limit: 10,
                    mode: SearchMode::Lexical,
                    ..Default::default()
                },
            )
            .unwrap();
        let expected = entry["results"].as_array().unwrap();
        assert_eq!(
            response.results.len(),
            expected.len(),
            "result count changed for {query:?}; a different hit count is exactly \
             how a divergent tokenizer shows up"
        );
        for (hit, want) in response.results.iter().zip(expected) {
            assert_eq!(
                hit.passage_id,
                want["passage_id"].as_str().unwrap(),
                "{query:?}"
            );
            // Compare the IEEE-754 bit pattern, not the decimal. Exactness is
            // required (a reader with the wrong boost constant ranks identically
            // but scores differently), and decimal comparison is unreliable
            // across JSON parsers: serde_json without `float_roundtrip` loses up
            // to 1 ULP reading a double back. The bit pattern is unambiguous in
            // every language.
            let want_bits = u64::from_str_radix(want["score_bits"].as_str().unwrap(), 16).unwrap();
            assert_eq!(
                hit.score.to_bits(),
                want_bits,
                "score drifted for {query:?} at rank {} ({:?} vs vector {:?})",
                hit.rank,
                hit.score,
                f64::from_bits(want_bits)
            );
            // The decimal form must still round-trip, or the published vector is
            // misleading to a reader that does parse it.
            assert_eq!(
                want["score"].as_f64().unwrap().to_bits(),
                want_bits,
                "decimal and bit-pattern disagree in the vectors for {query:?}"
            );
        }
    }
}

#[test]
fn the_corpus_actually_discriminates_a_splitting_tokenizer() {
    // Guard the packet's whole reason for existing. `std::move` and `foo_bar`
    // must each match exactly one passage; a tokenizer that splits on ':' or '_'
    // additionally matches the separate-words page, changing the hit count and,
    // for foo_bar, the top result. If someone edits the corpus and this property
    // is lost, the packet silently stops discriminating.
    let vectors = vectors("scoring.json");
    for query in ["std::move", "foo_bar"] {
        let entry = vectors["queries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["query"] == query)
            .unwrap_or_else(|| panic!("scoring vectors must cover {query:?}"));
        assert_eq!(
            entry["result_count"].as_u64().unwrap(),
            1,
            "{query:?} must match exactly one passage for a conformant tokenizer"
        );
    }
    // And the decoy page must exist, or the discrimination is vacuous.
    let engine =
        SearchEngine::open_path(packet().join("artifacts/conformance-v2.annpack")).unwrap();
    let decoys = engine
        .search(
            "std move foo bar",
            &SearchOptions {
                limit: 10,
                mode: SearchMode::Lexical,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        decoys
            .results
            .iter()
            .any(|hit| hit.source_path == "separate-words.md"),
        "the decoy page must be reachable by the split tokens"
    );
}

#[test]
fn published_evidence_receipt_verifies_offline() {
    let vectors = vectors("evidence.json");
    let receipt: annpack::evidence::EvidenceReceipt =
        serde_json::from_value(vectors["receipt"].clone()).unwrap();
    let report = annpack::evidence::verify_receipt(&receipt, None).unwrap();
    assert!(
        report.verified,
        "published receipt must verify: {:?}",
        report.issues
    );
    assert!(
        !report.identity_trusted,
        "identity trust needs an external binding"
    );
    assert_eq!(
        receipt.passage_merkle_root,
        vectors["passage_merkle_root"].as_str().unwrap()
    );
}

#[test]
fn signing_does_not_change_the_artifact_root() {
    let vectors = vectors("signature.json");
    assert!(vectors["roots_match"].as_bool().unwrap());
    let unsigned =
        SearchEngine::open_path(packet().join("artifacts/conformance-v2.annpack")).unwrap();
    let signed =
        SearchEngine::open_path(packet().join("artifacts/conformance-v2-signed.annpack")).unwrap();
    assert_eq!(unsigned.reader().root_hex(), signed.reader().root_hex());
    for signature in vectors["signatures"].as_array().unwrap() {
        assert!(signature["cryptographically_valid"].as_bool().unwrap());
        assert!(
            !signature["identity_trusted_without_external_binding"]
                .as_bool()
                .unwrap(),
            "a valid signature must never by itself establish identity trust"
        );
    }
}

#[test]
fn every_corruption_artifact_is_rejected() {
    let vectors = vectors("corruption.json");
    let directory = packet().join("artifacts/corruption");
    for (name, reason) in vectors["artifacts"].as_object().unwrap() {
        let path = directory.join(name);
        assert!(path.is_file(), "missing corruption artifact {name}");
        // "Rejected" means the reader refuses to serve content, not necessarily
        // that `open` fails. Section hashes are verified lazily, before decoding
        // each payload, so a section-hash mismatch surfaces on use. Both points
        // are conformant; a reader must fail at one of them.
        let rejected = match annpack::format::PackReader::open_path(&path) {
            Err(_) => true,
            Ok(reader) => reader.verify_all().is_err(),
        };
        assert!(
            rejected,
            "{name} must be rejected at open or on use ({})",
            reason.as_str().unwrap_or("")
        );
    }
}
