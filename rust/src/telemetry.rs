//! OpenTelemetry attributes for a retrieval.
//!
//! The point of this module is to be small. A trace already records that a
//! retrieval happened and what it returned; what it cannot record today is
//! *which immutable artifact* the returned text came from, so a span outlives
//! its own evidence — the corpus moves and the recorded document becomes
//! uncheckable. Attaching an artifact root and per-passage hashes fixes that
//! without any new backend.
//!
//! Everything here stays inside the `annpack.*` namespace. The host application
//! already sets whatever `gen_ai.*` conventions it follows, and those are still
//! moving; mirroring them here would bake a guess into a wire contract. These
//! attributes compose with any of them.
//!
//! Attributes are OTel-typed: strings, booleans, and homogeneous string arrays.
//! Artifact-level facts go on the retrieval span, per-passage facts go on one
//! event per passage, because OTel attribute values cannot be objects.

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};
use crate::search::SearchResponse;

/// The immutable artifact root the retrieval read from. Pinning this in a trace
/// is what lets a later reader tell whether the corpus has moved since.
pub const ROOT: &str = "annpack.root";
pub const PACK: &str = "annpack.pack";
pub const SOURCE_REVISION: &str = "annpack.source_revision";
pub const PASSAGE_ID: &str = "annpack.passage_id";
pub const PASSAGE_HASH: &str = "annpack.passage_hash";
pub const RECEIPT_URI: &str = "annpack.receipt_uri";

const PASSAGE_ID_PLACEHOLDER: &str = "{passage_id}";
const ROOT_PLACEHOLDER: &str = "{root}";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalSpanAttributes {
    #[serde(rename = "annpack.root")]
    pub root: String,
    #[serde(rename = "annpack.pack")]
    pub pack: String,
    #[serde(
        rename = "annpack.source_revision",
        skip_serializing_if = "Option::is_none"
    )]
    pub source_revision: Option<String>,
    /// `signed`, `unsigned`, or whatever the reader reported. Recorded because
    /// "this artifact was signed" and "the signer is trusted" are different
    /// claims and a trace that conflates them is misleading later.
    #[serde(rename = "annpack.publisher.status")]
    pub publisher_status: String,
    #[serde(rename = "annpack.publisher.identity_trusted")]
    pub publisher_identity_trusted: bool,
    #[serde(rename = "annpack.passage_ids")]
    pub passage_ids: Vec<String>,
    #[serde(rename = "annpack.passage_hashes")]
    pub passage_hashes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassageEventAttributes {
    #[serde(rename = "annpack.root")]
    pub root: String,
    #[serde(rename = "annpack.passage_id")]
    pub passage_id: String,
    #[serde(rename = "annpack.passage_hash")]
    pub passage_hash: String,
    #[serde(rename = "annpack.rank")]
    pub rank: i64,
    #[serde(
        rename = "annpack.receipt_uri",
        skip_serializing_if = "Option::is_none"
    )]
    pub receipt_uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalTelemetry {
    /// Attributes for the retrieval span itself.
    pub span: RetrievalSpanAttributes,
    /// One `annpack.passage` event per returned passage, in rank order.
    pub events: Vec<PassageEventAttributes>,
}

/// Percent-encode a path segment against RFC 3986's unreserved set.
///
/// Passage IDs from the reference builder are hex, but an ID is a string from
/// another implementation's perspective, and this value is interpolated into a
/// URI. Encoding here means a hostile or merely unusual ID cannot escape its
/// path segment.
fn encode_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(byte as char);
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

/// A template missing `{passage_id}` would give every passage in the run the
/// same receipt URI, so the attribute would point at the wrong evidence for all
/// but one of them. Refuse rather than emit that.
fn validate_receipt_uri_template(template: &str) -> Result<()> {
    if template.contains(PASSAGE_ID_PLACEHOLDER) {
        return Ok(());
    }
    Err(AnnpackError::InvalidInput(format!(
        "receipt URI template must contain {PASSAGE_ID_PLACEHOLDER}"
    )))
}

/// Derive span and event attributes from a search response.
///
/// `receipt_uri_template` is where this deployment serves receipts. ANNPack does
/// not define that location, so the template is supplied rather than assumed; it
/// must contain `{passage_id}` and may contain `{root}`. Pass `None` to omit
/// `annpack.receipt_uri` entirely.
pub fn retrieval_telemetry(
    response: &SearchResponse,
    receipt_uri_template: Option<&str>,
) -> Result<RetrievalTelemetry> {
    if let Some(template) = receipt_uri_template {
        validate_receipt_uri_template(template)?;
    }

    let root = response.pack.root_hash.clone();
    let events = response
        .results
        .iter()
        .map(|hit| PassageEventAttributes {
            root: root.clone(),
            passage_id: hit.passage_id.clone(),
            passage_hash: hit.citation.passage_hash.clone(),
            rank: hit.rank as i64,
            receipt_uri: receipt_uri_template.map(|template| {
                template
                    .replace(ROOT_PLACEHOLDER, &encode_segment(&root))
                    .replace(PASSAGE_ID_PLACEHOLDER, &encode_segment(&hit.passage_id))
            }),
        })
        .collect();

    Ok(RetrievalTelemetry {
        span: RetrievalSpanAttributes {
            root: root.clone(),
            pack: format!("{}@{}", response.pack.name, response.pack.version),
            source_revision: response.pack.source_revision.clone(),
            publisher_status: response.pack.publisher.status.clone(),
            publisher_identity_trusted: response.pack.publisher.identity_trusted,
            passage_ids: response
                .results
                .iter()
                .map(|hit| hit.passage_id.clone())
                .collect(),
            passage_hashes: response
                .results
                .iter()
                .map(|hit| hit.citation.passage_hash.clone())
                .collect(),
        },
        events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreserved_characters_survive_encoding() {
        assert_eq!(encode_segment("aZ09-._~"), "aZ09-._~");
    }

    #[test]
    fn a_segment_cannot_escape_its_path() {
        assert_eq!(encode_segment("../etc"), "..%2Fetc");
        assert_eq!(encode_segment("a?b#c"), "a%3Fb%23c");
        assert_eq!(encode_segment("a b"), "a%20b");
    }

    #[test]
    fn non_ascii_is_percent_encoded_per_byte() {
        assert_eq!(encode_segment("é"), "%C3%A9");
    }

    #[test]
    fn a_template_without_the_passage_placeholder_is_refused() {
        assert!(validate_receipt_uri_template("https://e.test/{root}").is_err());
        assert!(validate_receipt_uri_template("https://e.test/r/{passage_id}").is_ok());
        assert!(validate_receipt_uri_template("https://e.test/{root}/{passage_id}").is_ok());
    }

    /// The exported constants and the `serde(rename)` attributes spell the same
    /// attribute names twice, and nothing in the type system ties them together.
    /// A rename on one side would silently ship traces whose keys disagree with
    /// the names this module publishes, so bind them here.
    #[test]
    fn exported_names_match_the_names_actually_serialized() {
        let span = serde_json::to_value(RetrievalSpanAttributes {
            root: "root".into(),
            pack: "demo@1.0.0".into(),
            source_revision: Some("git:abc".into()),
            publisher_status: "unsigned".into(),
            publisher_identity_trusted: false,
            passage_ids: Vec::new(),
            passage_hashes: Vec::new(),
        })
        .unwrap();
        for name in [ROOT, PACK, SOURCE_REVISION] {
            assert!(span.get(name).is_some(), "span is missing {name}");
        }

        let event = serde_json::to_value(PassageEventAttributes {
            root: "root".into(),
            passage_id: "id".into(),
            passage_hash: "hash".into(),
            rank: 1,
            receipt_uri: Some("https://e.test/id".into()),
        })
        .unwrap();
        for name in [ROOT, PASSAGE_ID, PASSAGE_HASH, RECEIPT_URI] {
            assert!(event.get(name).is_some(), "event is missing {name}");
        }
    }

    #[test]
    fn an_absent_receipt_uri_is_omitted_rather_than_null() {
        // A null attribute is not a valid OTel value; the key must be absent.
        let event = serde_json::to_value(PassageEventAttributes {
            root: "root".into(),
            passage_id: "id".into(),
            passage_hash: "hash".into(),
            rank: 1,
            receipt_uri: None,
        })
        .unwrap();
        assert!(event.get(RECEIPT_URI).is_none());
    }
}
