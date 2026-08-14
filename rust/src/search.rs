use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::conformance::{ConformanceReport, inspect_conformance_with_manifest};
use crate::error::{AdyarError, Result};
use crate::format::{PackReader, SectionType};
use crate::model::{
    DictionaryBlock, Document, IndexBlock, IvfIndex, LexicalBlockIndex, LexicalDictionary,
    Manifest, Passage, PostingMeta, RecordBlockIndex, StoredPassageIndex, StoredRecord,
    VectorProfileSection,
};
use crate::reader::{FileReader, ReadAt};
use crate::signing::verify_signatures;

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;
const MAX_RESULTS: usize = 1_000;
const MAX_QUERY_TERMS: usize = 256;
const MAX_PASSAGE_BLOCK_LOGICAL_SIZE: u64 = 1024 * 1024;
const MAX_PASSAGE_BLOCK_COMPRESSION_RATIO: u64 = 256;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Lexical,
    Vector,
    Hybrid,
}

/// AN-10 profile request. Which advertised `retrieval_profiles` entry (if any)
/// the runtime should activate for this search.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ProfileRequest {
    /// Default. Core lexical only — byte-identical to Core; never activates a
    /// vector or derived (expansion/splade) profile. This is what keeps derived
    /// retrieval off by default (AN-7/AN-8 policy) even for a fat pack.
    #[default]
    Lexical,
    /// Activate the named profile if the runtime can execute it; otherwise fall
    /// back deterministically to lexical (never a different derived profile).
    Named(String),
    /// Explicit opt-in: walk `retrieval_profiles` in order and activate the first
    /// profile the runtime can execute. May activate a derived profile — the
    /// caller asked for it, and the choice is always reported.
    Auto,
}

impl ProfileRequest {
    fn label(&self) -> String {
        match self {
            Self::Lexical => "lexical".into(),
            Self::Named(id) => id.clone(),
            Self::Auto => "auto".into(),
        }
    }
}

/// Default effective overlay weight applied when a derived profile (expansion or
/// splade) is selected via AN-10. Weight *calibration* is deliberately out of
/// scope for the selection contract; this is a neutral default and the effective
/// value is always reported in `SearchResponse.profile_selection`.
const DERIVED_PROFILE_WEIGHT: f64 = 1.0;

/// Capabilities the reference runtime can actually EXECUTE during search. Note
const RUNTIME_SEARCH_CAPABILITIES: &[&str] = &[
    "lexical-bm25",
    "vector-ivf-flat-dot",
    "term-overlay-expansion",
    "term-overlay-splade",
];

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit: usize,
    pub mode: SearchMode,
    pub query_vector: Option<Vec<f32>>,
    pub vector_profile: Option<String>,
    pub candidate_depth: usize,
    pub lexical_weight: f64,
    pub vector_weight: f64,
    pub vector_probes: usize,
    /// AN-7 expansion overlay weight. Defaults to 0.0: no effect, Core results.
    /// Superseded by AN-10 profile selection on a fat pack (see `profile`).
    pub expansion_weight: f64,
    /// AN-8 vocabulary overlay weight. Defaults to 0.0: no effect, Core results.
    /// Superseded by AN-10 profile selection on a fat pack (see `profile`).
    pub splade_weight: f64,
    /// AN-10 profile request. On a fat pack this determines the effective mode
    /// and overlay weights; on a non-fat pack it is a no-op and the raw
    /// mode/weights above apply (legacy behavior).
    pub profile: ProfileRequest,
    pub debug: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            mode: SearchMode::Hybrid,
            query_vector: None,
            vector_profile: None,
            candidate_depth: 50,
            lexical_weight: 1.0,
            vector_weight: 1.0,
            vector_probes: 4,
            expansion_weight: 0.0,
            splade_weight: 0.0,
            profile: ProfileRequest::Lexical,
            debug: false,
        }
    }
}

/// The outcome of AN-10 profile selection, always returned on `SearchResponse`
/// so callers see which profile ran, why, and with what effective weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSelection {
    /// The profile the caller asked for (`"lexical"`, `"auto"`, or an id).
    pub requested: String,
    /// The profile actually activated. `None` when the pack is not a fat pack
    /// (no `retrieval_profiles`), in which case raw mode/weights applied.
    pub selected: Option<String>,
    pub selected_kind: Option<String>,
    /// Human-readable deterministic explanation of the selection/fallback.
    pub reason: String,
    pub effective_mode: SearchMode,
    pub effective_expansion_weight: f64,
    pub effective_splade_weight: f64,
    /// Exactly the sections the selected profile declares. The overlay loader
    /// reads only these, so selecting one derived profile never fetches another
    /// profile's ranges. Empty when no profile was selected.
    pub selected_section_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub pack: SearchPackIdentity,
    pub query: String,
    pub requested_mode: SearchMode,
    pub effective_mode: SearchMode,
    pub results: Vec<SearchHit>,
    /// AN-10: which profile was selected, why, and its effective weights.
    /// Always present so selection is auditable without enabling debug.
    pub profile_selection: ProfileSelection,
    pub diagnostics: Option<SearchDiagnostics>,
}

/// Deterministic AN-10 profile selection. Pure function of the pack's advertised
/// profiles, the caller's request, and whether a query vector is available.
///
/// Contract:
/// - Non-fat pack (`profiles` empty): no-op. `selected = None`, raw mode/weights
///   pass through unchanged (legacy behavior).
/// - `Lexical` (default): force the lexical profile — never a vector/derived one.
/// - `Named(id)`: activate it if present and runtime-supported; otherwise fall
///   back to lexical (never a *different* derived profile the caller did not ask
///   for).
/// - `Auto`: first runtime-supported profile in array order (may be derived).
fn select_profile(
    profiles: &[crate::model::RetrievalProfile],
    request: &ProfileRequest,
    opts_mode: SearchMode,
    opts_expansion: f64,
    opts_splade: f64,
    has_vector: bool,
) -> ProfileSelection {
    // Effective execution config for a chosen profile kind.
    let effective = |kind: &str| -> (SearchMode, f64, f64) {
        match kind {
            "vector" => (SearchMode::Vector, 0.0, 0.0),
            "expansion" => (SearchMode::Lexical, DERIVED_PROFILE_WEIGHT, 0.0),
            "splade" => (SearchMode::Lexical, 0.0, DERIVED_PROFILE_WEIGHT),
            _ => (SearchMode::Lexical, 0.0, 0.0), // lexical and any unknown kind
        }
    };
    let supported = |p: &crate::model::RetrievalProfile| -> bool {
        // An empty `requires` would satisfy `all()` vacuously, and an
        // unrecognized kind has no execution path — reporting either as
        // "selected" would name a retrieval strategy that never ran.
        !p.requires.is_empty()
            && matches!(
                p.kind.as_str(),
                "lexical" | "vector" | "expansion" | "splade"
            )
            && p.requires
                .iter()
                .all(|cap| RUNTIME_SEARCH_CAPABILITIES.contains(&cap.as_str()))
            && (p.kind != "vector" || has_vector)
    };
    let requested = request.label();

    if profiles.is_empty() {
        return ProfileSelection {
            requested,
            selected: None,
            selected_kind: None,
            reason: "pack advertises no retrieval profiles; using requested mode/weights".into(),
            effective_mode: opts_mode,
            effective_expansion_weight: opts_expansion,
            effective_splade_weight: opts_splade,
            // No descriptor to scope by: the legacy weight path may read any
            // overlay the pack carries.
            selected_section_ids: Vec::new(),
        };
    }

    // The lexical profile is the guaranteed terminal fallback for any walk.
    let lexical = || {
        profiles
            .iter()
            .rev()
            .find(|p| p.kind == "lexical")
            .unwrap_or(&profiles[profiles.len() - 1])
    };
    // Deterministic forward walk from `start` to the first supported profile.
    let walk_from = |start: usize| profiles[start..].iter().find(|p| supported(p));

    let (chosen, reason): (&crate::model::RetrievalProfile, String) = match request {
        // Force lexical directly — never a vector/derived profile by default.
        ProfileRequest::Lexical => (lexical(), "default: core lexical".into()),
        ProfileRequest::Auto => {
            let c = walk_from(0).unwrap_or_else(lexical);
            (
                c,
                format!("auto: selected first supported profile {:?}", c.id),
            )
        }
        // A named request yields the named profile or lexical — never a
        // *different* derived profile the caller did not ask for.
        ProfileRequest::Named(id) => match profiles.iter().find(|p| &p.id == id) {
            Some(p) if supported(p) => (p, format!("requested profile {id:?}")),
            Some(_) => (
                lexical(),
                format!("requested profile {id:?} not runtime-supported; fell back to lexical"),
            ),
            None => (
                lexical(),
                format!("requested profile {id:?} absent from pack; fell back to lexical"),
            ),
        },
    };
    let (mode, exp, spl) = effective(&chosen.kind);
    ProfileSelection {
        requested,
        selected: Some(chosen.id.clone()),
        selected_kind: Some(chosen.kind.clone()),
        reason,
        effective_mode: mode,
        effective_expansion_weight: exp,
        effective_splade_weight: spl,
        selected_section_ids: chosen.section_ids.clone(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPackIdentity {
    pub name: String,
    pub version: String,
    pub root_hash: String,
    pub source_revision: Option<String>,
    pub publisher: PublisherEvidence,
    pub conformance: ConformanceReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub rank: usize,
    pub score: f64,
    pub lexical_score: Option<f64>,
    pub vector_score: Option<f64>,
    pub lexical_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub document_id: String,
    pub passage_id: String,
    pub title: String,
    pub heading_path: Vec<String>,
    pub url: Option<String>,
    pub source_path: String,
    pub text: String,
    pub citation: Citation,
    pub evidence: EvidenceEnvelope,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub canonical_url: Option<String>,
    pub pack: String,
    pub pack_root: String,
    pub passage_hash: String,
    pub source_revision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEnvelope {
    pub schema: String,
    pub pack: String,
    pub pack_root: String,
    pub source_revision: Option<String>,
    pub passage_id: String,
    pub passage_hash: String,
    pub canonical_url: Option<String>,
    pub publisher: PublisherEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublisherEvidence {
    pub status: String,
    pub key_ids: Vec<String>,
    pub asserted_identities: Vec<String>,
    pub identity_trusted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchDiagnostics {
    pub query_terms: Vec<String>,
    pub lexical_candidates: usize,
    pub vector_candidates: usize,
    pub vector_profile: Option<String>,
}

#[derive(Debug, Clone)]
struct RankedCandidate {
    ordinal: usize,
    score: f64,
}

#[derive(Debug, Default, Clone)]
struct FusionCandidate {
    fused_score: f64,
    lexical_score: Option<f64>,
    vector_score: Option<f64>,
    lexical_rank: Option<usize>,
    vector_rank: Option<usize>,
}

pub struct SearchEngine {
    reader: PackReader,
    manifest: Manifest,
    documents: Vec<Document>,
    documents_by_id: HashMap<String, usize>,
    passage_index: StoredPassageIndex,
    records: RecordTable,
    dictionary: LexicalDictionary,
    lexical: LexicalIndex,
    conformance: ConformanceReport,
    publisher: PublisherEvidence,
    passage_block_cache: Mutex<HashMap<u32, Arc<Vec<u8>>>>,
}

/// How a pack's term table and posting stream are reached.
///
/// The distinction is entirely about transfer, not semantics: both variants
/// answer the same two questions (what is this term's posting metadata, and
/// what bytes are its posting list) and must produce identical results. What
/// differs is whether answering costs the whole index or one block.
enum LexicalIndex {
    /// Lexical index format 1: the term table and posting stream were read in
    /// full at open. Retained so packs built before format 2 keep working.
    Inline {
        terms: BTreeMap<String, PostingMeta>,
        postings: Vec<u8>,
    },
    /// Lexical index format 2: block tables only. Blocks are fetched on demand,
    /// verified against their own hash, and cached for the session.
    Blocked {
        terms_section: u32,
        postings_section: u32,
        blocks: LexicalBlockIndex,
        /// Logical start offset of each postings block, cumulative over the
        /// table. Lets a posting range map to the blocks that carry it without
        /// assuming the writer's block size.
        postings_starts: Vec<u64>,
        term_cache: Mutex<HashMap<usize, Arc<BTreeMap<String, PostingMeta>>>>,
        postings_cache: Mutex<HashMap<usize, Arc<Vec<u8>>>>,
    },
}

/// A validated AN-7/AN-8 term overlay: matching-only, never citable.
#[derive(Debug, Clone)]
struct LoadedOverlay {
    kind: String,
    /// Dequantization scale: 1.0 for expansion, `vocabulary.scale` for splade.
    scale: f64,
    terms: HashMap<String, Vec<(u32, u32)>>,
}

impl SearchEngine {
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_path_with_trusted_key(path, None)
    }

    pub fn open_path_with_trusted_key(
        path: impl AsRef<Path>,
        trusted_public_key: Option<&Path>,
    ) -> Result<Self> {
        Self::open_source_with_trusted_key(Arc::new(FileReader::open(path)?), trusted_public_key)
    }

    pub fn open_source(source: Arc<dyn ReadAt>) -> Result<Self> {
        Self::open_source_with_trusted_key(source, None)
    }

    pub fn open_source_with_trusted_key(
        source: Arc<dyn ReadAt>,
        trusted_public_key: Option<&Path>,
    ) -> Result<Self> {
        let reader = PackReader::open(source)?;
        let manifest = reader.manifest()?;
        let conformance = inspect_conformance_with_manifest(&reader, &manifest);
        if !conformance.core_conformant {
            return Err(AdyarError::InvalidFormat(format!(
                "pack does not conform to {}: {}",
                conformance.core_profile,
                conformance.issues.join("; ")
            )));
        }
        let signature_section_count = reader.entries_of_type(SectionType::Signature).count();
        let signature_reports = verify_signatures(&reader, trusted_public_key)?;
        let publisher = PublisherEvidence {
            status: if signature_section_count == 0 {
                "unsigned".to_string()
            } else if signature_reports.is_empty() {
                "not_verified".to_string()
            } else {
                "cryptographically_verified".to_string()
            },
            key_ids: signature_reports
                .iter()
                .map(|report| report.key_id.clone())
                .collect(),
            asserted_identities: signature_reports
                .iter()
                .filter_map(|report| report.identity.clone())
                .collect(),
            identity_trusted: signature_reports
                .iter()
                .any(|report| report.identity_trusted),
        };
        let documents_entry = required_profile_section(&reader, SectionType::Documents)?;
        let passage_index_entry = required_profile_section(&reader, SectionType::PassageIndex)?;
        required_profile_section(&reader, SectionType::PassageData)?;
        let dictionary_entry = required_profile_section(&reader, SectionType::LexicalDictionary)?;
        let postings_entry = required_profile_section(&reader, SectionType::LexicalPostings)?;
        let documents: Vec<Document> =
            serde_json::from_slice(&reader.read_section(documents_entry)?)?;
        let mut passage_index: StoredPassageIndex =
            serde_json::from_slice(&reader.read_section(passage_index_entry)?)?;
        let mut dictionary: LexicalDictionary =
            serde_json::from_slice(&reader.read_section(dictionary_entry)?)?;

        // Resolve the lexical layout before reading anything large. A pack that
        // declares block tables and carries a terms section is format 2, and its
        // postings section must never be read whole — that read is the cost the
        // layout removes. Anything else is format 1 and is read as before.
        let terms_entry = reader
            .first_entry(SectionType::LexicalTerms)
            .map(|e| e.section_id);
        let records_section = reader
            .first_entry(SectionType::PassageRecords)
            .map(|e| e.section_id);
        let (lexical_layout, postings) = match (&passage_index.lexical_blocks, terms_entry) {
            (Some(blocks), Some(terms_section)) => (
                LexicalLayout::Blocked {
                    terms_section,
                    postings_section: postings_entry,
                    blocks: blocks.clone(),
                },
                Vec::new(),
            ),
            (Some(_), None) => {
                return Err(AdyarError::InvalidFormat(
                    "pack declares lexical block tables but carries no lexical terms section"
                        .into(),
                ));
            }
            (None, _) => (LexicalLayout::Inline, reader.read_section(postings_entry)?),
        };
        if documents.len() != manifest.document_count as usize {
            return Err(AdyarError::InvalidFormat(
                "document section and manifest counts disagree".into(),
            ));
        }
        // The manifest's passage count is the reference all three must agree on.
        // In passage index format 2 the record table is not resident here, so
        // its coverage is checked against this same count in
        // `validate_record_blocks`; only the inline layout can be compared
        // directly.
        if dictionary.passage_lengths.len() != manifest.passage_count as usize
            || (passage_index.record_blocks.is_none()
                && passage_index.records.len() != manifest.passage_count as usize)
        {
            return Err(AdyarError::InvalidFormat(
                "passage index, lexical index, and manifest counts disagree".into(),
            ));
        }
        if passage_index.codec != "deflate-zlib" {
            return Err(AdyarError::Unsupported(format!(
                "passage block codec {:?}",
                passage_index.codec
            )));
        }
        if !dictionary.average_passage_length.is_finite() || dictionary.average_passage_length < 0.0
        {
            return Err(AdyarError::InvalidFormat(
                "lexical index has an invalid average passage length".into(),
            ));
        }
        let passage_data_entry = reader
            .first_entry(SectionType::PassageData)
            .ok_or_else(|| AdyarError::InvalidFormat("passage data section is missing".into()))?;
        if passage_data_entry.codec != crate::format::Codec::None {
            return Err(AdyarError::InvalidFormat(
                "passage data must use independently compressed blocks".into(),
            ));
        }
        let mut block_ranges = Vec::with_capacity(passage_index.blocks.len());
        for (index, block) in passage_index.blocks.iter().enumerate() {
            if block.logical_length > MAX_PASSAGE_BLOCK_LOGICAL_SIZE {
                return Err(AdyarError::InvalidFormat(format!(
                    "passage block {index} exceeds the logical size limit"
                )));
            }
            if block.stored_length == 0 && block.logical_length != 0 {
                return Err(AdyarError::InvalidFormat(format!(
                    "passage block {index} has no stored bytes"
                )));
            }
            if block.logical_length
                > block
                    .stored_length
                    .saturating_mul(MAX_PASSAGE_BLOCK_COMPRESSION_RATIO)
                    .max(MAX_PASSAGE_BLOCK_LOGICAL_SIZE)
            {
                return Err(AdyarError::InvalidFormat(format!(
                    "passage block {index} exceeds the compression-ratio limit"
                )));
            }
            let hash = hex::decode(&block.hash).map_err(|_| {
                AdyarError::InvalidFormat(format!("passage block {index} has an invalid hash"))
            })?;
            if hash.len() != 32 {
                return Err(AdyarError::InvalidFormat(format!(
                    "passage block {index} has an invalid hash length"
                )));
            }
            let end = block
                .offset
                .checked_add(block.stored_length)
                .ok_or_else(|| {
                    AdyarError::InvalidFormat("passage block range overflow".into())
                })?;
            if end > passage_data_entry.stored_length {
                return Err(AdyarError::InvalidFormat(format!(
                    "passage block {index} exceeds the passage data section"
                )));
            }
            block_ranges.push((block.offset, end, index));
        }
        block_ranges.sort_by_key(|range| range.0);
        for pair in block_ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(AdyarError::InvalidFormat(format!(
                    "passage blocks {} and {} overlap",
                    pair[0].2, pair[1].2
                )));
            }
        }
        for record in &passage_index.records {
            let block = passage_index
                .blocks
                .get(record.block as usize)
                .ok_or_else(|| {
                    AdyarError::InvalidFormat(format!(
                        "passage {} references missing block {}",
                        record.id, record.block
                    ))
                })?;
            let end = (record.offset as u64)
                .checked_add(record.length as u64)
                .ok_or_else(|| {
                    AdyarError::InvalidFormat("passage record range overflow".into())
                })?;
            if end > block.logical_length {
                return Err(AdyarError::InvalidFormat(format!(
                    "passage {} exceeds logical block {}",
                    record.id, record.block
                )));
            }
            let id = hex::decode(&record.id).map_err(|_| {
                AdyarError::InvalidFormat(format!(
                    "passage record has an invalid ID {:?}",
                    record.id
                ))
            })?;
            if id.len() != 32 {
                return Err(AdyarError::InvalidFormat(format!(
                    "passage record has an invalid ID length {:?}",
                    record.id
                )));
            }
        }
        // Structural validation of the lexical index.
        //
        // In the inline layout this can afford to be exhaustive, because the
        // whole term table and posting stream are already resident. In the
        // blocked layout an exhaustive walk would defeat the point — it would
        // fetch every block at open, which is exactly the cost the layout
        // exists to avoid. So the blocked path validates the block tables
        // (cheap, and already in memory) and defers per-posting ordinal checks
        // to the point of use, where `decode_postings` results are bounds-checked
        // against the passage count before they are scored.
        let lexical = match lexical_layout {
            LexicalLayout::Inline => {
                let mut posting_cursor = 0_u64;
                for (term, meta) in &dictionary.terms {
                    if meta.offset != posting_cursor || meta.document_frequency == 0 {
                        return Err(AdyarError::InvalidFormat(format!(
                            "posting metadata for term {term:?} is non-canonical"
                        )));
                    }
                    let end = meta.offset.checked_add(meta.length).ok_or_else(|| {
                        AdyarError::InvalidFormat("posting metadata range overflow".into())
                    })?;
                    let start = usize::try_from(meta.offset).map_err(|_| {
                        AdyarError::InvalidFormat("posting offset exceeds address space".into())
                    })?;
                    let end_usize = usize::try_from(end).map_err(|_| {
                        AdyarError::InvalidFormat("posting end exceeds address space".into())
                    })?;
                    let list = postings.get(start..end_usize).ok_or_else(|| {
                        AdyarError::InvalidFormat(format!(
                            "posting list for term {term:?} exceeds its section"
                        ))
                    })?;
                    for (ordinal, _) in decode_postings(list, meta.document_frequency as usize)? {
                        if ordinal >= passage_index.records.len() {
                            return Err(AdyarError::InvalidFormat(format!(
                                "posting list for term {term:?} has an invalid passage ordinal"
                            )));
                        }
                    }
                    posting_cursor = end;
                }
                if posting_cursor != postings.len() as u64 {
                    return Err(AdyarError::InvalidFormat(
                        "lexical dictionary does not cover the postings section exactly".into(),
                    ));
                }
                LexicalIndex::Inline {
                    terms: std::mem::take(&mut dictionary.terms),
                    postings,
                }
            }
            LexicalLayout::Blocked {
                terms_section,
                postings_section,
                blocks,
            } => {
                let postings_starts = validate_lexical_blocks(&reader, &blocks)?;
                LexicalIndex::Blocked {
                    terms_section,
                    postings_section,
                    blocks,
                    postings_starts,
                    term_cache: Mutex::new(HashMap::new()),
                    postings_cache: Mutex::new(HashMap::new()),
                }
            }
        };
        let mut documents_by_id = HashMap::new();
        for (index, document) in documents.iter().enumerate() {
            if documents_by_id.insert(document.id.clone(), index).is_some() {
                return Err(AdyarError::InvalidFormat(format!(
                    "duplicate document ID {}",
                    document.id
                )));
            }
        }
        // Passage record layout. Format 2 keeps the table out of memory and out
        // of the open path; format 1 packs still carry it inline.
        let records = match (&passage_index.record_blocks, records_section) {
            (Some(index), Some(section)) => {
                validate_record_blocks(&reader, index, manifest.passage_count as usize)?;
                RecordTable::Blocked {
                    section,
                    index: index.clone(),
                    count: manifest.passage_count as usize,
                    cache: Mutex::new(HashMap::new()),
                    id_cache: Mutex::new(HashMap::new()),
                }
            }
            (Some(_), None) => {
                return Err(AdyarError::InvalidFormat(
                    "pack declares record block tables but carries no passage records section"
                        .into(),
                ));
            }
            (None, _) => {
                let mut seen = std::collections::HashSet::new();
                for record in &passage_index.records {
                    if !seen.insert(record.id.clone()) {
                        return Err(AdyarError::InvalidFormat(format!(
                            "duplicate passage ID {}",
                            record.id
                        )));
                    }
                }
                RecordTable::Inline {
                    records: std::mem::take(&mut passage_index.records),
                }
            }
        };
        Ok(Self {
            reader,
            manifest,
            documents,
            documents_by_id,
            passage_index,
            records,
            dictionary,
            lexical,
            conformance,
            publisher,
            passage_block_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn reader(&self) -> &PackReader {
        &self.reader
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn conformance(&self) -> &ConformanceReport {
        &self.conformance
    }

    pub fn get_passage(&self, passage_id: &str) -> Result<Passage> {
        let ordinal = self
            .records
            .ordinal_of(&self.reader, passage_id)?
            .ok_or_else(|| AdyarError::Search(format!("unknown passage ID {passage_id}")))?;
        self.load_passage(ordinal)
    }

    pub fn passages(&self) -> Result<Vec<Passage>> {
        (0..self.records.len())
            .map(|ordinal| self.load_passage(ordinal))
            .collect()
    }

    pub fn search(&self, query: &str, options: &SearchOptions) -> Result<SearchResponse> {
        if query.trim().is_empty() {
            return Err(AdyarError::InvalidInput("query must not be empty".into()));
        }
        if options.limit == 0 || options.limit > MAX_RESULTS {
            return Err(AdyarError::InvalidInput(format!(
                "result limit must be between 1 and {MAX_RESULTS}"
            )));
        }
        let query_terms = tokenize(query);
        if query_terms.len() > MAX_QUERY_TERMS {
            return Err(AdyarError::InvalidInput(format!(
                "query contains more than {MAX_QUERY_TERMS} terms"
            )));
        }
        // Reject non-finite / negative scoring weights. Left unchecked, a +inf
        // weight poisons every score and a NaN weight silently disables a path.
        for (name, weight) in [
            ("lexical_weight", options.lexical_weight),
            ("vector_weight", options.vector_weight),
            ("expansion_weight", options.expansion_weight),
            ("splade_weight", options.splade_weight),
        ] {
            if !weight.is_finite() || weight < 0.0 {
                return Err(AdyarError::InvalidInput(format!(
                    "{name} must be a finite, non-negative number"
                )));
            }
        }

        // AN-10 safety boundary. A malformed optional descriptor must not be
        // able to influence retrieval at all: if the extension surface does not
        // validate, a profile request is refused outright and the default
        // lexical path runs from Core sections only. Default lexical retrieval
        // is therefore never reachable from an invalid descriptor.
        //
        // The guard covers every route into non-Core retrieval, not only the
        // AN-10 profile request. Selecting the default lexical profile while
        // asking for vector mode or a non-zero overlay weight would otherwise
        // still activate optional retrieval on a pack whose optional metadata
        // does not validate.
        //
        // Hybrid mode without a query vector is Core lexical and stays allowed;
        // it is the library default and reaches no optional section.
        let requests_vector_retrieval = match options.mode {
            SearchMode::Lexical => false,
            SearchMode::Vector => true,
            SearchMode::Hybrid => options.query_vector.is_some(),
        };
        if !self.conformance.extensions_conformant
            && (options.profile != ProfileRequest::Lexical
                || requests_vector_retrieval
                || options.expansion_weight > 0.0
                || options.splade_weight > 0.0)
        {
            return Err(AdyarError::InvalidInput(format!(
                "pack extension metadata is invalid, so only Core lexical retrieval is \
                 available: {}",
                self.conformance.extension_issues.join("; ")
            )));
        }
        let descriptor_usable =
            self.conformance.extensions_conformant && !self.manifest.retrieval_profiles.is_empty();

        // AN-10: resolve the effective execution config from profile selection.
        // On a non-fat pack this is a no-op and the raw options pass through.
        let selection = select_profile(
            if descriptor_usable {
                &self.manifest.retrieval_profiles
            } else {
                &[]
            },
            &options.profile,
            options.mode,
            options.expansion_weight,
            options.splade_weight,
            options.query_vector.is_some(),
        );
        let eff_mode = selection.effective_mode;
        let eff_expansion = selection.effective_expansion_weight;
        let eff_splade = selection.effective_splade_weight;

        let (lexical, lexical_achievable) = match eff_mode {
            SearchMode::Vector => (Vec::new(), 0.0),
            SearchMode::Lexical | SearchMode::Hybrid => self.lexical_candidates(
                &query_terms,
                options.candidate_depth.max(options.limit),
                eff_expansion,
                eff_splade,
                // Scope overlay reads to the selected profile's own sections so
                // choosing `expansion` never fetches the SPLADE ranges, and vice
                // versa. `None` is the legacy unscoped weight path.
                descriptor_usable.then_some(selection.selected_section_ids.as_slice()),
            )?,
        };
        let vector = match (&eff_mode, &options.query_vector) {
            (SearchMode::Lexical, _) => Vec::new(),
            (SearchMode::Vector | SearchMode::Hybrid, Some(vector)) => self.vector_candidates(
                vector,
                options.vector_profile.as_deref(),
                options.candidate_depth.max(options.limit),
                options.vector_probes,
            )?,
            (SearchMode::Vector, None) => {
                return Err(AdyarError::InvalidInput(
                    "vector mode requires a query vector".into(),
                ));
            }
            (SearchMode::Hybrid, None) => Vec::new(),
        };
        let effective_mode = if !lexical.is_empty() && !vector.is_empty() {
            SearchMode::Hybrid
        } else if !vector.is_empty() {
            SearchMode::Vector
        } else {
            SearchMode::Lexical
        };

        let fused = fuse_candidates(
            &lexical,
            lexical_achievable,
            &vector,
            options,
            effective_mode,
        );
        let mut results = Vec::new();
        for (rank, (ordinal, candidate)) in fused.into_iter().take(options.limit).enumerate() {
            let passage = self.load_passage(ordinal)?;
            let document_index =
                self.documents_by_id
                    .get(&passage.document_id)
                    .ok_or_else(|| {
                        AdyarError::InvalidFormat(format!(
                            "passage {} references unknown document {}",
                            passage.id, passage.document_id
                        ))
                    })?;
            let document = &self.documents[*document_index];
            let url = citation_url(document, &passage);
            let evidence = self.evidence_for_passage(&passage)?;
            let passage_hash = evidence.passage_hash.clone();
            results.push(SearchHit {
                rank: rank + 1,
                score: candidate.fused_score,
                lexical_score: candidate.lexical_score,
                vector_score: candidate.vector_score,
                lexical_rank: candidate.lexical_rank,
                vector_rank: candidate.vector_rank,
                document_id: document.id.clone(),
                passage_id: passage.id.clone(),
                title: document.title.clone(),
                heading_path: passage.heading_path.clone(),
                url: url.clone(),
                source_path: document.source_path.clone(),
                text: passage.text,
                citation: Citation {
                    canonical_url: url,
                    pack: format!("{}@{}", self.manifest.name, self.manifest.version),
                    pack_root: self.reader.root_hex(),
                    passage_hash,
                    source_revision: self.manifest.source_revision.clone(),
                },
                evidence,
            });
        }

        Ok(SearchResponse {
            pack: SearchPackIdentity {
                name: self.manifest.name.clone(),
                version: self.manifest.version.clone(),
                root_hash: self.reader.root_hex(),
                source_revision: self.manifest.source_revision.clone(),
                publisher: self.publisher.clone(),
                conformance: self.conformance.clone(),
            },
            query: query.to_string(),
            requested_mode: options.mode,
            effective_mode,
            results,
            profile_selection: selection,
            diagnostics: options.debug.then_some(SearchDiagnostics {
                query_terms,
                lexical_candidates: lexical.len(),
                vector_candidates: vector.len(),
                vector_profile: options.vector_profile.clone(),
            }),
        })
    }

    pub fn evidence_for_passage(&self, passage: &Passage) -> Result<EvidenceEnvelope> {
        let document_index = self
            .documents_by_id
            .get(&passage.document_id)
            .ok_or_else(|| {
                AdyarError::InvalidFormat(format!(
                    "passage {} references unknown document {}",
                    passage.id, passage.document_id
                ))
            })?;
        let document = &self.documents[*document_index];
        let canonical_url = citation_url(document, passage);
        let encoded = serde_json::to_vec(passage)?;
        let passage_hash = crate::evidence::passage_evidence_hash(&encoded);
        Ok(EvidenceEnvelope {
            schema: "annpack-evidence-v1".to_string(),
            pack: format!("{}@{}", self.manifest.name, self.manifest.version),
            pack_root: self.reader.root_hex(),
            source_revision: self.manifest.source_revision.clone(),
            passage_id: passage.id.clone(),
            passage_hash: hex::encode(passage_hash),
            canonical_url,
            publisher: self.publisher.clone(),
        })
    }

    /// Per-passage evidence hashes in deterministic corpus order: the leaves of
    /// the logical content root.
    fn passage_evidence_leaves(&self) -> Result<Vec<[u8; 32]>> {
        (0..self.records.len())
            .map(|ordinal| {
                let passage = self.load_passage(ordinal)?;
                let encoded = serde_json::to_vec(&passage)?;
                Ok(crate::evidence::passage_evidence_hash(&encoded))
            })
            .collect()
    }

    /// Build a standalone, offline-verifiable receipt for one passage.
    ///
    /// The receipt carries every byte needed to re-derive the chain from the
    /// passage record up to the signed artifact root, so a third party can check
    /// a citation without the pack, without the network, and without trusting
    /// this implementation.
    pub fn receipt_for_passage(
        &self,
        passage_id: &str,
    ) -> Result<crate::evidence::EvidenceReceipt> {
        let ordinal = self
            .records
            .ordinal_of(&self.reader, passage_id)?
            .ok_or_else(|| AdyarError::Search(format!("unknown passage ID {passage_id}")))?;
        let passage = self.load_passage(ordinal)?;
        let record = serde_json::to_vec(&passage)?;
        let evidence = self.evidence_for_passage(&passage)?;

        let leaves = self.passage_evidence_leaves()?;
        let root = crate::evidence::merkle_root(&leaves).ok_or_else(|| {
            AdyarError::InvalidFormat("pack carries no passages to commit".into())
        })?;
        // A pack built before manifest format 2 has no committed logical root.
        // Refuse rather than emit a receipt whose chain cannot close.
        let declared = self
            .manifest
            .passage_merkle_root
            .as_deref()
            .ok_or_else(|| {
                AdyarError::Unsupported(
                    "pack predates manifest format 2 and commits no passage_merkle_root, \
                 so no standalone receipt can be issued"
                        .into(),
                )
            })?;
        if declared != hex::encode(root) {
            return Err(AdyarError::Integrity(
                "recomputed passage merkle root does not match the manifest".into(),
            ));
        }
        let proof = crate::evidence::merkle_proof(&leaves, ordinal)?;

        let manifest_bytes = self
            .reader
            .read_section(self.reader.header.manifest_section_id)?;
        let directory = self.reader.directory_bytes()?;
        // Carry the Documents section's stored (compressed) bytes so a packless
        // verifier can authenticate `canonical_url`: they hash to the section's
        // directory entry, which `pack_root` already commits.
        let documents_section_id = self
            .reader
            .first_entry(SectionType::Documents)
            .ok_or_else(|| AdyarError::InvalidFormat("pack has no Documents section".into()))?
            .section_id;
        let documents_bytes = self.reader.read_stored_section(documents_section_id)?;
        let signature = self.first_signature()?;

        Ok(crate::evidence::EvidenceReceipt {
            schema: "annpack-receipt-v2".to_string(),
            pack: evidence.pack.clone(),
            pack_root: evidence.pack_root.clone(),
            passage_merkle_root: declared.to_string(),
            source_revision: evidence.source_revision.clone(),
            passage_id: evidence.passage_id.clone(),
            passage_hash: evidence.passage_hash.clone(),
            passage_ordinal: passage.ordinal,
            canonical_url: evidence.canonical_url.clone(),
            passage_record_b64: crate::evidence::b64_encode(&record),
            inclusion_proof: proof,
            manifest_bytes_b64: crate::evidence::b64_encode(&manifest_bytes),
            directory_b64: crate::evidence::b64_encode(&directory),
            manifest_section_id: self.reader.header.manifest_section_id,
            documents_section_id: Some(documents_section_id),
            documents_bytes_b64: Some(crate::evidence::b64_encode(&documents_bytes)),
            signature,
        })
    }

    fn first_signature(&self) -> Result<Option<crate::evidence::ReceiptSignature>> {
        let Some(entry) = self.reader.first_entry(SectionType::Signature) else {
            return Ok(None);
        };
        let envelope: crate::model::SignatureEnvelope =
            serde_json::from_slice(&self.reader.read_section(entry.section_id)?)?;
        Ok(Some(crate::evidence::ReceiptSignature {
            algorithm: envelope.algorithm,
            public_key: envelope.public_key,
            signature: envelope.signature,
            key_id: envelope.key_id,
            identity: envelope.identity,
        }))
    }

    fn lexical_candidates(
        &self,
        terms: &[String],
        depth: usize,
        expansion_weight: f64,
        splade_weight: f64,
        overlay_section_ids: Option<&[u32]>,
    ) -> Result<(Vec<RankedCandidate>, f64)> {
        if terms.is_empty() {
            return Ok((Vec::new(), 0.0));
        }
        let unique_terms: HashSet<&String> = terms.iter().collect();
        let passage_count = self.dictionary.passage_lengths.len() as f64;
        let average_length = self.dictionary.average_passage_length.max(1.0);
        let mut scores = HashMap::<usize, f64>::new();
        let mut achievable = 0.0_f64;
        for term in unique_terms.iter().copied() {
            let Some(meta) = self.lexical.lookup(&self.reader, term)? else {
                continue;
            };
            if meta.document_frequency == 0 {
                return Err(AdyarError::InvalidFormat(format!(
                    "posting metadata for term {term:?} is non-canonical"
                )));
            }
            let bytes = self.lexical.posting_bytes(&self.reader, &meta)?;
            let postings = decode_postings(&bytes, meta.document_frequency as usize)?;
            let df = meta.document_frequency as f64;
            let idf =
                (1.0 + (passage_count - df + 0.5) / (df + 0.5)).ln() * technical_term_boost(term);
            // The most this term could contribute to any passage: the BM25 term
            // saturates at idf * (k1 + 1) as term frequency grows. Summed over
            // the query, this is the score a hypothetical passage that fully
            // answered every term would earn, and it is what makes a raw score
            // comparable across queries.
            achievable += idf * (BM25_K1 + 1.0);
            for (ordinal, frequency) in postings {
                let passage_length =
                    *self
                        .dictionary
                        .passage_lengths
                        .get(ordinal)
                        .ok_or_else(|| {
                            AdyarError::InvalidFormat(format!(
                                "posting ordinal {ordinal} exceeds passage count"
                            ))
                        })? as f64;
                let tf = frequency as f64;
                let denominator =
                    tf + BM25_K1 * (1.0 - BM25_B + BM25_B * passage_length / average_length);
                *scores.entry(ordinal).or_default() += idf * tf * (BM25_K1 + 1.0) / denominator;
            }
        }
        // AN-7 / AN-8: pure-BM25 overlay contribution. No query-time model.
        // Weights default to 0.0, which reproduces Core results exactly and never
        // fetches the overlay sections. Overlays affect ranking only; they never
        // contribute citable text or evidence.
        let overlays = if expansion_weight > 0.0 || splade_weight > 0.0 {
            self.load_overlays(overlay_section_ids)?
        } else {
            Vec::new()
        };
        for overlay in &overlays {
            let weight = match overlay.kind.as_str() {
                crate::derive::EXPANSION_KIND => expansion_weight,
                crate::derive::SPLADE_KIND => splade_weight,
                _ => 0.0,
            };
            if weight <= 0.0 {
                continue;
            }
            for term in unique_terms.iter().copied() {
                let Some(postings) = overlay.terms.get(term) else {
                    continue;
                };
                let idf = self.term_idf(term, passage_count);
                for (ordinal, stored_weight) in postings {
                    let w = *stored_weight as f64 * overlay.scale;
                    let contribution = weight * idf * (w / (w + 1.0));
                    *scores.entry(*ordinal as usize).or_default() += contribution;
                }
            }
        }
        let mut candidates: Vec<_> = scores
            .into_iter()
            .map(|(ordinal, score)| RankedCandidate { ordinal, score })
            .collect();
        sort_candidates(&mut candidates);
        candidates.truncate(depth);
        Ok((candidates, achievable))
    }

    /// Core BM25 idf for a term, using its lexical document frequency (0 if the
    /// term never appears in original passage text).
    fn term_idf(&self, term: &str, passage_count: f64) -> f64 {
        let df = self
            .dictionary
            .terms
            .get(term)
            .map(|meta| meta.document_frequency as f64)
            .unwrap_or(0.0);
        (1.0 + (passage_count - df + 0.5) / (df + 0.5)).ln() * technical_term_boost(term)
    }

    fn vector_candidates(
        &self,
        query: &[f32],
        profile_id: Option<&str>,
        depth: usize,
        probes: usize,
    ) -> Result<Vec<RankedCandidate>> {
        if query.iter().any(|value| !value.is_finite()) {
            return Err(AdyarError::InvalidInput(
                "query vector contains a non-finite value".into(),
            ));
        }
        let profile_entry = self
            .reader
            .first_entry(SectionType::VectorProfile)
            .ok_or_else(|| AdyarError::Search("pack has no vector profile".into()))?;
        let profile: VectorProfileSection =
            serde_json::from_slice(&self.reader.read_section(profile_entry.section_id)?)?;
        if let Some(requested) = profile_id
            && requested != profile.profile.id
        {
            return Err(AdyarError::Search(format!(
                "vector profile {requested:?} is unavailable"
            )));
        }
        let dimensions = profile.profile.dimensions as usize;
        if dimensions == 0 || dimensions > 65_536 || profile.profile.dtype != "float32" {
            return Err(AdyarError::InvalidFormat(format!(
                "unsupported vector profile shape (dimensions {dimensions}, dtype {})",
                profile.profile.dtype
            )));
        }
        if query.len() != dimensions {
            return Err(AdyarError::InvalidInput(format!(
                "query vector dimension {} does not match profile dimension {dimensions}",
                query.len()
            )));
        }
        if probes == 0 || probes > 1_024 {
            return Err(AdyarError::InvalidInput(
                "vector probes must be between 1 and 1024".into(),
            ));
        }
        // AN-1 requires the vector rows to be in exact passage order, so every
        // identity is compared. In the blocked layout this walks the record
        // blocks; it runs only when a vector search is actually requested.
        if profile.passage_ids.len() != self.records.len() {
            return Err(AdyarError::InvalidFormat(
                "vector profile passage identities do not match the passage index".into(),
            ));
        }
        for (ordinal, profile_id) in profile.passage_ids.iter().enumerate() {
            if self.records.ordinal_of(&self.reader, profile_id)? != Some(ordinal) {
                return Err(AdyarError::InvalidFormat(
                    "vector profile passage identities do not match the passage index".into(),
                ));
            }
        }
        let vector_entry = self
            .reader
            .first_entry(SectionType::VectorData)
            .ok_or_else(|| {
                AdyarError::InvalidFormat("vector profile has no vector data".into())
            })?;
        let bytes = self.reader.read_section(vector_entry.section_id)?;
        if bytes.len() < 8 {
            return Err(AdyarError::InvalidFormat(
                "truncated vector section".into(),
            ));
        }
        let count = u32::from_le_bytes(bytes[0..4].try_into().expect("slice length")) as usize;
        let stored_dimensions =
            u32::from_le_bytes(bytes[4..8].try_into().expect("slice length")) as usize;
        let value_count = count
            .checked_mul(stored_dimensions)
            .ok_or_else(|| AdyarError::InvalidFormat("vector size overflow".into()))?;
        let expected_length = 8_usize
            .checked_add(
                value_count.checked_mul(4).ok_or_else(|| {
                    AdyarError::InvalidFormat("vector byte size overflow".into())
                })?,
            )
            .ok_or_else(|| AdyarError::InvalidFormat("vector section size overflow".into()))?;
        if count != self.records.len()
            || stored_dimensions != dimensions
            || bytes.len() != expected_length
        {
            return Err(AdyarError::InvalidFormat(
                "vector section shape does not match its profile".into(),
            ));
        }
        let ordinals = if let Some(index_entry) = self.reader.first_entry(SectionType::VectorIndex)
        {
            let index: IvfIndex =
                serde_json::from_slice(&self.reader.read_section(index_entry.section_id)?)?;
            select_ivf_ordinals(&index, query, count, dimensions, probes)?
        } else {
            (0..count).collect()
        };
        let mut candidates = Vec::with_capacity(ordinals.len());
        for ordinal in ordinals {
            let mut dot = 0.0_f64;
            let base = 8 + ordinal * dimensions * 4;
            for (dimension, query_value) in query.iter().enumerate() {
                let offset = base + dimension * 4;
                let value = f32::from_le_bytes(
                    bytes[offset..offset + 4]
                        .try_into()
                        .expect("validated vector bounds"),
                );
                if !value.is_finite() {
                    return Err(AdyarError::InvalidFormat(format!(
                        "stored vector {ordinal} contains a non-finite value"
                    )));
                }
                dot += *query_value as f64 * value as f64;
            }
            candidates.push(RankedCandidate {
                ordinal,
                score: dot,
            });
        }
        sort_candidates(&mut candidates);
        candidates.truncate(depth);
        Ok(candidates)
    }

    fn load_passage(&self, ordinal: usize) -> Result<Passage> {
        let record = self.records.get(&self.reader, ordinal)?;
        let record = &record;
        let passage_section = self
            .reader
            .first_entry(SectionType::PassageData)
            .map(|entry| entry.section_id)
            .ok_or_else(|| AdyarError::InvalidFormat("passage data section is missing".into()))?;
        let block = self
            .passage_index
            .blocks
            .get(record.block as usize)
            .ok_or_else(|| AdyarError::InvalidFormat("passage block is missing".into()))?;
        let logical = {
            let cached = self
                .passage_block_cache
                .lock()
                .map_err(|_| AdyarError::Search("passage block cache lock poisoned".into()))?
                .get(&record.block)
                .cloned();
            if let Some(cached) = cached {
                cached
            } else {
                let compressed = self.reader.read_section_range(
                    passage_section,
                    block.offset,
                    block.stored_length,
                )?;
                if blake3::hash(&compressed).to_hex().as_str() != block.hash {
                    return Err(AdyarError::Integrity(format!(
                        "passage block {} hash mismatch",
                        record.block
                    )));
                }
                let limit = usize::try_from(block.logical_length).map_err(|_| {
                    AdyarError::InvalidFormat("passage block exceeds address space".into())
                })?;
                let decompressed =
                    miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&compressed, limit)
                        .map_err(|error| {
                            AdyarError::InvalidFormat(format!(
                                "passage block {} deflate decode failed: {error:?}",
                                record.block
                            ))
                        })?;
                if decompressed.len() != limit {
                    return Err(AdyarError::InvalidFormat(format!(
                        "passage block {} decompressed to {}, expected {} bytes",
                        record.block,
                        decompressed.len(),
                        limit
                    )));
                }
                let decompressed = Arc::new(decompressed);
                self.passage_block_cache
                    .lock()
                    .map_err(|_| AdyarError::Search("passage block cache lock poisoned".into()))?
                    .insert(record.block, decompressed.clone());
                decompressed
            }
        };
        let start = record.offset as usize;
        let end = start
            .checked_add(record.length as usize)
            .ok_or_else(|| AdyarError::InvalidFormat("passage record range overflow".into()))?;
        let bytes = logical
            .get(start..end)
            .ok_or_else(|| AdyarError::InvalidFormat("passage record exceeds block".into()))?;
        let passage: Passage = serde_json::from_slice(bytes)?;
        if (!record.id.is_empty() && passage.id != record.id) || passage.ordinal as usize != ordinal
        {
            return Err(AdyarError::InvalidFormat(format!(
                "passage index record {ordinal} does not match passage payload"
            )));
        }
        Ok(passage)
    }
}

impl SearchEngine {
    /// Load and strictly validate every AN-7/AN-8 term overlay. Called lazily,
    /// only when an overlay weight is non-zero, so a lexical-only client never
    /// fetches these ranges (AN-10). A malformed overlay is an explicit error
    /// (attacker-controlled ordinals and weights are bounds-checked here). Overlays
    /// never contribute citable text.
    /// `section_ids` scopes the read to exactly the sections an AN-10 profile
    /// declares. `None` reads every overlay, which is the legacy raw-weight path
    /// used only when the pack advertises no usable descriptor.
    fn load_overlays(&self, section_ids: Option<&[u32]>) -> Result<Vec<LoadedOverlay>> {
        let reader = &self.reader;
        let passage_count = self.records.len();
        let mut overlays = Vec::new();
        for entry in reader.entries_of_type(SectionType::TermOverlay) {
            if let Some(allowed) = section_ids
                && !allowed.contains(&entry.section_id)
            {
                continue;
            }
            if !entry.derived() {
                return Err(AdyarError::InvalidFormat(
                    "term overlay section must be flagged derived".into(),
                ));
            }
            let section: crate::model::TermOverlaySection =
                serde_json::from_slice(&reader.read_section(entry.section_id)?)?;
            if section.kind != crate::derive::EXPANSION_KIND
                && section.kind != crate::derive::SPLADE_KIND
            {
                return Err(AdyarError::InvalidFormat(format!(
                    "unrecognized term overlay kind {:?}",
                    section.kind
                )));
            }
            let mut scale = 1.0_f64;
            if section.kind == crate::derive::SPLADE_KIND {
                match &section.vocabulary {
                    Some(vocabulary) if !vocabulary.id.trim().is_empty() => {
                        if !vocabulary.scale.is_finite() || vocabulary.scale <= 0.0 {
                            return Err(AdyarError::InvalidFormat(
                                "splade vocabulary scale must be positive and finite".into(),
                            ));
                        }
                        if vocabulary.quantization != "linear-u16" {
                            return Err(AdyarError::Unsupported(format!(
                                "splade vocabulary quantization {:?}",
                                vocabulary.quantization
                            )));
                        }
                        scale = vocabulary.scale;
                    }
                    _ => {
                        return Err(AdyarError::InvalidFormat(
                            "splade overlay requires a non-empty vocabulary id".into(),
                        ));
                    }
                }
            }
            let mut terms = HashMap::with_capacity(section.terms.len());
            for (term, postings) in section.terms {
                if postings.is_empty() {
                    return Err(AdyarError::InvalidFormat(format!(
                        "term overlay entry {term:?} has an empty posting list"
                    )));
                }
                let mut previous: Option<u32> = None;
                for (ordinal, weight) in &postings {
                    if *ordinal as usize >= passage_count {
                        return Err(AdyarError::InvalidFormat(format!(
                            "term overlay entry {term:?} has an out-of-range ordinal"
                        )));
                    }
                    if let Some(previous) = previous
                        && *ordinal <= previous
                    {
                        return Err(AdyarError::InvalidFormat(format!(
                            "term overlay entry {term:?} ordinals must be strictly increasing"
                        )));
                    }
                    if *weight == 0 {
                        return Err(AdyarError::InvalidFormat(format!(
                            "term overlay entry {term:?} has a zero weight"
                        )));
                    }
                    previous = Some(*ordinal);
                }
                terms.insert(term, postings);
            }
            overlays.push(LoadedOverlay {
                kind: section.kind,
                scale,
                terms,
            });
        }
        Ok(overlays)
    }
}

/// Fetch one stored block by range, verify it against the hash recorded in the
/// block table, then inflate it.
///
/// The hash check is the whole reason a partial read is safe here. `read_section_range`
/// cannot verify anything — a section hash only authenticates the section in
/// full — so authenticity for a block comes from the block table, which was
/// itself read from a hash-verified section.
fn read_index_block(reader: &PackReader, section_id: u32, block: &IndexBlock) -> Result<Vec<u8>> {
    let stored = reader.read_section_range(section_id, block.offset, block.stored_length)?;
    let expected = hex::decode(&block.hash)
        .map_err(|_| AdyarError::InvalidFormat("index block hash is not hex".into()))?;
    if expected.len() != 32 || blake3::hash(&stored).as_bytes() != expected.as_slice() {
        return Err(AdyarError::Integrity(format!(
            "index block at offset {} failed verification",
            block.offset
        )));
    }
    let limit = usize::try_from(block.logical_length)
        .map_err(|_| AdyarError::InvalidFormat("index block exceeds address space".into()))?;
    let logical = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&stored, limit).map_err(
        |error| {
            AdyarError::InvalidFormat(format!("index block deflate decode failed: {error:?}"))
        },
    )?;
    if logical.len() != limit {
        return Err(AdyarError::InvalidFormat(
            "index block decompressed to an unexpected length".into(),
        ));
    }
    Ok(logical)
}

impl LexicalIndex {
    /// Posting metadata for one term, or `None` if the term is absent.
    ///
    /// In the blocked layout this costs at most one block read: `first_term` is
    /// a sparse index over the sorted table, so the block that could contain a
    /// term is the last one whose first term is not greater than it.
    fn lookup(&self, reader: &PackReader, term: &str) -> Result<Option<PostingMeta>> {
        match self {
            Self::Inline { terms, .. } => Ok(terms.get(term).cloned()),
            Self::Blocked {
                terms_section,
                blocks,
                term_cache,
                ..
            } => {
                let Some(index) = sparse_block_for_term(&blocks.dictionary, term) else {
                    return Ok(None);
                };
                let cached = term_cache
                    .lock()
                    .map_err(|_| AdyarError::Search("term block cache poisoned".into()))?
                    .get(&index)
                    .cloned();
                let block = match cached {
                    Some(block) => block,
                    None => {
                        let logical =
                            read_index_block(reader, *terms_section, &blocks.dictionary[index])?;
                        let parsed: DictionaryBlock = serde_json::from_slice(&logical)?;
                        let shared = Arc::new(parsed.terms);
                        term_cache
                            .lock()
                            .map_err(|_| AdyarError::Search("term block cache poisoned".into()))?
                            .insert(index, Arc::clone(&shared));
                        shared
                    }
                };
                Ok(block.get(term).cloned())
            }
        }
    }

    /// The exact posting-list bytes for `meta`.
    fn posting_bytes(&self, reader: &PackReader, meta: &PostingMeta) -> Result<Vec<u8>> {
        let start = meta.offset;
        let end = start
            .checked_add(meta.length)
            .ok_or_else(|| AdyarError::InvalidFormat("posting range overflow".into()))?;
        match self {
            Self::Inline { postings, .. } => {
                let start = usize::try_from(start).map_err(|_| {
                    AdyarError::InvalidFormat("posting offset exceeds address space".into())
                })?;
                let end = usize::try_from(end).map_err(|_| {
                    AdyarError::InvalidFormat("posting end exceeds address space".into())
                })?;
                postings.get(start..end).map(<[u8]>::to_vec).ok_or_else(|| {
                    AdyarError::InvalidFormat("posting list exceeds postings section".into())
                })
            }
            Self::Blocked {
                postings_section,
                blocks,
                postings_starts,
                postings_cache,
                ..
            } => {
                let mut out = Vec::with_capacity(meta.length as usize);
                for (index, block) in blocks.postings.iter().enumerate() {
                    let block_start = postings_starts[index];
                    let block_end = block_start + block.logical_length;
                    if block_end <= start || block_start >= end {
                        continue;
                    }
                    let cached = postings_cache
                        .lock()
                        .map_err(|_| AdyarError::Search("postings block cache poisoned".into()))?
                        .get(&index)
                        .cloned();
                    let bytes = match cached {
                        Some(bytes) => bytes,
                        None => {
                            let shared =
                                Arc::new(read_index_block(reader, *postings_section, block)?);
                            postings_cache
                                .lock()
                                .map_err(|_| {
                                    AdyarError::Search("postings block cache poisoned".into())
                                })?
                                .insert(index, Arc::clone(&shared));
                            shared
                        }
                    };
                    let from = start.saturating_sub(block_start) as usize;
                    let to = (end.min(block_end) - block_start) as usize;
                    out.extend_from_slice(bytes.get(from..to).ok_or_else(|| {
                        AdyarError::InvalidFormat("posting range exceeds its block".into())
                    })?);
                }
                if out.len() as u64 != meta.length {
                    return Err(AdyarError::InvalidFormat(
                        "posting list is not covered by the postings block table".into(),
                    ));
                }
                Ok(out)
            }
        }
    }
}

/// How a pack's passage record table is reached. Same split as [`LexicalIndex`]:
/// both variants answer the same questions, and differ only in whether
/// answering costs the whole table or one block.
enum RecordTable {
    /// Passage index format 1: records were read in full at open.
    Inline { records: Vec<StoredRecord> },
    /// Passage index format 2: fixed-width blocks, fetched and verified on demand.
    Blocked {
        section: u32,
        index: RecordBlockIndex,
        count: usize,
        cache: Mutex<HashMap<usize, Arc<Vec<u8>>>>,
        id_cache: Mutex<HashMap<usize, Arc<Vec<u8>>>>,
    },
}

impl RecordTable {
    fn len(&self) -> usize {
        match self {
            Self::Inline { records } => records.len(),
            Self::Blocked { count, .. } => *count,
        }
    }

    /// The record at a passage ordinal. In the blocked layout the containing
    /// block is arithmetic, not a search: records are fixed width and uniformly
    /// packed, which is the whole reason for the encoding.
    fn get(&self, reader: &PackReader, ordinal: usize) -> Result<StoredRecord> {
        match self {
            Self::Inline { records } => records.get(ordinal).cloned().ok_or_else(|| {
                AdyarError::InvalidFormat(format!("passage ordinal {ordinal} is out of range"))
            }),
            Self::Blocked {
                section,
                index,
                count,
                cache,
                ..
            } => {
                if ordinal >= *count {
                    return Err(AdyarError::InvalidFormat(format!(
                        "passage ordinal {ordinal} is out of range"
                    )));
                }
                let per_block = index.per_block as usize;
                let block_index = ordinal / per_block;
                let within = (ordinal % per_block) * index.stride as usize;
                let block = index.records.get(block_index).ok_or_else(|| {
                    AdyarError::InvalidFormat("passage record block is missing".into())
                })?;
                let bytes = cached_block(reader, *section, block, cache, block_index)?;
                let end = within + index.stride as usize;
                let raw = bytes.get(within..end).ok_or_else(|| {
                    AdyarError::InvalidFormat("passage record exceeds its block".into())
                })?;
                Ok(StoredRecord {
                    // Not stored in a format-2 record; see build.rs RECORD_STRIDE.
                    // Callers that need it read the payload or the id index.
                    id: String::new(),
                    block: u32::from_le_bytes(raw[0..4].try_into().unwrap()),
                    offset: u32::from_le_bytes(raw[4..8].try_into().unwrap()),
                    length: u32::from_le_bytes(raw[8..12].try_into().unwrap()),
                })
            }
        }
    }

    /// The ordinal for a passage id, or `None` if the pack does not carry it.
    fn ordinal_of(&self, reader: &PackReader, passage_id: &str) -> Result<Option<usize>> {
        match self {
            Self::Inline { records } => {
                Ok(records.iter().position(|record| record.id == passage_id))
            }
            Self::Blocked {
                section,
                index,
                id_cache,
                ..
            } => {
                let Ok(target) = hex::decode(passage_id) else {
                    return Ok(None);
                };
                if target.len() != 32 {
                    return Ok(None);
                }
                let Some(block_index) = sparse_block_for_term(&index.ids, passage_id) else {
                    return Ok(None);
                };
                let block = &index.ids[block_index];
                let bytes = cached_block(reader, *section, block, id_cache, block_index)?;
                // Entries are fixed width and sorted, so this is a binary search
                // over the block rather than a scan.
                let stride = ID_ENTRY_STRIDE;
                if bytes.len() % stride != 0 {
                    return Err(AdyarError::InvalidFormat(
                        "passage id index block is not a whole number of entries".into(),
                    ));
                }
                let entries = bytes.len() / stride;
                let (mut low, mut high) = (0_usize, entries);
                while low < high {
                    let middle = (low + high) / 2;
                    let at = middle * stride;
                    match bytes[at..at + 32].cmp(target.as_slice()) {
                        std::cmp::Ordering::Less => low = middle + 1,
                        std::cmp::Ordering::Greater => high = middle,
                        std::cmp::Ordering::Equal => {
                            let ordinal =
                                u32::from_le_bytes(bytes[at + 32..at + 36].try_into().unwrap());
                            return Ok(Some(ordinal as usize));
                        }
                    }
                }
                Ok(None)
            }
        }
    }
}

/// Width of one entry in the id-sorted region: 32-byte id plus a u32 ordinal.
const ID_ENTRY_STRIDE: usize = 36;

/// Fetch a block through a cache keyed by block index.
fn cached_block(
    reader: &PackReader,
    section: u32,
    block: &IndexBlock,
    cache: &Mutex<HashMap<usize, Arc<Vec<u8>>>>,
    key: usize,
) -> Result<Arc<Vec<u8>>> {
    let hit = cache
        .lock()
        .map_err(|_| AdyarError::Search("index block cache poisoned".into()))?
        .get(&key)
        .cloned();
    if let Some(hit) = hit {
        return Ok(hit);
    }
    let shared = Arc::new(read_index_block(reader, section, block)?);
    cache
        .lock()
        .map_err(|_| AdyarError::Search("index block cache poisoned".into()))?
        .insert(key, Arc::clone(&shared));
    Ok(shared)
}

/// Validate the record block tables against the section directory before any
/// block is fetched, and check that they describe exactly the declared number
/// of passages. Mirrors [`validate_lexical_blocks`].
fn validate_record_blocks(
    reader: &PackReader,
    index: &RecordBlockIndex,
    passage_count: usize,
) -> Result<()> {
    if index.stride == 0 || index.per_block == 0 {
        return Err(AdyarError::InvalidFormat(
            "record block index has a zero stride or block size".into(),
        ));
    }
    let entry = reader
        .first_entry(SectionType::PassageRecords)
        .ok_or_else(|| AdyarError::InvalidFormat("passage records section missing".into()))?;

    let mut cursor = 0_u64;
    let mut record_bytes = 0_u64;
    let mut id_bytes = 0_u64;
    for (label, blocks) in [("record", &index.records), ("id", &index.ids)] {
        for block in blocks {
            if block.offset != cursor {
                return Err(AdyarError::InvalidFormat(format!(
                    "{label} blocks are not contiguous"
                )));
            }
            if block.stored_length == 0 || block.logical_length == 0 {
                return Err(AdyarError::InvalidFormat(format!(
                    "{label} block is empty"
                )));
            }
            if hex::decode(&block.hash).map(|h| h.len()) != Ok(32) {
                return Err(AdyarError::InvalidFormat(format!(
                    "{label} block has an invalid hash"
                )));
            }
            cursor = cursor.checked_add(block.stored_length).ok_or_else(|| {
                AdyarError::InvalidFormat("record block offset overflow".into())
            })?;
            if label == "record" {
                record_bytes += block.logical_length;
            } else {
                id_bytes += block.logical_length;
            }
        }
    }
    if cursor != entry.stored_length {
        return Err(AdyarError::InvalidFormat(
            "record blocks do not cover their section exactly".into(),
        ));
    }
    // Both regions must describe exactly the corpus: a short table would make
    // some ordinals silently unreachable rather than fail.
    if record_bytes != passage_count as u64 * index.stride as u64 {
        return Err(AdyarError::InvalidFormat(
            "record blocks do not cover every passage".into(),
        ));
    }
    if id_bytes != passage_count as u64 * ID_ENTRY_STRIDE as u64 {
        return Err(AdyarError::InvalidFormat(
            "id index does not cover every passage".into(),
        ));
    }
    let mut previous: Option<&str> = None;
    for block in &index.ids {
        let first = block.first_term.as_deref().ok_or_else(|| {
            AdyarError::InvalidFormat("id index block is missing its first id".into())
        })?;
        if let Some(previous) = previous
            && first <= previous
        {
            return Err(AdyarError::InvalidFormat(
                "id index block first ids must be strictly increasing".into(),
            ));
        }
        previous = Some(first);
    }
    Ok(())
}

/// Which lexical layout a pack uses, resolved once at open.
enum LexicalLayout {
    Inline,
    Blocked {
        terms_section: u32,
        postings_section: u32,
        blocks: LexicalBlockIndex,
    },
}

/// Validate the lexical block tables and return each postings block's logical
/// start offset.
///
/// Everything here is checked against the section directory, which is already
/// authenticated by the artifact root, so a malformed table is rejected before
/// any block is fetched. Blocks must tile their section exactly: contiguous,
/// in order, covering it with no gap and no overlap. Dictionary blocks must
/// additionally carry strictly increasing `first_term` values, since the sparse
/// search assumes that ordering.
fn validate_lexical_blocks(reader: &PackReader, blocks: &LexicalBlockIndex) -> Result<Vec<u64>> {
    fn tile(
        reader: &PackReader,
        section: SectionType,
        list: &[IndexBlock],
    ) -> Result<(Vec<u64>, u64)> {
        let entry = reader.first_entry(section).ok_or_else(|| {
            AdyarError::InvalidFormat(format!("{} section missing", section.name()))
        })?;
        let mut starts = Vec::with_capacity(list.len());
        let mut stored_cursor = 0_u64;
        let mut logical_cursor = 0_u64;
        for block in list {
            if block.offset != stored_cursor {
                return Err(AdyarError::InvalidFormat(format!(
                    "{} blocks are not contiguous",
                    section.name()
                )));
            }
            if block.stored_length == 0 || block.logical_length == 0 {
                return Err(AdyarError::InvalidFormat(format!(
                    "{} block is empty",
                    section.name()
                )));
            }
            if hex::decode(&block.hash).map(|h| h.len()) != Ok(32) {
                return Err(AdyarError::InvalidFormat(format!(
                    "{} block has an invalid hash",
                    section.name()
                )));
            }
            starts.push(logical_cursor);
            stored_cursor = stored_cursor
                .checked_add(block.stored_length)
                .ok_or_else(|| AdyarError::InvalidFormat("index block offset overflow".into()))?;
            logical_cursor = logical_cursor
                .checked_add(block.logical_length)
                .ok_or_else(|| {
                    AdyarError::InvalidFormat("index block logical overflow".into())
                })?;
        }
        if stored_cursor != entry.stored_length {
            return Err(AdyarError::InvalidFormat(format!(
                "{} blocks do not cover their section exactly",
                section.name()
            )));
        }
        Ok((starts, logical_cursor))
    }

    tile(reader, SectionType::LexicalTerms, &blocks.dictionary)?;
    let (postings_starts, _) = tile(reader, SectionType::LexicalPostings, &blocks.postings)?;

    let mut previous: Option<&str> = None;
    for block in &blocks.dictionary {
        let first = block.first_term.as_deref().ok_or_else(|| {
            AdyarError::InvalidFormat("dictionary block is missing its first term".into())
        })?;
        if let Some(previous) = previous
            && first <= previous
        {
            return Err(AdyarError::InvalidFormat(
                "dictionary block first terms must be strictly increasing".into(),
            ));
        }
        previous = Some(first);
    }
    Ok(postings_starts)
}

/// The one dictionary block that can contain `term`: the last block whose
/// `first_term` is less than or equal to it. Returns `None` when the term sorts
/// before every block, which means it is absent.
fn sparse_block_for_term(blocks: &[IndexBlock], term: &str) -> Option<usize> {
    let mut candidate = None;
    for (index, block) in blocks.iter().enumerate() {
        match block.first_term.as_deref() {
            Some(first) if first <= term => candidate = Some(index),
            Some(_) => break,
            None => return None,
        }
    }
    candidate
}

fn required_profile_section(reader: &PackReader, section_type: SectionType) -> Result<u32> {
    let entry = reader.first_entry(section_type).ok_or_else(|| {
        AdyarError::InvalidFormat(format!(
            "required {} section is missing",
            section_type.name()
        ))
    })?;
    // The postings section carries its own schema version: format 2 is the
    // block-addressable layout. Every other profile section is v1 only.
    let accepted: &[u16] = if section_type == SectionType::LexicalPostings {
        crate::format::SUPPORTED_LEXICAL_FORMAT_VERSIONS
    } else {
        &[1]
    };
    if !entry.required() || !accepted.contains(&entry.format_version) {
        return Err(AdyarError::InvalidFormat(format!(
            "{} section is not a required profile section at a supported format version",
            section_type.name()
        )));
    }
    Ok(entry.section_id)
}

fn select_ivf_ordinals(
    index: &IvfIndex,
    query: &[f32],
    vector_count: usize,
    dimensions: usize,
    requested_probes: usize,
) -> Result<Vec<usize>> {
    if index.algorithm != "ivf-flat-v1"
        || index.distance != "dot"
        || index.dimensions as usize != dimensions
        || index.centroids.is_empty()
        || index.centroids.len() != index.lists.len()
        || index.default_probes == 0
        || index.default_probes as usize > index.centroids.len()
    {
        return Err(AdyarError::InvalidFormat(
            "invalid or unsupported IVF vector index".into(),
        ));
    }
    let mut seen = vec![false; vector_count];
    for (cluster, (centroid, list)) in index.centroids.iter().zip(&index.lists).enumerate() {
        if centroid.len() != dimensions || centroid.iter().any(|value| !value.is_finite()) {
            return Err(AdyarError::InvalidFormat(format!(
                "IVF centroid {cluster} has invalid values or dimensions"
            )));
        }
        for ordinal in list {
            let ordinal = *ordinal as usize;
            if ordinal >= vector_count || std::mem::replace(&mut seen[ordinal], true) {
                return Err(AdyarError::InvalidFormat(
                    "IVF lists contain duplicate or out-of-range ordinals".into(),
                ));
            }
        }
    }
    if seen.iter().any(|value| !value) {
        return Err(AdyarError::InvalidFormat(
            "IVF lists do not cover every vector".into(),
        ));
    }
    let mut centroid_scores: Vec<_> = index
        .centroids
        .iter()
        .enumerate()
        .map(|(cluster, centroid)| {
            let score = query
                .iter()
                .zip(centroid)
                .map(|(left, right)| *left as f64 * *right as f64)
                .sum::<f64>();
            (cluster, score)
        })
        .collect();
    centroid_scores.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    let probes = requested_probes.min(index.centroids.len());
    let mut ordinals = Vec::new();
    for (cluster, _) in centroid_scores.into_iter().take(probes) {
        ordinals.extend(index.lists[cluster].iter().map(|ordinal| *ordinal as usize));
    }
    Ok(ordinals)
}

fn technical_term_boost(term: &str) -> f64 {
    if term.chars().any(|character| character.is_ascii_digit())
        || term
            .chars()
            .any(|character| matches!(character, '_' | '-' | '.' | ':' | '/' | '@' | '#'))
    {
        3.0
    } else {
        1.0
    }
}

/// Whether a character is General_Category `L*` or `N*`, which is what
/// FORMAT-v3 §6.1 means by "Unicode alphanumeric (`\p{L}` or `\p{N}`)".
///
/// Deliberately not `char::is_alphanumeric()`. That is `Alphabetic | N`, and
/// the Alphabetic property additionally includes Other_Alphabetic — most
/// combining marks among them. The two differ on real text: U+0903 DEVANAGARI
/// SIGN VISARGA is `Mc`, so the specification trims it from a token edge, while
/// `is_alphanumeric()` reports true and keeps it. This builder indexed
/// `रामः` while the browser runtime, which uses `\p{L}\p{N}` directly, queried
/// for `राम` — the same query against the same pack returned one result from
/// the CLI and none from the browser. The hybrid-parity smoke never saw it
/// because its corpus is English.
fn is_letter_or_number(character: char) -> bool {
    use unicode_general_category::{GeneralCategory as C, get_general_category};
    matches!(
        get_general_category(character),
        C::UppercaseLetter
            | C::LowercaseLetter
            | C::TitlecaseLetter
            | C::ModifierLetter
            | C::OtherLetter
            | C::DecimalNumber
            | C::LetterNumber
            | C::OtherNumber
    )
}

pub fn tokenize(text: &str) -> Vec<String> {
    let normalized: String = text.nfkc().flat_map(char::to_lowercase).collect();
    normalized
        .split_whitespace()
        .filter_map(|raw| {
            let token = raw.trim_matches(|character: char| {
                !is_letter_or_number(character)
                    && !matches!(character, '_' | '-' | '.' | ':' | '/' | '@' | '#')
            });
            (!token.is_empty()).then(|| token.to_string())
        })
        .collect()
}

pub fn decode_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut value = 0_u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| AdyarError::InvalidFormat("truncated varint".into()))?;
        *cursor += 1;
        if shift == 63 && byte > 1 {
            return Err(AdyarError::InvalidFormat("varint overflow".into()));
        }
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(AdyarError::InvalidFormat("non-terminating varint".into()))
}

fn decode_postings(bytes: &[u8], expected_count: usize) -> Result<Vec<(usize, u32)>> {
    let mut postings = Vec::with_capacity(expected_count);
    let mut cursor = 0;
    let mut ordinal = 0_u64;
    for index in 0..expected_count {
        let delta = decode_varint(bytes, &mut cursor)?;
        if index != 0 && delta == 0 {
            return Err(AdyarError::InvalidFormat(
                "posting ordinals must be strictly increasing".into(),
            ));
        }
        ordinal = if index == 0 {
            delta
        } else {
            ordinal
                .checked_add(delta)
                .ok_or_else(|| AdyarError::InvalidFormat("posting ordinal overflow".into()))?
        };
        let frequency = decode_varint(bytes, &mut cursor)?;
        let ordinal = usize::try_from(ordinal)
            .map_err(|_| AdyarError::InvalidFormat("posting ordinal exceeds platform".into()))?;
        let frequency = u32::try_from(frequency)
            .map_err(|_| AdyarError::InvalidFormat("term frequency exceeds u32".into()))?;
        if frequency == 0 {
            return Err(AdyarError::InvalidFormat("zero term frequency".into()));
        }
        postings.push((ordinal, frequency));
    }
    if cursor != bytes.len() {
        return Err(AdyarError::InvalidFormat(
            "posting list has trailing bytes or incorrect document frequency".into(),
        ));
    }
    Ok(postings)
}

fn sort_candidates(candidates: &mut [RankedCandidate]) {
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.ordinal.cmp(&right.ordinal))
    });
}

/// Fuse lexical and vector candidates into one ranking.
///
/// This is not reciprocal-rank fusion. RRF scores a document by its rank in each
/// list and sums those terms, which values presence in both lists at
/// approximately twice a top position in one. That weighting is appropriate when
/// both retrievers carry signal and harmful when either does not.
///
/// Measured on the hard-negative corpus (`evals/corpora/`), RRF ranked a passage
/// placed 47th by lexical retrieval above the passage placed 1st by vector
/// retrieval, and excluded a vector-rank-1 passage from the top 8. Hybrid scored
/// 0.556 recall@5 against vector-only at 0.794.
///
/// RRF discards score magnitude, which is the signal distinguishing a retriever
/// that found a match from one that did not. Each mode is therefore placed on a
/// comparable absolute scale and summed:
///
/// * Lexical scores divide by the query's maximum achievable BM25 score, the
///   total `idf * (k1 + 1)` across query terms. The ratio expresses the fraction
///   of the query's achievable score a passage accounts for, so a passage
///   matching only corpus-common terms scores near zero regardless of rank.
/// * Vector scores are dot products of normalized embeddings, already cosine
///   similarities on a fixed scale.
///
/// This raised hybrid to 0.730 recall@5 and improved both strata: 0.286 to 0.571
/// where lexical retrieval has no signal, 0.893 to 0.929 where it does.
///
/// Two alternatives were measured and rejected:
///
/// * Min-max positioning within each list, weighted by a per-query mode
///   confidence: identical recall, marginally higher MRR, and it makes ranking
///   depend on `candidate_depth`, since the bottom of the candidate list shifts
///   the normalization. The same query at a different depth would rank
///   differently.
/// * Reduced lexical weight: a sweep converges to vector-only rather than
///   exceeding it. At weight 0.25 the technical stratum falls to 0.821, equal to
///   vector-only, indicating lexical has been discarded rather than balanced.
///
/// Hybrid therefore remains disabled by default. On a query distribution where
/// most queries are not lexically answerable, its gain where lexical retrieval
/// contributes (+0.108 over 28 queries) is smaller than its loss where lexical
/// retrieval misleads (-0.200 over 35 queries). No static weighting resolves
/// this: the selection would have to be per-query, and lexical scores do not
/// carry sufficient information for it. A passage accounting for 27% of a
/// query's achievable score is indistinguishable on that basis from a correct
/// one.
fn fuse_candidates(
    lexical: &[RankedCandidate],
    lexical_achievable: f64,
    vector: &[RankedCandidate],
    options: &SearchOptions,
    effective_mode: SearchMode,
) -> Vec<(usize, FusionCandidate)> {
    // A query whose terms are all absent from the dictionary has nothing
    // achievable, and every lexical score is zero anyway.
    let achievable = if lexical_achievable > 0.0 {
        lexical_achievable
    } else {
        1.0
    };
    let mut candidates = BTreeMap::<usize, FusionCandidate>::new();
    for (index, candidate) in lexical.iter().enumerate() {
        let rank = index + 1;
        let entry = candidates.entry(candidate.ordinal).or_default();
        entry.lexical_score = Some(candidate.score);
        entry.lexical_rank = Some(rank);
        entry.fused_score += match effective_mode {
            SearchMode::Lexical => candidate.score,
            _ => options.lexical_weight * (candidate.score / achievable).clamp(0.0, 1.0),
        };
    }
    for (index, candidate) in vector.iter().enumerate() {
        let rank = index + 1;
        let entry = candidates.entry(candidate.ordinal).or_default();
        entry.vector_score = Some(candidate.score);
        entry.vector_rank = Some(rank);
        entry.fused_score += match effective_mode {
            SearchMode::Vector => candidate.score,
            // Cosine below zero points away from the query: no evidence, not
            // evidence against.
            _ => options.vector_weight * candidate.score.clamp(0.0, 1.0),
        };
    }
    let mut output: Vec<_> = candidates.into_iter().collect();
    output.sort_by(|left, right| {
        right
            .1
            .fused_score
            .partial_cmp(&left.1.fused_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    output
}

fn citation_url(document: &Document, passage: &Passage) -> Option<String> {
    let base = document.url.clone()?;
    match &passage.anchor {
        Some(anchor) if !anchor.is_empty() && !base.contains('#') => {
            Some(format!("{base}#{anchor}"))
        }
        _ => Some(base),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn technical_tokens_remain_whole() {
        assert_eq!(
            tokenize("AP-104 std::move useEffect foo_bar package.module"),
            vec![
                "ap-104",
                "std::move",
                "useeffect",
                "foo_bar",
                "package.module"
            ]
        );
    }

    #[test]
    fn rejects_non_terminating_varint() {
        let mut cursor = 0;
        assert!(decode_varint(&[0x80; 10], &mut cursor).is_err());
    }

    #[test]
    fn ranking_ties_are_deterministic() {
        let mut candidates = vec![
            RankedCandidate {
                ordinal: 2,
                score: 1.0,
            },
            RankedCandidate {
                ordinal: 1,
                score: 1.0,
            },
        ];
        sort_candidates(&mut candidates);
        assert_eq!(candidates[0].ordinal, 1);
    }
}
