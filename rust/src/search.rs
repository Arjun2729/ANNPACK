use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::conformance::{ConformanceReport, inspect_conformance_with_manifest};
use crate::error::{AnnpackError, Result};
use crate::format::{PackReader, SectionType};
use crate::model::{
    Document, IvfIndex, LexicalDictionary, Manifest, Passage, StoredPassageIndex,
    VectorProfileSection,
};
use crate::reader::{FileReader, ReadAt};
use crate::signing::verify_signatures;

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;
const RRF_CONSTANT: f64 = 60.0;
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

/// ANN-10 profile request. Which advertised `retrieval_profiles` entry (if any)
/// the runtime should activate for this search.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ProfileRequest {
    /// Default. Core lexical only — byte-identical to Core; never activates a
    /// vector or derived (expansion/splade) profile. This is what keeps derived
    /// retrieval off by default (ANN-7/ANN-8 policy) even for a fat pack.
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
/// splade) is selected via ANN-10. Weight *calibration* is deliberately out of
/// scope for the selection contract; this is a neutral default and the effective
/// value is always reported in `SearchResponse.profile_selection`.
const DERIVED_PROFILE_WEIGHT: f64 = 1.0;

/// Capabilities the reference runtime can actually EXECUTE during search. Note
/// `anchor-relative` is intentionally absent: it is decode-only, never a search
/// path, so anchor profiles are never selected.
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
    /// ANN-7 expansion overlay weight. Defaults to 0.0: no effect, Core results.
    /// Superseded by ANN-10 profile selection on a fat pack (see `profile`).
    pub expansion_weight: f64,
    /// ANN-8 vocabulary overlay weight. Defaults to 0.0: no effect, Core results.
    /// Superseded by ANN-10 profile selection on a fat pack (see `profile`).
    pub splade_weight: f64,
    /// ANN-10 profile request. On a fat pack this determines the effective mode
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

/// The outcome of ANN-10 profile selection, always returned on `SearchResponse`
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub pack: SearchPackIdentity,
    pub query: String,
    pub requested_mode: SearchMode,
    pub effective_mode: SearchMode,
    pub results: Vec<SearchHit>,
    /// ANN-10: which profile was selected, why, and its effective weights.
    /// Always present so selection is auditable without enabling debug.
    pub profile_selection: ProfileSelection,
    pub diagnostics: Option<SearchDiagnostics>,
}

/// Deterministic ANN-10 profile selection. Pure function of the pack's advertised
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
        p.requires
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
    passage_by_id: HashMap<String, usize>,
    dictionary: LexicalDictionary,
    postings: Vec<u8>,
    conformance: ConformanceReport,
    publisher: PublisherEvidence,
    passage_block_cache: Mutex<HashMap<u32, Arc<Vec<u8>>>>,
}

/// A validated ANN-7/ANN-8 term overlay: matching-only, never citable.
#[derive(Debug, Clone)]
struct LoadedOverlay {
    kind: String,
    /// Dequantization scale: 1.0 for expansion, `vocabulary.scale` for splade.
    scale: f64,
    terms: HashMap<String, Vec<(u32, u32)>>,
}

/// A validated ANN-9 anchor representation. Research-grade and unvalidated for
/// retrieval quality; decode and scoring path only.
#[derive(Debug, Clone)]
pub struct LoadedAnchors {
    pub space_id: String,
    pub metric: String,
    pub scale: f64,
    pub anchors: Vec<String>,
    pub coordinates: Vec<Vec<i32>>,
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
            return Err(AnnpackError::InvalidFormat(format!(
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
        let passage_index: StoredPassageIndex =
            serde_json::from_slice(&reader.read_section(passage_index_entry)?)?;
        let dictionary: LexicalDictionary =
            serde_json::from_slice(&reader.read_section(dictionary_entry)?)?;
        let postings = reader.read_section(postings_entry)?;
        if documents.len() != manifest.document_count as usize {
            return Err(AnnpackError::InvalidFormat(
                "document section and manifest counts disagree".into(),
            ));
        }
        if passage_index.records.len() != dictionary.passage_lengths.len()
            || passage_index.records.len() != manifest.passage_count as usize
        {
            return Err(AnnpackError::InvalidFormat(
                "passage index, lexical index, and manifest counts disagree".into(),
            ));
        }
        if passage_index.codec != "deflate-zlib" {
            return Err(AnnpackError::Unsupported(format!(
                "passage block codec {:?}",
                passage_index.codec
            )));
        }
        if !dictionary.average_passage_length.is_finite() || dictionary.average_passage_length < 0.0
        {
            return Err(AnnpackError::InvalidFormat(
                "lexical index has an invalid average passage length".into(),
            ));
        }
        let passage_data_entry = reader
            .first_entry(SectionType::PassageData)
            .ok_or_else(|| AnnpackError::InvalidFormat("passage data section is missing".into()))?;
        if passage_data_entry.codec != crate::format::Codec::None {
            return Err(AnnpackError::InvalidFormat(
                "passage data must use independently compressed blocks".into(),
            ));
        }
        let mut block_ranges = Vec::with_capacity(passage_index.blocks.len());
        for (index, block) in passage_index.blocks.iter().enumerate() {
            if block.logical_length > MAX_PASSAGE_BLOCK_LOGICAL_SIZE {
                return Err(AnnpackError::InvalidFormat(format!(
                    "passage block {index} exceeds the logical size limit"
                )));
            }
            if block.stored_length == 0 && block.logical_length != 0 {
                return Err(AnnpackError::InvalidFormat(format!(
                    "passage block {index} has no stored bytes"
                )));
            }
            if block.logical_length
                > block
                    .stored_length
                    .saturating_mul(MAX_PASSAGE_BLOCK_COMPRESSION_RATIO)
                    .max(MAX_PASSAGE_BLOCK_LOGICAL_SIZE)
            {
                return Err(AnnpackError::InvalidFormat(format!(
                    "passage block {index} exceeds the compression-ratio limit"
                )));
            }
            let hash = hex::decode(&block.hash).map_err(|_| {
                AnnpackError::InvalidFormat(format!("passage block {index} has an invalid hash"))
            })?;
            if hash.len() != 32 {
                return Err(AnnpackError::InvalidFormat(format!(
                    "passage block {index} has an invalid hash length"
                )));
            }
            let end = block
                .offset
                .checked_add(block.stored_length)
                .ok_or_else(|| {
                    AnnpackError::InvalidFormat("passage block range overflow".into())
                })?;
            if end > passage_data_entry.stored_length {
                return Err(AnnpackError::InvalidFormat(format!(
                    "passage block {index} exceeds the passage data section"
                )));
            }
            block_ranges.push((block.offset, end, index));
        }
        block_ranges.sort_by_key(|range| range.0);
        for pair in block_ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(AnnpackError::InvalidFormat(format!(
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
                    AnnpackError::InvalidFormat(format!(
                        "passage {} references missing block {}",
                        record.id, record.block
                    ))
                })?;
            let end = (record.offset as u64)
                .checked_add(record.length as u64)
                .ok_or_else(|| {
                    AnnpackError::InvalidFormat("passage record range overflow".into())
                })?;
            if end > block.logical_length {
                return Err(AnnpackError::InvalidFormat(format!(
                    "passage {} exceeds logical block {}",
                    record.id, record.block
                )));
            }
            let id = hex::decode(&record.id).map_err(|_| {
                AnnpackError::InvalidFormat(format!(
                    "passage record has an invalid ID {:?}",
                    record.id
                ))
            })?;
            if id.len() != 32 {
                return Err(AnnpackError::InvalidFormat(format!(
                    "passage record has an invalid ID length {:?}",
                    record.id
                )));
            }
        }
        let mut posting_cursor = 0_u64;
        for (term, meta) in &dictionary.terms {
            if meta.offset != posting_cursor || meta.document_frequency == 0 {
                return Err(AnnpackError::InvalidFormat(format!(
                    "posting metadata for term {term:?} is non-canonical"
                )));
            }
            let end = meta.offset.checked_add(meta.length).ok_or_else(|| {
                AnnpackError::InvalidFormat("posting metadata range overflow".into())
            })?;
            let start = usize::try_from(meta.offset).map_err(|_| {
                AnnpackError::InvalidFormat("posting offset exceeds address space".into())
            })?;
            let end_usize = usize::try_from(end).map_err(|_| {
                AnnpackError::InvalidFormat("posting end exceeds address space".into())
            })?;
            let list = postings.get(start..end_usize).ok_or_else(|| {
                AnnpackError::InvalidFormat(format!(
                    "posting list for term {term:?} exceeds its section"
                ))
            })?;
            for (ordinal, _) in decode_postings(list, meta.document_frequency as usize)? {
                if ordinal >= passage_index.records.len() {
                    return Err(AnnpackError::InvalidFormat(format!(
                        "posting list for term {term:?} has an invalid passage ordinal"
                    )));
                }
            }
            posting_cursor = end;
        }
        if posting_cursor != postings.len() as u64 {
            return Err(AnnpackError::InvalidFormat(
                "lexical dictionary does not cover the postings section exactly".into(),
            ));
        }
        let mut documents_by_id = HashMap::new();
        for (index, document) in documents.iter().enumerate() {
            if documents_by_id.insert(document.id.clone(), index).is_some() {
                return Err(AnnpackError::InvalidFormat(format!(
                    "duplicate document ID {}",
                    document.id
                )));
            }
        }
        let mut passage_by_id = HashMap::new();
        for (index, record) in passage_index.records.iter().enumerate() {
            if passage_by_id.insert(record.id.clone(), index).is_some() {
                return Err(AnnpackError::InvalidFormat(format!(
                    "duplicate passage ID {}",
                    record.id
                )));
            }
        }
        Ok(Self {
            reader,
            manifest,
            documents,
            documents_by_id,
            passage_index,
            passage_by_id,
            dictionary,
            postings,
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

    /// The validated ANN-9 anchor representation, if the pack carries one.
    /// Reads and validates the anchor sections on demand.
    /// Decode-only access to a pack's shipped anchor set and coordinates.
    ///
    /// ANN-9 relative-coordinate *retrieval* was withdrawn (it is dominated by
    /// raw same-dimension comparison and by anchor-supervised adapters), so there
    /// is no anchor scoring path. This accessor is retained because the anchor
    /// texts are the supervision an anchor-supervised cross-model adapter needs.
    pub fn anchors(&self) -> Result<Option<LoadedAnchors>> {
        self.load_anchors()
    }

    pub fn get_passage(&self, passage_id: &str) -> Result<Passage> {
        let ordinal = self
            .passage_by_id
            .get(passage_id)
            .copied()
            .ok_or_else(|| AnnpackError::Search(format!("unknown passage ID {passage_id}")))?;
        self.load_passage(ordinal)
    }

    pub fn passages(&self) -> Result<Vec<Passage>> {
        (0..self.passage_index.records.len())
            .map(|ordinal| self.load_passage(ordinal))
            .collect()
    }

    pub fn search(&self, query: &str, options: &SearchOptions) -> Result<SearchResponse> {
        if query.trim().is_empty() {
            return Err(AnnpackError::InvalidInput("query must not be empty".into()));
        }
        if options.limit == 0 || options.limit > MAX_RESULTS {
            return Err(AnnpackError::InvalidInput(format!(
                "result limit must be between 1 and {MAX_RESULTS}"
            )));
        }
        let query_terms = tokenize(query);
        if query_terms.len() > MAX_QUERY_TERMS {
            return Err(AnnpackError::InvalidInput(format!(
                "query contains more than {MAX_QUERY_TERMS} terms"
            )));
        }

        // ANN-10: resolve the effective execution config from profile selection.
        // On a non-fat pack this is a no-op and the raw options pass through.
        let selection = select_profile(
            &self.manifest.retrieval_profiles,
            &options.profile,
            options.mode,
            options.expansion_weight,
            options.splade_weight,
            options.query_vector.is_some(),
        );
        let eff_mode = selection.effective_mode;
        let eff_expansion = selection.effective_expansion_weight;
        let eff_splade = selection.effective_splade_weight;

        let lexical = match eff_mode {
            SearchMode::Vector => Vec::new(),
            SearchMode::Lexical | SearchMode::Hybrid => self.lexical_candidates(
                &query_terms,
                options.candidate_depth.max(options.limit),
                eff_expansion,
                eff_splade,
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
                return Err(AnnpackError::InvalidInput(
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

        let fused = fuse_candidates(&lexical, &vector, options, effective_mode);
        let mut results = Vec::new();
        for (rank, (ordinal, candidate)) in fused.into_iter().take(options.limit).enumerate() {
            let passage = self.load_passage(ordinal)?;
            let document_index =
                self.documents_by_id
                    .get(&passage.document_id)
                    .ok_or_else(|| {
                        AnnpackError::InvalidFormat(format!(
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
                AnnpackError::InvalidFormat(format!(
                    "passage {} references unknown document {}",
                    passage.id, passage.document_id
                ))
            })?;
        let document = &self.documents[*document_index];
        let canonical_url = citation_url(document, passage);
        let encoded = serde_json::to_vec(passage)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ANNPACK3-PASSAGE-EVIDENCE\0");
        hasher.update(&encoded);
        Ok(EvidenceEnvelope {
            schema: "annpack-evidence-v1".to_string(),
            pack: format!("{}@{}", self.manifest.name, self.manifest.version),
            pack_root: self.reader.root_hex(),
            source_revision: self.manifest.source_revision.clone(),
            passage_id: passage.id.clone(),
            passage_hash: hasher.finalize().to_hex().to_string(),
            canonical_url,
            publisher: self.publisher.clone(),
        })
    }

    fn lexical_candidates(
        &self,
        terms: &[String],
        depth: usize,
        expansion_weight: f64,
        splade_weight: f64,
    ) -> Result<Vec<RankedCandidate>> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let unique_terms: HashSet<&String> = terms.iter().collect();
        let passage_count = self.dictionary.passage_lengths.len() as f64;
        let average_length = self.dictionary.average_passage_length.max(1.0);
        let mut scores = HashMap::<usize, f64>::new();
        for term in unique_terms.iter().copied() {
            let Some(meta) = self.dictionary.terms.get(term) else {
                continue;
            };
            let start = usize::try_from(meta.offset).map_err(|_| {
                AnnpackError::InvalidFormat("posting offset exceeds address space".into())
            })?;
            let length = usize::try_from(meta.length).map_err(|_| {
                AnnpackError::InvalidFormat("posting length exceeds address space".into())
            })?;
            let end = start
                .checked_add(length)
                .ok_or_else(|| AnnpackError::InvalidFormat("posting range overflow".into()))?;
            let bytes = self.postings.get(start..end).ok_or_else(|| {
                AnnpackError::InvalidFormat(format!(
                    "posting list for term {term:?} exceeds postings section"
                ))
            })?;
            let postings = decode_postings(bytes, meta.document_frequency as usize)?;
            let df = meta.document_frequency as f64;
            let idf =
                (1.0 + (passage_count - df + 0.5) / (df + 0.5)).ln() * technical_term_boost(term);
            for (ordinal, frequency) in postings {
                let passage_length =
                    *self
                        .dictionary
                        .passage_lengths
                        .get(ordinal)
                        .ok_or_else(|| {
                            AnnpackError::InvalidFormat(format!(
                                "posting ordinal {ordinal} exceeds passage count"
                            ))
                        })? as f64;
                let tf = frequency as f64;
                let denominator =
                    tf + BM25_K1 * (1.0 - BM25_B + BM25_B * passage_length / average_length);
                *scores.entry(ordinal).or_default() += idf * tf * (BM25_K1 + 1.0) / denominator;
            }
        }
        // ANN-7 / ANN-8: pure-BM25 overlay contribution. No query-time model.
        // Weights default to 0.0, which reproduces Core results exactly and never
        // fetches the overlay sections. Overlays affect ranking only; they never
        // contribute citable text or evidence.
        let overlays = if expansion_weight > 0.0 || splade_weight > 0.0 {
            self.load_overlays()?
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
        Ok(candidates)
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
            return Err(AnnpackError::InvalidInput(
                "query vector contains a non-finite value".into(),
            ));
        }
        let profile_entry = self
            .reader
            .first_entry(SectionType::VectorProfile)
            .ok_or_else(|| AnnpackError::Search("pack has no vector profile".into()))?;
        let profile: VectorProfileSection =
            serde_json::from_slice(&self.reader.read_section(profile_entry.section_id)?)?;
        if let Some(requested) = profile_id
            && requested != profile.profile.id
        {
            return Err(AnnpackError::Search(format!(
                "vector profile {requested:?} is unavailable"
            )));
        }
        let dimensions = profile.profile.dimensions as usize;
        if dimensions == 0 || dimensions > 65_536 || profile.profile.dtype != "float32" {
            return Err(AnnpackError::InvalidFormat(format!(
                "unsupported vector profile shape (dimensions {dimensions}, dtype {})",
                profile.profile.dtype
            )));
        }
        if query.len() != dimensions {
            return Err(AnnpackError::InvalidInput(format!(
                "query vector dimension {} does not match profile dimension {dimensions}",
                query.len()
            )));
        }
        if probes == 0 || probes > 1_024 {
            return Err(AnnpackError::InvalidInput(
                "vector probes must be between 1 and 1024".into(),
            ));
        }
        if profile.passage_ids.len() != self.passage_index.records.len()
            || profile
                .passage_ids
                .iter()
                .zip(&self.passage_index.records)
                .any(|(profile_id, record)| profile_id != &record.id)
        {
            return Err(AnnpackError::InvalidFormat(
                "vector profile passage identities do not match the passage index".into(),
            ));
        }
        let vector_entry = self
            .reader
            .first_entry(SectionType::VectorData)
            .ok_or_else(|| {
                AnnpackError::InvalidFormat("vector profile has no vector data".into())
            })?;
        let bytes = self.reader.read_section(vector_entry.section_id)?;
        if bytes.len() < 8 {
            return Err(AnnpackError::InvalidFormat(
                "truncated vector section".into(),
            ));
        }
        let count = u32::from_le_bytes(bytes[0..4].try_into().expect("slice length")) as usize;
        let stored_dimensions =
            u32::from_le_bytes(bytes[4..8].try_into().expect("slice length")) as usize;
        let value_count = count
            .checked_mul(stored_dimensions)
            .ok_or_else(|| AnnpackError::InvalidFormat("vector size overflow".into()))?;
        let expected_length = 8_usize
            .checked_add(
                value_count.checked_mul(4).ok_or_else(|| {
                    AnnpackError::InvalidFormat("vector byte size overflow".into())
                })?,
            )
            .ok_or_else(|| AnnpackError::InvalidFormat("vector section size overflow".into()))?;
        if count != self.passage_index.records.len()
            || stored_dimensions != dimensions
            || bytes.len() != expected_length
        {
            return Err(AnnpackError::InvalidFormat(
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
                    return Err(AnnpackError::InvalidFormat(format!(
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
        let record = self.passage_index.records.get(ordinal).ok_or_else(|| {
            AnnpackError::InvalidFormat(format!("passage ordinal {ordinal} is out of range"))
        })?;
        let passage_section = self
            .reader
            .first_entry(SectionType::PassageData)
            .map(|entry| entry.section_id)
            .ok_or_else(|| AnnpackError::InvalidFormat("passage data section is missing".into()))?;
        let block = self
            .passage_index
            .blocks
            .get(record.block as usize)
            .ok_or_else(|| AnnpackError::InvalidFormat("passage block is missing".into()))?;
        let logical = {
            let cached = self
                .passage_block_cache
                .lock()
                .map_err(|_| AnnpackError::Search("passage block cache lock poisoned".into()))?
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
                    return Err(AnnpackError::Integrity(format!(
                        "passage block {} hash mismatch",
                        record.block
                    )));
                }
                let limit = usize::try_from(block.logical_length).map_err(|_| {
                    AnnpackError::InvalidFormat("passage block exceeds address space".into())
                })?;
                let decompressed =
                    miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&compressed, limit)
                        .map_err(|error| {
                            AnnpackError::InvalidFormat(format!(
                                "passage block {} deflate decode failed: {error:?}",
                                record.block
                            ))
                        })?;
                if decompressed.len() != limit {
                    return Err(AnnpackError::InvalidFormat(format!(
                        "passage block {} decompressed to {}, expected {} bytes",
                        record.block,
                        decompressed.len(),
                        limit
                    )));
                }
                let decompressed = Arc::new(decompressed);
                self.passage_block_cache
                    .lock()
                    .map_err(|_| AnnpackError::Search("passage block cache lock poisoned".into()))?
                    .insert(record.block, decompressed.clone());
                decompressed
            }
        };
        let start = record.offset as usize;
        let end = start
            .checked_add(record.length as usize)
            .ok_or_else(|| AnnpackError::InvalidFormat("passage record range overflow".into()))?;
        let bytes = logical
            .get(start..end)
            .ok_or_else(|| AnnpackError::InvalidFormat("passage record exceeds block".into()))?;
        let passage: Passage = serde_json::from_slice(bytes)?;
        if passage.id != record.id || passage.ordinal as usize != ordinal {
            return Err(AnnpackError::InvalidFormat(format!(
                "passage index record {ordinal} does not match passage payload"
            )));
        }
        Ok(passage)
    }
}

impl SearchEngine {
    /// Load and strictly validate every ANN-7/ANN-8 term overlay. Called lazily,
    /// only when an overlay weight is non-zero, so a lexical-only client never
    /// fetches these ranges (ANN-10). A malformed overlay is an explicit error
    /// (attacker-controlled ordinals and weights are bounds-checked here). Overlays
    /// never contribute citable text.
    fn load_overlays(&self) -> Result<Vec<LoadedOverlay>> {
        let reader = &self.reader;
        let passage_count = self.passage_index.records.len();
        let mut overlays = Vec::new();
        for entry in reader.entries_of_type(SectionType::TermOverlay) {
            if !entry.derived() {
                return Err(AnnpackError::InvalidFormat(
                    "term overlay section must be flagged derived".into(),
                ));
            }
            let section: crate::model::TermOverlaySection =
                serde_json::from_slice(&reader.read_section(entry.section_id)?)?;
            if section.kind != crate::derive::EXPANSION_KIND
                && section.kind != crate::derive::SPLADE_KIND
            {
                return Err(AnnpackError::InvalidFormat(format!(
                    "unrecognized term overlay kind {:?}",
                    section.kind
                )));
            }
            let mut scale = 1.0_f64;
            if section.kind == crate::derive::SPLADE_KIND {
                match &section.vocabulary {
                    Some(vocabulary) if !vocabulary.id.trim().is_empty() => {
                        if !vocabulary.scale.is_finite() || vocabulary.scale <= 0.0 {
                            return Err(AnnpackError::InvalidFormat(
                                "splade vocabulary scale must be positive and finite".into(),
                            ));
                        }
                        scale = vocabulary.scale;
                    }
                    _ => {
                        return Err(AnnpackError::InvalidFormat(
                            "splade overlay requires a non-empty vocabulary id".into(),
                        ));
                    }
                }
            }
            let mut terms = HashMap::with_capacity(section.terms.len());
            for (term, postings) in section.terms {
                if postings.is_empty() {
                    return Err(AnnpackError::InvalidFormat(format!(
                        "term overlay entry {term:?} has an empty posting list"
                    )));
                }
                let mut previous: Option<u32> = None;
                for (ordinal, weight) in &postings {
                    if *ordinal as usize >= passage_count {
                        return Err(AnnpackError::InvalidFormat(format!(
                            "term overlay entry {term:?} has an out-of-range ordinal"
                        )));
                    }
                    if let Some(previous) = previous
                        && *ordinal <= previous
                    {
                        return Err(AnnpackError::InvalidFormat(format!(
                            "term overlay entry {term:?} ordinals must be strictly increasing"
                        )));
                    }
                    if *weight == 0 {
                        return Err(AnnpackError::InvalidFormat(format!(
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

    /// Load and validate the ANN-9 anchor representation, if present. Lazy: only
    /// read when `anchor_scores` is invoked, so a lexical-only client never fetches
    /// these ranges.
    fn load_anchors(&self) -> Result<Option<LoadedAnchors>> {
        let reader = &self.reader;
        let passage_count = self.passage_index.records.len();
        let set_entry = reader.first_entry(SectionType::AnchorSet);
        let coord_entry = reader.first_entry(SectionType::AnchorCoordinates);
        let (set_entry, coord_entry) = match (set_entry, coord_entry) {
            (Some(set), Some(coords)) => (set, coords),
            (None, None) => return Ok(None),
            _ => {
                return Err(AnnpackError::InvalidFormat(
                    "ANN-9 anchor sections are incomplete".into(),
                ));
            }
        };
        if !coord_entry.derived() {
            return Err(AnnpackError::InvalidFormat(
                "anchor coordinates section must be flagged derived".into(),
            ));
        }
        let set: crate::model::AnchorSetSection =
            serde_json::from_slice(&reader.read_section(set_entry.section_id)?)?;
        let coords: crate::model::AnchorCoordinatesSection =
            serde_json::from_slice(&reader.read_section(coord_entry.section_id)?)?;
        if set.space_id != coords.space_id {
            return Err(AnnpackError::InvalidFormat(
                "anchor set and coordinates declare different spaces".into(),
            ));
        }
        if set.anchors.is_empty() {
            return Err(AnnpackError::InvalidFormat("anchor set is empty".into()));
        }
        if coords.metric != "cosine" {
            return Err(AnnpackError::Unsupported(format!(
                "anchor metric {:?}",
                coords.metric
            )));
        }
        if coords.coordinates.len() != passage_count {
            return Err(AnnpackError::InvalidFormat(
                "anchor coordinates row count does not match passage count".into(),
            ));
        }
        for row in &coords.coordinates {
            if row.len() != set.anchors.len() {
                return Err(AnnpackError::InvalidFormat(
                    "anchor coordinate row length does not match the anchor count".into(),
                ));
            }
        }
        if !coords.scale.is_finite() || coords.scale <= 0.0 {
            return Err(AnnpackError::InvalidFormat(
                "anchor scale must be positive and finite".into(),
            ));
        }
        Ok(Some(LoadedAnchors {
            space_id: set.space_id,
            metric: coords.metric,
            scale: coords.scale,
            anchors: set.anchors,
            coordinates: coords.coordinates,
        }))
    }
}

fn required_profile_section(reader: &PackReader, section_type: SectionType) -> Result<u32> {
    let entry = reader.first_entry(section_type).ok_or_else(|| {
        AnnpackError::InvalidFormat(format!(
            "required {} section is missing",
            section_type.name()
        ))
    })?;
    if !entry.required() || entry.format_version != 1 {
        return Err(AnnpackError::InvalidFormat(format!(
            "{} section is not a required v1 profile section",
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
        return Err(AnnpackError::InvalidFormat(
            "invalid or unsupported IVF vector index".into(),
        ));
    }
    let mut seen = vec![false; vector_count];
    for (cluster, (centroid, list)) in index.centroids.iter().zip(&index.lists).enumerate() {
        if centroid.len() != dimensions || centroid.iter().any(|value| !value.is_finite()) {
            return Err(AnnpackError::InvalidFormat(format!(
                "IVF centroid {cluster} has invalid values or dimensions"
            )));
        }
        for ordinal in list {
            let ordinal = *ordinal as usize;
            if ordinal >= vector_count || std::mem::replace(&mut seen[ordinal], true) {
                return Err(AnnpackError::InvalidFormat(
                    "IVF lists contain duplicate or out-of-range ordinals".into(),
                ));
            }
        }
    }
    if seen.iter().any(|value| !value) {
        return Err(AnnpackError::InvalidFormat(
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

pub fn tokenize(text: &str) -> Vec<String> {
    let normalized: String = text.nfkc().flat_map(char::to_lowercase).collect();
    normalized
        .split_whitespace()
        .filter_map(|raw| {
            let token = raw.trim_matches(|character: char| {
                !character.is_alphanumeric()
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
            .ok_or_else(|| AnnpackError::InvalidFormat("truncated varint".into()))?;
        *cursor += 1;
        if shift == 63 && byte > 1 {
            return Err(AnnpackError::InvalidFormat("varint overflow".into()));
        }
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(AnnpackError::InvalidFormat("non-terminating varint".into()))
}

fn decode_postings(bytes: &[u8], expected_count: usize) -> Result<Vec<(usize, u32)>> {
    let mut postings = Vec::with_capacity(expected_count);
    let mut cursor = 0;
    let mut ordinal = 0_u64;
    for index in 0..expected_count {
        let delta = decode_varint(bytes, &mut cursor)?;
        if index != 0 && delta == 0 {
            return Err(AnnpackError::InvalidFormat(
                "posting ordinals must be strictly increasing".into(),
            ));
        }
        ordinal = if index == 0 {
            delta
        } else {
            ordinal
                .checked_add(delta)
                .ok_or_else(|| AnnpackError::InvalidFormat("posting ordinal overflow".into()))?
        };
        let frequency = decode_varint(bytes, &mut cursor)?;
        let ordinal = usize::try_from(ordinal)
            .map_err(|_| AnnpackError::InvalidFormat("posting ordinal exceeds platform".into()))?;
        let frequency = u32::try_from(frequency)
            .map_err(|_| AnnpackError::InvalidFormat("term frequency exceeds u32".into()))?;
        if frequency == 0 {
            return Err(AnnpackError::InvalidFormat("zero term frequency".into()));
        }
        postings.push((ordinal, frequency));
    }
    if cursor != bytes.len() {
        return Err(AnnpackError::InvalidFormat(
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

fn fuse_candidates(
    lexical: &[RankedCandidate],
    vector: &[RankedCandidate],
    options: &SearchOptions,
    effective_mode: SearchMode,
) -> Vec<(usize, FusionCandidate)> {
    let mut candidates = BTreeMap::<usize, FusionCandidate>::new();
    for (index, candidate) in lexical.iter().enumerate() {
        let rank = index + 1;
        let entry = candidates.entry(candidate.ordinal).or_default();
        entry.lexical_score = Some(candidate.score);
        entry.lexical_rank = Some(rank);
        entry.fused_score += match effective_mode {
            SearchMode::Lexical => candidate.score,
            _ => options.lexical_weight / (RRF_CONSTANT + rank as f64),
        };
    }
    for (index, candidate) in vector.iter().enumerate() {
        let rank = index + 1;
        let entry = candidates.entry(candidate.ordinal).or_default();
        entry.vector_score = Some(candidate.score);
        entry.vector_rank = Some(rank);
        entry.fused_score += match effective_mode {
            SearchMode::Vector => candidate.score,
            _ => options.vector_weight / (RRF_CONSTANT + rank as f64),
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
