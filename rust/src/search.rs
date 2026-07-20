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
            debug: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub pack: SearchPackIdentity,
    pub query: String,
    pub requested_mode: SearchMode,
    pub effective_mode: SearchMode,
    pub results: Vec<SearchHit>,
    pub diagnostics: Option<SearchDiagnostics>,
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

        let lexical = match options.mode {
            SearchMode::Vector => Vec::new(),
            SearchMode::Lexical | SearchMode::Hybrid => {
                self.lexical_candidates(&query_terms, options.candidate_depth.max(options.limit))?
            }
        };
        let vector = match (&options.mode, &options.query_vector) {
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

    fn lexical_candidates(&self, terms: &[String], depth: usize) -> Result<Vec<RankedCandidate>> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let unique_terms: HashSet<&String> = terms.iter().collect();
        let passage_count = self.dictionary.passage_lengths.len() as f64;
        let average_length = self.dictionary.average_passage_length.max(1.0);
        let mut scores = HashMap::<usize, f64>::new();
        for term in unique_terms {
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
        let mut candidates: Vec<_> = scores
            .into_iter()
            .map(|(ordinal, score)| RankedCandidate { ordinal, score })
            .collect();
        sort_candidates(&mut candidates);
        candidates.truncate(depth);
        Ok(candidates)
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
