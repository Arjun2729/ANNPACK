use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};
use crate::format::{PackWriter, SectionData, SectionType};
use crate::ingest::{IngestOptions, IngestedCorpus, InputFormat, ingest_directory};
use crate::model::{
    AccessClass, DictionaryBlock, EmbeddingProfile, IndexBlock, IvfIndex, LexicalBlockIndex,
    LexicalDictionary, Manifest, PackPolicy, PostingMeta, RecordBlockIndex, StoredBlock,
    StoredPassageIndex, StoredRecord, VectorProfileSection,
};
use crate::search::tokenize;

// Section IDs this builder assigns. Section IDs are artifact-local coordinates
// and are independent of section-type numbers (FORMAT-v3 §2). Early reference-
// builder IDs happen to equal their section types; later assignments do not.
// In particular, section ID 14 identifies a section of type 13 (term overlay)
// even though section *type* 14 is retired, and IDs 17 and 18 identify types 16
// and 17 respectively.
//
// The equality below the vector sections is coincidence, not contract. A reader
// that resolves sections by treating an ID as a type is wrong in a way this
// builder's own output will not reveal until it reaches a pack carrying both
// overlays or a format-2 index:
//
//     id   type                        note
//      1   1  manifest                 id == type
//      …
//      9   9  vector index             id == type
//     13  13  term overlay (expansion) id == type, by coincidence
//     14  13  term overlay (splade)    two sections share one type
//     17  16  lexical terms            type 17 is passage records, not this
//     18  17  passage records
//
// IDs 15 and 16 are absent because they were AN-9 anchor sections, withdrawn.
// The gap is deliberate; the numbers are not reused.
pub const MANIFEST_SECTION_ID: u32 = 1;
pub const DOCUMENTS_SECTION_ID: u32 = 2;
pub const PASSAGE_INDEX_SECTION_ID: u32 = 3;
pub const PASSAGE_DATA_SECTION_ID: u32 = 4;
pub const LEXICAL_DICTIONARY_SECTION_ID: u32 = 5;
pub const LEXICAL_POSTINGS_SECTION_ID: u32 = 6;
pub const VECTOR_PROFILE_SECTION_ID: u32 = 7;
pub const VECTOR_DATA_SECTION_ID: u32 = 8;
pub const VECTOR_INDEX_SECTION_ID: u32 = 9;
pub const EXPANSION_SECTION_ID: u32 = 13;
pub const SPLADE_SECTION_ID: u32 = 14;
pub const LEXICAL_TERMS_SECTION_ID: u32 = 17;
pub const PASSAGE_RECORDS_SECTION_ID: u32 = 18;

/// Fixed-width passage record: block, offset, length as little-endian u32.
///
/// Deliberately carries no passage id. The id is already stored twice — in the
/// id index below, and in the passage payload itself — and a third copy cost 32
/// incompressible bytes per passage, which made packs larger than their source.
/// A mis-seek is caught by checking the ordinal the payload carries, which is
/// the property the id comparison was actually providing.
pub const RECORD_STRIDE: u32 = 12;
/// Fixed-width id-index entry: 32-byte raw id, then its u32 passage ordinal.
pub const ID_ENTRY_STRIDE: u32 = 36;
/// Records per block. At 12 bytes this is a ~64 KiB block, matching the other
/// index regions.
pub const RECORDS_PER_BLOCK: u32 = 5_461;
const PASSAGE_BLOCK_TARGET: usize = 64 * 1024;

/// Section format version for the block-addressable lexical index.
///
/// Format 1 stored the term table inline in the dictionary section and the
/// posting stream as one deflated section, so resolving a single term required
/// downloading and inflating both in full. Format 2 partitions each into
/// independently hashed blocks. Readers accept both; the version is what tells
/// them which layout they are looking at.
pub const LEXICAL_INDEX_FORMAT_VERSION: u16 = 2;

/// Section format version for the block-addressable passage record table.
///
/// Format 1 stored records inline in the passage index as JSON with hex ids,
/// which had to be downloaded and parsed in full before any result could be
/// resolved. Format 2 moves them to fixed-width blocks addressed by ordinal,
/// plus an id-sorted index for `get_passage`.
pub const PASSAGE_INDEX_FORMAT_VERSION: u16 = 2;

#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub source_revision: Option<String>,
    pub base_url: Option<String>,
    pub created_at: Option<String>,
    pub license: Option<String>,
    pub access: AccessClass,
    pub redistributable: Option<bool>,
    pub policy_expires_at: Option<String>,
    pub policy_url: Option<String>,
    pub policy_override: Option<PackPolicy>,
    pub vector_input: Option<PathBuf>,
    pub expansion_input: Option<PathBuf>,
    pub splade_input: Option<PathBuf>,
    pub target_chars: usize,
    pub max_chars: usize,
    pub input_format: InputFormat,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuildReport {
    pub output: String,
    pub root_hash: String,
    pub bytes: u64,
    pub documents: usize,
    pub passages: usize,
    pub terms: usize,
    pub ignored_files: Vec<String>,
    pub deterministic: bool,
    pub capabilities: Vec<String>,
    pub input_format: String,
    pub input_format_version: Option<String>,
    pub source_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorInput {
    pub profile: EmbeddingProfile,
    pub vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub passage_ids: Vec<String>,
}

pub fn build_pack(options: &BuildOptions) -> Result<BuildReport> {
    validate_build_options(options)?;
    let ingest_options = IngestOptions {
        target_chars: options.target_chars,
        max_chars: options.max_chars,
        base_url: options.base_url.clone(),
        input_format: options.input_format,
    };
    let corpus = ingest_directory(&options.input, &ingest_options)?;
    if corpus.documents.is_empty() {
        return Err(AnnpackError::InvalidInput(
            "input contains no supported knowledge documents".into(),
        ));
    }
    if corpus.passages.is_empty() {
        return Err(AnnpackError::InvalidInput(
            "input contains no searchable passages".into(),
        ));
    }

    let vector_input = options
        .vector_input
        .as_ref()
        .map(|path| read_vector_input(path, &corpus))
        .transpose()?;
    let (writer, term_count, capabilities) = assemble_pack(options, &corpus, vector_input)?;
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let root_hash = writer.write_path(&options.output)?;
    let bytes = fs::metadata(&options.output)?.len();
    Ok(BuildReport {
        output: options.output.display().to_string(),
        root_hash: hex::encode(root_hash),
        bytes,
        documents: corpus.documents.len(),
        passages: corpus.passages.len(),
        terms: term_count,
        ignored_files: corpus.ignored,
        deterministic: options.created_at.is_none(),
        capabilities,
        input_format: corpus.input_format.as_str().into(),
        input_format_version: corpus.input_format_version,
        source_digest: corpus.source_digest,
    })
}

pub fn build_pack_bytes(options: &BuildOptions) -> Result<Vec<u8>> {
    validate_build_options(options)?;
    let ingest_options = IngestOptions {
        target_chars: options.target_chars,
        max_chars: options.max_chars,
        base_url: options.base_url.clone(),
        input_format: options.input_format,
    };
    let corpus = ingest_directory(&options.input, &ingest_options)?;
    let vector_input = options
        .vector_input
        .as_ref()
        .map(|path| read_vector_input(path, &corpus))
        .transpose()?;
    let (writer, _, _) = assemble_pack(options, &corpus, vector_input)?;
    writer.build_bytes()
}

fn validate_build_options(options: &BuildOptions) -> Result<()> {
    if options.name.trim().is_empty() || options.version.trim().is_empty() {
        return Err(AnnpackError::InvalidInput(
            "pack name and version must not be empty".into(),
        ));
    }
    if options.target_chars == 0 || options.max_chars < options.target_chars {
        return Err(AnnpackError::InvalidInput(
            "chunk maximum must be at least the non-zero target".into(),
        ));
    }
    Ok(())
}

fn assemble_pack(
    options: &BuildOptions,
    corpus: &IngestedCorpus,
    vector_input: Option<VectorInput>,
) -> Result<(PackWriter, usize, Vec<String>)> {
    let documents = serde_json::to_vec(&corpus.documents)?;
    let (mut passage_index, passage_data, passage_leaves) = encode_passages(corpus)?;
    // Logical content root over the exact stored passage records. Computed from
    // the same bytes the reader hashes, so a receipt's leaf always reproduces.
    let passage_merkle_root = crate::evidence::merkle_root(&passage_leaves).ok_or_else(|| {
        AnnpackError::InvalidInput("cannot commit a passage merkle root for an empty corpus".into())
    })?;
    let (mut lexical_dictionary, lexical_postings) = build_lexical_index(corpus)?;
    let term_count = lexical_dictionary.terms.len();
    // Move the term table out of the monolithic dictionary section and into
    // independently addressable blocks. What stays inline is only what every
    // query needs regardless of terms: the per-passage lengths and their mean.
    let (dictionary_payload, postings_payload, lexical_blocks) =
        partition_lexical_index(&lexical_dictionary.terms, &lexical_postings)?;
    lexical_dictionary.terms = BTreeMap::new();
    passage_index.lexical_blocks = Some(lexical_blocks);
    // Same treatment for the record table: it was the largest thing a reader
    // still had to download in full before it could resolve a single result.
    let (records_payload, record_blocks) = partition_passage_records(&passage_index.records)?;
    passage_index.records = Vec::new();
    passage_index.record_blocks = Some(record_blocks);
    let lexical_header = serde_json::to_vec(&lexical_dictionary)?;

    let mut capabilities = vec![
        "content".to_string(),
        "citations".to_string(),
        "lexical-bm25".to_string(),
        "range-addressable-passages".to_string(),
        "section-integrity".to_string(),
    ];
    let embedding_profiles = vector_input
        .as_ref()
        .map(|input| vec![input.profile.clone()])
        .unwrap_or_default();
    if vector_input.is_some() {
        capabilities.push("vector-flat-dot".to_string());
        capabilities.push("vector-ivf-flat-dot".to_string());
        // Named for the fusion this builder actually performs. It was
        // `hybrid-rrf` until the ranker moved to absolute-scale fusion and the
        // name did not follow -- see `fuse_candidates` for why RRF was dropped.
        capabilities.push("hybrid-absolute-scale".to_string());
    }
    if corpus.input_format == InputFormat::Okf {
        capabilities.push("source-okf".to_string());
    }

    // AN-7/8/9: consume pinned, hashed sidecars. No model runs here; the
    // sections are pure functions of the committed sidecars, so the build stays
    // byte-identical. Missing sidecars simply produce a Core-only pack.
    let mut derived_sections: Vec<SectionData> = Vec::new();
    let mut derived_inputs: Vec<crate::model::DerivedInput> = Vec::new();
    if let Some(path) = &options.expansion_input {
        let (sidecar, digest) = crate::derive::read_overlay_sidecar(path)?;
        if sidecar.kind != crate::derive::EXPANSION_KIND {
            return Err(AnnpackError::InvalidInput(
                "--expansion sidecar is not an expansion-v1 overlay".into(),
            ));
        }
        let built = crate::derive::build_overlay(
            &sidecar,
            &digest,
            EXPANSION_SECTION_ID,
            &corpus.passages,
        )?;
        capabilities.push("term-overlay-expansion".to_string());
        derived_sections.push(built.section);
        derived_inputs.push(built.derived_input);
    }
    if let Some(path) = &options.splade_input {
        let (sidecar, digest) = crate::derive::read_overlay_sidecar(path)?;
        if sidecar.kind != crate::derive::SPLADE_KIND {
            return Err(AnnpackError::InvalidInput(
                "--splade sidecar is not a splade-v1 overlay".into(),
            ));
        }
        let built =
            crate::derive::build_overlay(&sidecar, &digest, SPLADE_SECTION_ID, &corpus.passages)?;
        capabilities.push("term-overlay-splade".to_string());
        derived_sections.push(built.section);
        derived_inputs.push(built.derived_input);
    }
    capabilities.sort();

    // AN-10: fat-pack fallback order. Highest-capability profile first, always
    // ending at Core lexical so selection terminates for every conformant reader.
    let mut retrieval_profiles: Vec<crate::model::RetrievalProfile> = Vec::new();
    if vector_input.is_some() {
        retrieval_profiles.push(crate::model::RetrievalProfile {
            id: "vectors".into(),
            kind: "vector".into(),
            section_ids: vec![
                VECTOR_PROFILE_SECTION_ID,
                VECTOR_DATA_SECTION_ID,
                VECTOR_INDEX_SECTION_ID,
            ],
            requires: vec!["vector-ivf-flat-dot".into()],
        });
    }
    if options.splade_input.is_some() {
        retrieval_profiles.push(crate::model::RetrievalProfile {
            id: "splade".into(),
            kind: "splade".into(),
            section_ids: vec![SPLADE_SECTION_ID],
            requires: vec!["term-overlay-splade".into()],
        });
    }
    if options.expansion_input.is_some() {
        retrieval_profiles.push(crate::model::RetrievalProfile {
            id: "expansion".into(),
            kind: "expansion".into(),
            section_ids: vec![EXPANSION_SECTION_ID],
            requires: vec!["term-overlay-expansion".into()],
        });
    }
    // Only advertise the fat-pack descriptor when two or more optional
    // representations coexist and the runtime must actually choose. A pack with
    // a single optional profile (e.g. AN-1 vectors only) is not a fat pack.
    if retrieval_profiles.len() >= 2 {
        retrieval_profiles.push(crate::model::RetrievalProfile {
            id: "lexical".into(),
            kind: "lexical".into(),
            // The sections lexical retrieval owns. 17 and 18 were missing
            // while a format-2 query read both of them, which the isolation
            // test could not catch: it asserts only that lexical avoids other
            // profiles' ranges, never that its own list is complete.
            section_ids: vec![
                PASSAGE_INDEX_SECTION_ID,
                PASSAGE_DATA_SECTION_ID,
                LEXICAL_DICTIONARY_SECTION_ID,
                LEXICAL_POSTINGS_SECTION_ID,
                LEXICAL_TERMS_SECTION_ID,
                PASSAGE_RECORDS_SECTION_ID,
            ],
            requires: vec!["lexical-bm25".into()],
        });
    } else {
        retrieval_profiles.clear();
    }

    let manifest = Manifest {
        name: options.name.clone(),
        version: options.version.clone(),
        description: options.description.clone(),
        source_revision: options.source_revision.clone(),
        base_url: options.base_url.clone(),
        created_at: options.created_at.clone(),
        document_count: corpus.documents.len() as u64,
        passage_count: corpus.passages.len() as u64,
        capabilities: capabilities.clone(),
        embedding_profiles,
        policy: options.policy_override.clone().unwrap_or(PackPolicy {
            license: options.license.clone(),
            access: options.access.clone(),
            redistributable: options.redistributable,
            expires_at: options.policy_expires_at.clone(),
            policy_url: options.policy_url.clone(),
        }),
        passage_merkle_root: Some(hex::encode(passage_merkle_root)),
        // Authenticated for every input format since manifest format 4. The
        // digest was always computed here; only OKF artifacts used to commit to
        // it, which made provenance for a Markdown artifact a builder claim the
        // artifact could not corroborate (ADR-0005).
        //
        // `corpus.input_format` is the resolved format, never `auto`: ingestion
        // resolves it before returning. `version` stays OKF-specific because
        // Markdown has no corpus-format version to state.
        source: Some(crate::model::SourceDescriptor {
            format: corpus.input_format.as_str().into(),
            version: corpus.input_format_version.clone(),
            digest_algorithm: "blake3".into(),
            // The same value `build --json` reports. One computation, read
            // twice: two would be two things to keep in agreement.
            digest: corpus.source_digest.clone(),
        }),
        derived_inputs,
        retrieval_profiles,
    };

    let mut writer = PackWriter::new();
    writer.push(SectionData::required_versioned(
        MANIFEST_SECTION_ID,
        SectionType::Manifest,
        1,
        crate::format::MANIFEST_FORMAT_VERSION,
        serde_json::to_vec(&manifest)?,
    ))?;
    writer.push(SectionData::required_deflate(
        DOCUMENTS_SECTION_ID,
        SectionType::Documents,
        corpus.documents.len() as u64,
        documents,
    ))?;
    writer.push(SectionData::required_deflate(
        PASSAGE_INDEX_SECTION_ID,
        SectionType::PassageIndex,
        corpus.passages.len() as u64,
        serde_json::to_vec(&passage_index)?,
    ))?;
    writer.push(SectionData::required(
        PASSAGE_DATA_SECTION_ID,
        SectionType::PassageData,
        corpus.passages.len() as u64,
        passage_data,
    ))?;
    // Lexical index format 2. The dictionary section keeps only what a query
    // needs unconditionally (per-passage lengths and their mean); the term
    // table and the posting stream become independently hashed blocks that a
    // reader fetches by range. Section-level codec must be None for both, since
    // partial reads of a section-compressed payload are not addressable.
    writer.push(SectionData::required_deflate(
        LEXICAL_DICTIONARY_SECTION_ID,
        SectionType::LexicalDictionary,
        term_count as u64,
        lexical_header,
    ))?;
    writer.push(SectionData::required_versioned(
        PASSAGE_RECORDS_SECTION_ID,
        SectionType::PassageRecords,
        corpus.passages.len() as u64,
        PASSAGE_INDEX_FORMAT_VERSION,
        records_payload,
    ))?;
    writer.push(SectionData::required_versioned(
        LEXICAL_TERMS_SECTION_ID,
        SectionType::LexicalTerms,
        term_count as u64,
        LEXICAL_INDEX_FORMAT_VERSION,
        dictionary_payload,
    ))?;
    writer.push(SectionData::required_versioned(
        LEXICAL_POSTINGS_SECTION_ID,
        SectionType::LexicalPostings,
        term_count as u64,
        LEXICAL_INDEX_FORMAT_VERSION,
        postings_payload,
    ))?;

    if let Some(input) = vector_input {
        let ivf_index = build_ivf_index(&input.vectors, input.profile.dimensions)?;
        let profile_section = VectorProfileSection {
            profile: input.profile.clone(),
            passage_ids: corpus
                .passages
                .iter()
                .map(|passage| passage.id.clone())
                .collect(),
        };
        writer.push(SectionData::optional_deflate(
            VECTOR_PROFILE_SECTION_ID,
            SectionType::VectorProfile,
            1,
            serde_json::to_vec(&profile_section)?,
        ))?;
        writer.push(SectionData::optional(
            VECTOR_DATA_SECTION_ID,
            SectionType::VectorData,
            input.vectors.len() as u64,
            encode_vectors(&input)?,
        ))?;
        writer.push(SectionData::optional_deflate(
            VECTOR_INDEX_SECTION_ID,
            SectionType::VectorIndex,
            ivf_index.centroids.len() as u64,
            serde_json::to_vec(&ivf_index)?,
        ))?;
    }
    for section in derived_sections {
        writer.push(section)?;
    }
    Ok((writer, term_count, capabilities))
}

/// Encodes passage blocks and, from the same serialized bytes, the per-passage
/// evidence hashes that form the logical content root. Deriving both here keeps
/// the leaf and the stored record from ever diverging.
fn encode_passages(
    corpus: &IngestedCorpus,
) -> Result<(StoredPassageIndex, Vec<u8>, Vec<[u8; 32]>)> {
    let mut data = Vec::new();
    let mut records = Vec::with_capacity(corpus.passages.len());
    let mut blocks = Vec::new();
    let mut logical_block = Vec::new();
    let mut leaves = Vec::with_capacity(corpus.passages.len());
    for passage in &corpus.passages {
        let bytes = serde_json::to_vec(passage)?;
        leaves.push(crate::evidence::passage_evidence_hash(&bytes));
        if !logical_block.is_empty() && logical_block.len() + bytes.len() > PASSAGE_BLOCK_TARGET {
            flush_passage_block(&mut logical_block, &mut data, &mut blocks);
        }
        let block = u32::try_from(blocks.len())
            .map_err(|_| AnnpackError::InvalidInput("too many passage blocks".into()))?;
        let offset = u32::try_from(logical_block.len())
            .map_err(|_| AnnpackError::InvalidInput("passage block offset exceeds u32".into()))?;
        let length = u32::try_from(bytes.len())
            .map_err(|_| AnnpackError::InvalidInput("passage record exceeds u32".into()))?;
        logical_block.extend_from_slice(&bytes);
        records.push(StoredRecord {
            id: passage.id.clone(),
            block,
            offset,
            length,
        });
    }
    flush_passage_block(&mut logical_block, &mut data, &mut blocks);
    Ok((
        StoredPassageIndex {
            codec: "deflate-zlib".into(),
            records,
            blocks,
            record_blocks: None,
            lexical_blocks: None,
        },
        data,
        leaves,
    ))
}

fn flush_passage_block(logical: &mut Vec<u8>, output: &mut Vec<u8>, blocks: &mut Vec<StoredBlock>) {
    if logical.is_empty() {
        return;
    }
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(logical, 6);
    let offset = output.len() as u64;
    blocks.push(StoredBlock {
        offset,
        stored_length: compressed.len() as u64,
        logical_length: logical.len() as u64,
        hash: blake3::hash(&compressed).to_hex().to_string(),
    });
    output.extend_from_slice(&compressed);
    logical.clear();
}

fn build_lexical_index(corpus: &IngestedCorpus) -> Result<(LexicalDictionary, Vec<u8>)> {
    let mut terms: BTreeMap<String, BTreeMap<u32, u32>> = BTreeMap::new();
    let mut passage_lengths = Vec::with_capacity(corpus.passages.len());
    let documents = corpus
        .documents
        .iter()
        .map(|document| (document.id.as_str(), document))
        .collect::<HashMap<_, _>>();
    for passage in &corpus.passages {
        let source_context = if corpus.input_format == InputFormat::Okf {
            let document = documents.get(passage.document_id.as_str()).ok_or_else(|| {
                AnnpackError::InvalidInput(format!(
                    "passage {} references missing document {}",
                    passage.id, passage.document_id
                ))
            })?;
            let metadata = document
                .metadata
                .iter()
                .flat_map(|(key, value)| [key.as_str(), value.as_str()])
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} {} {}", document.title, document.source_path, metadata)
        } else {
            String::new()
        };
        let tokens = tokenize(&format!(
            "{} {} {}",
            source_context,
            passage.heading_path.join(" "),
            passage.text
        ));
        passage_lengths.push(tokens.len() as u32);
        let mut frequencies = HashMap::<String, u32>::new();
        for token in tokens {
            *frequencies.entry(token).or_default() += 1;
        }
        for (term, frequency) in frequencies {
            terms
                .entry(term)
                .or_default()
                .insert(passage.ordinal, frequency);
        }
    }

    let mut posting_bytes = Vec::new();
    let mut dictionary_terms = BTreeMap::new();
    for (term, postings) in terms {
        let offset = posting_bytes.len() as u64;
        let mut previous = 0_u32;
        for (index, (ordinal, frequency)) in postings.iter().enumerate() {
            let delta = if index == 0 {
                *ordinal
            } else {
                ordinal - previous
            };
            encode_varint(delta as u64, &mut posting_bytes);
            encode_varint(*frequency as u64, &mut posting_bytes);
            previous = *ordinal;
        }
        let length = posting_bytes.len() as u64 - offset;
        dictionary_terms.insert(
            term,
            PostingMeta {
                offset,
                length,
                document_frequency: postings.len() as u32,
            },
        );
    }
    let total: u64 = passage_lengths.iter().map(|length| *length as u64).sum();
    let average_passage_length = total as f64 / passage_lengths.len().max(1) as f64;
    Ok((
        LexicalDictionary {
            passage_lengths,
            average_passage_length,
            terms: dictionary_terms,
        },
        posting_bytes,
    ))
}

/// Encode the passage record table as two independently addressable regions:
/// fixed-width records in ordinal order, then the same records keyed by id and
/// sorted by it.
///
/// Returns the section payload and the block tables that authenticate it.
fn partition_passage_records(records: &[StoredRecord]) -> Result<(Vec<u8>, RecordBlockIndex)> {
    fn raw_id(record: &StoredRecord) -> Result<[u8; 32]> {
        let bytes = hex::decode(&record.id).map_err(|_| {
            AnnpackError::InvalidInput(format!("passage id {:?} is not hex", record.id))
        })?;
        <[u8; 32]>::try_from(bytes.as_slice()).map_err(|_| {
            AnnpackError::InvalidInput(format!("passage id {:?} is not 32 bytes", record.id))
        })
    }

    let mut payload = Vec::new();
    let mut record_blocks = Vec::new();
    for chunk in records.chunks(RECORDS_PER_BLOCK as usize) {
        let mut logical = Vec::with_capacity(chunk.len() * RECORD_STRIDE as usize);
        for record in chunk {
            logical.extend_from_slice(&record.block.to_le_bytes());
            logical.extend_from_slice(&record.offset.to_le_bytes());
            logical.extend_from_slice(&record.length.to_le_bytes());
        }
        flush_index_block(&logical, None, &mut payload, &mut record_blocks);
    }

    // Id-sorted region. Sorting on the raw bytes rather than the hex string
    // keeps the reader's comparison and the writer's order identical without
    // either having to agree on an encoding.
    let mut by_id: Vec<([u8; 32], u32)> = records
        .iter()
        .enumerate()
        .map(|(ordinal, record)| {
            Ok((
                raw_id(record)?,
                u32::try_from(ordinal).map_err(|_| {
                    AnnpackError::InvalidInput("passage ordinal exceeds u32".into())
                })?,
            ))
        })
        .collect::<Result<_>>()?;
    by_id.sort_unstable_by_key(|(id, _)| *id);

    let mut id_blocks = Vec::new();
    let per_id_block = (LEXICAL_BLOCK_TARGET / ID_ENTRY_STRIDE as usize).max(1);
    for chunk in by_id.chunks(per_id_block) {
        let mut logical = Vec::with_capacity(chunk.len() * ID_ENTRY_STRIDE as usize);
        for (id, ordinal) in chunk {
            logical.extend_from_slice(id);
            logical.extend_from_slice(&ordinal.to_le_bytes());
        }
        let first = hex::encode(chunk[0].0);
        flush_index_block(&logical, Some(first), &mut payload, &mut id_blocks);
    }

    Ok((
        payload,
        RecordBlockIndex {
            stride: RECORD_STRIDE,
            per_block: RECORDS_PER_BLOCK,
            records: record_blocks,
            ids: id_blocks,
        },
    ))
}

/// Deflate one logical block, hash the stored bytes, and append both to the
/// running payload and its block table.
fn flush_index_block(
    logical: &[u8],
    first_term: Option<String>,
    payload: &mut Vec<u8>,
    blocks: &mut Vec<IndexBlock>,
) {
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(logical, 6);
    blocks.push(IndexBlock {
        offset: payload.len() as u64,
        stored_length: compressed.len() as u64,
        logical_length: logical.len() as u64,
        hash: blake3::hash(&compressed).to_hex().to_string(),
        first_term,
    });
    payload.extend_from_slice(&compressed);
}

/// Target logical size of one lexical index block, in bytes.
///
/// The trade-off is per-lookup transfer against block-table size. At 64 KiB a
/// 15k-term dictionary partitions into roughly a dozen blocks, so a term costs
/// one bounded read and the table stays a few hundred bytes. Shrinking this
/// mostly grows the table, which is fetched on every open.
const LEXICAL_BLOCK_TARGET: usize = 64 * 1024;

/// Split the sorted term table and the posting byte stream into independently
/// deflated, independently hashed blocks.
///
/// Returns the section payloads (concatenated stored blocks, section codec
/// `None`) and the block tables that authenticate them. Terms are partitioned
/// on their sorted order so `first_term` is a usable sparse index; postings are
/// partitioned on byte boundaries, since a posting list is addressed by range
/// rather than by key.
fn partition_lexical_index(
    terms: &BTreeMap<String, PostingMeta>,
    posting_bytes: &[u8],
) -> Result<(Vec<u8>, Vec<u8>, LexicalBlockIndex)> {
    // Dictionary: accumulate terms until the serialized block reaches target.
    let mut dictionary_payload = Vec::new();
    let mut dictionary_blocks = Vec::new();
    let mut pending: BTreeMap<String, PostingMeta> = BTreeMap::new();
    let mut pending_first: Option<String> = None;
    let mut pending_size = 0_usize;

    for (term, meta) in terms {
        if pending_first.is_none() {
            pending_first = Some(term.clone());
        }
        // Approximate the serialized cost so partitioning does not require
        // re-encoding the block on every insert.
        pending_size += term.len() + 48;
        pending.insert(term.clone(), meta.clone());
        if pending_size >= LEXICAL_BLOCK_TARGET {
            flush_dictionary_block(
                &mut pending,
                &mut pending_first,
                &mut dictionary_payload,
                &mut dictionary_blocks,
            )?;
            pending_size = 0;
        }
    }
    flush_dictionary_block(
        &mut pending,
        &mut pending_first,
        &mut dictionary_payload,
        &mut dictionary_blocks,
    )?;

    // Postings: fixed-size logical spans over the byte stream.
    let mut postings_payload = Vec::new();
    let mut postings_blocks = Vec::new();
    for chunk_start in (0..posting_bytes.len()).step_by(LEXICAL_BLOCK_TARGET) {
        let chunk_end = (chunk_start + LEXICAL_BLOCK_TARGET).min(posting_bytes.len());
        let logical = &posting_bytes[chunk_start..chunk_end];
        let compressed = miniz_oxide::deflate::compress_to_vec_zlib(logical, 6);
        postings_blocks.push(IndexBlock {
            offset: postings_payload.len() as u64,
            stored_length: compressed.len() as u64,
            logical_length: logical.len() as u64,
            hash: blake3::hash(&compressed).to_hex().to_string(),
            first_term: None,
        });
        postings_payload.extend_from_slice(&compressed);
    }

    Ok((
        dictionary_payload,
        postings_payload,
        LexicalBlockIndex {
            dictionary: dictionary_blocks,
            postings: postings_blocks,
        },
    ))
}

fn flush_dictionary_block(
    pending: &mut BTreeMap<String, PostingMeta>,
    first_term: &mut Option<String>,
    payload: &mut Vec<u8>,
    blocks: &mut Vec<IndexBlock>,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let block = DictionaryBlock {
        terms: std::mem::take(pending),
    };
    let logical = serde_json::to_vec(&block)?;
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&logical, 6);
    blocks.push(IndexBlock {
        offset: payload.len() as u64,
        stored_length: compressed.len() as u64,
        logical_length: logical.len() as u64,
        hash: blake3::hash(&compressed).to_hex().to_string(),
        first_term: first_term.take(),
    });
    payload.extend_from_slice(&compressed);
    Ok(())
}

fn read_vector_input(path: &Path, corpus: &IngestedCorpus) -> Result<VectorInput> {
    let bytes = fs::read(path)?;
    let mut input: VectorInput = serde_json::from_slice(&bytes)?;
    if input.profile.dimensions == 0 || input.profile.dimensions > 65_536 {
        return Err(AnnpackError::InvalidInput(
            "embedding dimensions must be between 1 and 65536".into(),
        ));
    }
    if input.profile.dtype != "float32" {
        return Err(AnnpackError::InvalidInput(
            "the reference vector runtime currently accepts dtype=float32".into(),
        ));
    }
    if input.profile.id.trim().is_empty()
        || input.profile.model.trim().is_empty()
        || input.profile.revision.trim().is_empty()
        || input.profile.pooling.trim().is_empty()
    {
        return Err(AnnpackError::InvalidInput(
            "embedding id, model, revision, and pooling must not be empty".into(),
        ));
    }
    if let Some(runtime) = &input.profile.runtime
        && (runtime.library.trim().is_empty()
            || runtime.library_version.trim().is_empty()
            || runtime.weights_dtype.trim().is_empty()
            || runtime.max_tokens == 0)
    {
        return Err(AnnpackError::InvalidInput(
            "embedding runtime descriptor is incomplete".into(),
        ));
    }
    if input.vectors.len() != corpus.passages.len() {
        return Err(AnnpackError::InvalidInput(format!(
            "vector count {} does not match passage count {}",
            input.vectors.len(),
            corpus.passages.len()
        )));
    }
    for (index, vector) in input.vectors.iter().enumerate() {
        if vector.len() != input.profile.dimensions as usize {
            return Err(AnnpackError::InvalidInput(format!(
                "vector {index} has dimension {}, expected {}",
                vector.len(),
                input.profile.dimensions
            )));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(AnnpackError::InvalidInput(format!(
                "vector {index} contains a non-finite value"
            )));
        }
    }
    let expected_ids: Vec<_> = corpus
        .passages
        .iter()
        .map(|passage| passage.id.clone())
        .collect();
    if input.passage_ids.is_empty() {
        input.passage_ids = expected_ids;
    } else if input.passage_ids != expected_ids {
        return Err(AnnpackError::InvalidInput(
            "vector passage IDs do not match deterministic corpus order".into(),
        ));
    }
    Ok(input)
}

fn encode_vectors(input: &VectorInput) -> Result<Vec<u8>> {
    let count = u32::try_from(input.vectors.len())
        .map_err(|_| AnnpackError::InvalidInput("too many vectors".into()))?;
    let value_count = input
        .vectors
        .len()
        .checked_mul(input.profile.dimensions as usize)
        .ok_or_else(|| AnnpackError::InvalidInput("vector count overflow".into()))?;
    let capacity = value_count
        .checked_mul(size_of::<f32>())
        .and_then(|bytes| bytes.checked_add(8))
        .ok_or_else(|| AnnpackError::InvalidInput("vector byte size overflow".into()))?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&input.profile.dimensions.to_le_bytes());
    for vector in &input.vectors {
        for value in vector {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn build_ivf_index(vectors: &[Vec<f32>], dimensions: u32) -> Result<IvfIndex> {
    if vectors.is_empty() {
        return Err(AnnpackError::InvalidInput(
            "cannot build a vector index without vectors".into(),
        ));
    }
    let cluster_count = (vectors.len() as f64).sqrt().ceil().clamp(1.0, 256.0) as usize;
    let dimensions = dimensions as usize;
    let mut centroids: Vec<Vec<f32>> = (0..cluster_count)
        .map(|cluster| vectors[cluster * vectors.len() / cluster_count].clone())
        .collect();
    let mut assignments = vec![0_usize; vectors.len()];
    for _ in 0..6 {
        for (index, vector) in vectors.iter().enumerate() {
            assignments[index] = best_centroid(vector, &centroids);
        }
        let mut sums = vec![vec![0.0_f64; dimensions]; cluster_count];
        let mut counts = vec![0_u64; cluster_count];
        for (vector, cluster) in vectors.iter().zip(&assignments) {
            counts[*cluster] += 1;
            for (dimension, value) in vector.iter().enumerate() {
                sums[*cluster][dimension] += *value as f64;
            }
        }
        for cluster in 0..cluster_count {
            if counts[cluster] == 0 {
                continue;
            }
            for dimension in 0..dimensions {
                centroids[cluster][dimension] =
                    (sums[cluster][dimension] / counts[cluster] as f64) as f32;
            }
        }
    }
    for (index, vector) in vectors.iter().enumerate() {
        assignments[index] = best_centroid(vector, &centroids);
    }
    let mut lists = vec![Vec::new(); cluster_count];
    for (ordinal, cluster) in assignments.into_iter().enumerate() {
        lists[cluster].push(u32::try_from(ordinal).map_err(|_| {
            AnnpackError::InvalidInput("vector ordinal exceeds the v3 IVF limit".into())
        })?);
    }
    let default_probes = (cluster_count as f64).sqrt().ceil() as u32;
    Ok(IvfIndex {
        algorithm: "ivf-flat-v1".into(),
        distance: "dot".into(),
        dimensions: dimensions as u32,
        default_probes,
        centroids,
        lists,
    })
}

fn best_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0_usize;
    let mut best_score = f64::NEG_INFINITY;
    for (index, centroid) in centroids.iter().enumerate() {
        let score = vector
            .iter()
            .zip(centroid)
            .map(|(left, right)| *left as f64 * *right as f64)
            .sum::<f64>();
        if score > best_score {
            best = index;
            best_score = score;
        }
    }
    best
}

pub fn encode_varint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_has_canonical_encoding() {
        let mut bytes = Vec::new();
        encode_varint(300, &mut bytes);
        assert_eq!(bytes, vec![0xac, 0x02]);
    }
}
