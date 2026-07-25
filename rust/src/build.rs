use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};
use crate::format::{PackWriter, SectionData, SectionType};
use crate::ingest::{IngestOptions, IngestedCorpus, InputFormat, ingest_directory};
use crate::model::{
    AccessClass, EmbeddingProfile, IvfIndex, LexicalDictionary, Manifest, PackDependency,
    PackPolicy, PostingMeta, StoredBlock, StoredPassageIndex, StoredRecord, VectorProfileSection,
};
use crate::search::tokenize;

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
pub const ANCHOR_SET_SECTION_ID: u32 = 15;
pub const ANCHOR_COORDINATES_SECTION_ID: u32 = 16;
const PASSAGE_BLOCK_TARGET: usize = 64 * 1024;

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
    pub dependencies: Vec<PackDependency>,
    pub policy_override: Option<PackPolicy>,
    pub vector_input: Option<PathBuf>,
    pub expansion_input: Option<PathBuf>,
    pub splade_input: Option<PathBuf>,
    pub anchors_input: Option<PathBuf>,
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
    let (passage_index, passage_data) = encode_passages(corpus)?;
    let (lexical_dictionary, lexical_postings) = build_lexical_index(corpus)?;
    let term_count = lexical_dictionary.terms.len();
    let lexical_dictionary = serde_json::to_vec(&lexical_dictionary)?;

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
        capabilities.push("hybrid-rrf".to_string());
    }
    if corpus.input_format == InputFormat::Okf {
        capabilities.push("source-okf".to_string());
    }

    // ANN-7/8/9: consume pinned, hashed sidecars. No model runs here; the
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
    if let Some(path) = &options.anchors_input {
        let (sidecar, digest) = crate::derive::read_anchor_sidecar(path)?;
        let built = crate::derive::build_anchors(
            &sidecar,
            &digest,
            ANCHOR_SET_SECTION_ID,
            ANCHOR_COORDINATES_SECTION_ID,
            &corpus.passages,
        )?;
        // ANN-9 relative-coordinate retrieval was withdrawn: the anchor sections
        // still ship (they are the supervision an anchor-supervised adapter uses),
        // but the pack no longer advertises "anchor-relative" as a retrieval
        // capability, and no anchor retrieval profile is emitted below.
        derived_sections.push(built.anchor_set);
        derived_sections.push(built.coordinates);
        derived_inputs.push(built.derived_input);
    }
    capabilities.sort();

    // ANN-10: fat-pack fallback order. Highest-capability profile first, always
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
    // ANN-9 anchors are intentionally NOT advertised as a retrieval profile:
    // relative-coordinate retrieval was withdrawn. The anchor sections still ship
    // as adapter supervision, but they are not a runtime-selectable profile.
    // Only advertise the fat-pack descriptor when two or more optional
    // representations coexist and the runtime must actually choose. A pack with
    // a single optional profile (e.g. ANN-1 vectors only) is not a fat pack.
    if retrieval_profiles.len() >= 2 {
        retrieval_profiles.push(crate::model::RetrievalProfile {
            id: "lexical".into(),
            kind: "lexical".into(),
            section_ids: vec![
                PASSAGE_INDEX_SECTION_ID,
                PASSAGE_DATA_SECTION_ID,
                LEXICAL_DICTIONARY_SECTION_ID,
                LEXICAL_POSTINGS_SECTION_ID,
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
        builder: format!("annpack-reference/{}", env!("CARGO_PKG_VERSION")),
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
            payment: None,
            encryption: None,
        }),
        dependencies: options.dependencies.clone(),
        source: (corpus.input_format == InputFormat::Okf).then(|| crate::model::SourceDescriptor {
            format: "okf".into(),
            version: corpus.input_format_version.clone(),
            digest_algorithm: "blake3".into(),
            digest: corpus.source_digest.clone(),
        }),
        derived_inputs,
        retrieval_profiles,
    };

    let mut writer = PackWriter::new();
    writer.push(SectionData::required(
        MANIFEST_SECTION_ID,
        SectionType::Manifest,
        1,
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
    writer.push(SectionData::required_deflate(
        LEXICAL_DICTIONARY_SECTION_ID,
        SectionType::LexicalDictionary,
        term_count as u64,
        lexical_dictionary,
    ))?;
    writer.push(SectionData::required_deflate(
        LEXICAL_POSTINGS_SECTION_ID,
        SectionType::LexicalPostings,
        term_count as u64,
        lexical_postings,
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

fn encode_passages(corpus: &IngestedCorpus) -> Result<(StoredPassageIndex, Vec<u8>)> {
    let mut data = Vec::new();
    let mut records = Vec::with_capacity(corpus.passages.len());
    let mut blocks = Vec::new();
    let mut logical_block = Vec::new();
    for passage in &corpus.passages {
        let bytes = serde_json::to_vec(passage)?;
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
        },
        data,
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
