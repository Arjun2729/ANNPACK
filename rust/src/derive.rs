//! AN-7 / AN-8 offline derivation.
//!
//! Semantic understanding is produced by an external model, out of band, into a
//! *raw* sidecar. The deterministic `generate` commands filter, quantize, and
//! canonicalize that raw input into a *pinned* sidecar and report its BLAKE3
//! digest. The `build` command consumes a pinned sidecar as an input, maps its
//! passage-id-keyed contents onto the deterministic corpus ordinals, records the
//! sidecar digest in `manifest.derived_inputs`, and writes a derived section.
//!
//! No model runs inside `generate` or `build`; both are pure transforms, so a
//! second build from identical inputs is byte-identical.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{AnnpackError, Result};
use crate::format::{SectionData, SectionType};
use crate::model::{DerivedInput, OverlayVocabulary, Passage, TermOverlaySection};
use crate::search::tokenize;

pub const EXPANSION_KIND: &str = "expansion-v1";
pub const SPLADE_KIND: &str = "splade-v1";
const MAX_OVERLAY_TERMS: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Raw sidecars (external model output).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct RawExpansion {
    pub generator: String,
    pub model: String,
    pub revision: String,
    pub passages: Vec<RawExpansionPassage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawExpansionPassage {
    pub passage_id: String,
    pub candidates: Vec<RawCandidate>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawCandidate {
    pub text: String,
    pub score: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawSplade {
    pub generator: String,
    pub model: String,
    pub revision: String,
    pub vocabulary: OverlayVocabulary,
    pub passages: Vec<RawSpladePassage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawSpladePassage {
    pub passage_id: String,
    pub weights: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OverlaySidecar {
    pub kind: String,
    pub generator: String,
    pub model: String,
    pub revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocabulary: Option<OverlayVocabulary>,
    /// passage_id -> (term -> weight), all maps lexicographically ordered.
    pub passages: BTreeMap<String, BTreeMap<String, u32>>,
}

pub fn sidecar_digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

// ---------------------------------------------------------------------------
// generate: raw -> pinned. Deterministic; no model runs here.
// ---------------------------------------------------------------------------

pub fn generate_expansion(raw: &RawExpansion, threshold: f64) -> Result<OverlaySidecar> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(AnnpackError::InvalidInput(
            "expansion threshold must be within [0, 1]".into(),
        ));
    }
    let mut passages = BTreeMap::new();
    for passage in &raw.passages {
        let mut terms: BTreeMap<String, u32> = BTreeMap::new();
        for candidate in &passage.candidates {
            // "When less is more": drop low-relevance generated queries before
            // inclusion so hallucinated terms do not enter the index.
            if !candidate.score.is_finite() || candidate.score < threshold {
                continue;
            }
            for token in tokenize(&candidate.text) {
                let entry = terms.entry(token).or_default();
                *entry = entry.saturating_add(1);
            }
        }
        if !terms.is_empty() {
            passages.insert(passage.passage_id.clone(), terms);
        }
    }
    Ok(OverlaySidecar {
        kind: EXPANSION_KIND.into(),
        generator: raw.generator.clone(),
        model: raw.model.clone(),
        revision: raw.revision.clone(),
        threshold: Some(threshold),
        vocabulary: None,
        passages,
    })
}

pub fn generate_splade(raw: &RawSplade) -> Result<OverlaySidecar> {
    let scale = raw.vocabulary.scale;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(AnnpackError::InvalidInput(
            "splade vocabulary scale must be a positive finite number".into(),
        ));
    }
    if raw.vocabulary.id.trim().is_empty() {
        return Err(AnnpackError::InvalidInput(
            "splade vocabulary id must not be empty".into(),
        ));
    }
    if raw.vocabulary.quantization != "linear-u16" {
        return Err(AnnpackError::InvalidInput(
            "the reference splade generator supports quantization=linear-u16".into(),
        ));
    }
    let mut passages = BTreeMap::new();
    for passage in &raw.passages {
        let mut terms: BTreeMap<String, u32> = BTreeMap::new();
        for (term, weight) in &passage.weights {
            if !weight.is_finite() || *weight <= 0.0 {
                continue;
            }
            let quantized = (weight / scale).round();
            if quantized < 1.0 {
                continue;
            }
            let quantized = quantized.min(u16::MAX as f64) as u32;
            terms.insert(term.clone(), quantized);
        }
        if !terms.is_empty() {
            passages.insert(passage.passage_id.clone(), terms);
        }
    }
    Ok(OverlaySidecar {
        kind: SPLADE_KIND.into(),
        generator: raw.generator.clone(),
        model: raw.model.clone(),
        revision: raw.revision.clone(),
        threshold: None,
        vocabulary: Some(raw.vocabulary.clone()),
        passages,
    })
}

pub struct BuiltOverlay {
    pub section: SectionData,
    pub derived_input: DerivedInput,
    pub kind: String,
}

pub fn read_overlay_sidecar(path: &Path) -> Result<(OverlaySidecar, String)> {
    let bytes = fs::read(path)?;
    let sidecar: OverlaySidecar = serde_json::from_slice(&bytes)?;
    Ok((sidecar, sidecar_digest(&bytes)))
}

fn passage_ordinals(passages: &[Passage]) -> HashMap<&str, u32> {
    passages
        .iter()
        .map(|passage| (passage.id.as_str(), passage.ordinal))
        .collect()
}

/// Invert a passage-keyed overlay sidecar into an ordinal-keyed term index.
pub fn build_overlay(
    sidecar: &OverlaySidecar,
    digest: &str,
    section_id: u32,
    passages: &[Passage],
) -> Result<BuiltOverlay> {
    if sidecar.kind != EXPANSION_KIND && sidecar.kind != SPLADE_KIND {
        return Err(AnnpackError::InvalidInput(format!(
            "unrecognized overlay kind {:?}",
            sidecar.kind
        )));
    }
    if sidecar.kind == SPLADE_KIND {
        match &sidecar.vocabulary {
            Some(vocabulary) if !vocabulary.id.trim().is_empty() => {}
            _ => {
                return Err(AnnpackError::InvalidInput(
                    "splade overlay requires a non-empty vocabulary id".into(),
                ));
            }
        }
    }
    let ordinals = passage_ordinals(passages);
    let mut terms: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
    for (passage_id, term_weights) in &sidecar.passages {
        let ordinal = *ordinals.get(passage_id.as_str()).ok_or_else(|| {
            AnnpackError::InvalidInput(format!(
                "overlay sidecar references passage {passage_id} that is not in the corpus"
            ))
        })?;
        for (term, weight) in term_weights {
            if *weight == 0 {
                continue;
            }
            terms
                .entry(term.clone())
                .or_default()
                .push((ordinal, *weight));
        }
    }
    if terms.len() > MAX_OVERLAY_TERMS {
        return Err(AnnpackError::InvalidInput(
            "overlay exceeds the term limit".into(),
        ));
    }
    for postings in terms.values_mut() {
        postings.sort_by_key(|(ordinal, _)| *ordinal);
    }
    let section_model = TermOverlaySection {
        kind: sidecar.kind.clone(),
        generator: sidecar.generator.clone(),
        model: sidecar.model.clone(),
        revision: sidecar.revision.clone(),
        threshold: sidecar.threshold,
        vocabulary: sidecar.vocabulary.clone(),
        terms,
    };
    let item_count = section_model.terms.len() as u64;
    let bytes = serde_json::to_vec(&section_model)?;
    let mut params = BTreeMap::new();
    if let Some(threshold) = sidecar.threshold {
        params.insert("threshold".into(), format!("{threshold}"));
    }
    if let Some(vocabulary) = &sidecar.vocabulary {
        params.insert("vocabulary_id".into(), vocabulary.id.clone());
        params.insert("quantization".into(), vocabulary.quantization.clone());
        params.insert("scale".into(), format!("{}", vocabulary.scale));
    }
    Ok(BuiltOverlay {
        section: SectionData::derived_deflate(
            section_id,
            SectionType::TermOverlay,
            item_count,
            bytes,
        ),
        derived_input: DerivedInput {
            kind: sidecar.kind.clone(),
            section_id,
            generator: sidecar.generator.clone(),
            model: sidecar.model.clone(),
            revision: sidecar.revision.clone(),
            params,
            sidecar_digest: digest.into(),
        },
        kind: sidecar.kind.clone(),
    })
}
