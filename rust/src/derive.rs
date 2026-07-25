//! ANN-7 / ANN-8 / ANN-9 offline derivation.
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
use crate::model::{
    AnchorCoordinatesSection, AnchorSetSection, DerivedInput, OverlayVocabulary, Passage,
    TermOverlaySection,
};
use crate::search::tokenize;

pub const EXPANSION_KIND: &str = "expansion-v1";
pub const SPLADE_KIND: &str = "splade-v1";
const MAX_OVERLAY_TERMS: usize = 1 << 20;
const MAX_ANCHORS: usize = 4_096;

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

#[derive(Debug, Clone, Deserialize)]
pub struct RawAnchors {
    pub space_id: String,
    #[serde(default = "default_metric")]
    pub metric: String,
    #[serde(default = "default_anchor_quantization")]
    pub quantization: String,
    #[serde(default = "default_anchor_scale")]
    pub scale: f64,
    pub anchors: Vec<String>,
    pub passages: Vec<RawAnchorPassage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawAnchorPassage {
    pub passage_id: String,
    pub similarities: Vec<f64>,
}

fn default_metric() -> String {
    "cosine".into()
}
fn default_anchor_quantization() -> String {
    "linear-i16".into()
}
fn default_anchor_scale() -> f64 {
    0.0001
}

// ---------------------------------------------------------------------------
// Pinned sidecars (canonical, hashed, committed). Keyed by passage id so they
// are independent of any particular corpus ordering.
// ---------------------------------------------------------------------------

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnchorSidecar {
    pub space_id: String,
    pub metric: String,
    pub quantization: String,
    pub scale: f64,
    pub anchors: Vec<String>,
    /// passage_id -> quantized similarity row (length == anchors.len()).
    pub passages: BTreeMap<String, Vec<i32>>,
}

/// Canonical serialization is deterministic UTF-8 JSON with sorted maps; the
/// digest is over exactly those bytes, so it pins the file that is committed.
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

pub fn generate_anchors(raw: &RawAnchors) -> Result<AnchorSidecar> {
    if raw.space_id.trim().is_empty() {
        return Err(AnnpackError::InvalidInput(
            "anchor space_id must not be empty".into(),
        ));
    }
    if raw.anchors.is_empty() || raw.anchors.len() > MAX_ANCHORS {
        return Err(AnnpackError::InvalidInput(format!(
            "anchor count must be between 1 and {MAX_ANCHORS}"
        )));
    }
    if raw.metric != "cosine" {
        return Err(AnnpackError::InvalidInput(
            "the reference anchor generator supports metric=cosine".into(),
        ));
    }
    if raw.quantization != "linear-i16" {
        return Err(AnnpackError::InvalidInput(
            "the reference anchor generator supports quantization=linear-i16".into(),
        ));
    }
    if !raw.scale.is_finite() || raw.scale <= 0.0 {
        return Err(AnnpackError::InvalidInput(
            "anchor scale must be a positive finite number".into(),
        ));
    }
    let anchor_count = raw.anchors.len();
    let mut passages = BTreeMap::new();
    for passage in &raw.passages {
        if passage.similarities.len() != anchor_count {
            return Err(AnnpackError::InvalidInput(format!(
                "passage {} has {} similarities, expected {anchor_count}",
                passage.passage_id,
                passage.similarities.len()
            )));
        }
        let mut row = Vec::with_capacity(anchor_count);
        for value in &passage.similarities {
            if !value.is_finite() {
                return Err(AnnpackError::InvalidInput(format!(
                    "passage {} has a non-finite anchor similarity",
                    passage.passage_id
                )));
            }
            let quantized = (value / raw.scale)
                .round()
                .clamp(i16::MIN as f64, i16::MAX as f64) as i32;
            row.push(quantized);
        }
        passages.insert(passage.passage_id.clone(), row);
    }
    Ok(AnchorSidecar {
        space_id: raw.space_id.clone(),
        metric: raw.metric.clone(),
        quantization: raw.quantization.clone(),
        scale: raw.scale,
        anchors: raw.anchors.clone(),
        passages,
    })
}

// ---------------------------------------------------------------------------
// consume: pinned sidecar -> derived section(s), mapped onto corpus ordinals.
// ---------------------------------------------------------------------------

/// Result of consuming one overlay sidecar during a build.
#[derive(Debug)]
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

pub fn read_anchor_sidecar(path: &Path) -> Result<(AnchorSidecar, String)> {
    let bytes = fs::read(path)?;
    let sidecar: AnchorSidecar = serde_json::from_slice(&bytes)?;
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

/// The two derived sections plus provenance produced by an anchor sidecar.
#[derive(Debug)]
pub struct BuiltAnchors {
    pub anchor_set: SectionData,
    pub coordinates: SectionData,
    pub derived_input: DerivedInput,
}

pub fn build_anchors(
    sidecar: &AnchorSidecar,
    digest: &str,
    anchor_set_section_id: u32,
    coordinates_section_id: u32,
    passages: &[Passage],
) -> Result<BuiltAnchors> {
    if sidecar.anchors.is_empty() || sidecar.anchors.len() > MAX_ANCHORS {
        return Err(AnnpackError::InvalidInput(
            "anchor set is empty or exceeds the anchor limit".into(),
        ));
    }
    let anchor_count = sidecar.anchors.len();
    let mut coordinates = Vec::with_capacity(passages.len());
    for passage in passages {
        let row = sidecar.passages.get(&passage.id).ok_or_else(|| {
            AnnpackError::InvalidInput(format!(
                "anchor sidecar is missing coordinates for passage {}",
                passage.id
            ))
        })?;
        if row.len() != anchor_count {
            return Err(AnnpackError::InvalidInput(format!(
                "anchor coordinates for passage {} have length {}, expected {anchor_count}",
                passage.id,
                row.len()
            )));
        }
        coordinates.push(row.clone());
    }
    let anchor_set = AnchorSetSection {
        space_id: sidecar.space_id.clone(),
        anchors: sidecar.anchors.clone(),
    };
    let coordinates_model = AnchorCoordinatesSection {
        space_id: sidecar.space_id.clone(),
        metric: sidecar.metric.clone(),
        quantization: sidecar.quantization.clone(),
        scale: sidecar.scale,
        coordinates,
    };
    let mut params = BTreeMap::new();
    params.insert("space_id".into(), sidecar.space_id.clone());
    params.insert("metric".into(), sidecar.metric.clone());
    params.insert("quantization".into(), sidecar.quantization.clone());
    params.insert("scale".into(), format!("{}", sidecar.scale));
    params.insert("anchors".into(), format!("{anchor_count}"));
    Ok(BuiltAnchors {
        anchor_set: SectionData::optional_deflate(
            anchor_set_section_id,
            SectionType::AnchorSet,
            anchor_count as u64,
            serde_json::to_vec(&anchor_set)?,
        ),
        coordinates: SectionData::derived_deflate(
            coordinates_section_id,
            SectionType::AnchorCoordinates,
            passages.len() as u64,
            serde_json::to_vec(&coordinates_model)?,
        ),
        derived_input: DerivedInput {
            kind: "anchor-v1".into(),
            section_id: coordinates_section_id,
            generator: "anchor-ref".into(),
            model: sidecar.space_id.clone(),
            revision: sidecar.space_id.clone(),
            params,
            sidecar_digest: digest.into(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passage(id: &str, ordinal: u32) -> Passage {
        Passage {
            id: id.into(),
            document_id: "doc".into(),
            ordinal,
            heading_path: Vec::new(),
            anchor: None,
            text: String::new(),
            source_byte_start: None,
            source_byte_end: None,
        }
    }

    #[test]
    fn expansion_threshold_drops_low_relevance_candidates() {
        let raw = RawExpansion {
            generator: "g".into(),
            model: "m".into(),
            revision: "r".into(),
            passages: vec![RawExpansionPassage {
                passage_id: "p0".into(),
                candidates: vec![
                    RawCandidate {
                        text: "keep this".into(),
                        score: 0.9,
                    },
                    RawCandidate {
                        text: "drop noise".into(),
                        score: 0.1,
                    },
                ],
            }],
        };
        let sidecar = generate_expansion(&raw, 0.5).unwrap();
        let terms = &sidecar.passages["p0"];
        assert!(terms.contains_key("keep"));
        assert!(!terms.contains_key("noise"));
    }

    #[test]
    fn expansion_generation_is_deterministic() {
        let raw = RawExpansion {
            generator: "g".into(),
            model: "m".into(),
            revision: "r".into(),
            passages: vec![RawExpansionPassage {
                passage_id: "p0".into(),
                candidates: vec![RawCandidate {
                    text: "alpha beta".into(),
                    score: 0.9,
                }],
            }],
        };
        let first = serde_json::to_vec(&generate_expansion(&raw, 0.5).unwrap()).unwrap();
        let second = serde_json::to_vec(&generate_expansion(&raw, 0.5).unwrap()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn build_overlay_rejects_unknown_passage_id() {
        let sidecar = OverlaySidecar {
            kind: EXPANSION_KIND.into(),
            generator: "g".into(),
            model: "m".into(),
            revision: "r".into(),
            threshold: Some(0.5),
            vocabulary: None,
            passages: [(
                "missing".to_string(),
                [("t".to_string(), 1_u32)].into_iter().collect(),
            )]
            .into_iter()
            .collect(),
        };
        let error = build_overlay(&sidecar, "digest", 13, &[passage("p0", 0)]).unwrap_err();
        assert!(matches!(error, AnnpackError::InvalidInput(_)));
    }

    #[test]
    fn anchors_reject_ragged_rows() {
        let raw = RawAnchors {
            space_id: "s".into(),
            metric: "cosine".into(),
            quantization: "linear-i16".into(),
            scale: 0.0001,
            anchors: vec!["a".into(), "b".into()],
            passages: vec![RawAnchorPassage {
                passage_id: "p0".into(),
                similarities: vec![0.1], // wrong length
            }],
        };
        assert!(matches!(
            generate_anchors(&raw),
            Err(AnnpackError::InvalidInput(_))
        ));
    }

    #[test]
    fn build_anchors_rejects_missing_passage() {
        let sidecar = AnchorSidecar {
            space_id: "s".into(),
            metric: "cosine".into(),
            quantization: "linear-i16".into(),
            scale: 0.0001,
            anchors: vec!["a".into()],
            passages: BTreeMap::new(),
        };
        let error = build_anchors(&sidecar, "d", 15, 16, &[passage("p0", 0)]).unwrap_err();
        assert!(matches!(error, AnnpackError::InvalidInput(_)));
    }

    #[test]
    fn splade_requires_positive_scale() {
        let raw = RawSplade {
            generator: "g".into(),
            model: "m".into(),
            revision: "r".into(),
            vocabulary: OverlayVocabulary {
                id: "v".into(),
                size: 10,
                quantization: "linear-u16".into(),
                scale: 0.0,
            },
            passages: Vec::new(),
        };
        assert!(matches!(
            generate_splade(&raw),
            Err(AnnpackError::InvalidInput(_))
        ));
    }
}
