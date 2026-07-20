use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::format::{FORMAT_VERSION, PackReader, SectionType};
use crate::model::Manifest;

pub const CORE_PROFILE: &str = "annpack-core-v1.0-draft";
pub const VECTOR_EXTENSION: &str = "ANN-1";
pub const POLICY_EXTENSION: &str = "ANN-5";
pub const DEPENDENCY_EXTENSION: &str = "ANN-6";

const CORE_CAPABILITIES: [&str; 5] = [
    "citations",
    "content",
    "lexical-bm25",
    "range-addressable-passages",
    "section-integrity",
];

const CORE_SECTIONS: [SectionType; 6] = [
    SectionType::Manifest,
    SectionType::Documents,
    SectionType::PassageIndex,
    SectionType::PassageData,
    SectionType::LexicalDictionary,
    SectionType::LexicalPostings,
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConformanceReport {
    pub wire_format: String,
    pub core_profile: String,
    pub core_conformant: bool,
    pub extensions: Vec<String>,
    pub issues: Vec<String>,
}

pub fn inspect_conformance(reader: &PackReader) -> Result<ConformanceReport> {
    let manifest = reader.manifest()?;
    Ok(inspect_conformance_with_manifest(reader, &manifest))
}

pub fn inspect_conformance_with_manifest(
    reader: &PackReader,
    manifest: &Manifest,
) -> ConformanceReport {
    let mut issues = Vec::new();
    for section_type in CORE_SECTIONS {
        match reader.first_entry(section_type) {
            Some(entry) if !entry.required() => issues.push(format!(
                "core section {} is not marked required",
                section_type.name()
            )),
            Some(entry) if entry.format_version != 1 => issues.push(format!(
                "core section {} uses unsupported section format {}",
                section_type.name(),
                entry.format_version
            )),
            Some(_) => {}
            None => issues.push(format!("core section {} is missing", section_type.name())),
        }
    }
    for capability in CORE_CAPABILITIES {
        if !manifest
            .capabilities
            .iter()
            .any(|value| value == capability)
        {
            issues.push(format!("core capability {capability} is not declared"));
        }
    }
    let core_conformant = issues.is_empty();

    let mut extensions = Vec::new();
    let vector_sections = [
        SectionType::VectorProfile,
        SectionType::VectorData,
        SectionType::VectorIndex,
    ];
    let vector_count = vector_sections
        .iter()
        .filter(|section_type| reader.first_entry(**section_type).is_some())
        .count();
    if vector_count == vector_sections.len() {
        extensions.push(VECTOR_EXTENSION.to_string());
    } else if vector_count != 0 {
        issues.push("ANN-1 vector sections are incomplete".to_string());
    }
    if manifest.policy.payment.is_some() || manifest.policy.encryption.is_some() {
        extensions.push(POLICY_EXTENSION.to_string());
    }
    if !manifest.dependencies.is_empty() {
        extensions.push(DEPENDENCY_EXTENSION.to_string());
    }
    extensions.sort();

    ConformanceReport {
        wire_format: format!("ANNPACK{FORMAT_VERSION}"),
        core_profile: CORE_PROFILE.to_string(),
        core_conformant,
        extensions,
        issues,
    }
}
