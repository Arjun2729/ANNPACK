use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::format::{FORMAT_VERSION, PackReader, SectionType};
use crate::model::Manifest;

// FROZEN WIRE IDENTIFIER: serialized and matched by third parties. It names a
// format version, not a project, and changes only when that version does.
pub const CORE_PROFILE: &str = "annpack-core-v1.0-draft";
pub const VECTOR_EXTENSION: &str = "AN-1";
pub const EXPANSION_EXTENSION: &str = "AN-7";
pub const SPLADE_EXTENSION: &str = "AN-8";
pub const MULTI_PROFILE_EXTENSION: &str = "AN-10";

const CORE_CAPABILITIES: [&str; 5] = [
    "citations",
    "content",
    "lexical-bm25",
    "range-addressable-passages",
    "section-integrity",
];

/// Capabilities an AN-10 profile may name in `requires`. A profile naming
/// anything outside this set cannot be evaluated for support, so it is a
/// descriptor error rather than a merely-unsupported profile.
const KNOWN_PROFILE_CAPABILITIES: [&str; 5] = [
    "lexical-bm25",
    "vector-flat-dot",
    "vector-ivf-flat-dot",
    "term-overlay-expansion",
    "term-overlay-splade",
];

const CORE_SECTIONS: [SectionType; 6] = [
    SectionType::Manifest,
    SectionType::Documents,
    SectionType::PassageIndex,
    SectionType::PassageData,
    SectionType::LexicalDictionary,
    SectionType::LexicalPostings,
];

/// How strongly an artifact binds the source bytes it was built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceBinding {
    /// The artifact commits to a well-formed source digest. Provenance naming a
    /// different digest is detectable.
    Authenticated,
    /// Manifest format is below 4, which predates the requirement. The artifact
    /// legitimately cannot say, and this is not corruption.
    AbsentLegacyArtifact,
    /// Format 4 or later, and the descriptor is missing or malformed.
    Malformed,
    /// The manifest format version is not one this reader implements, so no
    /// statement about source binding can be made.
    UnsupportedVersion,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConformanceReport {
    pub wire_format: String,
    pub core_profile: String,
    /// True when the Core profile is intact. Deliberately independent of every
    /// extension check: a malformed optional descriptor must never be able to
    /// make a structurally valid Core pack unopenable, and must never be able to
    /// influence the default lexical path.
    pub core_conformant: bool,
    /// True when every *present* extension also validates. A pack can be
    /// `core_conformant` and not `extensions_conformant`; in that state the
    /// runtime serves Core lexical and refuses profile-enabled retrieval.
    pub extensions_conformant: bool,
    pub extensions: Vec<String>,
    /// Core issues only. Empty whenever `core_conformant` is true.
    pub core_issues: Vec<String>,
    /// Extension issues only. Empty whenever `extensions_conformant` is true.
    pub extension_issues: Vec<String>,
    /// Core and extension issues combined, in that order.
    pub issues: Vec<String>,
    /// Whether the artifact commits to the source bytes it was built from.
    ///
    /// Reported separately from conformance because absence is legitimate
    /// history for a manifest older than format 4, not a defect. A verifier
    /// consuming build provenance needs to distinguish "the artifact agrees
    /// with the builder's digest" from "the artifact cannot say" (ADR-0005).
    pub source_binding: SourceBinding,
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
        // Two Core sections carry schema versions independent of the wire
        // format: the Manifest, and the postings section, whose format 2 is the
        // block-addressable layout. Every other Core section is v1 only.
        let accepted: &[u16] = match section_type {
            SectionType::Manifest => crate::format::SUPPORTED_MANIFEST_FORMAT_VERSIONS,
            SectionType::LexicalPostings => crate::format::SUPPORTED_LEXICAL_FORMAT_VERSIONS,
            _ => &[1],
        };
        match reader.first_entry(section_type) {
            Some(entry) if !entry.required() => issues.push(format!(
                "core section {} is not marked required",
                section_type.name()
            )),
            Some(entry) if !accepted.contains(&entry.format_version) => issues.push(format!(
                "core section {} uses unsupported section format {}",
                section_type.name(),
                entry.format_version
            )),
            Some(_) => {}
            None => issues.push(format!("core section {} is missing", section_type.name())),
        }
    }
    // The logical content root is required from manifest section format 2 on.
    // A format-2 pack missing it is not Core-conformant, however well-formed the
    // rest of the container is.
    if let Some(entry) = reader.first_entry(SectionType::Manifest)
        && let Some(issue) =
            crate::format::manifest_logical_root_issue(manifest, entry.format_version)
    {
        issues.push(issue);
    }
    let source_binding = match reader.first_entry(SectionType::Manifest) {
        None => SourceBinding::UnsupportedVersion,
        Some(entry)
            if !crate::format::SUPPORTED_MANIFEST_FORMAT_VERSIONS
                .contains(&entry.format_version) =>
        {
            SourceBinding::UnsupportedVersion
        }
        Some(entry) if entry.format_version < 4 => SourceBinding::AbsentLegacyArtifact,
        Some(entry) => {
            match crate::format::manifest_source_digest_issue(manifest, entry.format_version) {
                Some(issue) => {
                    // Format 4 requires it, so a missing or malformed descriptor
                    // is a Core defect and not merely a weaker binding.
                    issues.push(issue);
                    SourceBinding::Malformed
                }
                None => SourceBinding::Authenticated,
            }
        }
    };
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
    // Everything appended from here on is an extension issue. Splitting the
    // vectors at this index keeps the two verdicts genuinely independent.
    let core_issue_count = issues.len();

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
        issues.push("AN-1 vector sections are incomplete".to_string());
    }

    // AN-7 / AN-8: term overlays (section type 13) must be optional and
    // derived. A derived section is matching-only and never citable.
    let overlay_present = reader.first_entry(SectionType::TermOverlay).is_some();
    if overlay_present {
        for entry in reader.entries_of_type(SectionType::TermOverlay) {
            if entry.required() {
                issues.push("term overlay section must not be required".to_string());
            }
            if !entry.derived() {
                issues.push("term overlay section must be flagged derived".to_string());
            }
        }
    }
    if manifest
        .capabilities
        .iter()
        .any(|capability| capability == "term-overlay-expansion")
    {
        extensions.push(EXPANSION_EXTENSION.to_string());
    }
    if manifest
        .capabilities
        .iter()
        .any(|capability| capability == "term-overlay-splade")
    {
        extensions.push(SPLADE_EXTENSION.to_string());
    }

    // AN-10: fat-pack descriptor. Fallback order must end at Core lexical, ids
    // must reference present sections, and profile ids must be unique.
    if !manifest.retrieval_profiles.is_empty() {
        extensions.push(MULTI_PROFILE_EXTENSION.to_string());
        match manifest.retrieval_profiles.last() {
            Some(profile) if profile.kind == "lexical" => {}
            _ => issues.push(
                "retrieval_profiles fallback order must end at the Core lexical profile"
                    .to_string(),
            ),
        }
        let mut seen_ids = std::collections::BTreeSet::new();
        for profile in &manifest.retrieval_profiles {
            if !seen_ids.insert(profile.id.as_str()) {
                issues.push(format!("duplicate retrieval profile id {:?}", profile.id));
            }
            // `requires` drives runtime selection. An empty list satisfies an
            // `all()` check vacuously, which would make any profile look
            // supported, so it is rejected outright.
            if profile.requires.is_empty() {
                issues.push(format!(
                    "retrieval profile {:?} declares no required capabilities",
                    profile.id
                ));
            }
            for capability in &profile.requires {
                if !KNOWN_PROFILE_CAPABILITIES.contains(&capability.as_str()) {
                    issues.push(format!(
                        "retrieval profile {:?} requires unknown capability {capability:?}",
                        profile.id
                    ));
                }
            }
            // The section types a profile of this kind is permitted to reference.
            let allowed: Option<&[SectionType]> = match profile.kind.as_str() {
                "lexical" => Some(&[
                    SectionType::PassageIndex,
                    SectionType::PassageData,
                    SectionType::LexicalDictionary,
                    SectionType::LexicalPostings,
                    // Format-2 lexical retrieval reads both of these on every
                    // query, so a lexical profile owns them. Omitting them here
                    // made the complete declaration illegal to express.
                    SectionType::LexicalTerms,
                    SectionType::PassageRecords,
                ]),
                "vector" => Some(&[
                    SectionType::VectorProfile,
                    SectionType::VectorData,
                    SectionType::VectorIndex,
                ]),
                // Expansion and splade both ship as term overlays.
                "expansion" | "splade" => Some(&[SectionType::TermOverlay]),
                // An unrecognized kind is not runtime-selectable. Flag it rather
                // than leaving it unconstrained: a reader that silently executes
                // an unknown kind as lexical would report a retrieval strategy
                // that never ran.
                other => {
                    issues.push(format!(
                        "retrieval profile {:?} declares unrecognized kind {other:?}",
                        profile.id
                    ));
                    None
                }
            };
            if profile.section_ids.is_empty() {
                issues.push(format!(
                    "retrieval profile {:?} references no sections",
                    profile.id
                ));
            }
            for section_id in &profile.section_ids {
                match reader.entry(*section_id) {
                    Err(_) => issues.push(format!(
                        "retrieval profile {:?} references missing section {section_id}",
                        profile.id
                    )),
                    Ok(entry) => {
                        if let Some(allowed) = allowed
                            && !allowed.contains(&entry.section_type)
                        {
                            issues.push(format!(
                                "retrieval profile {:?} (kind {:?}) references section {section_id} \
                                 of incompatible type {:?}",
                                profile.id, profile.kind, entry.section_type
                            ));
                        }
                    }
                }
            }
        }
    }

    extensions.sort();

    let core_issues = issues[..core_issue_count].to_vec();
    let extension_issues = issues[core_issue_count..].to_vec();
    ConformanceReport {
        wire_format: format!("ANNPACK{FORMAT_VERSION}"),
        core_profile: CORE_PROFILE.to_string(),
        core_conformant,
        extensions_conformant: extension_issues.is_empty(),
        extensions,
        core_issues,
        extension_issues,
        issues,
        source_binding,
    }
}
