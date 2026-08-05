use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use crate::error::{AnnpackError, Result};
use crate::model::Manifest;
use crate::reader::{FileReader, SharedReader};

pub const MAGIC: &[u8; 8] = b"ANNPACK3";
pub const FORMAT_VERSION: u32 = 3;
pub const HEADER_SIZE: usize = 128;
pub const DIRECTORY_ENTRY_SIZE: usize = 80;
pub const FLAG_REQUIRED: u16 = 1;
/// Bit one marks a section as derived: produced from passage text by an offline
/// model, matching-only, and never citable in an evidence envelope. See ANN-7.
pub const FLAG_DERIVED: u16 = 2;
pub const MAX_SECTIONS: u32 = 16_384;
/// Section format version emitted for the Manifest section.
///
/// Version 2 (v0.4.0) removed the required `builder` field and added the
/// required `passage_merkle_root` logical content root. Bumping the section
/// format version is what makes that schema change explicit: a v1-only reader
/// declines a v2 manifest instead of failing deep in JSON deserialization.
/// v0.3.1 changed the schema *without* this bump, which is the compatibility
/// defect v0.4.0 corrects.
///
/// Version 3 (v0.5.0) removed the `dependencies` array and the policy
/// `payment` and `encryption` descriptors along with ANN-6 and ANN-5. A v2
/// reader requires `dependencies` to be present, so it must decline a v3
/// manifest rather than fail mid-deserialization -- the same discipline, applied
/// to a removal instead of an addition.
pub const MANIFEST_FORMAT_VERSION: u16 = 3;
/// Manifest section format versions this reader accepts.
pub const SUPPORTED_MANIFEST_FORMAT_VERSIONS: &[u16] = &[1, 2, 3];
/// Lexical index section format versions this reader accepts.
///
/// 1 is the original monolithic layout: the term table inline in the dictionary
/// section, the posting stream one deflated section. 2 partitions both into
/// independently hashed blocks so a term costs a bounded range read instead of
/// the whole index. Both remain readable.
pub const SUPPORTED_LEXICAL_FORMAT_VERSIONS: &[u16] = &[1, 2];
pub const MAX_MANIFEST_SIZE: u64 = 4 * 1024 * 1024;
pub const MAX_SECTION_SIZE: u64 = 64 * 1024 * 1024 * 1024;
pub const DECOMPRESSION_RATIO_LIMIT: u64 = 256;
pub const DECOMPRESSION_RATIO_FLOOR: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SectionType {
    Manifest,
    Documents,
    PassageIndex,
    PassageData,
    LexicalDictionary,
    LexicalPostings,
    VectorProfile,
    VectorData,
    VectorIndex,
    Signature,
    DeltaManifest,
    TermOverlay,
    /// Block-addressable term table (lexical index format 2). Optional: a
    /// format-1 pack carries its terms inline in the dictionary section.
    ///
    /// Types 11, 14, and 15 are retired: they were ANN-5 policy and ANN-9
    /// anchor sections, both withdrawn. Their numbers are not reused. A pack
    /// carrying one now decodes it as an unknown optional section and ignores
    /// it, which is the defined behavior for an optional section a reader does
    /// not recognize.
    LexicalTerms,
    /// Block-addressable passage record table (passage index format 2).
    PassageRecords,
    Other(u16),
}

impl SectionType {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::Manifest => 1,
            Self::Documents => 2,
            Self::PassageIndex => 3,
            Self::PassageData => 4,
            Self::LexicalDictionary => 5,
            Self::LexicalPostings => 6,
            Self::VectorProfile => 7,
            Self::VectorData => 8,
            Self::VectorIndex => 9,
            Self::Signature => 10,
            Self::DeltaManifest => 12,
            Self::TermOverlay => 13,
            Self::LexicalTerms => 16,
            Self::PassageRecords => 17,
            Self::Other(value) => value,
        }
    }

    pub fn from_u16(value: u16) -> Self {
        match value {
            1 => Self::Manifest,
            2 => Self::Documents,
            3 => Self::PassageIndex,
            4 => Self::PassageData,
            5 => Self::LexicalDictionary,
            6 => Self::LexicalPostings,
            7 => Self::VectorProfile,
            8 => Self::VectorData,
            9 => Self::VectorIndex,
            10 => Self::Signature,
            12 => Self::DeltaManifest,
            13 => Self::TermOverlay,
            16 => Self::LexicalTerms,
            17 => Self::PassageRecords,
            other => Self::Other(other),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::Documents => "documents",
            Self::PassageIndex => "passage_index",
            Self::PassageData => "passage_data",
            Self::LexicalDictionary => "lexical_dictionary",
            Self::LexicalPostings => "lexical_postings",
            Self::VectorProfile => "vector_profile",
            Self::VectorData => "vector_data",
            Self::VectorIndex => "vector_index",
            Self::Signature => "signature",
            Self::DeltaManifest => "delta_manifest",
            Self::TermOverlay => "term_overlay",
            Self::LexicalTerms => "lexical_terms",
            Self::PassageRecords => "passage_records",
            Self::Other(_) => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    None,
    Deflate,
    Other(u16),
}

impl Codec {
    fn as_u16(self) -> u16 {
        match self {
            Self::None => 0,
            Self::Deflate => 1,
            Self::Other(value) => value,
        }
    }

    fn from_u16(value: u16) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Deflate,
            other => Self::Other(other),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PackHeader {
    pub flags: u64,
    pub directory_offset: u64,
    pub directory_length: u64,
    pub manifest_section_id: u32,
    pub section_count: u32,
    pub root_hash: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct SectionEntry {
    pub section_id: u32,
    pub section_type: SectionType,
    pub format_version: u16,
    pub codec: Codec,
    pub flags: u16,
    pub offset: u64,
    pub stored_length: u64,
    pub logical_length: u64,
    pub item_count: u64,
    pub hash: [u8; 32],
}

impl SectionEntry {
    pub fn required(&self) -> bool {
        self.flags & FLAG_REQUIRED != 0
    }

    pub fn derived(&self) -> bool {
        self.flags & FLAG_DERIVED != 0
    }
}

#[derive(Debug, Clone)]
pub struct SectionData {
    pub section_id: u32,
    pub section_type: SectionType,
    pub format_version: u16,
    pub codec: Codec,
    pub flags: u16,
    pub item_count: u64,
    pub logical_length: u64,
    pub bytes: Vec<u8>,
}

impl SectionData {
    pub fn required(
        section_id: u32,
        section_type: SectionType,
        item_count: u64,
        bytes: Vec<u8>,
    ) -> Self {
        let logical_length = bytes.len() as u64;
        Self {
            section_id,
            section_type,
            format_version: 1,
            codec: Codec::None,
            flags: FLAG_REQUIRED,
            item_count,
            logical_length,
            bytes,
        }
    }

    /// A required, uncompressed section carrying an explicit section-format
    /// version. Used for the Manifest, whose schema is versioned independently
    /// of the `ANNPACK3` wire format.
    pub fn required_versioned(
        section_id: u32,
        section_type: SectionType,
        item_count: u64,
        format_version: u16,
        bytes: Vec<u8>,
    ) -> Self {
        let mut section = Self::required(section_id, section_type, item_count, bytes);
        section.format_version = format_version;
        section
    }

    pub fn optional(
        section_id: u32,
        section_type: SectionType,
        item_count: u64,
        bytes: Vec<u8>,
    ) -> Self {
        let logical_length = bytes.len() as u64;
        Self {
            section_id,
            section_type,
            format_version: 1,
            codec: Codec::None,
            flags: 0,
            item_count,
            logical_length,
            bytes,
        }
    }

    pub fn required_deflate(
        section_id: u32,
        section_type: SectionType,
        item_count: u64,
        logical_bytes: Vec<u8>,
    ) -> Self {
        Self::deflate(
            section_id,
            section_type,
            item_count,
            FLAG_REQUIRED,
            logical_bytes,
        )
    }

    pub fn optional_deflate(
        section_id: u32,
        section_type: SectionType,
        item_count: u64,
        logical_bytes: Vec<u8>,
    ) -> Self {
        Self::deflate(section_id, section_type, item_count, 0, logical_bytes)
    }

    /// An optional, derived (matching-only, never citable) DEFLATE section.
    pub fn derived_deflate(
        section_id: u32,
        section_type: SectionType,
        item_count: u64,
        logical_bytes: Vec<u8>,
    ) -> Self {
        Self::deflate(
            section_id,
            section_type,
            item_count,
            FLAG_DERIVED,
            logical_bytes,
        )
    }

    fn deflate(
        section_id: u32,
        section_type: SectionType,
        item_count: u64,
        flags: u16,
        logical_bytes: Vec<u8>,
    ) -> Self {
        let logical_length = logical_bytes.len() as u64;
        let bytes = miniz_oxide::deflate::compress_to_vec_zlib(&logical_bytes, 6);
        Self {
            section_id,
            section_type,
            format_version: 1,
            codec: Codec::Deflate,
            flags,
            item_count,
            logical_length,
            bytes,
        }
    }
}

#[derive(Debug, Default)]
pub struct PackWriter {
    sections: Vec<SectionData>,
    flags: u64,
}

impl PackWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_flags(mut self, flags: u64) -> Self {
        self.flags = flags;
        self
    }

    pub fn push(&mut self, section: SectionData) -> Result<()> {
        if self
            .sections
            .iter()
            .any(|item| item.section_id == section.section_id)
        {
            return Err(AnnpackError::InvalidInput(format!(
                "duplicate section ID {}",
                section.section_id
            )));
        }
        if section.bytes.len() as u64 > MAX_SECTION_SIZE
            || section.logical_length > MAX_SECTION_SIZE
        {
            return Err(AnnpackError::InvalidInput(
                "section exceeds size limit".into(),
            ));
        }
        validate_codec_lengths(
            section.section_id,
            section.codec,
            section.bytes.len() as u64,
            section.logical_length,
        )
        .map_err(|error| AnnpackError::InvalidInput(error.to_string()))?;
        if section.section_type == SectionType::Manifest
            && section.logical_length > MAX_MANIFEST_SIZE
        {
            return Err(AnnpackError::InvalidInput(
                "manifest exceeds size limit".into(),
            ));
        }
        self.sections.push(section);
        Ok(())
    }

    pub fn build_bytes(&self) -> Result<Vec<u8>> {
        if self.sections.is_empty() {
            return Err(AnnpackError::InvalidInput("pack has no sections".into()));
        }
        if self.sections.len() > MAX_SECTIONS as usize {
            return Err(AnnpackError::InvalidInput("too many sections".into()));
        }
        let manifest = self
            .sections
            .iter()
            .find(|section| section.section_type == SectionType::Manifest)
            .ok_or_else(|| AnnpackError::InvalidInput("pack has no manifest section".into()))?;
        if self
            .sections
            .iter()
            .filter(|section| section.section_type == SectionType::Manifest)
            .count()
            != 1
        {
            return Err(AnnpackError::InvalidInput(
                "pack must have exactly one manifest section".into(),
            ));
        }

        let mut sections = self.sections.clone();
        sections.sort_by_key(|section| section.section_id);
        let mut bytes = vec![0_u8; HEADER_SIZE];
        let mut entries = Vec::with_capacity(sections.len());
        for section in &sections {
            pad_to_eight(&mut bytes);
            let offset = bytes.len() as u64;
            bytes.extend_from_slice(&section.bytes);
            entries.push(SectionEntry {
                section_id: section.section_id,
                section_type: section.section_type,
                format_version: section.format_version,
                codec: section.codec,
                flags: section.flags,
                offset,
                stored_length: section.bytes.len() as u64,
                logical_length: section.logical_length,
                item_count: section.item_count,
                hash: *blake3::hash(&section.bytes).as_bytes(),
            });
        }

        pad_to_eight(&mut bytes);
        let directory_offset = bytes.len() as u64;
        let directory = encode_directory(&entries);
        let root_hash = compute_root_hash(&entries);
        bytes.extend_from_slice(&directory);

        let header = PackHeader {
            flags: self.flags,
            directory_offset,
            directory_length: directory.len() as u64,
            manifest_section_id: manifest.section_id,
            section_count: entries.len() as u32,
            root_hash,
        };
        bytes[..HEADER_SIZE].copy_from_slice(&encode_header(&header));
        Ok(bytes)
    }

    pub fn write_path(&self, path: impl AsRef<Path>) -> Result<[u8; 32]> {
        let path = path.as_ref();
        let bytes = self.build_bytes()?;
        let header = decode_header(&bytes[..HEADER_SIZE])?;
        let temporary = path.with_extension(format!("annpack-tmp-{}", std::process::id()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result?;
        Ok(header.root_hash)
    }
}

pub struct PackReader {
    source: SharedReader,
    pub header: PackHeader,
    pub entries: Vec<SectionEntry>,
    by_id: BTreeMap<u32, usize>,
}

impl PackReader {
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::open(Arc::new(FileReader::open(path)?))
    }

    pub fn open(source: SharedReader) -> Result<Self> {
        let source_len = source.len()?;
        if source_len < HEADER_SIZE as u64 {
            return Err(AnnpackError::InvalidFormat(
                "file is smaller than the header".into(),
            ));
        }
        let mut header_bytes = [0_u8; HEADER_SIZE];
        source.read_exact_at(0, &mut header_bytes)?;
        let header = decode_header(&header_bytes)?;
        if header.section_count == 0 || header.section_count > MAX_SECTIONS {
            return Err(AnnpackError::InvalidFormat(format!(
                "invalid section count {}",
                header.section_count
            )));
        }
        let expected_directory_length = (header.section_count as u64)
            .checked_mul(DIRECTORY_ENTRY_SIZE as u64)
            .ok_or_else(|| AnnpackError::InvalidFormat("directory length overflow".into()))?;
        if header.directory_length != expected_directory_length {
            return Err(AnnpackError::InvalidFormat(format!(
                "directory length {} does not match section count",
                header.directory_length
            )));
        }
        checked_file_range(
            header.directory_offset,
            header.directory_length,
            source_len,
            "directory",
        )?;
        let directory_len = usize::try_from(header.directory_length)
            .map_err(|_| AnnpackError::InvalidFormat("directory exceeds address space".into()))?;
        let mut directory_bytes = vec![0_u8; directory_len];
        source.read_exact_at(header.directory_offset, &mut directory_bytes)?;
        let entries = decode_directory(&directory_bytes)?;
        // The content-root check is a cryptographic gate: random mutation cannot
        // satisfy it, so a byte-mutation fuzzer never reaches the parsing behind
        // it. `fuzzing-unsafe` skips the comparison so those targets can exercise
        // directory validation, section decoding, and the index structures.
        //
        // The bypass requires `cfg(fuzzing)` as well as the feature, and only
        // cargo-fuzz sets that. Gating on the feature alone was wrong: features
        // are additive, so `cargo test --all-features` and `cargo build
        // --all-features` both silently produced a runtime with no artifact
        // integrity verification. CI ran both. `cargo` cannot be asked to
        // exclude a feature from `--all-features`, so the second condition is
        // what makes the bypass unreachable from an ordinary build.
        #[cfg(not(all(fuzzing, feature = "fuzzing-unsafe")))]
        if compute_root_hash(&entries) != header.root_hash {
            return Err(AnnpackError::Integrity(
                "root hash does not match directory".into(),
            ));
        }

        validate_entries(&entries, source_len, &header)?;
        let mut by_id = BTreeMap::new();
        let mut singleton_types = BTreeSet::new();
        for (index, entry) in entries.iter().enumerate() {
            if by_id.insert(entry.section_id, index).is_some() {
                return Err(AnnpackError::InvalidFormat(format!(
                    "duplicate section ID {}",
                    entry.section_id
                )));
            }
            if matches!(entry.section_type, SectionType::Other(_)) && entry.required() {
                return Err(AnnpackError::Unsupported(format!(
                    "required section type {}",
                    entry.section_type.as_u16()
                )));
            }
            if matches!(entry.codec, Codec::Other(_)) && entry.required() {
                return Err(AnnpackError::Unsupported(format!(
                    "required codec {}",
                    entry.codec.as_u16()
                )));
            }
            // An unknown *optional* section is ignored safely whether or not it
            // carries the derived flag (FORMAT-v3 §2). Only required-and-unknown
            // and derived-and-required are structural errors.
            if entry.derived() && entry.required() {
                return Err(AnnpackError::InvalidFormat(format!(
                    "section {} is both derived and required",
                    entry.section_id
                )));
            }
            if !matches!(
                entry.section_type,
                SectionType::Signature | SectionType::TermOverlay | SectionType::Other(_)
            ) && !singleton_types.insert(entry.section_type.as_u16())
            {
                return Err(AnnpackError::InvalidFormat(format!(
                    "duplicate singleton section type {}",
                    entry.section_type.name()
                )));
            }
        }
        let manifest_index = by_id
            .get(&header.manifest_section_id)
            .ok_or_else(|| AnnpackError::InvalidFormat("manifest section ID is missing".into()))?;
        let manifest = &entries[*manifest_index];
        if manifest.section_type != SectionType::Manifest
            || !manifest.required()
            || manifest.logical_length > MAX_MANIFEST_SIZE
        {
            return Err(AnnpackError::InvalidFormat(
                "manifest directory entry is invalid".into(),
            ));
        }
        // Refuse an unknown manifest schema at the container boundary rather
        // than failing later inside JSON deserialization with a missing-field
        // error. This is the explicit compatibility boundary v0.3.1 lacked.
        if !SUPPORTED_MANIFEST_FORMAT_VERSIONS.contains(&manifest.format_version) {
            return Err(AnnpackError::Unsupported(format!(
                "manifest section format version {} (this reader supports {:?})",
                manifest.format_version, SUPPORTED_MANIFEST_FORMAT_VERSIONS
            )));
        }
        Ok(Self {
            source,
            header,
            entries,
            by_id,
        })
    }

    pub fn root_hex(&self) -> String {
        hex::encode(self.header.root_hash)
    }

    pub fn source_identity(&self) -> Option<&str> {
        self.source.identity()
    }

    pub fn entry(&self, section_id: u32) -> Result<&SectionEntry> {
        self.by_id
            .get(&section_id)
            .map(|index| &self.entries[*index])
            .ok_or_else(|| AnnpackError::InvalidFormat(format!("section {section_id} not found")))
    }

    pub fn first_entry(&self, section_type: SectionType) -> Option<&SectionEntry> {
        self.entries
            .iter()
            .find(|entry| entry.section_type == section_type)
    }

    pub fn entries_of_type(
        &self,
        section_type: SectionType,
    ) -> impl Iterator<Item = &SectionEntry> {
        self.entries
            .iter()
            .filter(move |entry| entry.section_type == section_type)
    }

    pub fn read_section(&self, section_id: u32) -> Result<Vec<u8>> {
        let entry = self.entry(section_id)?;
        let bytes = self.read_stored_section(section_id)?;
        match entry.codec {
            Codec::None => Ok(bytes),
            Codec::Deflate => {
                let limit = usize::try_from(entry.logical_length).map_err(|_| {
                    AnnpackError::InvalidFormat("logical section exceeds address space".into())
                })?;
                let logical =
                    miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&bytes, limit)
                        .map_err(|error| {
                            AnnpackError::InvalidFormat(format!(
                                "section {} deflate decode failed: {error:?}",
                                entry.section_id
                            ))
                        })?;
                if logical.len() != limit {
                    return Err(AnnpackError::InvalidFormat(format!(
                        "section {} decompressed to {}, expected {} bytes",
                        entry.section_id,
                        logical.len(),
                        limit
                    )));
                }
                Ok(logical)
            }
            Codec::Other(value) => Err(AnnpackError::Unsupported(format!("codec {value}"))),
        }
    }

    pub fn read_stored_section(&self, section_id: u32) -> Result<Vec<u8>> {
        let entry = self.entry(section_id)?;
        let length = usize::try_from(entry.stored_length)
            .map_err(|_| AnnpackError::InvalidFormat("section exceeds address space".into()))?;
        let mut bytes = vec![0_u8; length];
        self.source.read_exact_at(entry.offset, &mut bytes)?;
        let actual = blake3::hash(&bytes);
        if actual.as_bytes() != &entry.hash {
            return Err(AnnpackError::Integrity(format!(
                "section {} ({}) hash mismatch",
                entry.section_id,
                entry.section_type.name()
            )));
        }
        Ok(bytes)
    }

    pub fn read_section_range(
        &self,
        section_id: u32,
        relative_offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let entry = self.entry(section_id)?;
        if entry.codec != Codec::None {
            return Err(AnnpackError::Unsupported(
                "partial reads of section-level compressed data are not supported".into(),
            ));
        }
        let end = relative_offset
            .checked_add(length)
            .ok_or_else(|| AnnpackError::InvalidFormat("section-relative range overflow".into()))?;
        if end > entry.stored_length {
            return Err(AnnpackError::InvalidFormat(format!(
                "range exceeds section {}",
                entry.section_id
            )));
        }
        let absolute = entry
            .offset
            .checked_add(relative_offset)
            .ok_or_else(|| AnnpackError::InvalidFormat("absolute range overflow".into()))?;
        let length = usize::try_from(length)
            .map_err(|_| AnnpackError::InvalidFormat("range exceeds address space".into()))?;
        let mut bytes = vec![0_u8; length];
        self.source.read_exact_at(absolute, &mut bytes)?;
        Ok(bytes)
    }

    /// Raw section-directory bytes: the exact preimage (minus signature
    /// entries) of the artifact root. An evidence receipt carries these so a
    /// verifier can recompute the root without the pack.
    pub fn directory_bytes(&self) -> Result<Vec<u8>> {
        let length = usize::try_from(self.header.directory_length)
            .map_err(|_| AnnpackError::InvalidFormat("directory exceeds address space".into()))?;
        let mut bytes = vec![0_u8; length];
        self.source
            .read_exact_at(self.header.directory_offset, &mut bytes)?;
        Ok(bytes)
    }

    pub fn manifest(&self) -> Result<Manifest> {
        let entry = self.entry(self.header.manifest_section_id)?;
        if entry.section_type != SectionType::Manifest {
            return Err(AnnpackError::InvalidFormat(
                "manifest section ID points to another section type".into(),
            ));
        }
        if entry.logical_length > MAX_MANIFEST_SIZE {
            return Err(AnnpackError::InvalidFormat(
                "manifest exceeds size limit".into(),
            ));
        }
        let manifest: Manifest = serde_json::from_slice(&self.read_section(entry.section_id)?)?;
        if let Some(issue) = manifest_logical_root_issue(&manifest, entry.format_version) {
            return Err(AnnpackError::InvalidFormat(issue));
        }
        Ok(manifest)
    }

    pub fn verify_all(&self) -> Result<VerificationReport> {
        let mut verified = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            if matches!(entry.codec, Codec::Other(_)) && !entry.required() {
                self.read_stored_section(entry.section_id)?;
            } else {
                self.read_section(entry.section_id)?;
            }
            verified.push(entry.section_id);
        }
        Ok(VerificationReport {
            root_hash: self.root_hex(),
            section_ids: verified,
            bytes: self.source.len()?,
        })
    }

    pub fn all_section_data(&self, include_signatures: bool) -> Result<Vec<SectionData>> {
        let mut sections = Vec::new();
        for entry in &self.entries {
            if !include_signatures && entry.section_type == SectionType::Signature {
                continue;
            }
            sections.push(SectionData {
                section_id: entry.section_id,
                section_type: entry.section_type,
                format_version: entry.format_version,
                codec: entry.codec,
                flags: entry.flags,
                item_count: entry.item_count,
                logical_length: entry.logical_length,
                bytes: self.read_stored_section(entry.section_id)?,
            });
        }
        Ok(sections)
    }
}

#[derive(Debug, serde::Serialize)]
pub struct VerificationReport {
    pub root_hash: String,
    pub section_ids: Vec<u32>,
    pub bytes: u64,
}

/// Checks a decoded manifest against the requirements its own section-format
/// version imposes, returning the failure text when it does not hold.
///
/// Manifest section format 2 and later MUST carry `passage_merkle_root` as
/// exactly 64 lowercase hexadecimal characters (FORMAT-v3 §4.1). Format 1
/// predates the field, so a v0.3.x artifact stays readable without it and simply
/// cannot issue standalone receipts (spec/COMPATIBILITY.md). Validating by
/// section-format version is what keeps `Option` in the shared model from
/// silently accepting a v2 manifest that omits the logical content root.
pub fn manifest_logical_root_issue(manifest: &Manifest, format_version: u16) -> Option<String> {
    if format_version < 2 {
        return None;
    }
    match manifest.passage_merkle_root.as_deref() {
        None => Some(format!(
            "manifest section format {format_version} requires passage_merkle_root"
        )),
        Some(value) if !is_lowercase_hex_32(value) => Some(
            "manifest passage_merkle_root must be 64 lowercase hexadecimal characters".to_string(),
        ),
        Some(_) => None,
    }
}

fn is_lowercase_hex_32(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn encode_header(header: &PackHeader) -> [u8; HEADER_SIZE] {
    let mut bytes = [0_u8; HEADER_SIZE];
    bytes[0..8].copy_from_slice(MAGIC);
    bytes[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes[12..16].copy_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
    bytes[16..24].copy_from_slice(&header.flags.to_le_bytes());
    bytes[24..32].copy_from_slice(&header.directory_offset.to_le_bytes());
    bytes[32..40].copy_from_slice(&header.directory_length.to_le_bytes());
    bytes[40..44].copy_from_slice(&header.manifest_section_id.to_le_bytes());
    bytes[44..48].copy_from_slice(&header.section_count.to_le_bytes());
    bytes[48..80].copy_from_slice(&header.root_hash);
    bytes
}

fn decode_header(bytes: &[u8]) -> Result<PackHeader> {
    if bytes.len() != HEADER_SIZE {
        return Err(AnnpackError::InvalidFormat("invalid header length".into()));
    }
    if &bytes[0..8] != MAGIC {
        return Err(AnnpackError::InvalidFormat("bad magic".into()));
    }
    let version = read_u32(bytes, 8)?;
    if version != FORMAT_VERSION {
        return Err(AnnpackError::Unsupported(format!(
            "format version {version}"
        )));
    }
    if read_u32(bytes, 12)? != HEADER_SIZE as u32 {
        return Err(AnnpackError::InvalidFormat("invalid header size".into()));
    }
    if bytes[80..].iter().any(|byte| *byte != 0) {
        return Err(AnnpackError::InvalidFormat(
            "reserved header bytes must be zero".into(),
        ));
    }
    let mut root_hash = [0_u8; 32];
    root_hash.copy_from_slice(&bytes[48..80]);
    Ok(PackHeader {
        flags: read_u64(bytes, 16)?,
        directory_offset: read_u64(bytes, 24)?,
        directory_length: read_u64(bytes, 32)?,
        manifest_section_id: read_u32(bytes, 40)?,
        section_count: read_u32(bytes, 44)?,
        root_hash,
    })
}

fn encode_directory(entries: &[SectionEntry]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(entries.len() * DIRECTORY_ENTRY_SIZE);
    for entry in entries {
        bytes.extend_from_slice(&encode_entry(entry));
    }
    bytes
}

fn decode_directory(bytes: &[u8]) -> Result<Vec<SectionEntry>> {
    if !bytes.len().is_multiple_of(DIRECTORY_ENTRY_SIZE) {
        return Err(AnnpackError::InvalidFormat("misaligned directory".into()));
    }
    bytes
        .chunks_exact(DIRECTORY_ENTRY_SIZE)
        .map(decode_entry)
        .collect()
}

fn encode_entry(entry: &SectionEntry) -> [u8; DIRECTORY_ENTRY_SIZE] {
    let mut bytes = [0_u8; DIRECTORY_ENTRY_SIZE];
    bytes[0..4].copy_from_slice(&entry.section_id.to_le_bytes());
    bytes[4..6].copy_from_slice(&entry.section_type.as_u16().to_le_bytes());
    bytes[6..8].copy_from_slice(&entry.format_version.to_le_bytes());
    bytes[8..10].copy_from_slice(&entry.codec.as_u16().to_le_bytes());
    bytes[10..12].copy_from_slice(&entry.flags.to_le_bytes());
    bytes[12..20].copy_from_slice(&entry.offset.to_le_bytes());
    bytes[20..28].copy_from_slice(&entry.stored_length.to_le_bytes());
    bytes[28..36].copy_from_slice(&entry.logical_length.to_le_bytes());
    bytes[36..44].copy_from_slice(&entry.item_count.to_le_bytes());
    bytes[44..76].copy_from_slice(&entry.hash);
    bytes
}

fn decode_entry(bytes: &[u8]) -> Result<SectionEntry> {
    if bytes.len() != DIRECTORY_ENTRY_SIZE {
        return Err(AnnpackError::InvalidFormat(
            "invalid directory entry size".into(),
        ));
    }
    if bytes[76..80].iter().any(|byte| *byte != 0) {
        return Err(AnnpackError::InvalidFormat(
            "reserved directory-entry bytes must be zero".into(),
        ));
    }
    let mut hash = [0_u8; 32];
    hash.copy_from_slice(&bytes[44..76]);
    Ok(SectionEntry {
        section_id: read_u32(bytes, 0)?,
        section_type: SectionType::from_u16(read_u16(bytes, 4)?),
        format_version: read_u16(bytes, 6)?,
        codec: Codec::from_u16(read_u16(bytes, 8)?),
        flags: read_u16(bytes, 10)?,
        offset: read_u64(bytes, 12)?,
        stored_length: read_u64(bytes, 20)?,
        logical_length: read_u64(bytes, 28)?,
        item_count: read_u64(bytes, 36)?,
        hash,
    })
}

fn compute_root_hash(entries: &[SectionEntry]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ANNPACK3-CONTENT-ROOT\0");
    for entry in entries {
        if entry.section_type != SectionType::Signature {
            hasher.update(&encode_entry(entry));
        }
    }
    *hasher.finalize().as_bytes()
}

fn validate_entries(entries: &[SectionEntry], source_len: u64, header: &PackHeader) -> Result<()> {
    if entries
        .windows(2)
        .any(|pair| pair[0].section_id >= pair[1].section_id)
    {
        return Err(AnnpackError::InvalidFormat(
            "directory entries must be in strictly increasing section-ID order".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut ranges = Vec::with_capacity(entries.len());
    for entry in entries {
        if !ids.insert(entry.section_id) {
            return Err(AnnpackError::InvalidFormat(format!(
                "duplicate section ID {}",
                entry.section_id
            )));
        }
        if entry.stored_length > MAX_SECTION_SIZE || entry.logical_length > MAX_SECTION_SIZE {
            return Err(AnnpackError::InvalidFormat(format!(
                "section {} exceeds size limit",
                entry.section_id
            )));
        }
        validate_codec_lengths(
            entry.section_id,
            entry.codec,
            entry.stored_length,
            entry.logical_length,
        )?;
        let end = checked_file_range(entry.offset, entry.stored_length, source_len, "section")?;
        if entry.offset < HEADER_SIZE as u64 {
            return Err(AnnpackError::InvalidFormat(format!(
                "section {} overlaps header",
                entry.section_id
            )));
        }
        ranges.push((entry.offset, end, entry.section_id));
    }
    ranges.sort_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(AnnpackError::InvalidFormat(format!(
                "sections {} and {} overlap",
                pair[0].2, pair[1].2
            )));
        }
    }
    let directory_end = header
        .directory_offset
        .checked_add(header.directory_length)
        .ok_or_else(|| AnnpackError::InvalidFormat("directory end overflow".into()))?;
    for (start, end, id) in ranges {
        if start < directory_end && end > header.directory_offset {
            return Err(AnnpackError::InvalidFormat(format!(
                "section {id} overlaps directory"
            )));
        }
    }
    Ok(())
}

fn validate_codec_lengths(
    section_id: u32,
    codec: Codec,
    stored_length: u64,
    logical_length: u64,
) -> Result<()> {
    if codec == Codec::None && stored_length != logical_length {
        return Err(AnnpackError::InvalidFormat(format!(
            "uncompressed section {section_id} has mismatched lengths"
        )));
    }
    if codec == Codec::Deflate {
        if stored_length == 0 && logical_length != 0 {
            return Err(AnnpackError::InvalidFormat(format!(
                "compressed section {section_id} has no stored bytes"
            )));
        }
        let ratio_limit = stored_length
            .saturating_mul(DECOMPRESSION_RATIO_LIMIT)
            .max(DECOMPRESSION_RATIO_FLOOR);
        if logical_length > ratio_limit {
            return Err(AnnpackError::InvalidFormat(format!(
                "compressed section {section_id} exceeds the decompression-ratio limit"
            )));
        }
    }
    Ok(())
}

fn checked_file_range(offset: u64, length: u64, file_len: u64, label: &str) -> Result<u64> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| AnnpackError::InvalidFormat(format!("{label} range overflow")))?;
    if end > file_len {
        return Err(AnnpackError::InvalidFormat(format!(
            "{label} range {offset}..{end} exceeds file length {file_len}"
        )));
    }
    Ok(end)
}

fn pad_to_eight(bytes: &mut Vec<u8>) {
    while !bytes.len().is_multiple_of(8) {
        bytes.push(0);
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| AnnpackError::InvalidFormat("truncated u16".into()))?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| AnnpackError::InvalidFormat("truncated u32".into()))?;
    Ok(u32::from_le_bytes(
        value.try_into().expect("slice has exact length"),
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| AnnpackError::InvalidFormat("truncated u64".into()))?;
    Ok(u64::from_le_bytes(
        value.try_into().expect("slice has exact length"),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use proptest::prelude::*;

    use super::*;
    use crate::reader::MemoryReader;

    fn minimal_writer() -> PackWriter {
        let manifest = br#"{"name":"test","version":"1","description":null,"source_revision":null,"base_url":null,"created_at":null,"document_count":0,"passage_count":0,"capabilities":[],"embedding_profiles":[],"policy":{"license":null,"access":"public","redistributable":null,"expires_at":null,"policy_url":null},"dependencies":[]}"#.to_vec();
        let mut writer = PackWriter::new();
        writer
            .push(SectionData::required(1, SectionType::Manifest, 1, manifest))
            .unwrap();
        writer
    }

    #[test]
    fn round_trip_minimal_pack() {
        let bytes = minimal_writer().build_bytes().unwrap();
        let reader = PackReader::open(Arc::new(MemoryReader::new(bytes))).unwrap();
        assert_eq!(reader.manifest().unwrap().name, "test");
        assert_eq!(reader.verify_all().unwrap().section_ids, vec![1]);
    }

    #[test]
    fn deterministic_pack_bytes() {
        assert_eq!(
            minimal_writer().build_bytes().unwrap(),
            minimal_writer().build_bytes().unwrap()
        );
    }

    proptest! {
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..4096)) {
            let _ = PackReader::open(Arc::new(MemoryReader::new(bytes)));
        }
    }
}
