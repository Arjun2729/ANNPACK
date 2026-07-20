use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use annpack::build::{BuildOptions, build_pack};
use annpack::format::{DIRECTORY_ENTRY_SIZE, HEADER_SIZE, PackReader};
use annpack::model::AccessClass;
use annpack::reader::MemoryReader;
use tempfile::TempDir;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/docs-v1")
}

fn valid_bytes() -> (TempDir, Vec<u8>) {
    let temp = TempDir::new().unwrap();
    let output = temp.path().join("valid.annpack");
    build_pack(&BuildOptions {
        input: fixture(),
        output: output.clone(),
        name: "corruption-fixture".into(),
        version: "1".into(),
        description: None,
        source_revision: None,
        base_url: None,
        created_at: None,
        license: None,
        access: AccessClass::Public,
        redistributable: None,
        policy_expires_at: None,
        policy_url: None,
        dependencies: Vec::new(),
        policy_override: None,
        vector_input: None,
        target_chars: 1_200,
        max_chars: 2_400,
        input_format: annpack::ingest::InputFormat::Auto,
    })
    .unwrap();
    let bytes = fs::read(output).unwrap();
    (temp, bytes)
}

fn open(bytes: Vec<u8>) -> annpack::Result<PackReader> {
    PackReader::open(Arc::new(MemoryReader::new(bytes)))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn directory_offset(bytes: &[u8]) -> usize {
    read_u64(bytes, 24) as usize
}

fn recompute_root(bytes: &mut [u8]) {
    let directory = directory_offset(bytes);
    let count = read_u32(bytes, 44) as usize;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ANNPACK3-CONTENT-ROOT\0");
    for index in 0..count {
        let start = directory + index * DIRECTORY_ENTRY_SIZE;
        let entry_type = u16::from_le_bytes(bytes[start + 4..start + 6].try_into().unwrap());
        if entry_type != 10 {
            hasher.update(&bytes[start..start + DIRECTORY_ENTRY_SIZE]);
        }
    }
    bytes[48..80].copy_from_slice(hasher.finalize().as_bytes());
}

#[test]
fn truncated_inputs_are_rejected() {
    let (_temp, bytes) = valid_bytes();
    for length in [0, 1, HEADER_SIZE - 1, HEADER_SIZE, bytes.len() - 1] {
        assert!(
            open(bytes[..length].to_vec()).is_err(),
            "length {length} was accepted"
        );
    }
}

#[test]
fn absurd_header_counts_and_offsets_are_rejected() {
    let (_temp, bytes) = valid_bytes();
    let mut count = bytes.clone();
    count[44..48].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(open(count).is_err());

    let mut offset = bytes;
    offset[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(open(offset).is_err());
}

#[test]
fn duplicate_overlapping_and_out_of_bounds_sections_are_rejected_after_valid_root() {
    let (_temp, bytes) = valid_bytes();
    let directory = directory_offset(&bytes);

    let mut duplicate = bytes.clone();
    let first_id = duplicate[directory..directory + 4].to_vec();
    duplicate[directory + DIRECTORY_ENTRY_SIZE..directory + DIRECTORY_ENTRY_SIZE + 4]
        .copy_from_slice(&first_id);
    recompute_root(&mut duplicate);
    assert!(open(duplicate).is_err());

    let mut duplicate_type = bytes.clone();
    duplicate_type[directory + DIRECTORY_ENTRY_SIZE + 4..directory + DIRECTORY_ENTRY_SIZE + 6]
        .copy_from_slice(&1_u16.to_le_bytes());
    recompute_root(&mut duplicate_type);
    assert!(open(duplicate_type).is_err());

    let mut overlap = bytes.clone();
    let first_offset = overlap[directory + 12..directory + 20].to_vec();
    let second = directory + DIRECTORY_ENTRY_SIZE;
    overlap[second + 12..second + 20].copy_from_slice(&first_offset);
    recompute_root(&mut overlap);
    assert!(open(overlap).is_err());

    let mut out_of_bounds = bytes;
    out_of_bounds[directory + 12..directory + 20].copy_from_slice(&u64::MAX.to_le_bytes());
    recompute_root(&mut out_of_bounds);
    assert!(open(out_of_bounds).is_err());
}

#[test]
fn uncompressed_length_mismatch_and_unknown_required_sections_are_rejected() {
    let (_temp, bytes) = valid_bytes();
    let directory = directory_offset(&bytes);

    let mut length = bytes.clone();
    let stored = read_u64(&length, directory + 20);
    length[directory + 28..directory + 36].copy_from_slice(&(stored + 1).to_le_bytes());
    recompute_root(&mut length);
    assert!(open(length).is_err());

    let mut unknown = bytes;
    let last = directory + 5 * DIRECTORY_ENTRY_SIZE;
    unknown[last + 4..last + 6].copy_from_slice(&65_000_u16.to_le_bytes());
    unknown[last + 10..last + 12].copy_from_slice(&1_u16.to_le_bytes());
    recompute_root(&mut unknown);
    assert!(open(unknown).is_err());
}

#[test]
fn unknown_optional_section_is_safely_ignorable() {
    let (_temp, mut bytes) = valid_bytes();
    let directory = directory_offset(&bytes);
    let last = directory + 5 * DIRECTORY_ENTRY_SIZE;
    bytes[last + 4..last + 6].copy_from_slice(&65_000_u16.to_le_bytes());
    bytes[last + 8..last + 10].copy_from_slice(&65_000_u16.to_le_bytes());
    bytes[last + 10..last + 12].copy_from_slice(&0_u16.to_le_bytes());
    recompute_root(&mut bytes);
    let reader = open(bytes).unwrap();
    reader.verify_all().unwrap();
}

#[test]
fn noncanonical_directory_order_and_decompression_bombs_are_rejected() {
    let (_temp, bytes) = valid_bytes();
    let directory = directory_offset(&bytes);

    let mut reordered = bytes.clone();
    let first = reordered[directory..directory + DIRECTORY_ENTRY_SIZE].to_vec();
    let second =
        reordered[directory + DIRECTORY_ENTRY_SIZE..directory + 2 * DIRECTORY_ENTRY_SIZE].to_vec();
    reordered[directory..directory + DIRECTORY_ENTRY_SIZE].copy_from_slice(&second);
    reordered[directory + DIRECTORY_ENTRY_SIZE..directory + 2 * DIRECTORY_ENTRY_SIZE]
        .copy_from_slice(&first);
    recompute_root(&mut reordered);
    assert!(open(reordered).is_err());

    let mut bomb = bytes;
    let documents = directory + DIRECTORY_ENTRY_SIZE;
    bomb[documents + 28..documents + 36].copy_from_slice(&(17_u64 * 1024 * 1024).to_le_bytes());
    bomb[documents + 20..documents + 28].copy_from_slice(&1_u64.to_le_bytes());
    recompute_root(&mut bomb);
    assert!(open(bomb).is_err());
}

#[test]
fn reserved_header_bytes_are_rejected() {
    let (_temp, mut bytes) = valid_bytes();
    bytes[80] = 1;
    assert!(open(bytes).is_err());

    let (_temp, mut bytes) = valid_bytes();
    let directory = directory_offset(&bytes);
    bytes[directory + 76] = 1;
    recompute_root(&mut bytes);
    assert!(open(bytes).is_err());
}

#[test]
fn directory_bit_flip_fails_root_binding() {
    let (_temp, mut bytes) = valid_bytes();
    let directory = directory_offset(&bytes);
    bytes[directory + 44] ^= 1;
    assert!(open(bytes).is_err());
}
