#![no_main]

use std::sync::Arc;

use annpack::format::PackReader;
use annpack::reader::MemoryReader;
use libfuzzer_sys::fuzz_target;

/// Directory-validation fuzzing behind a structurally valid header.
///
/// An earlier version of this target prepended only the magic and format
/// version and let the fuzzer supply everything else. Measured against its own
/// corpus, 48 of 53 inputs died on "reserved header bytes must be zero": the
/// header reserves bytes 80..128 and requires them zeroed, so the fuzzer had to
/// produce 48 consecutive zero bytes at an exact offset before reaching any
/// other check. Directory validation was effectively unreachable, and the
/// content-root check that follows it was never even the binding gate.
///
/// This version constructs the fixed header itself, so every structural
/// constant is correct by construction and the input drives the section
/// directory instead: entry counts, offsets, lengths, codecs, flags, section
/// types, and format versions.
///
/// Requires the `fuzzing-unsafe` feature, which removes the content-root check
/// from `PackReader::open`. Without it this target would die at that check
/// instead, one gate later.
const HEADER_SIZE: usize = 128;
const ENTRY_SIZE: usize = 80;

fuzz_target!(|data: &[u8]| {
    // One byte of section count; the rest becomes directory and section bytes.
    if data.len() < 2 {
        return;
    }
    // Bounded by the byte width. MAX_SECTIONS is 16,384, so counts above the
    // limit are exercised by `open_pack` rather than here.
    let count = data[0] as usize;
    let body = &data[1..];

    let directory_length = count * ENTRY_SIZE;

    let mut artifact = Vec::with_capacity(HEADER_SIZE + body.len());
    artifact.extend_from_slice(b"ANNPACK3");
    artifact.extend_from_slice(&3_u32.to_le_bytes()); // format version
    artifact.extend_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
    artifact.extend_from_slice(&0_u64.to_le_bytes()); // container flags
    artifact.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // directory offset
    artifact.extend_from_slice(&(directory_length as u64).to_le_bytes());
    artifact.extend_from_slice(&1_u32.to_le_bytes()); // manifest section id
    artifact.extend_from_slice(&(count as u32).to_le_bytes());
    artifact.extend_from_slice(&[0_u8; 32]); // root hash, unchecked under fuzzing-unsafe
    artifact.resize(HEADER_SIZE, 0); // reserved bytes 80..128 are zero
    artifact.extend_from_slice(body);

    let _ = PackReader::open(Arc::new(MemoryReader::new(artifact)));
});
