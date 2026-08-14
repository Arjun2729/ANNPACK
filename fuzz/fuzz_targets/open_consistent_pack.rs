#![no_main]

use std::sync::Arc;

use adyar::format::PackReader;
use adyar::reader::MemoryReader;
use adyar::search::{SearchEngine, SearchOptions};
use libfuzzer_sys::fuzz_target;

/// Structure-aware container fuzzing.
///
/// `open_pack` and `open_pack_prefixed` cannot reach most of the parser. Opening
/// a pack recomputes the BLAKE3 content root over the section directory and
/// compares it to the header, and random mutation does not produce a 256-bit
/// hash match. Every input therefore dies at that gate, which is why `format.rs`
/// region coverage sits near 10% from those entry points: section decoding,
/// section-format dispatch, the block tables, the record table and the whole
/// search path are unreachable behind it.
///
/// This target mutates a valid artifact and then *repairs* the container's
/// self-consistency — per-section hashes, then the root — so the fuzzer spends
/// its executions on everything the root check was hiding. The mutated bytes are
/// arbitrary, so the structures parsed downstream are hostile even though the
/// envelope is well-formed.
///
/// Deliberately additive: `open_pack` still covers the reject-early paths, which
/// this target skips by construction.
const GOLDEN: &[u8] = include_bytes!("../../spec/test-vectors/minimal-v3.annpack");

const HEADER_SIZE: usize = 128;
const ENTRY_SIZE: usize = 80;

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// Recompute every section hash from the bytes each entry points at, then the
/// content root over the non-signature entries, exactly as a writer would.
fn repair(bytes: &mut [u8]) -> Option<()> {
    let directory_offset = usize::try_from(u64_at(bytes, 24)).ok()?;
    let directory_length = usize::try_from(u64_at(bytes, 32)).ok()?;
    let count = u32_at(bytes, 44) as usize;
    if count != directory_length / ENTRY_SIZE {
        return None;
    }
    if directory_offset.checked_add(directory_length)? > bytes.len() {
        return None;
    }

    for index in 0..count {
        let entry = directory_offset + index * ENTRY_SIZE;
        let offset = usize::try_from(u64_at(bytes, entry + 12)).ok()?;
        let stored = usize::try_from(u64_at(bytes, entry + 20)).ok()?;
        let end = offset.checked_add(stored)?;
        if end > bytes.len() {
            return None;
        }
        let hash = blake3::hash(&bytes[offset..end]);
        bytes[entry + 44..entry + 76].copy_from_slice(hash.as_bytes());
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ANNPACK3-CONTENT-ROOT\0");
    for index in 0..count {
        let entry = directory_offset + index * ENTRY_SIZE;
        // Section type 10 is Signature and is excluded from the root.
        if u16::from_le_bytes(bytes[entry + 4..entry + 6].try_into().unwrap()) != 10 {
            hasher.update(&bytes[entry..entry + ENTRY_SIZE]);
        }
    }
    bytes[48..80].copy_from_slice(hasher.finalize().as_bytes());
    Some(())
}

fuzz_target!(|data: &[u8]| {
    // Need a splice position and at least one byte to write.
    if data.len() < 5 {
        return;
    }
    let mut artifact = GOLDEN.to_vec();
    let patch = &data[4..];
    // Splice anywhere after the header. Landing inside the directory is
    // intentional: it exercises offset, length, codec, flag and format-version
    // handling with a container that still hashes correctly.
    let span = artifact.len() - HEADER_SIZE;
    let at = HEADER_SIZE + (u32_at(data, 0) as usize % span);
    let end = (at + patch.len()).min(artifact.len());
    artifact[at..end].copy_from_slice(&patch[..end - at]);

    if repair(&mut artifact).is_none() {
        return;
    }

    let source = Arc::new(MemoryReader::new(artifact));
    let Ok(reader) = PackReader::open(source.clone()) else {
        return;
    };
    // Past the root gate. Everything below is what the previous targets could
    // not reach.
    let _ = reader.verify_all();
    let _ = reader.manifest();
    if let Ok(engine) = SearchEngine::open_source(source) {
        let _ = engine.search(
            "cache rotation AP-104",
            &SearchOptions {
                limit: 5,
                ..Default::default()
            },
        );
        let _ = engine.passages();
    }
});
