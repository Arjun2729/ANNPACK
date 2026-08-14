use std::collections::HashMap;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{AdyarError, Result};
use crate::format::PackReader;
use crate::reader::MemoryReader;

const DELTA_MAGIC: &[u8; 8] = b"ANNDELT1";
const DELTA_HEADER_SIZE: usize = 88;
const DELTA_VERSION: u32 = 1;
const CODEC_SNAPSHOT: u32 = 0;
const CODEC_COPY_ADD: u32 = 1;
const COPY_ADD_CHUNK: usize = 256;
const COPY_ADD_STRIDE: usize = 8;
const MAX_COPY_ADD_SCAN_SIZE: usize = 512 * 1024 * 1024;
const MAX_COPY_ADD_ANCHORS: usize = 4_000_000;
const MAX_DELTA_TARGET_SIZE: u64 = 512 * 1024 * 1024;
const MAX_DELTA_OPERATIONS: u64 = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaReport {
    pub kind: String,
    pub base_root: String,
    pub target_root: String,
    pub target_bytes: u64,
    pub delta_bytes: u64,
    pub copied_bytes: u64,
    pub inserted_bytes: u64,
}

#[derive(Debug, Clone)]
enum DeltaOperation {
    Copy { offset: u64, length: u64 },
    Add(Vec<u8>),
}

/// Creates the smaller of a compatible snapshot payload and a bounded copy/add
/// payload. Copy operations always refer to the exact verified base artifact.
pub fn create_delta(base: &Path, target: &Path, output: &Path) -> Result<DeltaReport> {
    let base_bytes = fs::read(base)?;
    let target_bytes = fs::read(target)?;
    if target_bytes.len() as u64 > MAX_DELTA_TARGET_SIZE {
        return Err(AdyarError::InvalidInput(
            "target pack exceeds delta size limit".into(),
        ));
    }
    let base_reader = PackReader::open(Arc::new(MemoryReader::new(base_bytes.clone())))?;
    let target_reader = PackReader::open(Arc::new(MemoryReader::new(target_bytes.clone())))?;
    base_reader.verify_all()?;
    target_reader.verify_all()?;

    let operations = (base_bytes.len() <= MAX_COPY_ADD_SCAN_SIZE
        && target_bytes.len() <= MAX_COPY_ADD_SCAN_SIZE)
        .then(|| build_copy_add(&base_bytes, &target_bytes));
    let copy_add_payload = operations.as_deref().map(encode_operations).transpose()?;
    let (codec, payload, kind) = match copy_add_payload {
        Some(payload) if payload.len() < target_bytes.len() => {
            (CODEC_COPY_ADD, payload, "copy_add_v1")
        }
        _ => (CODEC_SNAPSHOT, target_bytes.clone(), "snapshot_replacement"),
    };
    let bytes = encode_delta_header(
        codec,
        &base_reader.header.root_hash,
        &target_reader.header.root_hash,
        target_bytes.len() as u64,
        &payload,
    );
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_atomic(output, &bytes, false)?;
    let (copied_bytes, inserted_bytes) = operations
        .as_deref()
        .map(operation_totals)
        .unwrap_or((0, target_bytes.len() as u64));
    Ok(DeltaReport {
        kind: kind.into(),
        base_root: base_reader.root_hex(),
        target_root: target_reader.root_hex(),
        target_bytes: target_bytes.len() as u64,
        delta_bytes: bytes.len() as u64,
        copied_bytes: if codec == CODEC_COPY_ADD {
            copied_bytes
        } else {
            0
        },
        inserted_bytes: if codec == CODEC_COPY_ADD {
            inserted_bytes
        } else {
            target_bytes.len() as u64
        },
    })
}

pub fn inspect_delta(path: &Path) -> Result<DeltaReport> {
    let bytes = fs::read(path)?;
    inspect_delta_bytes(&bytes)
}

pub fn inspect_delta_bytes(bytes: &[u8]) -> Result<DeltaReport> {
    let parsed = parse_delta(bytes)?;
    let (kind, copied_bytes, inserted_bytes) = match &parsed.payload {
        ParsedPayload::Snapshot(target) => ("snapshot_replacement", 0, target.len() as u64),
        ParsedPayload::CopyAdd(operations) => {
            let (copied, inserted) = operation_totals(operations);
            ("copy_add_v1", copied, inserted)
        }
    };
    Ok(DeltaReport {
        kind: kind.into(),
        base_root: hex::encode(parsed.base_root),
        target_root: hex::encode(parsed.target_root),
        target_bytes: parsed.target_length,
        delta_bytes: bytes.len() as u64,
        copied_bytes,
        inserted_bytes,
    })
}

pub fn apply_delta(base: &Path, delta: &Path, output: &Path) -> Result<DeltaReport> {
    let base_bytes = fs::read(base)?;
    let base_reader = PackReader::open(Arc::new(MemoryReader::new(base_bytes.clone())))?;
    base_reader.verify_all()?;
    let delta_bytes = fs::read(delta)?;
    let parsed = parse_delta(&delta_bytes)?;
    if parsed.base_root != base_reader.header.root_hash {
        return Err(AdyarError::Integrity(format!(
            "delta expects base {}, received {}",
            hex::encode(parsed.base_root),
            base_reader.root_hex()
        )));
    }
    let (kind, copied_bytes, inserted_bytes, target) = match parsed.payload {
        ParsedPayload::Snapshot(target) => (
            "snapshot_replacement",
            0,
            target.len() as u64,
            target.to_vec(),
        ),
        ParsedPayload::CopyAdd(operations) => {
            let (copied, inserted) = operation_totals(&operations);
            let target = apply_operations(&base_bytes, &operations, parsed.target_length)?;
            ("copy_add_v1", copied, inserted, target)
        }
    };
    let target_reader = PackReader::open(Arc::new(MemoryReader::new(target.clone())))?;
    target_reader.verify_all()?;
    if target_reader.header.root_hash != parsed.target_root {
        return Err(AdyarError::Integrity(
            "delta target root does not match reconstructed target pack".into(),
        ));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_atomic(output, &target, false)?;
    Ok(DeltaReport {
        kind: kind.into(),
        base_root: base_reader.root_hex(),
        target_root: target_reader.root_hex(),
        target_bytes: target.len() as u64,
        delta_bytes: delta_bytes.len() as u64,
        copied_bytes,
        inserted_bytes,
    })
}

fn build_copy_add(base: &[u8], target: &[u8]) -> Vec<DeltaOperation> {
    let mut chunks = HashMap::<[u8; 16], Vec<usize>>::new();
    let desired_stride = (base.len() / MAX_COPY_ADD_ANCHORS).max(COPY_ADD_STRIDE);
    let anchor_stride = desired_stride.div_ceil(COPY_ADD_STRIDE) * COPY_ADD_STRIDE;
    if base.len() >= COPY_ADD_CHUNK {
        for offset in (0..=base.len() - COPY_ADD_CHUNK).step_by(anchor_stride) {
            chunks
                .entry(short_hash(&base[offset..offset + COPY_ADD_CHUNK]))
                .or_default()
                .push(offset);
        }
    }
    let mut operations = Vec::new();
    let mut inserted = Vec::new();
    let mut target_offset = 0_usize;
    while target_offset < target.len() {
        let matched = if target.len() - target_offset >= COPY_ADD_CHUNK {
            let key = short_hash(&target[target_offset..target_offset + COPY_ADD_CHUNK]);
            chunks.get(&key).and_then(|offsets| {
                offsets.iter().find_map(|base_offset| {
                    (base[*base_offset..*base_offset + COPY_ADD_CHUNK]
                        == target[target_offset..target_offset + COPY_ADD_CHUNK])
                        .then_some(*base_offset)
                })
            })
        } else {
            None
        };
        if let Some(base_offset) = matched {
            flush_add(&mut operations, &mut inserted);
            let mut length = COPY_ADD_CHUNK;
            while base_offset + length < base.len()
                && target_offset + length < target.len()
                && base[base_offset + length] == target[target_offset + length]
            {
                length += 1;
            }
            operations.push(DeltaOperation::Copy {
                offset: base_offset as u64,
                length: length as u64,
            });
            target_offset += length;
        } else {
            // Advance to the next stride-aligned offset, not a blind +stride.
            // Base anchors are indexed only at stride-aligned offsets, so a
            // cursor left misaligned by a previous odd-length match would
            // otherwise skip every remaining anchor and insert the rest of the
            // target verbatim. Re-aligning here keeps one partial match from
            // destroying all subsequent reuse.
            let step = COPY_ADD_STRIDE - (target_offset % COPY_ADD_STRIDE);
            let length = step.min(target.len() - target_offset);
            inserted.extend_from_slice(&target[target_offset..target_offset + length]);
            target_offset += length;
        }
    }
    flush_add(&mut operations, &mut inserted);
    operations
}

fn short_hash(bytes: &[u8]) -> [u8; 16] {
    let hash = blake3::hash(bytes);
    let mut short = [0_u8; 16];
    short.copy_from_slice(&hash.as_bytes()[..16]);
    short
}

fn flush_add(operations: &mut Vec<DeltaOperation>, inserted: &mut Vec<u8>) {
    if !inserted.is_empty() {
        operations.push(DeltaOperation::Add(std::mem::take(inserted)));
    }
}

fn encode_operations(operations: &[DeltaOperation]) -> Result<Vec<u8>> {
    if operations.len() as u64 > MAX_DELTA_OPERATIONS {
        return Err(AdyarError::InvalidInput(
            "copy/add delta has too many operations".into(),
        ));
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(operations.len() as u64).to_le_bytes());
    for operation in operations {
        match operation {
            DeltaOperation::Copy { offset, length } => {
                bytes.push(0);
                bytes.extend_from_slice(&offset.to_le_bytes());
                bytes.extend_from_slice(&length.to_le_bytes());
            }
            DeltaOperation::Add(value) => {
                bytes.push(1);
                bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
                bytes.extend_from_slice(value);
            }
        }
    }
    Ok(bytes)
}

fn encode_delta_header(
    codec: u32,
    base_root: &[u8; 32],
    target_root: &[u8; 32],
    target_length: u64,
    payload: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(DELTA_HEADER_SIZE + payload.len());
    bytes.extend_from_slice(DELTA_MAGIC);
    bytes.extend_from_slice(&DELTA_VERSION.to_le_bytes());
    bytes.extend_from_slice(&codec.to_le_bytes());
    bytes.extend_from_slice(base_root);
    bytes.extend_from_slice(target_root);
    bytes.extend_from_slice(&target_length.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn apply_operations(
    base: &[u8],
    operations: &[DeltaOperation],
    target_length: u64,
) -> Result<Vec<u8>> {
    let capacity = usize::try_from(target_length)
        .map_err(|_| AdyarError::InvalidFormat("delta target exceeds address space".into()))?;
    let mut target = Vec::with_capacity(capacity);
    for operation in operations {
        match operation {
            DeltaOperation::Copy { offset, length } => {
                let end = offset.checked_add(*length).ok_or_else(|| {
                    AdyarError::InvalidFormat("delta copy range overflow".into())
                })?;
                let start = usize::try_from(*offset).map_err(|_| {
                    AdyarError::InvalidFormat("delta copy offset exceeds address space".into())
                })?;
                let end = usize::try_from(end).map_err(|_| {
                    AdyarError::InvalidFormat("delta copy end exceeds address space".into())
                })?;
                let source = base.get(start..end).ok_or_else(|| {
                    AdyarError::InvalidFormat("delta copy exceeds base artifact".into())
                })?;
                extend_bounded(&mut target, source, capacity)?;
            }
            DeltaOperation::Add(bytes) => extend_bounded(&mut target, bytes, capacity)?,
        }
    }
    if target.len() != capacity {
        return Err(AdyarError::InvalidFormat(format!(
            "delta reconstructed {} bytes, expected {capacity}",
            target.len()
        )));
    }
    Ok(target)
}

fn extend_bounded(target: &mut Vec<u8>, bytes: &[u8], limit: usize) -> Result<()> {
    let end = target
        .len()
        .checked_add(bytes.len())
        .ok_or_else(|| AdyarError::InvalidFormat("delta output length overflow".into()))?;
    if end > limit {
        return Err(AdyarError::InvalidFormat(
            "delta operations exceed declared target length".into(),
        ));
    }
    target.extend_from_slice(bytes);
    Ok(())
}

fn operation_totals(operations: &[DeltaOperation]) -> (u64, u64) {
    operations
        .iter()
        .fold((0, 0), |(copied, inserted), operation| match operation {
            DeltaOperation::Copy { length, .. } => (copied + length, inserted),
            DeltaOperation::Add(bytes) => (copied, inserted + bytes.len() as u64),
        })
}

enum ParsedPayload<'a> {
    Snapshot(&'a [u8]),
    CopyAdd(Vec<DeltaOperation>),
}

struct ParsedDelta<'a> {
    base_root: [u8; 32],
    target_root: [u8; 32],
    target_length: u64,
    payload: ParsedPayload<'a>,
}

fn parse_delta(bytes: &[u8]) -> Result<ParsedDelta<'_>> {
    if bytes.len() < DELTA_HEADER_SIZE {
        return Err(AdyarError::InvalidFormat("truncated delta header".into()));
    }
    if &bytes[0..8] != DELTA_MAGIC {
        return Err(AdyarError::InvalidFormat("invalid delta magic".into()));
    }
    if read_u32(bytes, 8)? != DELTA_VERSION {
        return Err(AdyarError::Unsupported("delta version".into()));
    }
    let codec = read_u32(bytes, 12)?;
    let mut base_root = [0_u8; 32];
    base_root.copy_from_slice(&bytes[16..48]);
    let mut target_root = [0_u8; 32];
    target_root.copy_from_slice(&bytes[48..80]);
    let target_length = read_u64(bytes, 80)?;
    if target_length > MAX_DELTA_TARGET_SIZE {
        return Err(AdyarError::InvalidFormat(
            "delta target exceeds size limit".into(),
        ));
    }
    let body = &bytes[DELTA_HEADER_SIZE..];
    let payload = match codec {
        CODEC_SNAPSHOT => {
            if body.len() as u64 != target_length {
                return Err(AdyarError::InvalidFormat(
                    "snapshot delta length does not match target length".into(),
                ));
            }
            ParsedPayload::Snapshot(body)
        }
        CODEC_COPY_ADD => ParsedPayload::CopyAdd(parse_operations(body, target_length)?),
        other => {
            return Err(AdyarError::Unsupported(format!("delta codec {other}")));
        }
    };
    Ok(ParsedDelta {
        base_root,
        target_root,
        target_length,
        payload,
    })
}

fn parse_operations(bytes: &[u8], target_length: u64) -> Result<Vec<DeltaOperation>> {
    let operation_count = read_u64(bytes, 0)?;
    let maximum_encoded_operations = bytes.len().saturating_sub(8) as u64 / 9;
    if operation_count > MAX_DELTA_OPERATIONS
        || operation_count > target_length.saturating_add(1)
        || operation_count > maximum_encoded_operations
    {
        return Err(AdyarError::InvalidFormat(
            "invalid copy/add operation count".into(),
        ));
    }
    let capacity = usize::try_from(operation_count)
        .map_err(|_| AdyarError::InvalidFormat("operation count exceeds address space".into()))?;
    let mut operations = Vec::with_capacity(capacity);
    let mut cursor = 8_usize;
    let mut logical_length = 0_u64;
    for _ in 0..operation_count {
        let kind = *bytes
            .get(cursor)
            .ok_or_else(|| AdyarError::InvalidFormat("truncated delta operation".into()))?;
        cursor += 1;
        match kind {
            0 => {
                let offset = read_u64(bytes, cursor)?;
                let length = read_u64(bytes, cursor + 8)?;
                cursor = cursor
                    .checked_add(16)
                    .ok_or_else(|| AdyarError::InvalidFormat("delta cursor overflow".into()))?;
                if length == 0 {
                    return Err(AdyarError::InvalidFormat("zero-length delta copy".into()));
                }
                logical_length = logical_length.checked_add(length).ok_or_else(|| {
                    AdyarError::InvalidFormat("delta logical length overflow".into())
                })?;
                operations.push(DeltaOperation::Copy { offset, length });
            }
            1 => {
                let length = read_u64(bytes, cursor)?;
                cursor = cursor
                    .checked_add(8)
                    .ok_or_else(|| AdyarError::InvalidFormat("delta cursor overflow".into()))?;
                if length == 0 {
                    return Err(AdyarError::InvalidFormat(
                        "zero-length delta insert".into(),
                    ));
                }
                let length_usize = usize::try_from(length).map_err(|_| {
                    AdyarError::InvalidFormat("delta insert exceeds address space".into())
                })?;
                let end = cursor.checked_add(length_usize).ok_or_else(|| {
                    AdyarError::InvalidFormat("delta insert range overflow".into())
                })?;
                let value = bytes
                    .get(cursor..end)
                    .ok_or_else(|| AdyarError::InvalidFormat("truncated delta insert".into()))?;
                cursor = end;
                logical_length = logical_length.checked_add(length).ok_or_else(|| {
                    AdyarError::InvalidFormat("delta logical length overflow".into())
                })?;
                operations.push(DeltaOperation::Add(value.to_vec()));
            }
            other => {
                return Err(AdyarError::InvalidFormat(format!(
                    "unknown delta operation {other}"
                )));
            }
        }
        if logical_length > target_length {
            return Err(AdyarError::InvalidFormat(
                "delta operations exceed target length".into(),
            ));
        }
    }
    if cursor != bytes.len() || logical_length != target_length {
        return Err(AdyarError::InvalidFormat(
            "delta operations do not exactly cover the target".into(),
        ));
    }
    Ok(operations)
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| AdyarError::InvalidFormat("truncated delta u32".into()))?;
    Ok(u32::from_le_bytes(value.try_into().expect("slice length")))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| AdyarError::InvalidFormat("truncated delta u64".into()))?;
    Ok(u64::from_le_bytes(value.try_into().expect("slice length")))
}

fn write_atomic(path: &Path, bytes: &[u8], replace: bool) -> Result<()> {
    if path.exists() && !replace {
        return Err(AdyarError::InvalidInput(format!(
            "output {} already exists",
            path.display()
        )));
    }
    let temporary = path.with_extension(format!("adyar-delta-tmp-{}", std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        if replace && path.exists() {
            fs::remove_file(path)?;
        }
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_add_reuses_long_unchanged_regions() {
        let mut base = vec![0_u8; 32 * 1024];
        for (index, byte) in base.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        let mut target = base.clone();
        target[12_000..12_100].fill(0xff);
        let operations = build_copy_add(&base, &target);
        let encoded = encode_operations(&operations).unwrap();
        assert!(encoded.len() < target.len() / 5);
        assert_eq!(
            apply_operations(&base, &operations, target.len() as u64).unwrap(),
            target
        );
    }

    #[test]
    fn a_misaligning_partial_match_does_not_destroy_later_reuse() {
        // Regression: base anchors are indexed only at stride-aligned offsets.
        // A first match of non-stride-multiple length leaves the target cursor
        // misaligned; before the fix it then stepped by a fixed stride forever,
        // skipping every remaining anchor and inserting the rest verbatim.
        let filler: Vec<u8> = (0..4096_u32).map(|index| (index % 251) as u8).collect();
        let shared: Vec<u8> = (0..8192_u32).map(|index| (index % 241) as u8).collect();

        // Region A matches for 300 bytes (not a multiple of 8) then diverges.
        let mut base = filler.clone();
        base.extend_from_slice(&shared);
        let mut target = filler.clone();
        target[300] ^= 0xff; // divergence at a non-aligned offset
        target.extend_from_slice(&shared);

        let operations = build_copy_add(&base, &target);
        let (copied, _) = operation_totals(&operations);
        assert!(
            copied as usize > shared.len(),
            "expected the shared tail to be reused, only copied {copied} bytes"
        );
        assert_eq!(
            apply_operations(&base, &operations, target.len() as u64).unwrap(),
            target
        );
    }

    #[test]
    fn operation_parser_rejects_truncation_and_zero_lengths() {
        assert!(parse_operations(&[1, 0, 0, 0, 0, 0, 0, 0], 1).is_err());
        let mut zero_copy = 1_u64.to_le_bytes().to_vec();
        zero_copy.push(0);
        zero_copy.extend_from_slice(&0_u64.to_le_bytes());
        zero_copy.extend_from_slice(&0_u64.to_le_bytes());
        assert!(parse_operations(&zero_copy, 0).is_err());

        let allocation_attack = 1_000_000_u64.to_le_bytes();
        assert!(
            parse_operations(&allocation_attack, MAX_DELTA_TARGET_SIZE).is_err(),
            "operation count must be bounded by encoded payload before allocation"
        );
    }
}
