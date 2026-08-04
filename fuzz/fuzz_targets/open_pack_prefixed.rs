#![no_main]

use std::sync::Arc;

use annpack::format::PackReader;
use annpack::reader::MemoryReader;
use libfuzzer_sys::fuzz_target;

/// Variant of open_pack that prepends a valid ANNPACK3 magic + format-version
/// prefix so libfuzzer reaches directory and section parsing on every input
/// rather than spending most executions on magic-byte rejection.
///
/// This is additive: run alongside open_pack, not instead of it.
fuzz_target!(|data: &[u8]| {
    // Prefix: magic (8) + format_version=3 u32le (4) = 12 bytes
    let mut prefixed = Vec::with_capacity(12 + data.len());
    prefixed.extend_from_slice(b"ANNPACK3");
    prefixed.extend_from_slice(&3_u32.to_le_bytes());
    prefixed.extend_from_slice(data);
    let _ = PackReader::open(Arc::new(MemoryReader::new(prefixed)));
});
