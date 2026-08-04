#![no_main]

use std::sync::Arc;

use annpack::format::PackReader;
use annpack::reader::MemoryReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = PackReader::open(Arc::new(MemoryReader::new(data.to_vec())));
});

