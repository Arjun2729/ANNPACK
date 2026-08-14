#![no_main]

use adyar::search::decode_varint;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut cursor = 0;
    let _ = decode_varint(data, &mut cursor);
});

