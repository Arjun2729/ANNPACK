#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let _ = adyar::delta::inspect_delta_bytes(bytes);
});
