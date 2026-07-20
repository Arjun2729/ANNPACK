use std::sync::Arc;

use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::format::PackReader;
use crate::reader::MemoryReader;
use crate::search::{SearchEngine, SearchMode, SearchOptions};

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn json_compatible<T: Serialize>(value: &T) -> std::result::Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

#[wasm_bindgen]
pub fn inspect_pack(bytes: Vec<u8>) -> std::result::Result<JsValue, JsValue> {
    let reader = PackReader::open(Arc::new(MemoryReader::new(bytes))).map_err(js_error)?;
    let manifest = reader.manifest().map_err(js_error)?;
    json_compatible(&serde_json::json!({
        "root_hash": reader.root_hex(),
        "manifest": manifest,
        "sections": reader.entries.iter().map(|entry| serde_json::json!({
            "id": entry.section_id,
            "type": entry.section_type.name(),
            "offset": entry.offset,
            "length": entry.stored_length,
        })).collect::<Vec<_>>()
    }))
}

#[wasm_bindgen]
pub fn search_pack(
    bytes: Vec<u8>,
    query: String,
    limit: usize,
) -> std::result::Result<JsValue, JsValue> {
    let engine = SearchEngine::open_source(Arc::new(MemoryReader::new(bytes))).map_err(js_error)?;
    let response = engine
        .search(
            &query,
            &SearchOptions {
                limit,
                mode: SearchMode::Lexical,
                ..SearchOptions::default()
            },
        )
        .map_err(js_error)?;
    json_compatible(&response)
}

#[wasm_bindgen]
pub fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[wasm_bindgen]
pub fn inflate_zlib(bytes: &[u8], limit: usize) -> std::result::Result<Vec<u8>, JsValue> {
    miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(bytes, limit).map_err(js_error)
}
