use std::io::{BufRead, Read, Write};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{AnnpackError, Result};
use crate::search::{ProfileRequest, SearchEngine, SearchMode, SearchOptions};

/// Maximum bytes accepted for one JSON-RPC request line.
///
/// Requests are small control messages. The largest legitimate one is a
/// `knowledge_search` carrying a query vector, which even at the v3 ceiling of
/// 65,536 dimensions encodes well below this bound. Without a limit a peer that
/// never sends a newline grows the read buffer without end. This bounds one
/// line; it is not a general defence against memory exhaustion.
pub const MAX_REQUEST_LINE_BYTES: usize = 8 * 1024 * 1024;

/// Chunk size used when discarding the tail of an over-long line. Bounded so
/// skipping past a hostile request never allocates in proportion to it.
const DISCARD_CHUNK_BYTES: u64 = 64 * 1024;

/// Capabilities this runtime can execute, used to report AN-10 profile support
/// through `knowledge_pack_info` so an agent never has to guess a profile id.
const RUNTIME_CAPABILITIES: [&str; 4] = [
    "lexical-bm25",
    "vector-ivf-flat-dot",
    "term-overlay-expansion",
    "term-overlay-splade",
];

pub struct McpServer {
    engine: SearchEngine,
}

impl McpServer {
    pub fn new(engine: SearchEngine) -> Self {
        Self { engine }
    }

    pub fn run<R: BufRead, W: Write>(&self, mut input: R, mut output: W) -> Result<()> {
        let mut line = Vec::new();
        loop {
            line.clear();
            // Read at most one byte past the limit, so an over-long line is
            // detected without ever buffering more than the limit plus one.
            let read = (&mut input)
                .take(MAX_REQUEST_LINE_BYTES as u64 + 1)
                .read_until(b'\n', &mut line)?;
            if read == 0 {
                break;
            }
            if line.len() > MAX_REQUEST_LINE_BYTES {
                // Skip the rest of the request so the next line is still framed
                // correctly, then report the refusal rather than closing.
                if line.last() != Some(&b'\n') {
                    discard_to_newline(&mut input)?;
                }
                write_response(
                    &mut output,
                    &error_response(
                        Value::Null,
                        -32600,
                        format!("request exceeds the {MAX_REQUEST_LINE_BYTES}-byte line limit"),
                    ),
                )?;
                continue;
            }
            let value: Value = match serde_json::from_slice(&line) {
                Ok(value) => value,
                Err(error) => {
                    write_response(
                        &mut output,
                        &error_response(Value::Null, -32700, format!("parse error: {error}")),
                    )?;
                    continue;
                }
            };
            let id = value.get("id").cloned();
            let response = match self.handle(value) {
                Ok(result) => id.clone().map(|id| success_response(id, result)),
                Err(error) => id
                    .clone()
                    .map(|id| error_response(id, -32602, error.to_string())),
            };
            if let Some(response) = response {
                write_response(&mut output, &response)?;
            }
        }
        Ok(())
    }

    fn handle(&self, request: Value) -> Result<Value> {
        if request.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Err(AnnpackError::Protocol("jsonrpc must be 2.0".into()));
        }
        let method = request
            .get("method")
            .and_then(Value::as_str)
            .ok_or_else(|| AnnpackError::Protocol("request method is missing".into()))?;
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {
                    "name": "annpack",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
            "notifications/initialized" | "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tool_definitions()})),
            "tools/call" => self.call_tool(
                request
                    .get("params")
                    .cloned()
                    .ok_or_else(|| AnnpackError::Protocol("tool params are missing".into()))?,
            ),
            other => Err(AnnpackError::Protocol(format!("unknown method {other}"))),
        }
    }

    fn call_tool(&self, params: Value) -> Result<Value> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| AnnpackError::Protocol("tool name is missing".into()))?;
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let structured = match name {
            "knowledge_pack_info" => {
                let manifest = self.engine.manifest();
                let conformance = self.engine.conformance();
                json!({
                    "name": manifest.name,
                    "version": manifest.version,
                    "root_hash": self.engine.reader().root_hex(),
                    "source_revision": manifest.source_revision,
                    "document_count": manifest.document_count,
                    "passage_count": manifest.passage_count,
                    "capabilities": manifest.capabilities,
                    "embedding_profiles": manifest.embedding_profiles,
                    "policy": manifest.policy,
                    "conformance": conformance,
                    // Logical content root. Present from manifest format 2 on;
                    // its absence means this pack cannot issue standalone
                    // receipts.
                    "passage_merkle_root": manifest.passage_merkle_root,
                    "supports_evidence_receipts": manifest.passage_merkle_root.is_some(),
                    // AN-10 discovery. Without this an agent has to guess
                    // profile ids, so every advertised profile is listed with
                    // whether this runtime can actually execute it and why.
                    "retrieval_profiles": manifest
                        .retrieval_profiles
                        .iter()
                        .map(|profile| {
                            let unmet: Vec<&str> = profile
                                .requires
                                .iter()
                                .filter(|capability| {
                                    !RUNTIME_CAPABILITIES.contains(&capability.as_str())
                                })
                                .map(String::as_str)
                                .collect();
                            json!({
                                "id": profile.id,
                                "kind": profile.kind,
                                "requires": profile.requires,
                                "section_ids": profile.section_ids,
                                "supported": unmet.is_empty()
                                    && conformance.extensions_conformant,
                                "unmet_capabilities": unmet,
                                "derived": matches!(
                                    profile.kind.as_str(),
                                    "expansion" | "splade"
                                ),
                                "provenance": manifest
                                    .derived_inputs
                                    .iter()
                                    .filter(|input| {
                                        profile.section_ids.contains(&input.section_id)
                                    })
                                    .map(|input| json!({
                                        "kind": input.kind,
                                        "generator": input.generator,
                                        "model": input.model,
                                        "revision": input.revision,
                                        "params": input.params,
                                        "sidecar_digest": input.sidecar_digest,
                                    }))
                                    .collect::<Vec<_>>(),
                            })
                        })
                        .collect::<Vec<_>>(),
                    "profile_selection_help":
                        "Pass `profile` to knowledge_search: a profile id above, \"auto\" \
                         (first supported), or \"lexical\" (default; never activates a \
                         derived profile). Profile-enabled search is refused when \
                         conformance.extensions_conformant is false.",
                })
            }
            "knowledge_search" => {
                let arguments: SearchArguments = serde_json::from_value(arguments)?;
                let mode = match arguments.mode.as_deref().unwrap_or("lexical") {
                    "lexical" => SearchMode::Lexical,
                    "vector" => SearchMode::Vector,
                    "hybrid" => SearchMode::Hybrid,
                    other => {
                        return Err(AnnpackError::InvalidInput(format!(
                            "unknown search mode {other:?}"
                        )));
                    }
                };
                let profile = match arguments.profile.as_deref() {
                    None | Some("lexical") => ProfileRequest::Lexical,
                    Some("auto") => ProfileRequest::Auto,
                    Some(id) => ProfileRequest::Named(id.to_string()),
                };
                let response = self.engine.search(
                    &arguments.query,
                    &SearchOptions {
                        limit: arguments.limit.unwrap_or(5),
                        mode,
                        query_vector: arguments.query_vector,
                        vector_profile: arguments.vector_profile,
                        vector_probes: arguments.vector_probes.unwrap_or(4),
                        profile,
                        expansion_weight: arguments.expansion_weight.unwrap_or(0.0),
                        splade_weight: arguments.splade_weight.unwrap_or(0.0),
                        debug: arguments.debug.unwrap_or(false),
                        ..SearchOptions::default()
                    },
                )?;
                serde_json::to_value(response)?
            }
            "knowledge_evidence_receipt" => {
                let passage_id = arguments
                    .get("passage_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AnnpackError::InvalidInput("passage_id is required".into()))?;
                serde_json::to_value(self.engine.receipt_for_passage(passage_id)?)?
            }
            "knowledge_get_passage" => {
                let passage_id = arguments
                    .get("passage_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AnnpackError::InvalidInput("passage_id is required".into()))?;
                let passage = self.engine.get_passage(passage_id)?;
                let evidence = self.engine.evidence_for_passage(&passage)?;
                json!({
                    "pack": {
                        "name": self.engine.manifest().name,
                        "version": self.engine.manifest().version,
                        "root_hash": self.engine.reader().root_hex()
                    },
                    "passage": passage,
                    "evidence": evidence
                })
            }
            other => return Err(AnnpackError::Protocol(format!("unknown tool {other}"))),
        };
        // The response carries the same value twice: once structured and once
        // as text. Serialize the text mirror compactly so a large payload — an
        // evidence receipt embeds the stored Documents section — is not doubled
        // and then inflated again by pretty-printing.
        let text = serde_json::to_string(&structured)?;
        Ok(json!({
            "content": [{"type": "text", "text": text}],
            "structuredContent": structured,
            "isError": false
        }))
    }
}

#[derive(Debug, Deserialize)]
struct SearchArguments {
    query: String,
    limit: Option<usize>,
    mode: Option<String>,
    query_vector: Option<Vec<f32>>,
    vector_profile: Option<String>,
    vector_probes: Option<usize>,
    /// AN-10 profile: a profile id, "auto", or "lexical" (default).
    profile: Option<String>,
    expansion_weight: Option<f64>,
    splade_weight: Option<f64>,
    debug: Option<bool>,
}

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "knowledge_pack_info",
            "description": "Inspect the exact identity, version, capabilities, policy, conformance, and available AN-10 retrieval profiles of the mounted knowledge pack. Call this first to discover valid `profile` values for knowledge_search.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
        }),
        json!({
            "name": "knowledge_search",
            "description": "Search the mounted knowledge pack and return exact passages with versioned citations.",
            "inputSchema": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 5},
                    "mode": {"type": "string", "enum": ["lexical", "vector", "hybrid"], "default": "lexical", "description": "Lexical is the portable Core default and requires no query embedding. Vector and hybrid need a pack with an AN-1 profile and a query vector."},
                    "query_vector": {"type": "array", "items": {"type": "number"}},
                    "vector_profile": {"type": "string"},
                    "vector_probes": {"type": "integer", "minimum": 1, "maximum": 1024, "default": 4},
                    "profile": {"type": "string", "description": "AN-10 profile: a profile id, \"auto\" (first supported), or \"lexical\" (default; never activates a derived profile)."},
                    "expansion_weight": {"type": "number", "minimum": 0, "description": "Advanced: AN-7 overlay weight on a non-fat pack (superseded by profile on a fat pack)."},
                    "splade_weight": {"type": "number", "minimum": 0, "description": "Advanced: AN-8 overlay weight on a non-fat pack (superseded by profile on a fat pack)."},
                    "debug": {"type": "boolean", "default": false}
                },
                "additionalProperties": false
            }
        }),
        json!({
            "name": "knowledge_evidence_receipt",
            "description": "Issue a standalone, offline-verifiable receipt proving a passage existed unmodified in this exact artifact. The receipt verifies with `annpack verify-evidence` without the pack, without network access, and without trusting this server.",
            "inputSchema": {
                "type": "object",
                "required": ["passage_id"],
                "properties": {"passage_id": {"type": "string", "minLength": 64, "maxLength": 64}},
                "additionalProperties": false
            }
        }),
        json!({
            "name": "knowledge_get_passage",
            "description": "Retrieve one exact passage by its stable content-derived identifier.",
            "inputSchema": {
                "type": "object",
                "required": ["passage_id"],
                "properties": {"passage_id": {"type": "string", "minLength": 64, "maxLength": 64}},
                "additionalProperties": false
            }
        }),
    ]
}

fn success_response(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn error_response(id: Value, code: i32, message: String) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// Consumes input up to and including the next newline in bounded chunks,
/// without retaining it.
fn discard_to_newline(input: &mut impl BufRead) -> Result<()> {
    let mut scratch = Vec::new();
    loop {
        scratch.clear();
        let read = (&mut *input)
            .take(DISCARD_CHUNK_BYTES)
            .read_until(b'\n', &mut scratch)?;
        if read == 0 || scratch.last() == Some(&b'\n') {
            return Ok(());
        }
    }
}

fn write_response(output: &mut impl Write, response: &Value) -> Result<()> {
    serde_json::to_writer(&mut *output, response)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_the_documented_tool_surface() {
        let definitions = tool_definitions();
        let names: Vec<&str> = definitions
            .iter()
            .map(|definition| definition["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec![
                "knowledge_pack_info",
                "knowledge_search",
                "knowledge_evidence_receipt",
                "knowledge_get_passage",
            ]
        );
    }
}
