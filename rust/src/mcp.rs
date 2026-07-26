use std::io::{BufRead, Write};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{AnnpackError, Result};
use crate::search::{ProfileRequest, SearchEngine, SearchMode, SearchOptions};

pub struct McpServer {
    engine: SearchEngine,
}

impl McpServer {
    pub fn new(engine: SearchEngine) -> Self {
        Self { engine }
    }

    pub fn run<R: BufRead, W: Write>(&self, mut input: R, mut output: W) -> Result<()> {
        let mut line = String::new();
        loop {
            line.clear();
            let read = input.read_line(&mut line)?;
            if read == 0 {
                break;
            }
            let value: Value = match serde_json::from_str(line.trim()) {
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
        let text = serde_json::to_string_pretty(&structured)?;
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
    /// ANN-10 profile: a profile id, "auto", or "lexical" (default).
    profile: Option<String>,
    expansion_weight: Option<f64>,
    splade_weight: Option<f64>,
    debug: Option<bool>,
}

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "knowledge_pack_info",
            "description": "Inspect the exact identity, version, capabilities, and policy of the mounted authoritative knowledge pack.",
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
                    "mode": {"type": "string", "enum": ["lexical", "vector", "hybrid"], "default": "lexical", "description": "Default lexical: BM25 is the measured best mode on real docs; vector/hybrid underperform without a strong embedding profile."},
                    "query_vector": {"type": "array", "items": {"type": "number"}},
                    "vector_profile": {"type": "string"},
                    "vector_probes": {"type": "integer", "minimum": 1, "maximum": 1024, "default": 4},
                    "profile": {"type": "string", "description": "ANN-10 profile: a profile id, \"auto\" (first supported), or \"lexical\" (default; never activates a derived profile)."},
                    "expansion_weight": {"type": "number", "minimum": 0, "description": "Advanced: ANN-7 overlay weight on a non-fat pack (superseded by profile on a fat pack)."},
                    "splade_weight": {"type": "number", "minimum": 0, "description": "Advanced: ANN-8 overlay weight on a non-fat pack (superseded by profile on a fat pack)."},
                    "debug": {"type": "boolean", "default": false}
                },
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
    fn exposes_three_tools() {
        let definitions = tool_definitions();
        assert_eq!(definitions.len(), 3);
        assert_eq!(definitions[0]["name"], "knowledge_pack_info");
    }
}
