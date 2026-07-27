use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use walkdir::WalkDir;

use crate::error::{AnnpackError, Result};
use crate::model::{Document, Passage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    Auto,
    Markdown,
    Okf,
}

impl InputFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Markdown => "markdown",
            Self::Okf => "okf",
        }
    }
}

#[derive(Debug, Clone)]
pub struct IngestOptions {
    pub target_chars: usize,
    pub max_chars: usize,
    pub base_url: Option<String>,
    pub input_format: InputFormat,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            target_chars: 1_200,
            max_chars: 2_400,
            base_url: None,
            input_format: InputFormat::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IngestedCorpus {
    pub documents: Vec<Document>,
    pub passages: Vec<Passage>,
    pub ignored: Vec<String>,
    pub input_format: InputFormat,
    pub input_format_version: Option<String>,
    pub source_digest: String,
}

#[derive(Debug, Clone)]
struct Block {
    text: String,
    start: usize,
    end: usize,
    heading_path: Vec<String>,
    anchor: Option<String>,
    indivisible: bool,
}

pub fn ingest_directory(root: impl AsRef<Path>, options: &IngestOptions) -> Result<IngestedCorpus> {
    let root = root.as_ref();
    if !root.is_dir() {
        return Err(AnnpackError::InvalidInput(format!(
            "input {} is not a directory",
            root.display()
        )));
    }
    let mut paths = Vec::new();
    let mut ignored = Vec::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry.map_err(|error| AnnpackError::InvalidInput(error.to_string()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        match path.extension().and_then(|value| value.to_str()) {
            Some("md" | "mdx") => paths.push(path.to_path_buf()),
            _ => ignored.push(relative_path(root, path)?),
        }
    }
    paths.sort_by_key(|path| relative_path(root, path).unwrap_or_default());
    ignored.sort();

    let input_format = match options.input_format {
        InputFormat::Auto if looks_like_okf_bundle(root, &paths)? => InputFormat::Okf,
        InputFormat::Auto => InputFormat::Markdown,
        explicit => explicit,
    };
    let input_format_version = if input_format == InputFormat::Okf {
        Some(detect_okf_version(root)?.unwrap_or_else(|| "0.1".into()))
    } else {
        None
    };

    let mut documents = Vec::with_capacity(paths.len());
    let mut passages = Vec::new();
    let mut source_hasher = blake3::Hasher::new();
    for path in paths {
        let source_path = relative_path(root, &path)?;
        if input_format == InputFormat::Okf
            && path.extension().and_then(|value| value.to_str()) != Some("md")
        {
            ignored.push(source_path);
            continue;
        }
        let source = fs::read_to_string(&path).map_err(|error| {
            AnnpackError::InvalidInput(format!("cannot read {source_path}: {error}"))
        })?;
        source_hasher.update(&(source_path.len() as u64).to_le_bytes());
        source_hasher.update(source_path.as_bytes());
        source_hasher.update(&(source.len() as u64).to_le_bytes());
        source_hasher.update(source.as_bytes());

        let (mut front_matter, body, body_offset, has_front_matter) = split_front_matter(&source)
            .map_err(|error| {
            AnnpackError::InvalidInput(format!("invalid frontmatter in {source_path}: {error}"))
        })?;
        if input_format == InputFormat::Okf {
            validate_okf_document(&source_path, &front_matter, has_front_matter, body)?;
        }
        let mut blocks = parse_blocks(body, body_offset);
        let title = front_matter
            .get("title")
            .cloned()
            .or_else(|| {
                blocks
                    .iter()
                    .find_map(|block| block.heading_path.first().cloned())
            })
            .unwrap_or_else(|| {
                Path::new(&source_path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("Untitled")
                    .replace(['-', '_'], " ")
            });
        if blocks.is_empty()
            && let Some(description) = front_matter
                .get("description")
                .filter(|value| !value.trim().is_empty())
        {
            blocks.push(Block {
                text: description.clone(),
                start: body_offset,
                end: body_offset,
                heading_path: vec![title.clone()],
                anchor: Some(slugify(&title)),
                indivisible: true,
            });
        }
        let explicit_url = front_matter
            .get("url")
            .or_else(|| front_matter.get("canonical_url"))
            .or_else(|| front_matter.get("permalink"))
            .or_else(|| front_matter.get("resource"))
            .cloned();
        let url = explicit_url.or_else(|| {
            options.base_url.as_ref().map(|base| {
                format!(
                    "{}/{}",
                    base.trim_end_matches('/'),
                    source_path.trim_start_matches('/')
                )
            })
        });
        let document_id = stable_hash(&[b"document\0", source_path.as_bytes()]);
        if input_format == InputFormat::Okf {
            let filename = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            let kind = match filename {
                "index.md" => "index",
                "log.md" => "log",
                _ => "concept",
            };
            front_matter.insert("okf.kind".into(), kind.into());
            front_matter.insert(
                "okf.concept_id".into(),
                source_path
                    .strip_suffix(".md")
                    .unwrap_or(&source_path)
                    .into(),
            );
            front_matter.insert(
                "okf.version".into(),
                input_format_version.clone().unwrap_or_else(|| "0.1".into()),
            );
        }
        let mut metadata = front_matter;
        metadata.remove("title");
        metadata.remove("url");
        metadata.remove("canonical_url");
        metadata.remove("permalink");
        documents.push(Document {
            id: document_id.clone(),
            source_path: source_path.clone(),
            title,
            url,
            metadata,
        });

        let chunks = chunk_blocks(&blocks, options);
        for chunk in chunks {
            if chunk.text.trim().is_empty() {
                continue;
            }
            let passage_id = stable_hash(&[
                b"passage\0",
                document_id.as_bytes(),
                b"\0",
                chunk.heading_path.join("\x1f").as_bytes(),
                b"\0",
                normalize_text(&chunk.text).as_bytes(),
            ]);
            passages.push(Passage {
                id: passage_id,
                document_id: document_id.clone(),
                ordinal: passages.len() as u32,
                heading_path: chunk.heading_path,
                anchor: chunk.anchor,
                text: chunk.text.trim().to_string(),
                source_byte_start: Some(chunk.start as u64),
                source_byte_end: Some(chunk.end as u64),
            });
        }
    }

    Ok(IngestedCorpus {
        documents,
        passages,
        ignored,
        input_format,
        input_format_version,
        source_digest: source_hasher.finalize().to_hex().to_string(),
    })
}

fn looks_like_okf_bundle(root: &Path, paths: &[std::path::PathBuf]) -> Result<bool> {
    if detect_okf_version(root)?.is_some() {
        return Ok(true);
    }
    let mut concepts = 0_usize;
    for path in paths {
        if path.extension().and_then(|value| value.to_str()) != Some("md") || is_okf_reserved(path)
        {
            continue;
        }
        concepts += 1;
        let source = fs::read_to_string(path)?;
        let (front_matter, _, _, present) = split_front_matter(&source).map_err(|error| {
            AnnpackError::InvalidInput(format!(
                "invalid frontmatter in {}: {error}",
                path.display()
            ))
        })?;
        if !present
            || front_matter
                .get("type")
                .is_none_or(|value| value.trim().is_empty())
        {
            return Ok(false);
        }
    }
    Ok(concepts > 0)
}

fn detect_okf_version(root: &Path) -> Result<Option<String>> {
    let index = root.join("index.md");
    if !index.is_file() {
        return Ok(None);
    }
    let source = fs::read_to_string(&index)?;
    let (front_matter, _, _, _) = split_front_matter(&source).map_err(|error| {
        AnnpackError::InvalidInput(format!("invalid frontmatter in index.md: {error}"))
    })?;
    Ok(front_matter.get("okf_version").cloned())
}

fn is_okf_reserved(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|value| value.to_str()),
        Some("index.md" | "log.md")
    )
}

fn validate_okf_document(
    source_path: &str,
    front_matter: &BTreeMap<String, String>,
    has_front_matter: bool,
    body: &str,
) -> Result<()> {
    let filename = Path::new(source_path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    match filename {
        "index.md" => {
            if source_path != "index.md" && has_front_matter {
                return Err(AnnpackError::InvalidInput(format!(
                    "OKF {source_path} is a nested index.md and must not contain frontmatter"
                )));
            }
        }
        "log.md" => {
            if has_front_matter {
                return Err(AnnpackError::InvalidInput(format!(
                    "OKF {source_path} is a log.md and must not contain frontmatter"
                )));
            }
            for line in body.lines().filter(|line| line.starts_with("## ")) {
                let date = line.trim_start_matches("## ").trim();
                if !is_iso_date(date) {
                    return Err(AnnpackError::InvalidInput(format!(
                        "OKF {source_path} contains non-ISO date heading {date:?}"
                    )));
                }
            }
        }
        _ => {
            if !has_front_matter {
                return Err(AnnpackError::InvalidInput(format!(
                    "OKF concept {source_path} is missing YAML frontmatter"
                )));
            }
            if front_matter
                .get("type")
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err(AnnpackError::InvalidInput(format!(
                    "OKF concept {source_path} is missing required non-empty type"
                )));
            }
        }
    }
    Ok(())
}

fn is_iso_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
}

fn relative_path(root: &Path, path: &Path) -> Result<String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| AnnpackError::InvalidInput("path escaped input root".into()))?;
    Ok(relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}

fn split_front_matter(source: &str) -> Result<(BTreeMap<String, String>, &str, usize, bool)> {
    let normalized_start = source.strip_prefix('\u{feff}').unwrap_or(source);
    if !normalized_start.starts_with("---\n") && !normalized_start.starts_with("---\r\n") {
        return Ok((BTreeMap::new(), source, 0, false));
    }
    let first_newline = normalized_start.find('\n').unwrap_or(3) + 1;
    let rest = &normalized_start[first_newline..];
    let mut relative_end = None;
    let mut cursor = 0_usize;
    for line in rest.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            relative_end = Some(cursor);
            break;
        }
        cursor += line.len();
    }
    let Some(relative_end) = relative_end else {
        return Err(AnnpackError::InvalidInput(
            "frontmatter is missing its closing --- delimiter".into(),
        ));
    };
    let yaml = &rest[..relative_end];
    let closing_start = first_newline + relative_end;
    let after_closing = &normalized_start[closing_start + 3..];
    let newline_len = if after_closing.starts_with("\r\n") {
        2
    } else if after_closing.starts_with('\n') {
        1
    } else {
        0
    };
    let body_offset = source.len() - after_closing.len() + newline_len;
    let yaml_value: serde_yaml_ng::Value = serde_yaml_ng::from_str(yaml)
        .map_err(|error| AnnpackError::InvalidInput(error.to_string()))?;
    let mapping = match yaml_value {
        serde_yaml_ng::Value::Null => serde_yaml_ng::Mapping::new(),
        serde_yaml_ng::Value::Mapping(mapping) => mapping,
        _ => {
            return Err(AnnpackError::InvalidInput(
                "frontmatter root must be a YAML mapping".into(),
            ));
        }
    };
    let mut metadata = BTreeMap::new();
    for (key, value) in mapping {
        let serde_yaml_ng::Value::String(key) = key else {
            return Err(AnnpackError::InvalidInput(
                "frontmatter keys must be strings".into(),
            ));
        };
        let value = match value {
            serde_yaml_ng::Value::Null => String::new(),
            serde_yaml_ng::Value::Bool(value) => value.to_string(),
            serde_yaml_ng::Value::Number(value) => value.to_string(),
            serde_yaml_ng::Value::String(value) => value,
            value => serde_json::to_string(&value)?,
        };
        metadata.insert(key, value);
    }
    Ok((metadata, &source[body_offset..], body_offset, true))
}

fn parse_blocks(body: &str, base_offset: usize) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut heading_path: Vec<String> = Vec::new();
    let mut paragraph = String::new();
    let mut paragraph_start = 0;
    let mut byte_cursor = 0;
    let mut in_code = false;
    let mut code = String::new();
    let mut code_start = 0;

    let flush_paragraph = |blocks: &mut Vec<Block>,
                           paragraph: &mut String,
                           paragraph_start: usize,
                           end: usize,
                           heading_path: &[String]| {
        if !paragraph.trim().is_empty() {
            blocks.push(Block {
                text: paragraph.trim_end().to_string(),
                start: base_offset + paragraph_start,
                end: base_offset + end,
                heading_path: heading_path.to_vec(),
                anchor: heading_path.last().map(|heading| slugify(heading)),
                indivisible: false,
            });
        }
        paragraph.clear();
    };

    for line_with_newline in body.split_inclusive('\n') {
        let line = line_with_newline.trim_end_matches(['\r', '\n']);
        let line_start = byte_cursor;
        byte_cursor += line_with_newline.len();

        if in_code {
            code.push_str(line_with_newline);
            if line.trim_start().starts_with("```") && code.lines().count() > 1 {
                blocks.push(Block {
                    text: code.trim_end().to_string(),
                    start: base_offset + code_start,
                    end: base_offset + byte_cursor,
                    heading_path: heading_path.clone(),
                    anchor: heading_path.last().map(|heading| slugify(heading)),
                    indivisible: true,
                });
                code.clear();
                in_code = false;
            }
            continue;
        }

        if line.trim_start().starts_with("```") {
            flush_paragraph(
                &mut blocks,
                &mut paragraph,
                paragraph_start,
                line_start,
                &heading_path,
            );
            in_code = true;
            code_start = line_start;
            code.push_str(line_with_newline);
            continue;
        }

        if let Some((level, heading)) = parse_heading(line) {
            flush_paragraph(
                &mut blocks,
                &mut paragraph,
                paragraph_start,
                line_start,
                &heading_path,
            );
            heading_path.truncate(level.saturating_sub(1));
            while heading_path.len() < level.saturating_sub(1) {
                heading_path.push(String::new());
            }
            heading_path.push(heading.to_string());
            continue;
        }

        if line.trim().is_empty() {
            flush_paragraph(
                &mut blocks,
                &mut paragraph,
                paragraph_start,
                byte_cursor,
                &heading_path,
            );
        } else {
            if paragraph.is_empty() {
                paragraph_start = line_start;
            }
            paragraph.push_str(line_with_newline);
        }
    }

    if in_code && !code.trim().is_empty() {
        blocks.push(Block {
            text: code.trim_end().to_string(),
            start: base_offset + code_start,
            end: base_offset + body.len(),
            heading_path: heading_path.clone(),
            anchor: heading_path.last().map(|heading| slugify(heading)),
            indivisible: true,
        });
    }
    flush_paragraph(
        &mut blocks,
        &mut paragraph,
        paragraph_start,
        body.len(),
        &heading_path,
    );
    blocks
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let hashes = trimmed.bytes().take_while(|byte| *byte == b'#').count();
    if hashes == 0 || hashes > 6 || trimmed.as_bytes().get(hashes) != Some(&b' ') {
        return None;
    }
    let heading = trimmed[hashes + 1..].trim().trim_end_matches('#').trim();
    (!heading.is_empty()).then_some((hashes, heading))
}

fn chunk_blocks(blocks: &[Block], options: &IngestOptions) -> Vec<Block> {
    let mut chunks = Vec::new();
    let mut current: Option<Block> = None;
    for block in blocks {
        if block.text.chars().count() > options.max_chars && !block.indivisible {
            if let Some(existing) = current.take() {
                chunks.push(existing);
            }
            chunks.extend(split_oversized_block(block, options.max_chars));
            continue;
        }
        let should_flush = current.as_ref().is_some_and(|existing| {
            existing.heading_path != block.heading_path
                || existing.text.chars().count() + 2 + block.text.chars().count()
                    > options.target_chars
        });
        if should_flush {
            chunks.push(current.take().expect("current exists"));
        }
        if let Some(existing) = current.as_mut() {
            existing.text.push_str("\n\n");
            existing.text.push_str(&block.text);
            existing.end = block.end;
        } else {
            current = Some(block.clone());
        }
    }
    if let Some(existing) = current {
        chunks.push(existing);
    }
    chunks
}

fn split_oversized_block(block: &Block, max_chars: usize) -> Vec<Block> {
    if max_chars == 0 {
        return vec![block.clone()];
    }
    let chars: Vec<char> = block.text.chars().collect();
    // `start`/`end` are byte offsets into the source document while chunking is
    // by character count. Accumulate the real UTF-8 length of each chunk instead
    // of multiplying the chunk index by a character count: for any non-ASCII
    // text the two differ, and the resulting offsets would misattribute the
    // source span a passage claims to come from.
    let mut byte_cursor = block.start;
    chars
        .chunks(max_chars)
        .map(|chunk| {
            let text: String = chunk.iter().collect();
            let byte_length = text.len();
            let mut result = block.clone();
            result.text = text;
            result.start = byte_cursor;
            result.end = byte_cursor + byte_length;
            byte_cursor += byte_length;
            result
        })
        .collect()
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn stable_hash(parts: &[&[u8]]) -> String {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().to_hex().to_string()
}

fn slugify(value: &str) -> String {
    let mut slug = String::new();
    let mut hyphen = false;
    for character in value.chars().flat_map(char::to_lowercase) {
        if character.is_alphanumeric() {
            slug.push(character);
            hyphen = false;
        } else if !hyphen && !slug.is_empty() {
            slug.push('-');
            hyphen = true;
        }
    }
    slug.trim_end_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_front_matter_and_structural_blocks() {
        let source = "---\ntitle: Demo\nurl: https://example.test/demo\n---\n# Root\n\nIntro.\n\n## API\n\nUse `rotateKey`.\n";
        let (metadata, body, offset, present) = split_front_matter(source).unwrap();
        assert!(present);
        assert_eq!(metadata["title"], "Demo");
        let blocks = parse_blocks(body, offset);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1].heading_path, vec!["Root", "API"]);
    }

    #[test]
    fn keeps_fenced_code_as_one_block() {
        let source = "# API\n\n```rust\nfn main() {\n    println!(\"ok\");\n}\n```\n";
        let blocks = parse_blocks(source, 0);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].indivisible);
        assert!(blocks[0].text.contains("println"));
    }

    #[test]
    fn oversized_multibyte_blocks_keep_byte_accurate_source_spans() {
        // Three-byte characters: a char-count chunk boundary is nowhere near the
        // byte offset it used to be multiplied into.
        let text = "。".repeat(10);
        let block = Block {
            text: text.clone(),
            start: 100,
            end: 100 + text.len(),
            heading_path: vec!["H".into()],
            anchor: None,
            indivisible: false,
        };
        let chunks = split_oversized_block(&block, 4);
        assert_eq!(chunks.len(), 3);
        // Spans must be contiguous, byte-accurate, and land inside the original.
        assert_eq!(chunks[0].start, 100);
        let mut cursor = 100;
        for chunk in &chunks {
            assert_eq!(
                chunk.start, cursor,
                "chunk start must follow the previous end"
            );
            assert_eq!(
                chunk.end - chunk.start,
                chunk.text.len(),
                "span width must equal the chunk's UTF-8 byte length"
            );
            cursor = chunk.end;
        }
        assert_eq!(
            cursor, block.end,
            "chunks must exactly cover the source span"
        );
    }

    #[test]
    fn stable_hash_is_deterministic() {
        assert_eq!(stable_hash(&[b"a", b"b"]), stable_hash(&[b"a", b"b"]));
        assert_ne!(stable_hash(&[b"a", b"b"]), stable_hash(&[b"b", b"a"]));
    }

    #[test]
    fn parses_okf_yaml_sequences_and_folded_scalars() {
        let source = "---\ntype: BigQuery Table\ntags:\n- sales\n- orders\ndescription: >-\n  One row per\n  completed order.\n---\nBody.\n";
        let (metadata, body, _, present) = split_front_matter(source).unwrap();
        assert!(present);
        assert_eq!(metadata["tags"], "[\"sales\",\"orders\"]");
        assert_eq!(metadata["description"], "One row per completed order.");
        assert_eq!(body, "Body.\n");
    }
}
