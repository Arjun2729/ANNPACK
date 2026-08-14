use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::conformance::{ConformanceReport, inspect_conformance_with_manifest};
use crate::error::Result;
use crate::format::PackReader;
use crate::signing::verify_signatures;

// FROZEN WIRE IDENTIFIER: serialized and matched by third parties. It names a
// format version, not a project, and changes only when that version does.
pub const DISCOVERY_MEDIA_TYPE: &str = "application/vnd.annpack.discovery+json";
pub const PACK_MEDIA_TYPE: &str = "application/vnd.annpack.v3";
pub const MANIFEST_MEDIA_TYPE: &str = "application/vnd.annpack.manifest+json";
pub const DELTA_MEDIA_TYPE: &str = "application/vnd.annpack.delta.v1";
pub const WELL_KNOWN_PATH: &str = "/.well-known/annpack.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryDocument {
    pub schema: String,
    pub media_type: String,
    pub publisher: Option<String>,
    pub generated_by: String,
    pub corpora: Vec<DiscoveredCorpus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredCorpus {
    pub name: String,
    pub description: Option<String>,
    pub releases: Vec<DiscoveredRelease>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredRelease {
    pub version: String,
    pub root_hash: String,
    pub url: String,
    pub media_type: String,
    pub bytes: u64,
    pub capabilities: Vec<String>,
    pub source_revision: Option<String>,
    pub signature_key_ids: Vec<String>,
    pub access: String,
    pub license: Option<String>,
    pub conformance: ConformanceReport,
}

pub fn create_discovery(
    packs: &[impl AsRef<Path>],
    public_base_url: Option<&str>,
    publisher: Option<String>,
) -> Result<DiscoveryDocument> {
    let mut corpora = std::collections::BTreeMap::<String, DiscoveredCorpus>::new();
    for path in packs {
        let path = path.as_ref();
        let reader = PackReader::open_path(path)?;
        // Publishing a root in a discovery document asserts that the artifact
        // behind it is intact. Opening the container only checks the directory
        // binding, so verify every referenced section before listing the pack.
        reader.verify_all()?;
        let manifest = reader.manifest()?;
        let conformance = inspect_conformance_with_manifest(&reader, &manifest);
        let signature_key_ids = verify_signatures(&reader, None)?
            .into_iter()
            .map(|report| report.key_id)
            .collect();
        let filename = path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        let url = public_base_url
            .map(|base| format!("{}/{}", base.trim_end_matches('/'), filename))
            .unwrap_or(filename);
        corpora
            .entry(manifest.name.clone())
            .or_insert_with(|| DiscoveredCorpus {
                name: manifest.name.clone(),
                description: manifest.description.clone(),
                releases: Vec::new(),
            })
            .releases
            .push(DiscoveredRelease {
                version: manifest.version,
                root_hash: reader.root_hex(),
                url,
                media_type: PACK_MEDIA_TYPE.into(),
                bytes: std::fs::metadata(path)?.len(),
                capabilities: manifest.capabilities,
                source_revision: manifest.source_revision,
                signature_key_ids,
                access: serde_json::to_value(manifest.policy.access)?
                    .as_str()
                    .unwrap_or("public")
                    .to_string(),
                license: manifest.policy.license,
                conformance,
            });
    }
    let mut corpora: Vec<_> = corpora.into_values().collect();
    for corpus in &mut corpora {
        corpus
            .releases
            .sort_by(|left, right| left.version.cmp(&right.version));
    }
    Ok(DiscoveryDocument {
        schema: "https://annpack.dev/spec/discovery/v1".into(),
        media_type: DISCOVERY_MEDIA_TYPE.into(),
        publisher,
        generated_by: format!("adyar-reference/{}", env!("CARGO_PKG_VERSION")),
        corpora,
    })
}
