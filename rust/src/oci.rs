use std::fs;
#[cfg(feature = "http")]
use std::fs::OpenOptions;
#[cfg(feature = "http")]
use std::io::{Read, Write};
use std::path::Path;

#[cfg(feature = "http")]
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::discovery::PACK_MEDIA_TYPE;
use crate::error::{AnnpackError, Result};
use crate::format::PackReader;

pub const OCI_MANIFEST_MEDIA_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
pub const OCI_ARTIFACT_TYPE: &str = "application/vnd.annpack.v3";
const OCI_CONFIG_MEDIA_TYPE: &str = "application/vnd.annpack.config.v1+json";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OciManifest {
    pub schema_version: u32,
    pub media_type: String,
    pub artifact_type: String,
    pub config: OciDescriptor,
    pub layers: Vec<OciDescriptor>,
    pub annotations: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OciDescriptor {
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<std::collections::BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryCredentials {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OciPushReport {
    pub reference: String,
    pub manifest_digest: String,
    pub pack_digest: String,
    pub pack_root: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct OciPullReport {
    pub reference: String,
    pub output: String,
    pub manifest_digest: String,
    pub pack_digest: String,
    pub pack_root: String,
    pub bytes: u64,
}

pub fn create_oci_manifest(pack: &Path) -> Result<OciManifest> {
    let bytes = fs::read(pack)?;
    let reader = PackReader::open_path(pack)?;
    let manifest = reader.manifest()?;
    let empty_config = b"{}";
    let config_digest = format!("sha256:{:x}", Sha256::digest(empty_config));
    let pack_digest = format!("sha256:{:x}", Sha256::digest(&bytes));
    let mut annotations = std::collections::BTreeMap::new();
    annotations.insert(
        "org.opencontainers.image.title".into(),
        manifest.name.clone(),
    );
    annotations.insert(
        "org.opencontainers.image.version".into(),
        manifest.version.clone(),
    );
    annotations.insert("dev.annpack.root".into(), reader.root_hex());
    if let Some(revision) = manifest.source_revision {
        annotations.insert("org.opencontainers.image.revision".into(), revision);
    }
    Ok(OciManifest {
        schema_version: 2,
        media_type: OCI_MANIFEST_MEDIA_TYPE.into(),
        artifact_type: OCI_ARTIFACT_TYPE.into(),
        config: OciDescriptor {
            media_type: OCI_CONFIG_MEDIA_TYPE.into(),
            digest: config_digest,
            size: empty_config.len() as u64,
            data: Some("e30=".into()),
            annotations: None,
        },
        layers: vec![OciDescriptor {
            media_type: PACK_MEDIA_TYPE.into(),
            digest: pack_digest,
            size: bytes.len() as u64,
            data: None,
            annotations: Some({
                let mut values = std::collections::BTreeMap::new();
                values.insert(
                    "org.opencontainers.image.title".into(),
                    pack.file_name()
                        .map(|value| value.to_string_lossy().to_string())
                        .unwrap_or_else(|| "knowledge.annpack".into()),
                );
                values
            }),
        }],
        annotations,
    })
}

#[cfg(feature = "http")]
pub fn push_pack(
    pack: &Path,
    reference: &str,
    credentials: Option<RegistryCredentials>,
) -> Result<OciPushReport> {
    let pack_bytes = fs::read(pack)?;
    let reader = PackReader::open_path(pack)?;
    reader.verify_all()?;
    let pack_root = reader.root_hex();
    let pack_digest = format!("sha256:{:x}", Sha256::digest(&pack_bytes));
    let manifest = create_oci_manifest(pack)?;
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let manifest_digest = format!("sha256:{:x}", Sha256::digest(&manifest_bytes));
    let parsed = OciReference::parse(reference)?;
    reject_insecure_credentials(&parsed, credentials.as_ref())?;
    let mut client = RegistryClient::new(parsed.base_url.clone(), credentials);

    upload_blob(
        &mut client,
        &parsed,
        &format!("sha256:{:x}", Sha256::digest(b"{}")),
        b"{}",
    )?;
    upload_blob(&mut client, &parsed, &pack_digest, &pack_bytes)?;
    let manifest_url = parsed.endpoint(&format!("manifests/{}", parsed.reference));
    let response = client.request(
        "PUT",
        &manifest_url,
        Some(OCI_MANIFEST_MEDIA_TYPE),
        Some(OCI_MANIFEST_MEDIA_TYPE),
        Some(&manifest_bytes),
    )?;
    if response.status() != 201 && response.status() != 202 {
        return Err(AnnpackError::Protocol(format!(
            "registry manifest upload returned HTTP {}",
            response.status()
        )));
    }
    if let Some(actual) = response.header("Docker-Content-Digest")
        && actual != manifest_digest
    {
        return Err(AnnpackError::Integrity(format!(
            "registry reported manifest digest {actual}, expected {manifest_digest}"
        )));
    }
    Ok(OciPushReport {
        reference: parsed.display(),
        manifest_digest,
        pack_digest,
        pack_root,
        bytes: pack_bytes.len() as u64,
    })
}

#[cfg(not(feature = "http"))]
pub fn push_pack(
    _pack: &Path,
    _reference: &str,
    _credentials: Option<RegistryCredentials>,
) -> Result<OciPushReport> {
    Err(AnnpackError::Unsupported(
        "binary was built without HTTP registry support".into(),
    ))
}

#[cfg(feature = "http")]
pub fn pull_pack(
    reference: &str,
    output: &Path,
    credentials: Option<RegistryCredentials>,
    force: bool,
) -> Result<OciPullReport> {
    if output.exists() && !force {
        return Err(AnnpackError::InvalidInput(format!(
            "output {} already exists; pass --force to replace it",
            output.display()
        )));
    }
    let parsed = OciReference::parse(reference)?;
    reject_insecure_credentials(&parsed, credentials.as_ref())?;
    let mut client = RegistryClient::new(parsed.base_url.clone(), credentials);
    let manifest_url = parsed.endpoint(&format!("manifests/{}", parsed.reference));
    let response = client.request(
        "GET",
        &manifest_url,
        Some(&format!(
            "{OCI_MANIFEST_MEDIA_TYPE}, application/vnd.docker.distribution.manifest.v2+json"
        )),
        None,
        None,
    )?;
    let manifest_bytes = read_bounded_response(response, 16 * 1024 * 1024)?;
    let manifest_digest = format!("sha256:{:x}", Sha256::digest(&manifest_bytes));
    let manifest: OciManifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.artifact_type != OCI_ARTIFACT_TYPE || manifest.layers.len() != 1 {
        return Err(AnnpackError::InvalidFormat(
            "OCI manifest is not a single-layer ANNPack artifact".into(),
        ));
    }
    let layer = &manifest.layers[0];
    if layer.media_type != PACK_MEDIA_TYPE || !valid_sha256_digest(&layer.digest) {
        return Err(AnnpackError::InvalidFormat(
            "OCI manifest has an invalid ANNPack layer descriptor".into(),
        ));
    }
    let blob_url = parsed.endpoint(&format!("blobs/{}", layer.digest));
    let response = client.request("GET", &blob_url, Some(PACK_MEDIA_TYPE), None, None)?;
    let pack_bytes = read_bounded_response(response, layer.size)?;
    if pack_bytes.len() as u64 != layer.size {
        return Err(AnnpackError::Integrity(format!(
            "registry blob length {} does not match descriptor {}",
            pack_bytes.len(),
            layer.size
        )));
    }
    let pack_digest = format!("sha256:{:x}", Sha256::digest(&pack_bytes));
    if pack_digest != layer.digest {
        return Err(AnnpackError::Integrity(format!(
            "registry blob digest {pack_digest} does not match {}",
            layer.digest
        )));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension(format!("annpack-tmp-{}", std::process::id()));
    let result = (|| -> Result<String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&pack_bytes)?;
        file.sync_all()?;
        drop(file);
        let reader = PackReader::open_path(&temporary)?;
        reader.verify_all()?;
        let root = reader.root_hex();
        if let Some(expected) = manifest.annotations.get("dev.annpack.root")
            && expected != &root
        {
            return Err(AnnpackError::Integrity(format!(
                "pack root {root} does not match OCI annotation {expected}"
            )));
        }
        if force && output.exists() {
            fs::remove_file(output)?;
        }
        fs::rename(&temporary, output)?;
        Ok(root)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    let pack_root = result?;
    Ok(OciPullReport {
        reference: parsed.display(),
        output: output.display().to_string(),
        manifest_digest,
        pack_digest,
        pack_root,
        bytes: pack_bytes.len() as u64,
    })
}

#[cfg(not(feature = "http"))]
pub fn pull_pack(
    _reference: &str,
    _output: &Path,
    _credentials: Option<RegistryCredentials>,
    _force: bool,
) -> Result<OciPullReport> {
    Err(AnnpackError::Unsupported(
        "binary was built without HTTP registry support".into(),
    ))
}

#[cfg(feature = "http")]
fn upload_blob(
    client: &mut RegistryClient,
    reference: &OciReference,
    digest: &str,
    bytes: &[u8],
) -> Result<()> {
    if !valid_sha256_digest(digest) {
        return Err(AnnpackError::InvalidInput("invalid blob digest".into()));
    }
    let upload_url = reference.endpoint("blobs/uploads/");
    let response = client.request("POST", &upload_url, None, None, Some(&[]))?;
    if response.status() != 202 {
        return Err(AnnpackError::Protocol(format!(
            "registry blob upload start returned HTTP {}",
            response.status()
        )));
    }
    let location = response
        .header("Location")
        .ok_or_else(|| AnnpackError::Protocol("registry omitted upload Location".into()))?;
    let mut location = resolve_location(&client.base_url, location)?;
    location.query_pairs_mut().append_pair("digest", digest);
    let response = if same_origin(&client.base_url, location.as_str())? {
        client.request(
            "PUT",
            location.as_str(),
            None,
            Some("application/octet-stream"),
            Some(bytes),
        )?
    } else {
        if location.scheme() != "https" {
            return Err(AnnpackError::Http(
                "registry redirected blob upload to an insecure foreign origin".into(),
            ));
        }
        ureq::put(location.as_str())
            .set("Content-Type", "application/octet-stream")
            .send_bytes(bytes)
            .map_err(http_error)?
    };
    if response.status() != 201 {
        return Err(AnnpackError::Protocol(format!(
            "registry blob upload returned HTTP {}",
            response.status()
        )));
    }
    if let Some(actual) = response.header("Docker-Content-Digest")
        && actual != digest
    {
        return Err(AnnpackError::Integrity(format!(
            "registry reported blob digest {actual}, expected {digest}"
        )));
    }
    Ok(())
}

#[cfg(feature = "http")]
fn read_bounded_response(response: ureq::Response, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(AnnpackError::InvalidFormat(
            "registry response exceeds its allocation limit".into(),
        ));
    }
    Ok(bytes)
}

#[cfg(feature = "http")]
fn valid_sha256_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(feature = "http")]
fn resolve_location(base: &str, location: &str) -> Result<url::Url> {
    if let Ok(url) = url::Url::parse(location) {
        return Ok(url);
    }
    url::Url::parse(base)
        .and_then(|base| base.join(location))
        .map_err(|error| AnnpackError::Protocol(format!("invalid upload Location: {error}")))
}

#[cfg(feature = "http")]
fn same_origin(left: &str, right: &str) -> Result<bool> {
    let left = url::Url::parse(left)
        .map_err(|error| AnnpackError::Protocol(format!("invalid registry origin: {error}")))?;
    let right = url::Url::parse(right)
        .map_err(|error| AnnpackError::Protocol(format!("invalid upload origin: {error}")))?;
    Ok(left.origin() == right.origin())
}

#[cfg(feature = "http")]
#[derive(Debug, Clone)]
struct OciReference {
    scheme: String,
    registry: String,
    repository: String,
    reference: String,
    digest_reference: bool,
    base_url: String,
}

#[cfg(feature = "http")]
impl OciReference {
    fn parse(value: &str) -> Result<Self> {
        let (explicit_scheme, remainder) = if let Some(value) = value.strip_prefix("http://") {
            (Some("http"), value)
        } else if let Some(value) = value.strip_prefix("https://") {
            (Some("https"), value)
        } else if let Some(value) = value.strip_prefix("oci://") {
            (None, value)
        } else {
            (None, value)
        };
        let (registry, name) = remainder.split_once('/').ok_or_else(|| {
            AnnpackError::InvalidInput(
                "OCI reference must be REGISTRY/REPOSITORY[:TAG|@DIGEST]".into(),
            )
        })?;
        if registry.is_empty() || name.is_empty() || name.contains('?') || name.contains('#') {
            return Err(AnnpackError::InvalidInput("invalid OCI reference".into()));
        }
        let (repository, reference, digest_reference) =
            if let Some((repository, digest)) = name.rsplit_once('@') {
                (repository, digest, true)
            } else {
                let final_slash = name.rfind('/').map_or(0, |index| index + 1);
                if let Some(relative_colon) = name[final_slash..].rfind(':') {
                    let colon = final_slash + relative_colon;
                    (&name[..colon], &name[colon + 1..], false)
                } else {
                    (name, "latest", false)
                }
            };
        if repository.is_empty()
            || repository.starts_with('/')
            || repository.ends_with('/')
            || repository.contains("..")
            || !repository.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'/' | b'.' | b'_' | b'-')
            })
            || reference.is_empty()
            || (digest_reference && !valid_sha256_digest(reference))
            || (!digest_reference
                && !reference
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')))
        {
            return Err(AnnpackError::InvalidInput(
                "OCI repository or reference contains unsupported characters".into(),
            ));
        }
        let scheme = explicit_scheme.unwrap_or_else(|| {
            if registry.starts_with("localhost") || registry.starts_with("127.0.0.1") {
                "http"
            } else {
                "https"
            }
        });
        let base_url = format!("{scheme}://{registry}/");
        url::Url::parse(&base_url).map_err(|error| {
            AnnpackError::InvalidInput(format!("invalid registry URL: {error}"))
        })?;
        Ok(Self {
            scheme: scheme.into(),
            registry: registry.into(),
            repository: repository.into(),
            reference: reference.into(),
            digest_reference,
            base_url,
        })
    }

    fn endpoint(&self, suffix: &str) -> String {
        format!("{}v2/{}/{}", self.base_url, self.repository, suffix)
    }

    fn display(&self) -> String {
        let separator = if self.digest_reference { '@' } else { ':' };
        format!(
            "{}://{}/{}{}{}",
            self.scheme, self.registry, self.repository, separator, self.reference
        )
    }
}

#[cfg(feature = "http")]
fn reject_insecure_credentials(
    reference: &OciReference,
    credentials: Option<&RegistryCredentials>,
) -> Result<()> {
    let loopback = reference.registry.starts_with("localhost")
        || reference.registry.starts_with("127.0.0.1")
        || reference.registry.starts_with("[::1]");
    if credentials.is_some() && reference.scheme != "https" && !loopback {
        return Err(AnnpackError::InvalidInput(
            "refusing to send registry credentials over non-HTTPS transport".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "http")]
struct RegistryClient {
    base_url: String,
    credentials: Option<RegistryCredentials>,
    bearer_token: Option<String>,
}

#[cfg(feature = "http")]
impl RegistryClient {
    fn new(base_url: String, credentials: Option<RegistryCredentials>) -> Self {
        Self {
            base_url,
            credentials,
            bearer_token: None,
        }
    }

    fn request(
        &mut self,
        method: &str,
        url: &str,
        accept: Option<&str>,
        content_type: Option<&str>,
        body: Option<&[u8]>,
    ) -> Result<ureq::Response> {
        match self.request_once(method, url, accept, content_type, body) {
            Ok(response) => Ok(response),
            Err(error) if matches!(&*error, ureq::Error::Status(401, _)) => {
                let ureq::Error::Status(_, response) = *error else {
                    unreachable!("guard requires an HTTP status error")
                };
                let challenge = response.header("WWW-Authenticate").ok_or_else(|| {
                    AnnpackError::Http("registry returned 401 without an auth challenge".into())
                })?;
                self.bearer_token = Some(self.fetch_bearer_token(challenge)?);
                self.request_once(method, url, accept, content_type, body)
                    .map_err(|error| http_error(*error))
            }
            Err(error) => Err(http_error(*error)),
        }
    }

    fn request_once(
        &self,
        method: &str,
        url: &str,
        accept: Option<&str>,
        content_type: Option<&str>,
        body: Option<&[u8]>,
    ) -> std::result::Result<ureq::Response, Box<ureq::Error>> {
        let mut request = ureq::request(method, url);
        if let Some(accept) = accept {
            request = request.set("Accept", accept);
        }
        if let Some(content_type) = content_type {
            request = request.set("Content-Type", content_type);
        }
        if let Some(token) = &self.bearer_token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        } else if let Some(credentials) = &self.credentials {
            let encoded = base64::engine::general_purpose::STANDARD
                .encode(format!("{}:{}", credentials.username, credentials.password));
            request = request.set("Authorization", &format!("Basic {encoded}"));
        }
        if let Some(body) = body {
            request.send_bytes(body).map_err(Box::new)
        } else {
            request.call().map_err(Box::new)
        }
    }

    fn fetch_bearer_token(&self, challenge: &str) -> Result<String> {
        let parameters = parse_bearer_challenge(challenge)?;
        let realm = parameters
            .get("realm")
            .ok_or_else(|| AnnpackError::Http("registry bearer challenge omitted realm".into()))?;
        let mut url = url::Url::parse(realm)
            .map_err(|error| AnnpackError::Http(format!("invalid auth realm: {error}")))?;
        let realm_loopback = url
            .host_str()
            .is_some_and(|host| matches!(host, "localhost" | "127.0.0.1" | "::1"));
        if url.scheme() != "https"
            && !(realm_loopback && same_origin(&self.base_url, url.as_str())?)
        {
            return Err(AnnpackError::Http(
                "registry delegated authentication to an insecure realm".into(),
            ));
        }
        {
            let mut query = url.query_pairs_mut();
            for key in ["service", "scope"] {
                if let Some(value) = parameters.get(key) {
                    query.append_pair(key, value);
                }
            }
        }
        let mut request = ureq::get(url.as_str()).set("Accept", "application/json");
        if let Some(credentials) = &self.credentials {
            let encoded = base64::engine::general_purpose::STANDARD
                .encode(format!("{}:{}", credentials.username, credentials.password));
            request = request.set("Authorization", &format!("Basic {encoded}"));
        }
        let response = request.call().map_err(http_error)?;
        let response = read_bounded_response(response, 1024 * 1024)?;
        let value: serde_json::Value = serde_json::from_slice(&response)?;
        value
            .get("token")
            .or_else(|| value.get("access_token"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or_else(|| AnnpackError::Http("registry token response omitted token".into()))
    }
}

#[cfg(feature = "http")]
fn parse_bearer_challenge(value: &str) -> Result<std::collections::BTreeMap<String, String>> {
    let value = value.trim();
    let parameters = value
        .get(..7)
        .filter(|prefix| prefix.eq_ignore_ascii_case("Bearer "))
        .and_then(|_| value.get(7..))
        .ok_or_else(|| AnnpackError::Http("unsupported registry auth challenge".into()))?;
    let bytes = parameters.as_bytes();
    let mut cursor = 0_usize;
    let mut values = std::collections::BTreeMap::new();
    while cursor < bytes.len() {
        while cursor < bytes.len() && (bytes[cursor].is_ascii_whitespace() || bytes[cursor] == b',')
        {
            cursor += 1;
        }
        let key_start = cursor;
        while cursor < bytes.len()
            && (bytes[cursor].is_ascii_alphanumeric() || matches!(bytes[cursor], b'_' | b'-'))
        {
            cursor += 1;
        }
        if cursor == key_start || bytes.get(cursor) != Some(&b'=') {
            return Err(AnnpackError::Http(
                "malformed registry bearer challenge".into(),
            ));
        }
        let key = std::str::from_utf8(&bytes[key_start..cursor])
            .map_err(|_| AnnpackError::Http("non-UTF-8 bearer key".into()))?
            .to_ascii_lowercase();
        cursor += 1;
        if bytes.get(cursor) != Some(&b'"') {
            return Err(AnnpackError::Http("malformed registry bearer value".into()));
        }
        cursor += 1;
        let mut decoded = Vec::new();
        let mut terminated = false;
        while cursor < bytes.len() {
            match bytes[cursor] {
                b'"' => {
                    cursor += 1;
                    terminated = true;
                    break;
                }
                b'\\' => {
                    cursor += 1;
                    decoded.push(
                        *bytes
                            .get(cursor)
                            .ok_or_else(|| AnnpackError::Http("truncated bearer escape".into()))?,
                    );
                    cursor += 1;
                }
                byte => {
                    decoded.push(byte);
                    cursor += 1;
                }
            }
        }
        if !terminated {
            return Err(AnnpackError::Http(
                "unterminated registry bearer value".into(),
            ));
        }
        let decoded = String::from_utf8(decoded)
            .map_err(|_| AnnpackError::Http("non-UTF-8 bearer value".into()))?;
        values.insert(key, decoded);
    }
    Ok(values)
}

#[cfg(feature = "http")]
fn http_error(error: ureq::Error) -> AnnpackError {
    match error {
        ureq::Error::Status(status, response) => AnnpackError::Http(format!(
            "registry returned HTTP {status}: {}",
            response.status_text()
        )),
        ureq::Error::Transport(error) => AnnpackError::Http(error.to_string()),
    }
}

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::*;

    #[test]
    fn parses_registry_references_and_bearer_challenges() {
        let reference = OciReference::parse("ghcr.io/example/docs:1.2.3").unwrap();
        assert_eq!(reference.repository, "example/docs");
        assert_eq!(reference.reference, "1.2.3");
        assert_eq!(reference.scheme, "https");
        let challenge = parse_bearer_challenge(
            "Bearer realm=\"https://auth.example/token\",service=\"registry.example\",scope=\"repository:example/docs:pull\"",
        )
        .unwrap();
        assert_eq!(challenge["service"], "registry.example");
        assert_eq!(challenge["scope"], "repository:example/docs:pull");
        let challenge = parse_bearer_challenge(
            "Bearer realm=\"https://auth.example/token\",scope=\"repository:example/docs:pull,push\"",
        )
        .unwrap();
        assert_eq!(challenge["scope"], "repository:example/docs:pull,push");
    }

    #[test]
    fn rejects_ambiguous_or_unsafe_registry_references() {
        assert!(OciReference::parse("missing-repository").is_err());
        assert!(OciReference::parse("registry.example/Uppercase:tag").is_err());
        assert!(OciReference::parse("registry.example/a/../b:tag").is_err());
        assert!(OciReference::parse("registry.example/a/b@not-a-digest").is_err());
    }
}
