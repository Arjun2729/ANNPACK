#[cfg(feature = "signing")]
use std::fs::{self, OpenOptions};
#[cfg(feature = "signing")]
use std::io::Write;
#[cfg(all(feature = "signing", unix))]
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use crate::error::{AnnpackError, Result};
use crate::format::PackReader;
#[cfg(feature = "signing")]
use crate::format::{PackWriter, SectionData, SectionType};
#[cfg(feature = "signing")]
use crate::model::SignatureEnvelope;

#[derive(Debug, Clone, serde::Serialize)]
pub struct SignatureReport {
    pub section_id: u32,
    pub key_id: String,
    pub public_key: String,
    pub identity: Option<String>,
    pub cryptographically_valid: bool,
    pub identity_trusted: bool,
}

#[cfg(feature = "signing")]
pub fn generate_keypair(
    secret_path: &Path,
    public_path: Option<&Path>,
) -> Result<(String, String)> {
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    let signing_key = SigningKey::generate(&mut OsRng);
    let secret_hex = hex::encode(signing_key.to_bytes());
    let public_hex = hex::encode(signing_key.verifying_key().to_bytes());
    let public_path = public_path
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| secret_path.with_extension("pub"));
    if secret_path.exists() || public_path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "refusing to overwrite an existing key file",
        )
        .into());
    }
    let mut secret_options = OpenOptions::new();
    secret_options.write(true).create_new(true);
    #[cfg(unix)]
    secret_options.mode(0o600);
    let mut secret_file = secret_options.open(secret_path)?;
    secret_file.write_all(format!("{secret_hex}\n").as_bytes())?;
    secret_file.sync_all()?;
    let mut public_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(public_path)?;
    public_file.write_all(format!("{public_hex}\n").as_bytes())?;
    public_file.sync_all()?;
    Ok((secret_hex, public_hex))
}

#[cfg(not(feature = "signing"))]
pub fn generate_keypair(
    _secret_path: &Path,
    _public_path: Option<&Path>,
) -> Result<(String, String)> {
    Err(AnnpackError::Unsupported(
        "binary was built without signing support".into(),
    ))
}

#[cfg(feature = "signing")]
pub fn sign_pack(
    input: &Path,
    output: &Path,
    secret_key_path: &Path,
    identity: Option<String>,
    expires_at: Option<String>,
) -> Result<SignatureReport> {
    use ed25519_dalek::{Signer, SigningKey};

    let reader = PackReader::open_path(input)?;
    reader.verify_all()?;
    let key_bytes = read_hex_array::<32>(secret_key_path, "secret key")?;
    let signing_key = SigningKey::from_bytes(&key_bytes);
    let public_key = signing_key.verifying_key().to_bytes();
    let key_id = blake3::hash(&public_key).to_hex().to_string();
    let message = signature_message(&reader.header.root_hash);
    let signature = signing_key.sign(&message);
    let envelope = SignatureEnvelope {
        algorithm: "Ed25519".into(),
        public_key: hex::encode(public_key),
        signature: hex::encode(signature.to_bytes()),
        signed_root: reader.root_hex(),
        key_id: key_id.clone(),
        identity: identity.clone(),
        expires_at,
        transparency_log_url: None,
        revocation_url: None,
        build_attestation: None,
    };

    let mut sections = reader.all_section_data(true)?;
    let section_id = sections
        .iter()
        .map(|section| section.section_id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| {
            AnnpackError::InvalidFormat("no section ID available for signature".into())
        })?;
    sections.push(SectionData::optional(
        section_id,
        SectionType::Signature,
        1,
        serde_json::to_vec(&envelope)?,
    ));
    let mut writer = PackWriter::new().with_flags(reader.header.flags);
    for section in sections {
        writer.push(section)?;
    }
    let signed_root = writer.write_path(output)?;
    if signed_root != reader.header.root_hash {
        return Err(AnnpackError::Integrity(
            "adding a signature unexpectedly changed the content root".into(),
        ));
    }
    Ok(SignatureReport {
        section_id,
        key_id,
        public_key: hex::encode(public_key),
        identity,
        cryptographically_valid: true,
        identity_trusted: false,
    })
}

#[cfg(not(feature = "signing"))]
pub fn sign_pack(
    _input: &Path,
    _output: &Path,
    _secret_key_path: &Path,
    _identity: Option<String>,
    _expires_at: Option<String>,
) -> Result<SignatureReport> {
    Err(AnnpackError::Unsupported(
        "binary was built without signing support".into(),
    ))
}

#[cfg(feature = "signing")]
pub fn verify_signatures(
    reader: &PackReader,
    trusted_public_key: Option<&Path>,
) -> Result<Vec<SignatureReport>> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let trusted = trusted_public_key
        .map(|path| read_hex_array::<32>(path, "public key"))
        .transpose()?;
    let mut reports = Vec::new();
    let mut trusted_signature_found = false;
    for entry in reader.entries_of_type(SectionType::Signature) {
        let envelope: SignatureEnvelope =
            serde_json::from_slice(&reader.read_section(entry.section_id)?)?;
        if envelope.algorithm != "Ed25519" || envelope.signed_root != reader.root_hex() {
            return Err(AnnpackError::Signature(format!(
                "signature section {} has incompatible algorithm or root",
                entry.section_id
            )));
        }
        let public_key = decode_hex_array::<32>(&envelope.public_key, "embedded public key")?;
        let signature_bytes = decode_hex_array::<64>(&envelope.signature, "signature")?;
        let verifying_key = VerifyingKey::from_bytes(&public_key)
            .map_err(|error| AnnpackError::Signature(error.to_string()))?;
        let signature = Signature::from_bytes(&signature_bytes);
        verifying_key
            .verify(&signature_message(&reader.header.root_hash), &signature)
            .map_err(|error| AnnpackError::Signature(error.to_string()))?;
        let expected_key_id = blake3::hash(&public_key).to_hex().to_string();
        if expected_key_id != envelope.key_id {
            return Err(AnnpackError::Signature("signature key ID mismatch".into()));
        }
        let identity_trusted = trusted.as_ref().is_some_and(|value| value == &public_key);
        trusted_signature_found |= identity_trusted;
        reports.push(SignatureReport {
            section_id: entry.section_id,
            key_id: envelope.key_id,
            public_key: envelope.public_key,
            identity: envelope.identity,
            cryptographically_valid: true,
            identity_trusted,
        });
    }
    if trusted.is_some() && !trusted_signature_found {
        return Err(AnnpackError::Signature(
            "no valid signature uses the explicitly trusted public key".into(),
        ));
    }
    Ok(reports)
}

#[cfg(not(feature = "signing"))]
pub fn verify_signatures(
    _reader: &PackReader,
    _trusted_public_key: Option<&Path>,
) -> Result<Vec<SignatureReport>> {
    Ok(Vec::new())
}

#[cfg(feature = "signing")]
fn signature_message(root: &[u8; 32]) -> Vec<u8> {
    let mut message = b"ANNPACK3-SIGNATURE\0".to_vec();
    message.extend_from_slice(root);
    message
}

#[cfg(feature = "signing")]
fn read_hex_array<const N: usize>(path: &Path, label: &str) -> Result<[u8; N]> {
    let value = fs::read_to_string(path)?;
    decode_hex_array(value.trim(), label)
}

#[cfg(feature = "signing")]
fn decode_hex_array<const N: usize>(value: &str, label: &str) -> Result<[u8; N]> {
    let bytes = hex::decode(value)
        .map_err(|error| AnnpackError::Signature(format!("invalid {label}: {error}")))?;
    bytes.try_into().map_err(|bytes: Vec<u8>| {
        AnnpackError::Signature(format!(
            "invalid {label} length {}, expected {N} bytes",
            bytes.len()
        ))
    })
}
