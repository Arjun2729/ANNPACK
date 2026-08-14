use std::io;

#[derive(Debug, thiserror::Error)]
pub enum AdyarError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid ANNPack: {0}")]
    InvalidFormat(String),

    #[error("unsupported ANNPack feature: {0}")]
    Unsupported(String),

    #[error("integrity verification failed: {0}")]
    Integrity(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("search error: {0}")]
    Search(String),

    #[error("signature error: {0}")]
    Signature(String),

    #[error("protocol error: {0}")]
    Protocol(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[cfg(feature = "http")]
    #[error("HTTP error: {0}")]
    Http(String),
}

pub type Result<T> = std::result::Result<T, AdyarError>;
