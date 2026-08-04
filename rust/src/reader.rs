use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::error::{AnnpackError, Result};

pub trait ReadAt: Send + Sync {
    fn len(&self) -> Result<u64>;
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<()>;
    fn identity(&self) -> Option<&str> {
        None
    }
}

pub type SharedReader = Arc<dyn ReadAt>;

pub struct FileReader {
    file: Mutex<File>,
    length: u64,
    identity: String,
}

impl FileReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let length = file.metadata()?.len();
        Ok(Self {
            file: Mutex::new(file),
            length,
            identity: path.display().to_string(),
        })
    }
}

impl ReadAt for FileReader {
    fn len(&self) -> Result<u64> {
        Ok(self.length)
    }

    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        checked_range(offset, buffer.len() as u64, self.length)?;
        let mut file = self
            .file
            .lock()
            .map_err(|_| AnnpackError::Io(std::io::Error::other("file lock poisoned")))?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buffer)?;
        Ok(())
    }

    fn identity(&self) -> Option<&str> {
        Some(&self.identity)
    }
}

#[derive(Clone)]
pub struct MemoryReader {
    bytes: Arc<[u8]>,
}

impl MemoryReader {
    pub fn new(bytes: impl Into<Arc<[u8]>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }
}

impl ReadAt for MemoryReader {
    fn len(&self) -> Result<u64> {
        Ok(self.bytes.len() as u64)
    }

    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        let range = checked_range(offset, buffer.len() as u64, self.bytes.len() as u64)?;
        buffer.copy_from_slice(&self.bytes[range]);
        Ok(())
    }
}

pub fn checked_range(
    offset: u64,
    length: u64,
    source_length: u64,
) -> Result<std::ops::Range<usize>> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| AnnpackError::InvalidFormat("offset arithmetic overflow".into()))?;
    if end > source_length {
        return Err(AnnpackError::InvalidFormat(format!(
            "range {offset}..{end} exceeds source length {source_length}"
        )));
    }
    let start = usize::try_from(offset)
        .map_err(|_| AnnpackError::InvalidFormat("offset exceeds platform address space".into()))?;
    let end = usize::try_from(end).map_err(|_| {
        AnnpackError::InvalidFormat("range end exceeds platform address space".into())
    })?;
    Ok(start..end)
}

#[cfg(feature = "http")]
pub struct HttpRangeReader {
    url: String,
    length: u64,
    etag: Option<String>,
}

#[cfg(feature = "http")]
impl HttpRangeReader {
    pub fn open(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        let response = ureq::head(&url)
            .call()
            .map_err(|e| AnnpackError::Http(e.to_string()))?;
        let length = response
            .header("Content-Length")
            .ok_or_else(|| AnnpackError::Http("server did not provide Content-Length".into()))?
            .parse::<u64>()
            .map_err(|_| AnnpackError::Http("invalid Content-Length".into()))?;
        let etag = response.header("ETag").map(ToOwned::to_owned);
        Ok(Self { url, length, etag })
    }
}

#[cfg(feature = "http")]
impl ReadAt for HttpRangeReader {
    fn len(&self) -> Result<u64> {
        Ok(self.length)
    }

    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        checked_range(offset, buffer.len() as u64, self.length)?;
        if buffer.is_empty() {
            return Ok(());
        }
        let end = offset + buffer.len() as u64 - 1;
        let mut request = ureq::get(&self.url).set("Range", &format!("bytes={offset}-{end}"));
        if let Some(etag) = &self.etag {
            request = request.set("If-Match", etag);
        }
        let response = request
            .call()
            .map_err(|e| AnnpackError::Http(e.to_string()))?;
        if response.status() != 206 {
            return Err(AnnpackError::Http(format!(
                "range server returned HTTP {}, expected 206",
                response.status()
            )));
        }
        let expected_content_range = format!("bytes {offset}-{end}/{}", self.length);
        if response.header("Content-Range") != Some(expected_content_range.as_str()) {
            return Err(AnnpackError::Http(format!(
                "incorrect Content-Range, expected {expected_content_range:?}"
            )));
        }
        if let (Some(expected), Some(actual)) = (&self.etag, response.header("ETag"))
            && actual != expected
        {
            return Err(AnnpackError::Http(
                "ETag changed during read session".into(),
            ));
        }
        let mut received = Vec::with_capacity(buffer.len());
        response
            .into_reader()
            .take(buffer.len() as u64 + 1)
            .read_to_end(&mut received)?;
        if received.len() != buffer.len() {
            return Err(AnnpackError::Http(format!(
                "range response contained {} bytes, expected {}",
                received.len(),
                buffer.len()
            )));
        }
        buffer.copy_from_slice(&received);
        Ok(())
    }

    fn identity(&self) -> Option<&str> {
        Some(&self.url)
    }
}
