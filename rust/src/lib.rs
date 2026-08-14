//! ANNPack v3 reference implementation.
//!
//! The crate deliberately separates the immutable container from the indexes it
//! carries. A consumer can verify and inspect a pack even when it does not
//! understand every optional retrieval section.

pub mod attestation;
pub mod build;
pub mod bundle;
pub mod compat;
pub mod config;
pub mod conformance;
pub mod delta;
pub mod derive;
pub mod discovery;
pub mod error;
pub mod evidence;
pub mod fleet;
pub mod format;
pub mod ingest;
pub mod mcp;
pub mod model;
pub mod monitor;
pub mod oci;
pub mod policy;
pub mod provenance;
pub mod reader;
pub mod release;
pub mod run_attestation;
pub mod search;
pub mod signing;
pub mod telemetry;
pub mod transparency;
pub mod trust;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;

pub use error::{AdyarError, Result};
pub use format::{PackReader, PackWriter};
pub use search::{SearchEngine, SearchMode, SearchOptions, SearchResponse};
