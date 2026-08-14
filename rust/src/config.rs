//! Optional project configuration for `annpack build`.
//!
//! `--name` and `--version` are mandatory and stable per project, so every
//! build retypes them. This file supplies them once. It is a CLI convenience
//! only: nothing here reaches the wire format, and a value read from
//! configuration produces byte-identical output to the same value passed as an
//! argument.
//!
//! What it deliberately does not do is invent identity. Every field is a value
//! a person wrote down. `version` in particular is named by channel-state
//! statements alongside an artifact root, so a synthesized default would let a
//! release advertise a version nobody chose. There is no `source_revision`
//! field either: it changes per commit, so keeping it in a checked-in file
//! would make it wrong by default.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{AnnpackError, Result};

/// The file name searched for in the working directory.
pub const CONFIG_FILE: &str = "annpack.toml";

/// Refuses a file large enough to suggest something other than configuration,
/// consistent with the bounded-read discipline the rest of the CLI applies to
/// untrusted input.
const MAX_CONFIG_BYTES: u64 = 64 * 1024;

/// Project defaults for `annpack build`. Every field is optional: the file may
/// supply as much or as little as a project finds worth writing down.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct BuildConfig {
    pub name: Option<String>,
    pub version: Option<String>,
    /// Default input directory, used when `annpack build` is given no path.
    pub source: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub description: Option<String>,
    pub base_url: Option<String>,
    pub license: Option<String>,
    pub redistributable: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(default)]
    build: BuildConfig,
}

impl BuildConfig {
    /// Loads `annpack.toml` from `directory`, returning defaults when absent.
    ///
    /// A missing file is the ordinary case and not an error. A malformed one
    /// is: silently ignoring it would build an artifact whose identity differs
    /// from what the project wrote down, which is worse than refusing.
    pub fn load_from(directory: &Path) -> Result<Self> {
        let path = directory.join(CONFIG_FILE);
        if !path.is_file() {
            return Ok(Self::default());
        }
        let size = std::fs::metadata(&path)?.len();
        if size > MAX_CONFIG_BYTES {
            return Err(AnnpackError::InvalidInput(format!(
                "{} is {size} bytes, above the {MAX_CONFIG_BYTES}-byte limit",
                path.display()
            )));
        }
        let text = std::fs::read_to_string(&path)?;
        let parsed: ConfigFile = toml::from_str(&text)
            .map_err(|error| AnnpackError::InvalidInput(format!("{}: {error}", path.display())))?;
        Ok(parsed.build)
    }

    /// Loads from the current working directory.
    pub fn load() -> Result<Self> {
        Self::load_from(&std::env::current_dir()?)
    }
}

/// Reports a required build field that neither the command line nor the
/// configuration supplied, naming both ways to provide it.
pub fn missing_field(field: &str, flag: &str, example: &str) -> AnnpackError {
    AnnpackError::InvalidInput(format!(
        "{field} is required: pass {flag} or set `{field} = \"{example}\"` \
         under [build] in {CONFIG_FILE}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, body: &str) {
        std::fs::write(dir.join(CONFIG_FILE), body).unwrap();
    }

    #[test]
    fn a_missing_file_is_not_an_error() {
        let temp = tempfile::tempdir().unwrap();
        let config = BuildConfig::load_from(temp.path()).unwrap();
        assert!(config.name.is_none() && config.version.is_none());
    }

    #[test]
    fn fields_are_read_from_the_build_table() {
        let temp = tempfile::tempdir().unwrap();
        write(
            temp.path(),
            "[build]\nname = \"refund-policy\"\nversion = \"1.0.0\"\nsource = \"docs\"\n",
        );
        let config = BuildConfig::load_from(temp.path()).unwrap();
        assert_eq!(config.name.as_deref(), Some("refund-policy"));
        assert_eq!(config.version.as_deref(), Some("1.0.0"));
        assert_eq!(config.source, Some(PathBuf::from("docs")));
    }

    #[test]
    fn kebab_case_keys_are_accepted() {
        let temp = tempfile::tempdir().unwrap();
        write(
            temp.path(),
            "[build]\nbase-url = \"https://vendor.example/docs\"\n",
        );
        let config = BuildConfig::load_from(temp.path()).unwrap();
        assert_eq!(
            config.base_url.as_deref(),
            Some("https://vendor.example/docs")
        );
    }

    /// A typo that silently did nothing would produce an artifact whose
    /// identity differs from what the file says, which is the failure this
    /// configuration exists to prevent.
    #[test]
    fn an_unknown_key_is_refused_rather_than_ignored() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "[build]\nnmae = \"typo\"\n");
        let error = BuildConfig::load_from(temp.path()).unwrap_err();
        assert!(matches!(error, AnnpackError::InvalidInput(_)));
    }

    #[test]
    fn malformed_toml_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "[build\nname =");
        assert!(BuildConfig::load_from(temp.path()).is_err());
    }

    /// There is deliberately no source-revision field; it would be stale on
    /// every commit after the one that wrote it.
    #[test]
    fn source_revision_is_not_a_configurable_field() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "[build]\nsource-revision = \"git:abc123\"\n");
        assert!(BuildConfig::load_from(temp.path()).is_err());
    }
}
