//! Optional project configuration for `adyar build`.
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

use crate::error::{AdyarError, Result};

// The file name, and its pre-rename spelling, live in `compat` so that closing
// the transition window is one deletion rather than an audit.
pub use crate::compat::CONFIG_FILE;

/// Refuses a file large enough to suggest something other than configuration,
/// consistent with the bounded-read discipline the rest of the CLI applies to
/// untrusted input.
const MAX_CONFIG_BYTES: u64 = 64 * 1024;

/// Project defaults for `adyar build`. Every field is optional: the file may
/// supply as much or as little as a project finds worth writing down.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct BuildConfig {
    pub name: Option<String>,
    pub version: Option<String>,
    /// Default input directory, used when `adyar build` is given no path.
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
    /// Loads project configuration from `directory`, returning defaults when
    /// there is none.
    ///
    /// A missing file is the ordinary case and not an error. A malformed one
    /// is: silently ignoring it would build an artifact whose identity differs
    /// from what the project wrote down, which is worse than refusing.
    ///
    /// Which file that is — and what happens when a pre-rename `annpack.toml`
    /// is also present — is decided by [`crate::compat::config_path`].
    pub fn load_from(directory: &Path) -> Result<Self> {
        let Some(path) = crate::compat::config_path(directory) else {
            return Ok(Self::default());
        };
        let size = std::fs::metadata(&path)?.len();
        if size > MAX_CONFIG_BYTES {
            return Err(AdyarError::InvalidInput(format!(
                "{} is {size} bytes, above the {MAX_CONFIG_BYTES}-byte limit",
                path.display()
            )));
        }
        let text = std::fs::read_to_string(&path)?;
        let parsed: ConfigFile = toml::from_str(&text)
            .map_err(|error| AdyarError::InvalidInput(format!("{}: {error}", path.display())))?;
        Ok(parsed.build)
    }

    /// Loads from the current working directory.
    pub fn load() -> Result<Self> {
        Self::load_from(&std::env::current_dir()?)
    }
}

/// Reports a required build field that neither the command line nor the
/// configuration supplied, naming both ways to provide it.
pub fn missing_field(field: &str, flag: &str, example: &str) -> AdyarError {
    AdyarError::InvalidInput(format!(
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

    fn write_legacy(dir: &Path, body: &str) {
        std::fs::write(dir.join(crate::compat::LEGACY_CONFIG_FILE), body).unwrap();
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
        assert!(matches!(error, AdyarError::InvalidInput(_)));
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

    #[test]
    fn the_canonical_file_is_read() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "[build]\nname = \"canonical\"\n");
        let config = BuildConfig::load_from(temp.path()).unwrap();
        assert_eq!(config.name.as_deref(), Some("canonical"));
    }

    #[test]
    fn a_pre_rename_file_is_still_read() {
        let temp = tempfile::tempdir().unwrap();
        write_legacy(temp.path(), "[build]\nname = \"legacy\"\n");
        let config = BuildConfig::load_from(temp.path()).unwrap();
        assert_eq!(config.name.as_deref(), Some("legacy"));
    }

    /// Precedence is decided by which file exists, never by which was written
    /// most recently, so the same tree configures the same build on every
    /// machine.
    #[test]
    fn the_canonical_file_wins_when_both_exist() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "[build]\nname = \"canonical\"\n");
        write_legacy(temp.path(), "[build]\nname = \"legacy\"\n");
        let config = BuildConfig::load_from(temp.path()).unwrap();
        assert_eq!(config.name.as_deref(), Some("canonical"));
    }

    /// The property that makes the legacy file a genuine fallback rather than a
    /// second mandatory config source: once a valid `adyar.toml` exists, an
    /// abandoned `annpack.toml` is never parsed and so cannot break the build.
    #[test]
    fn a_malformed_legacy_file_is_inert_beside_a_valid_canonical_one() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "[build]\nname = \"canonical\"\n");
        write_legacy(temp.path(), "[build\nthis is not toml at all =");
        let config = BuildConfig::load_from(temp.path()).unwrap();
        assert_eq!(config.name.as_deref(), Some("canonical"));
    }

    #[test]
    fn a_malformed_legacy_file_alone_is_still_refused() {
        let temp = tempfile::tempdir().unwrap();
        write_legacy(temp.path(), "[build\nname =");
        assert!(BuildConfig::load_from(temp.path()).is_err());
    }

    /// A typo in the current file must not silently build from superseded
    /// configuration.
    #[test]
    fn a_malformed_canonical_file_does_not_fall_back_to_the_legacy_one() {
        let temp = tempfile::tempdir().unwrap();
        write(temp.path(), "[build\nname =");
        write_legacy(temp.path(), "[build]\nname = \"legacy\"\n");
        assert!(BuildConfig::load_from(temp.path()).is_err());
    }
}
