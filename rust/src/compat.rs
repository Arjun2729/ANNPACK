//! Compatibility shims for the ANNPack -> Adyar rename.
//!
//! Configuration written against the old name keeps working. This module is
//! deliberately the only place that knows the legacy spelling, so closing the
//! transition window is a single deletion rather than an audit.
//!
//! Wire identifiers are *not* handled here and are not deprecated: the
//! `ANNPACK3`/`ANNDELT1` magic, the `annpack-core-v1.0-draft` profile id, and
//! the `https://annpack.dev/attestations/build/v1` predicate type are frozen
//! format constants that outlive the project's name.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Prefix for Adyar configuration variables.
const CANONICAL_PREFIX: &str = "ADYAR_";
/// Prefix these variables carried when the project was called ANNPack.
const LEGACY_PREFIX: &str = "ANNPACK_";

fn warned() -> &'static Mutex<HashSet<String>> {
    static WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    WARNED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// A variable set to the empty string is treated as unset for precedence.
/// Otherwise `ADYAR_X=""` would silently mask a populated `ANNPACK_X`, which is
/// the opposite of what an operator mid-migration means by it.
fn nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// Warn once per variable that a legacy name was used.
///
/// The warning names the variable and never its value. These carry signing keys
/// and registry passwords, and a deprecation notice is not worth leaking a
/// secret into CI logs.
fn warn_once(legacy: &str, canonical: &str) {
    let mut seen = match warned().lock() {
        Ok(seen) => seen,
        // A poisoned lock means another thread panicked mid-warning. The
        // deprecation notice is not worth propagating that panic.
        Err(poisoned) => poisoned.into_inner(),
    };
    if seen.insert(legacy.to_string()) {
        eprintln!("warning: {legacy} is deprecated; use {canonical}");
    }
}

/// Read a configuration variable by its suffix, e.g. `REGISTRY_PASSWORD`.
///
/// `ADYAR_*` wins when both are set. A fallback to `ANNPACK_*` warns once.
pub fn env_var(suffix: &str) -> Option<String> {
    let canonical = format!("{CANONICAL_PREFIX}{suffix}");
    if let Some(value) = nonempty(&canonical) {
        return Some(value);
    }
    let legacy = format!("{LEGACY_PREFIX}{suffix}");
    let value = nonempty(&legacy)?;
    warn_once(&legacy, &canonical);
    Some(value)
}

/// The canonical name for a variable, for use in diagnostics.
pub fn canonical_name(suffix: &str) -> String {
    format!("{CANONICAL_PREFIX}{suffix}")
}

/// Read a variable the caller named explicitly.
///
/// No fallback and no deprecation warning: an operator who names a variable
/// means that variable, exactly as an explicitly supplied output path is never
/// rewritten.
pub fn env_var_exact(name: &str) -> Option<String> {
    nonempty(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests set process-global environment variables, so they are
    // serialized behind one mutex rather than run as separate `#[test]` cases.
    // Cargo runs tests in threads within a single process, and a concurrent
    // `remove_var` in another case would make precedence assertions flaky.
    static ENV_GUARD: Mutex<()> = Mutex::new(());

    #[test]
    fn precedence_and_fallback() {
        let _guard = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());

        // SAFETY: guarded by ENV_GUARD; no other test in this module mutates
        // the environment concurrently.
        unsafe {
            std::env::remove_var("ADYAR_COMPAT_TEST");
            std::env::remove_var("ANNPACK_COMPAT_TEST");
        }
        assert_eq!(env_var("COMPAT_TEST"), None);

        // Legacy alone is honoured.
        unsafe { std::env::set_var("ANNPACK_COMPAT_TEST", "legacy") };
        assert_eq!(env_var("COMPAT_TEST").as_deref(), Some("legacy"));

        // Canonical wins when both are set.
        unsafe { std::env::set_var("ADYAR_COMPAT_TEST", "canonical") };
        assert_eq!(env_var("COMPAT_TEST").as_deref(), Some("canonical"));

        // An empty canonical does not mask a populated legacy value.
        unsafe { std::env::set_var("ADYAR_COMPAT_TEST", "") };
        assert_eq!(env_var("COMPAT_TEST").as_deref(), Some("legacy"));

        unsafe {
            std::env::remove_var("ADYAR_COMPAT_TEST");
            std::env::remove_var("ANNPACK_COMPAT_TEST");
        }
    }

    #[test]
    fn explicit_names_are_not_rewritten() {
        let _guard = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());

        // SAFETY: guarded by ENV_GUARD.
        unsafe { std::env::set_var("ANNPACK_EXPLICIT_TEST", "value") };
        // Reading the legacy name explicitly returns it without consulting any
        // canonical counterpart.
        assert_eq!(
            env_var_exact("ANNPACK_EXPLICIT_TEST").as_deref(),
            Some("value")
        );
        assert_eq!(env_var_exact("ADYAR_EXPLICIT_TEST"), None);
        unsafe { std::env::remove_var("ANNPACK_EXPLICIT_TEST") };
    }

    #[test]
    fn warns_only_once_per_variable() {
        let _guard = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());

        // SAFETY: guarded by ENV_GUARD.
        unsafe { std::env::set_var("ANNPACK_WARN_ONCE_TEST", "value") };
        assert!(env_var("WARN_ONCE_TEST").is_some());
        assert!(env_var("WARN_ONCE_TEST").is_some());
        let seen = warned().lock().unwrap_or_else(|e| e.into_inner());
        assert!(seen.contains("ANNPACK_WARN_ONCE_TEST"));
        drop(seen);
        unsafe { std::env::remove_var("ANNPACK_WARN_ONCE_TEST") };
    }
}
