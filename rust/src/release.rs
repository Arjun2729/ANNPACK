//! Channel state: which artifact a publisher currently stands behind.
//!
//! An artifact cannot say whether it is current. Anything it claims about its
//! own supersession is served by whoever serves the artifact, so an attacker
//! replaying an old release simply replays the old, un-superseded bytes. Currency
//! therefore lives in a separate signed statement, scoped to a publisher, corpus
//! and channel, and carrying a sequence number.
//!
//! # What a statement can and cannot establish
//!
//! A verified statement establishes that an authorised key said *this artifact
//! is the current one for this channel*, at a stated time, at a stated sequence.
//! It does not establish that no newer statement exists — nothing offline can,
//! since withholding is invisible. Expiry bounds how long the claim may be
//! believed and monotonic client state stops replay after first contact; neither
//! turns absence of newer information into evidence of currency.
//!
//! [`Currency::Unknown`] is therefore a first-class outcome and must never be
//! reported as [`Currency::Current`]. A statement that does not mention an
//! artifact says nothing about it.
//!
//! # Role separation is enforced, not merely declared
//!
//! The architecture contract permits a statement to be signed by the
//! `release_state` role or the `emergency_revocation` role. Honouring both
//! equally would defeat the separation: a compromised revocation key could then
//! declare any artifact current, which is exactly the authority the split exists
//! to withhold. So the roles are honoured asymmetrically —
//! `emergency_revocation` can take an artifact out of service, and only
//! `release_state` can put one into service. See [`SigningAuthority`].

use serde::{Deserialize, Serialize};

use crate::error::{AdyarError, Result};
use crate::trust::{
    ROLE_EMERGENCY_REVOCATION, ROLE_RELEASE_STATE, TrustRoot, TrustRootVerification,
    authorized_signers, parse_utc_timestamp,
};

// FROZEN WIRE IDENTIFIER: serialized and matched by third parties. It names a
// format version, not a project, and changes only when that version does.
pub const CHANNEL_STATE_SCHEMA_V1: &str = "annpack-channel-state-v1";

const CHANNEL_STATE_CONTEXT: &[u8] = b"ANNPACK3-CHANNEL-STATE\0";

const MAX_LISTED_ROOTS: usize = 4_096;
const MAX_SIGNATURES: usize = 128;

/// Maximum channel-state file size the reference CLI reads.
pub const MAX_CHANNEL_STATE_FILE_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CurrentRelease {
    pub version: String,
    pub artifact_root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Supersession {
    pub artifact_root: String,
    pub by: String,
    pub at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Revocation {
    pub artifact_root: String,
    pub at: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatementSignature {
    pub key_id: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChannelState {
    pub schema: String,
    pub publisher: String,
    pub corpus: String,
    pub channel: String,
    pub sequence: u64,
    pub issued_at: String,
    pub valid_until: String,
    pub current: CurrentRelease,
    #[serde(default)]
    pub superseded: Vec<Supersession>,
    #[serde(default)]
    pub revoked: Vec<Revocation>,
    #[serde(default)]
    pub signatures: Vec<StatementSignature>,
}

/// The bytes a channel-state signature covers: everything except signatures.
#[derive(Serialize)]
struct SignedPayload<'a> {
    schema: &'a str,
    publisher: &'a str,
    corpus: &'a str,
    channel: &'a str,
    sequence: u64,
    issued_at: &'a str,
    valid_until: &'a str,
    current: &'a CurrentRelease,
    superseded: &'a [Supersession],
    revoked: &'a [Revocation],
}

/// What the statement's signers are entitled to assert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SigningAuthority {
    /// No authorised role met its threshold. The statement asserts nothing.
    None,
    /// Signed only by `emergency_revocation`. Revocations are honoured; the
    /// `current` claim is not, because taking an artifact out of service and
    /// putting one into service are deliberately different powers.
    RevocationOnly,
    /// Signed by `release_state`. The whole statement is honoured.
    Full,
}

/// A root's status according to one verified statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Currency {
    Current,
    Superseded,
    /// The publisher withdrew this artifact. A security event, not policy.
    Revoked,
    /// No verified statement covers this root. Never equivalent to `Current`.
    Unknown,
}

/// Retained per-channel client state, keyed by publisher + corpus + channel.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetainedState {
    pub publisher: String,
    pub corpus: String,
    pub channel: String,
    pub highest_sequence: u64,
    pub statement_digest: String,
    pub artifact_root: String,
    pub accepted_at: String,
}

impl RetainedState {
    pub fn key(publisher: &str, corpus: &str, channel: &str) -> String {
        format!("{publisher}\0{corpus}\0{channel}")
    }

    pub fn matches(&self, publisher: &str, corpus: &str, channel: &str) -> bool {
        self.publisher == publisher && self.corpus == corpus && self.channel == channel
    }
}

/// Where a statement's sequence number places it relative to retained state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceVerdict {
    /// Scope did not match, so retained state was deliberately not consulted.
    ///
    /// Distinct from `FirstContact`, which asserts that state was looked for and
    /// none existed. Reporting a scope-mismatched statement as first contact
    /// would describe a comparison that never happened.
    NotEvaluated,
    /// Nothing retained. Rollback resistance is not available for this decision.
    FirstContact,
    Advanced,
    /// Same sequence, same bytes. Safe to re-accept.
    Idempotent,
    /// An older sequence after a newer one was accepted.
    Rollback,
    /// Same sequence, different bytes: the publisher signed two conflicting
    /// statements. A security event that no amount of valid signing excuses.
    Equivocation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStateVerification {
    pub publisher: String,
    pub corpus: String,
    pub channel: String,
    pub sequence: u64,
    pub statement_digest: String,
    pub schema_supported: bool,
    pub structurally_valid: bool,
    /// The trust root that authorised the signers had itself verified.
    pub trust_root_verified: bool,
    pub authority: SigningAuthority,
    /// Distinct authorised key ids that signed, by role.
    pub signers: Vec<String>,
    /// Statement scope equals the externally established scope the consumer
    /// asked about. Mandatory: there is no "not checked" state.
    pub scope_matches: bool,
    /// `None` when no trusted clock was supplied.
    pub within_validity: Option<bool>,
    pub sequence_verdict: SequenceVerdict,
    pub verified: bool,
    pub assumptions: Vec<String>,
    pub issues: Vec<String>,
}

fn signed_payload_bytes(statement: &ChannelState) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&SignedPayload {
        schema: &statement.schema,
        publisher: &statement.publisher,
        corpus: &statement.corpus,
        channel: &statement.channel,
        sequence: statement.sequence,
        issued_at: &statement.issued_at,
        valid_until: &statement.valid_until,
        current: &statement.current,
        superseded: &statement.superseded,
        revoked: &statement.revoked,
    })?)
}

/// The exact message a channel-state signature covers.
pub fn channel_state_signing_message(statement: &ChannelState) -> Result<Vec<u8>> {
    let mut message = CHANNEL_STATE_CONTEXT.to_vec();
    message.extend_from_slice(&signed_payload_bytes(statement)?);
    Ok(message)
}

/// Digest of the signed payload. Two statements agree iff their digests do.
pub fn statement_digest(statement: &ChannelState) -> Result<String> {
    Ok(blake3::hash(&signed_payload_bytes(statement)?)
        .to_hex()
        .to_string())
}

/// Raw bytes underlying [`statement_digest`]. What a transparency-log proof
/// (`crate::transparency`) binds to, so the same identity that already
/// distinguishes equivocation from idempotency (`SequenceVerdict`) is what
/// gets externally logged and witnessed -- not a second, independently
/// computed representation of the statement that could drift from the first.
pub fn statement_digest_bytes(statement: &ChannelState) -> Result<[u8; 32]> {
    Ok(*blake3::hash(&signed_payload_bytes(statement)?).as_bytes())
}

#[cfg(feature = "signing")]
pub fn sign_channel_state(statement: &mut ChannelState, secret_key: &[u8; 32]) -> Result<String> {
    use ed25519_dalek::{Signer, SigningKey};

    let signing_key = SigningKey::from_bytes(secret_key);
    let (key_id, _) = crate::trust::key_identity(secret_key);
    let message = channel_state_signing_message(statement)?;
    let signature = hex::encode(signing_key.sign(&message).to_bytes());
    statement.signatures.retain(|entry| entry.key_id != key_id);
    statement.signatures.push(StatementSignature {
        key_id: key_id.clone(),
        signature,
    });
    Ok(key_id)
}

fn is_hex_root(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn structural_issues(statement: &ChannelState) -> Vec<String> {
    let mut issues = Vec::new();
    if statement.publisher.trim().is_empty() {
        issues.push("statement names no publisher".into());
    }
    if statement.corpus.trim().is_empty() {
        issues.push("statement names no corpus".into());
    }
    if statement.channel.trim().is_empty() {
        issues.push("statement names no channel".into());
    }
    if statement.signatures.len() > MAX_SIGNATURES {
        issues.push(format!(
            "statement carries more than {MAX_SIGNATURES} signatures"
        ));
    }
    if statement.superseded.len() + statement.revoked.len() > MAX_LISTED_ROOTS {
        issues.push(format!(
            "statement lists more than {MAX_LISTED_ROOTS} roots"
        ));
    }
    if !is_hex_root(&statement.current.artifact_root) {
        issues.push("current artifact_root is not a 64-character hex digest".into());
    }
    for entry in &statement.superseded {
        if !is_hex_root(&entry.artifact_root) || !is_hex_root(&entry.by) {
            issues.push("a superseded entry carries a malformed root".into());
            break;
        }
    }
    for entry in &statement.revoked {
        if !is_hex_root(&entry.artifact_root) {
            issues.push("a revoked entry carries a malformed root".into());
            break;
        }
    }
    // A statement that both advertises and revokes the same artifact is
    // self-contradictory. Resolving it silently in either direction would hide a
    // publisher error at exactly the moment it matters most.
    if statement.revoked.iter().any(|entry| {
        entry
            .artifact_root
            .eq_ignore_ascii_case(&statement.current.artifact_root)
    }) {
        issues.push("statement advertises a revoked artifact as current".into());
    }
    issues
}

/// Where a statement sits relative to retained state for the **expected** scope.
///
/// `expected` is the scope the consumer asked about, never the one the statement
/// declares. Keying on the statement's own scope would let a statement select
/// which state it is compared against, which is the comparison it is supposed to
/// be subject to.
fn sequence_verdict(
    statement: &ChannelState,
    expected: (&str, &str, &str),
    digest: &str,
    retained: Option<&RetainedState>,
    issues: &mut Vec<String>,
    assumptions: &mut Vec<String>,
) -> SequenceVerdict {
    let Some(retained) = retained else {
        assumptions.push(
            "no retained state for this channel: accepted on first contact, so this decision \
             has no rollback resistance"
                .into(),
        );
        return SequenceVerdict::FirstContact;
    };
    if !retained.matches(expected.0, expected.1, expected.2) {
        // State for a different channel says nothing about this one, and using
        // it would compare unrelated sequence numbers.
        issues.push("retained state belongs to a different publisher, corpus, or channel".into());
        return SequenceVerdict::NotEvaluated;
    }
    match statement.sequence.cmp(&retained.highest_sequence) {
        std::cmp::Ordering::Less => {
            issues.push(format!(
                "statement sequence {} is below the accepted sequence {}",
                statement.sequence, retained.highest_sequence
            ));
            SequenceVerdict::Rollback
        }
        std::cmp::Ordering::Equal => {
            if digest == retained.statement_digest {
                SequenceVerdict::Idempotent
            } else {
                issues.push(format!(
                    "two different statements exist at sequence {}: publisher equivocation",
                    statement.sequence
                ));
                SequenceVerdict::Equivocation
            }
        }
        std::cmp::Ordering::Greater => SequenceVerdict::Advanced,
    }
}

/// Verify a channel-state statement against a verified trust root.
///
/// `expected` is the publisher, corpus and channel the consumer is asking about,
/// and it is mandatory. It must be established outside the statement — from a
/// trusted root and from configuration — because a statement that supplies its
/// own expectations is only ever compared against itself.
///
/// An earlier signature made this optional, and both reference-CLI callers
/// passed the statement's own fields. `scope_matches` was therefore always true
/// and a `staging` statement verified cleanly for a consumer asking about
/// `production`. The library test covering the check called the library
/// directly, so it passed while the shipped binary could not fail the check at
/// all. Requiring the argument removes the shape of that mistake.
#[allow(clippy::too_many_arguments)]
pub fn verify_channel_state(
    statement: &ChannelState,
    trust_root: &TrustRoot,
    trust: &TrustRootVerification,
    retained: Option<&RetainedState>,
    now: Option<&str>,
    expected: (&str, &str, &str),
) -> Result<ChannelStateVerification> {
    let mut issues = Vec::new();
    let mut assumptions = Vec::new();

    let schema_supported = statement.schema == CHANNEL_STATE_SCHEMA_V1;
    if !schema_supported {
        issues.push(format!(
            "channel-state schema {:?}; this verifier supports {CHANNEL_STATE_SCHEMA_V1}",
            statement.schema
        ));
    }

    let digest = statement_digest(statement)?;
    let structural = structural_issues(statement);
    let structurally_valid = structural.is_empty();
    issues.extend(structural);

    let trust_root_verified = trust.verified;
    if !trust_root_verified {
        issues.push("the trust root authorising this statement did not verify".into());
    }
    if trust.publisher != statement.publisher {
        issues.push(format!(
            "statement publisher {:?} is not the trust root's publisher {:?}",
            statement.publisher, trust.publisher
        ));
    }

    let evaluate_signatures = schema_supported && structurally_valid && trust_root_verified;
    let (authority, signers) = if evaluate_signatures {
        let message = channel_state_signing_message(statement)?;
        let pairs: Vec<(&str, &str)> = statement
            .signatures
            .iter()
            .map(|entry| (entry.key_id.as_str(), entry.signature.as_str()))
            .collect();

        let release = authorized_signers(trust_root, ROLE_RELEASE_STATE, &message, &pairs);
        let revocation =
            authorized_signers(trust_root, ROLE_EMERGENCY_REVOCATION, &message, &pairs);

        let release_threshold = trust_root
            .roles
            .get(ROLE_RELEASE_STATE)
            .map(|role| role.threshold as usize)
            .unwrap_or(usize::MAX);
        let revocation_threshold = trust_root
            .roles
            .get(ROLE_EMERGENCY_REVOCATION)
            .map(|role| role.threshold as usize)
            .unwrap_or(usize::MAX);

        let mut signers: Vec<String> = release.union(&revocation).cloned().collect();
        signers.sort();

        if release.len() >= release_threshold {
            (SigningAuthority::Full, signers)
        } else if revocation.len() >= revocation_threshold {
            assumptions.push(
                "signed only by the emergency-revocation role: revocations are honoured and \
                 the current-release claim is not"
                    .into(),
            );
            (SigningAuthority::RevocationOnly, signers)
        } else {
            issues.push(format!(
                "statement has {} release-state and {} revocation signatures, needing \
                 {release_threshold} or {revocation_threshold}",
                release.len(),
                revocation.len()
            ));
            (SigningAuthority::None, signers)
        }
    } else {
        (SigningAuthority::None, Vec::new())
    };

    let (publisher, corpus, channel) = expected;
    let scope_matches = statement.publisher == publisher
        && statement.corpus == corpus
        && statement.channel == channel;
    if !scope_matches {
        issues.push(format!(
            "statement scopes {}/{}/{} but the consumer asked about {publisher}/{corpus}/{channel}",
            statement.publisher, statement.corpus, statement.channel
        ));
    }

    let within_validity = match now {
        None => {
            assumptions.push(
                "no trusted clock supplied: expiry was not evaluated, so this statement's \
                 currency claim is unbounded in time"
                    .into(),
            );
            None
        }
        Some(now) => {
            let now = parse_utc_timestamp(now)?;
            let issued = parse_utc_timestamp(&statement.issued_at)?;
            let expires = parse_utc_timestamp(&statement.valid_until)?;
            if expires <= issued {
                issues.push("statement expires no later than it was issued".into());
            }
            let valid = now >= issued && now < expires;
            if !valid {
                issues.push(format!(
                    "statement validity {}..{} does not contain the supplied time",
                    statement.issued_at, statement.valid_until
                ));
            }
            Some(valid)
        }
    };

    // Retained state is not consulted at all when the scope does not match. A
    // statement for another channel must not be compared against this channel's
    // sequence, and must not cause this channel's state to be read or written.
    let verdict = if scope_matches {
        sequence_verdict(
            statement,
            expected,
            &digest,
            retained,
            &mut issues,
            &mut assumptions,
        )
    } else {
        SequenceVerdict::NotEvaluated
    };

    let sequence_acceptable = matches!(
        verdict,
        SequenceVerdict::FirstContact | SequenceVerdict::Advanced | SequenceVerdict::Idempotent
    );

    let verified = schema_supported
        && structurally_valid
        && trust_root_verified
        && authority != SigningAuthority::None
        && scope_matches
        && within_validity.unwrap_or(false)
        && sequence_acceptable;

    Ok(ChannelStateVerification {
        publisher: statement.publisher.clone(),
        corpus: statement.corpus.clone(),
        channel: statement.channel.clone(),
        sequence: statement.sequence,
        statement_digest: digest,
        schema_supported,
        structurally_valid,
        trust_root_verified,
        authority,
        signers,
        scope_matches,
        within_validity,
        sequence_verdict: verdict,
        verified,
        assumptions,
        issues,
    })
}

/// The status of one artifact root according to a verified statement.
///
/// Takes the verification rather than the statement alone so an unverified
/// statement cannot be consulted for a currency verdict by accident — the same
/// discipline [`crate::trust::role_authorizes`] applies to authorisation.
pub fn currency_for_root(
    statement: &ChannelState,
    verification: &ChannelStateVerification,
    artifact_root: &str,
) -> Currency {
    if !verification.verified {
        return Currency::Unknown;
    }
    // Revocation is checked first and honoured under either authority: a
    // security event outranks a policy one, and refusing to act on a revocation
    // because the same statement's current-release claim is unauthorised would
    // invert the risk.
    if statement
        .revoked
        .iter()
        .any(|entry| entry.artifact_root.eq_ignore_ascii_case(artifact_root))
    {
        return Currency::Revoked;
    }
    if statement
        .superseded
        .iter()
        .any(|entry| entry.artifact_root.eq_ignore_ascii_case(artifact_root))
    {
        return Currency::Superseded;
    }
    if verification.authority == SigningAuthority::Full
        && statement
            .current
            .artifact_root
            .eq_ignore_ascii_case(artifact_root)
    {
        return Currency::Current;
    }
    Currency::Unknown
}

/// The state to retain after accepting a statement.
///
/// Returns `None` unless the statement verified and its sequence advanced.
/// Persisting on an idempotent re-acceptance is harmless but pointless, and
/// persisting anything else would record a statement the verifier rejected.
pub fn state_to_retain(
    statement: &ChannelState,
    verification: &ChannelStateVerification,
    expected: (&str, &str, &str),
    accepted_at: &str,
) -> Option<RetainedState> {
    if !verification.verified || verification.sequence_verdict == SequenceVerdict::Rollback {
        return None;
    }
    if !matches!(
        verification.sequence_verdict,
        SequenceVerdict::Advanced | SequenceVerdict::FirstContact
    ) {
        return None;
    }
    Some(RetainedState {
        // Keyed on the expected scope, not the statement's. Verification already
        // required the two to be equal; keying on the external one means a
        // future change that weakens the scope check cannot also silently
        // repoint which channel's state a statement overwrites.
        publisher: expected.0.to_string(),
        corpus: expected.1.to_string(),
        channel: expected.2.to_string(),
        highest_sequence: statement.sequence,
        statement_digest: verification.statement_digest.clone(),
        artifact_root: statement.current.artifact_root.clone(),
        accepted_at: accepted_at.to_string(),
    })
}

/// Persist retained state so that a crash cannot leave it partially written.
///
/// Write to a sibling temporary file, flush it to disk, then rename over the
/// target. A rename within a directory is atomic, so a reader sees either the
/// old state or the new one. Writing in place would allow a truncated file,
/// which on the next start reads as "no retained state" — silently downgrading
/// a client to first contact, which is precisely the rollback exposure the state
/// exists to close.
pub fn persist_retained_state(path: &std::path::Path, state: &RetainedState) -> Result<()> {
    use std::io::Write;

    let parent = path.parent().ok_or_else(|| {
        AdyarError::InvalidInput("retained state path has no parent directory".into())
    })?;
    std::fs::create_dir_all(parent)?;
    let temporary = path.with_extension("tmp");
    let mut file = std::fs::File::create(&temporary)?;
    file.write_all(&serde_json::to_vec_pretty(state)?)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, path)?;
    Ok(())
}

pub fn load_retained_state(path: &std::path::Path) -> Result<Option<RetainedState>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn statement() -> ChannelState {
        ChannelState {
            schema: CHANNEL_STATE_SCHEMA_V1.into(),
            publisher: "example.com".into(),
            corpus: "support-manual".into(),
            channel: "production".into(),
            sequence: 4,
            issued_at: "2026-08-06T00:00:00Z".into(),
            valid_until: "2026-08-06T01:00:00Z".into(),
            current: CurrentRelease {
                version: "4.3.0".into(),
                artifact_root: "aa".repeat(32),
            },
            superseded: Vec::new(),
            revoked: Vec::new(),
            signatures: Vec::new(),
        }
    }

    #[test]
    fn signatures_are_excluded_from_the_signed_payload() {
        let mut input = statement();
        let before = statement_digest(&input).unwrap();
        input.signatures.push(StatementSignature {
            key_id: "k".into(),
            signature: "s".into(),
        });
        assert_eq!(before, statement_digest(&input).unwrap());
    }

    #[test]
    fn the_signing_message_is_domain_separated() {
        assert!(
            channel_state_signing_message(&statement())
                .unwrap()
                .starts_with(CHANNEL_STATE_CONTEXT)
        );
    }

    #[test]
    fn any_payload_change_changes_the_digest() {
        // Equivocation detection compares digests, so a field that does not
        // affect the digest is a field two conflicting statements could differ
        // in undetected.
        type Mutate = Box<dyn Fn(&mut ChannelState)>;

        let base = statement_digest(&statement()).unwrap();
        let mutate: Vec<Mutate> = vec![
            Box::new(|s| s.sequence += 1),
            Box::new(|s| s.publisher = "other.example".into()),
            Box::new(|s| s.corpus = "other".into()),
            Box::new(|s| s.channel = "staging".into()),
            Box::new(|s| s.issued_at = "2026-08-06T00:00:01Z".into()),
            Box::new(|s| s.valid_until = "2026-08-06T02:00:00Z".into()),
            Box::new(|s| s.current.version = "9.9.9".into()),
            Box::new(|s| s.current.artifact_root = "bb".repeat(32)),
            Box::new(|s| {
                s.revoked.push(Revocation {
                    artifact_root: "cc".repeat(32),
                    at: "2026-08-06T00:30:00Z".into(),
                    reason: "incorrect-content".into(),
                })
            }),
            Box::new(|s| {
                s.superseded.push(Supersession {
                    artifact_root: "dd".repeat(32),
                    by: "aa".repeat(32),
                    at: "2026-08-06T00:30:00Z".into(),
                })
            }),
        ];
        for (index, apply) in mutate.iter().enumerate() {
            let mut changed = statement();
            apply(&mut changed);
            assert_ne!(
                base,
                statement_digest(&changed).unwrap(),
                "mutation {index} did not change the digest"
            );
        }
    }

    #[test]
    fn a_statement_advertising_a_revoked_root_as_current_is_malformed() {
        let mut input = statement();
        input.revoked.push(Revocation {
            artifact_root: input.current.artifact_root.clone(),
            at: "2026-08-06T00:30:00Z".into(),
            reason: "incorrect-content".into(),
        });
        assert!(
            structural_issues(&input)
                .iter()
                .any(|issue| issue.contains("revoked artifact as current"))
        );
    }

    #[test]
    fn retained_state_from_another_channel_is_not_consulted() {
        let mut issues = Vec::new();
        let mut assumptions = Vec::new();
        let retained = RetainedState {
            publisher: "example.com".into(),
            corpus: "support-manual".into(),
            channel: "staging".into(),
            highest_sequence: 99,
            statement_digest: "x".into(),
            artifact_root: "bb".repeat(32),
            accepted_at: "2026-08-06T00:00:00Z".into(),
        };
        // Sequence 4 against a retained 99 would be a rollback if the state were
        // wrongly treated as applying to this channel.
        let verdict = sequence_verdict(
            &statement(),
            ("example.com", "support-manual", "production"),
            "digest",
            Some(&retained),
            &mut issues,
            &mut assumptions,
        );
        // `NotEvaluated`, not `FirstContact`: state existed, it simply belonged
        // to another channel. Reporting first contact would claim a comparison
        // was made against nothing, when in fact none was made at all.
        assert_eq!(verdict, SequenceVerdict::NotEvaluated);
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("different publisher"))
        );
    }

    #[test]
    fn equal_sequence_with_different_bytes_is_equivocation() {
        let mut issues = Vec::new();
        let mut assumptions = Vec::new();
        let retained = RetainedState {
            publisher: "example.com".into(),
            corpus: "support-manual".into(),
            channel: "production".into(),
            highest_sequence: 4,
            statement_digest: "a-different-digest".into(),
            artifact_root: "bb".repeat(32),
            accepted_at: "2026-08-06T00:00:00Z".into(),
        };
        assert_eq!(
            sequence_verdict(
                &statement(),
                ("example.com", "support-manual", "production"),
                "this-digest",
                Some(&retained),
                &mut issues,
                &mut assumptions
            ),
            SequenceVerdict::Equivocation
        );
        assert_eq!(
            sequence_verdict(
                &statement(),
                ("example.com", "support-manual", "production"),
                "a-different-digest",
                Some(&retained),
                &mut issues,
                &mut assumptions
            ),
            SequenceVerdict::Idempotent
        );
    }
}
