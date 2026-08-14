//! Cross-observation consistency monitoring for release-state statements
//! (Step 9b).
//!
//! # What this answers that a single verification cannot
//!
//! [`crate::release::verify_channel_state`] answers whether one statement,
//! shown to one verifier, is authentic and currently selected. It cannot
//! answer whether a publisher showed a *different*, equally authentic
//! statement to someone else — that requires comparing multiple independent
//! observations over time, which is exactly what this module does. It is the
//! second half of the gap [ADR-0007](../../spec/decisions/0007-transparency-log-integration.md)
//! named: a Sigsum proof shows a statement was logged publicly; this module
//! is what would notice if two different statements were each logged, at the
//! same or overlapping sequence, for the same scope.
//!
//! # What this does NOT do
//!
//! It does not fetch statements from anywhere. Callers append what they have
//! already independently obtained and verified enough to parse
//! ([`append_observation`]); this module only compares what it is given.
//! Nothing here proves the observation history is complete — a monitor that
//! has only ever seen one side of an equivocating publisher's story will
//! report no incident, honestly, because none is visible to it.
//! [`IncidentKind::SequenceGap`] exists precisely to flag when the history looks
//! incomplete, not to paper over it.

use serde::{Deserialize, Serialize};

use crate::error::{AdyarError, Result};
use crate::release::{
    ChannelState, RetainedState, SigningAuthority, statement_digest, verify_channel_state,
};
use crate::trust::{TrustRoot, TrustRootVerification};

pub const MAX_OBSERVATIONS_FILE_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_OBSERVATIONS: usize = 100_000;

/// One observed channel-state statement, with when it was observed. Stored
/// one per line as JSON Lines: an ever-growing history is meant to be
/// appended to, not rewritten, and a line-oriented format survives a
/// truncated last write without corrupting everything before it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub observed_at: String,
    pub statement: ChannelState,
}

/// Append one observation to `existing` file contents, returning the new
/// contents. Never deduplicates: two identical observations are two data
/// points that the history really did see the same statement twice, and
/// collapsing them would be this module deciding what counts as
/// corroboration instead of reporting what was seen.
pub fn append_observation(
    existing: &str,
    statement: &ChannelState,
    observed_at: &str,
) -> Result<String> {
    let line = serde_json::to_string(&Observation {
        observed_at: observed_at.to_string(),
        statement: statement.clone(),
    })?;
    let mut updated = existing.to_string();
    if !updated.is_empty() && !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str(&line);
    updated.push('\n');
    Ok(updated)
}

/// Parse a JSON Lines observation history. Blank lines are skipped; anything
/// else that fails to parse is a malformed file, not a partial one.
pub fn parse_observations(text: &str) -> Result<Vec<Observation>> {
    if text.len() as u64 > MAX_OBSERVATIONS_FILE_BYTES {
        return Err(AdyarError::InvalidFormat(
            "observation history exceeds size limit".into(),
        ));
    }
    let mut observations = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let observation: Observation = serde_json::from_str(line).map_err(|error| {
            AdyarError::InvalidFormat(format!(
                "observation history line {}: {error}",
                line_number + 1
            ))
        })?;
        observations.push(observation);
        if observations.len() > MAX_OBSERVATIONS {
            return Err(AdyarError::InvalidFormat(
                "observation history exceeds observation count limit".into(),
            ));
        }
    }
    Ok(observations)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncidentKind {
    /// Same publisher, corpus, channel and sequence; different signed
    /// content. A publisher signed two conflicting statements.
    Equivocation,
    /// More than one artifact root is, per the observed history, never
    /// explicitly superseded by anything at a higher sequence -- two
    /// statements each claim to still be current with no chain between them.
    Conflict,
    /// A statement's signatures did not meet any authorised role's
    /// threshold, so it asserts nothing, yet it was observed as if it were a
    /// real statement.
    AuthorityViolation,
    /// A gap between consecutively observed sequence numbers. Not
    /// necessarily an attack -- most often it means this monitor's view of
    /// history is incomplete, not that anything is wrong with the publisher.
    SequenceGap,
    /// The observation history contains an authorised, higher-sequence
    /// statement than the supplied retained state reflects.
    StaleLocalState,
    /// A statement revoked a root; a later-or-equal-sequence statement still
    /// advertises that root as current.
    RevokedRootAdvertised,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Incident {
    pub kind: IncidentKind,
    pub description: String,
    /// Statement digests this incident's evidence rests on -- enough for an
    /// operator to go find the exact observations and inspect them, without
    /// this report having to re-embed full statement bodies.
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelReport {
    pub publisher: String,
    pub corpus: String,
    pub channel: String,
    pub observation_count: usize,
    pub distinct_sequences: usize,
    pub highest_sequence: Option<u64>,
    pub incidents: Vec<Incident>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorReport {
    pub channels: Vec<ChannelReport>,
    pub total_incidents: usize,
}

fn scope_key(statement: &ChannelState) -> (String, String, String) {
    (
        statement.publisher.clone(),
        statement.corpus.clone(),
        statement.channel.clone(),
    )
}

/// Compare every observation against every other observation for the same
/// publisher/corpus/channel and report the incidents in
/// [`IncidentKind`]. `retained`, when supplied, is checked against whichever
/// group matches its own scope; groups it does not match simply get no
/// [`IncidentKind::StaleLocalState`] check, the same "unavailable input, no
/// check performed" shape used throughout this codebase rather than an error.
pub fn monitor(
    observations: &[Observation],
    trust_root: &TrustRoot,
    trust: &TrustRootVerification,
    retained: Option<&RetainedState>,
) -> Result<MonitorReport> {
    let mut scopes: std::collections::BTreeMap<(String, String, String), Vec<&Observation>> =
        std::collections::BTreeMap::new();
    for observation in observations {
        scopes
            .entry(scope_key(&observation.statement))
            .or_default()
            .push(observation);
    }

    let mut channels = Vec::new();
    let mut total_incidents = 0;
    for ((publisher, corpus, channel), group) in scopes {
        let incidents = channel_incidents(&group, trust_root, trust, retained)?;
        let mut sequences: Vec<u64> = group.iter().map(|o| o.statement.sequence).collect();
        sequences.sort_unstable();
        sequences.dedup();
        total_incidents += incidents.len();
        channels.push(ChannelReport {
            publisher,
            corpus,
            channel,
            observation_count: group.len(),
            distinct_sequences: sequences.len(),
            highest_sequence: sequences.last().copied(),
            incidents,
        });
    }

    Ok(MonitorReport {
        channels,
        total_incidents,
    })
}

fn channel_incidents(
    group: &[&Observation],
    trust_root: &TrustRoot,
    trust: &TrustRootVerification,
    retained: Option<&RetainedState>,
) -> Result<Vec<Incident>> {
    let mut incidents = Vec::new();
    incidents.extend(equivocation_incidents(group)?);
    incidents.extend(conflict_incidents(group)?);
    incidents.extend(authority_incidents(group, trust_root, trust)?);
    incidents.extend(sequence_gap_incidents(group));
    incidents.extend(revoked_root_incidents(group)?);
    if let Some(retained) = retained {
        incidents.extend(stale_local_state_incidents(
            group, retained, trust_root, trust,
        )?);
    }
    Ok(incidents)
}

/// Same sequence, different statement digest.
fn equivocation_incidents(group: &[&Observation]) -> Result<Vec<Incident>> {
    let mut by_sequence: std::collections::BTreeMap<u64, Vec<(String, &ChannelState)>> =
        std::collections::BTreeMap::new();
    for observation in group {
        let digest = statement_digest(&observation.statement)?;
        let entries = by_sequence
            .entry(observation.statement.sequence)
            .or_default();
        if !entries.iter().any(|(existing, _)| existing == &digest) {
            entries.push((digest, &observation.statement));
        }
    }
    let mut incidents = Vec::new();
    for (sequence, entries) in by_sequence {
        if entries.len() > 1 {
            incidents.push(Incident {
                kind: IncidentKind::Equivocation,
                description: format!(
                    "{} different statements observed at sequence {sequence}",
                    entries.len()
                ),
                evidence: entries.into_iter().map(|(digest, _)| digest).collect(),
            });
        }
    }
    Ok(incidents)
}

/// A root is "still advertised as current" if no other observed statement at
/// a higher sequence lists it in `superseded`. More than one such root means
/// two statements each claim the role with no chain between them.
fn conflict_incidents(group: &[&Observation]) -> Result<Vec<Incident>> {
    let mut by_digest: std::collections::BTreeMap<String, &ChannelState> =
        std::collections::BTreeMap::new();
    for observation in group {
        let digest = statement_digest(&observation.statement)?;
        by_digest.entry(digest).or_insert(&observation.statement);
    }

    let mut still_current: Vec<(String, String)> = Vec::new();
    for (digest, statement) in &by_digest {
        let superseded_elsewhere = by_digest.values().any(|other| {
            other.sequence > statement.sequence
                && other
                    .superseded
                    .iter()
                    .any(|entry| entry.artifact_root == statement.current.artifact_root)
        });
        if !superseded_elsewhere {
            still_current.push((digest.clone(), statement.current.artifact_root.clone()));
        }
    }

    let mut distinct_roots: Vec<&String> = still_current.iter().map(|(_, root)| root).collect();
    distinct_roots.sort();
    distinct_roots.dedup();

    if distinct_roots.len() > 1 {
        Ok(vec![Incident {
            kind: IncidentKind::Conflict,
            description: format!(
                "{} distinct artifact roots are each advertised as current with no statement superseding the other",
                distinct_roots.len()
            ),
            evidence: still_current
                .into_iter()
                .map(|(digest, _)| digest)
                .collect(),
        }])
    } else {
        Ok(Vec::new())
    }
}

fn authority_incidents(
    group: &[&Observation],
    trust_root: &TrustRoot,
    trust: &TrustRootVerification,
) -> Result<Vec<Incident>> {
    let mut incidents = Vec::new();
    for observation in group {
        let statement = &observation.statement;
        let expected = (
            statement.publisher.as_str(),
            statement.corpus.as_str(),
            statement.channel.as_str(),
        );
        let verification =
            verify_channel_state(statement, trust_root, trust, None, None, expected)?;
        if verification.authority == SigningAuthority::None {
            incidents.push(Incident {
                kind: IncidentKind::AuthorityViolation,
                description: format!(
                    "statement at sequence {} did not meet any authorised role's threshold",
                    statement.sequence
                ),
                evidence: vec![statement_digest(statement)?],
            });
        }
    }
    Ok(incidents)
}

fn sequence_gap_incidents(group: &[&Observation]) -> Vec<Incident> {
    let mut sequences: Vec<u64> = group.iter().map(|o| o.statement.sequence).collect();
    sequences.sort_unstable();
    sequences.dedup();
    let mut incidents = Vec::new();
    for pair in sequences.windows(2) {
        if pair[1] > pair[0] + 1 {
            incidents.push(Incident {
                kind: IncidentKind::SequenceGap,
                description: format!(
                    "no statement observed between sequence {} and {}",
                    pair[0], pair[1]
                ),
                evidence: Vec::new(),
            });
        }
    }
    incidents
}

fn revoked_root_incidents(group: &[&Observation]) -> Result<Vec<Incident>> {
    let mut incidents = Vec::new();
    for revoker in group {
        for revocation in &revoker.statement.revoked {
            for advertiser in group {
                if advertiser.statement.sequence >= revoker.statement.sequence
                    && advertiser.statement.current.artifact_root == revocation.artifact_root
                {
                    incidents.push(Incident {
                        kind: IncidentKind::RevokedRootAdvertised,
                        description: format!(
                            "root {} was revoked at sequence {} but is advertised as current at sequence {}",
                            revocation.artifact_root,
                            revoker.statement.sequence,
                            advertiser.statement.sequence,
                        ),
                        evidence: vec![
                            statement_digest(&revoker.statement)?,
                            statement_digest(&advertiser.statement)?,
                        ],
                    });
                }
            }
        }
    }
    Ok(incidents)
}

fn stale_local_state_incidents(
    group: &[&Observation],
    retained: &RetainedState,
    trust_root: &TrustRoot,
    trust: &TrustRootVerification,
) -> Result<Vec<Incident>> {
    let Some(first) = group.first() else {
        return Ok(Vec::new());
    };
    if !retained.matches(
        &first.statement.publisher,
        &first.statement.corpus,
        &first.statement.channel,
    ) {
        return Ok(Vec::new());
    }

    let mut incidents = Vec::new();
    for observation in group {
        let statement = &observation.statement;
        if statement.sequence <= retained.highest_sequence {
            continue;
        }
        let expected = (
            statement.publisher.as_str(),
            statement.corpus.as_str(),
            statement.channel.as_str(),
        );
        let verification =
            verify_channel_state(statement, trust_root, trust, None, None, expected)?;
        if verification.authority == SigningAuthority::Full {
            incidents.push(Incident {
                kind: IncidentKind::StaleLocalState,
                description: format!(
                    "an authorised statement at sequence {} was observed; retained state is at sequence {}",
                    statement.sequence, retained.highest_sequence
                ),
                evidence: vec![statement_digest(statement)?],
            });
        }
    }
    Ok(incidents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::release::CHANNEL_STATE_SCHEMA_V1;
    #[cfg(feature = "signing")]
    use crate::trust::{
        KeyDescriptor, ROLE_ARTIFACT, ROLE_EMERGENCY_REVOCATION, ROLE_RELEASE_STATE, ROLE_ROOT,
        RoleDescriptor, verify_trust_root,
    };
    #[cfg(feature = "signing")]
    use std::collections::BTreeMap;

    /// A minimal, genuinely self-signed trust root authorising one
    /// release-state key and one revocation key. It must actually verify --
    /// `verify_channel_state` (what `authority_incidents` calls) only trusts
    /// a statement's signers when the trust root backing them verified, so
    /// an unsigned or unverified test root would make every statement look
    /// unauthorized regardless of who signed it.
    #[cfg(feature = "signing")]
    struct Keys {
        release_secret: [u8; 32],
        revocation_secret: [u8; 32],
        outsider_secret: [u8; 32],
        trust_root: TrustRoot,
    }

    #[cfg(feature = "signing")]
    fn keypair(seed: u8) -> ([u8; 32], [u8; 32]) {
        use ed25519_dalek::SigningKey;
        let signing = SigningKey::from_bytes(&[seed; 32]);
        (signing.verifying_key().to_bytes(), [seed; 32])
    }

    #[cfg(feature = "signing")]
    fn key_id(public: &[u8; 32]) -> String {
        blake3::hash(public).to_hex().to_string()
    }

    #[cfg(feature = "signing")]
    fn keys() -> Keys {
        let (release_public, release_secret) = keypair(1);
        let (revocation_public, revocation_secret) = keypair(2);
        let (_, outsider_secret) = keypair(3);
        let (root_public, root_secret) = keypair(9);
        let (artifact_public, _) = keypair(10);

        let mut trust_keys = BTreeMap::new();
        for public in [
            &root_public,
            &artifact_public,
            &release_public,
            &revocation_public,
        ] {
            trust_keys.insert(
                key_id(public),
                KeyDescriptor {
                    algorithm: "Ed25519".into(),
                    public_key: hex::encode(public),
                },
            );
        }
        let mut roles = BTreeMap::new();
        roles.insert(
            ROLE_ROOT.to_string(),
            RoleDescriptor {
                threshold: 1,
                keys: vec![key_id(&root_public)],
            },
        );
        roles.insert(
            ROLE_ARTIFACT.to_string(),
            RoleDescriptor {
                threshold: 1,
                keys: vec![key_id(&artifact_public)],
            },
        );
        roles.insert(
            ROLE_RELEASE_STATE.to_string(),
            RoleDescriptor {
                threshold: 1,
                keys: vec![key_id(&release_public)],
            },
        );
        roles.insert(
            ROLE_EMERGENCY_REVOCATION.to_string(),
            RoleDescriptor {
                threshold: 1,
                keys: vec![key_id(&revocation_public)],
            },
        );

        let mut trust_root = TrustRoot {
            schema: crate::trust::TRUST_ROOT_SCHEMA_V1.into(),
            publisher: "example.com".into(),
            version: 1,
            issued_at: "2026-08-06T00:00:00Z".into(),
            valid_until: "2099-01-01T00:00:00Z".into(),
            roles,
            keys: trust_keys,
            signatures: Vec::new(),
        };
        crate::trust::sign_trust_root(&mut trust_root, &root_secret).unwrap();

        Keys {
            release_secret,
            revocation_secret,
            outsider_secret,
            trust_root,
        }
    }

    fn base_statement(sequence: u64, root: &str) -> ChannelState {
        ChannelState {
            schema: CHANNEL_STATE_SCHEMA_V1.into(),
            publisher: "example.com".into(),
            corpus: "support-manual".into(),
            channel: "production".into(),
            sequence,
            issued_at: "2026-08-06T00:00:00Z".into(),
            valid_until: "2099-01-01T00:00:00Z".into(),
            current: crate::release::CurrentRelease {
                version: "1.0.0".into(),
                artifact_root: root.repeat(64)[..64].to_string(),
            },
            superseded: Vec::new(),
            revoked: Vec::new(),
            signatures: Vec::new(),
        }
    }

    #[cfg(feature = "signing")]
    fn sign(statement: &mut ChannelState, secret: &[u8; 32]) {
        crate::release::sign_channel_state(statement, secret).unwrap();
    }

    #[cfg(feature = "signing")]
    fn supersede(statement: &mut ChannelState, root: &str) {
        statement.superseded.push(crate::release::Supersession {
            artifact_root: root.repeat(64)[..64].to_string(),
            by: statement.current.artifact_root.clone(),
            at: statement.issued_at.clone(),
        });
    }

    #[cfg(feature = "signing")]
    fn revoke(statement: &mut ChannelState, root: &str) {
        statement.revoked.push(crate::release::Revocation {
            artifact_root: root.repeat(64)[..64].to_string(),
            at: statement.issued_at.clone(),
            reason: "test".into(),
        });
    }

    #[cfg(feature = "signing")]
    fn observation(statement: &ChannelState) -> Observation {
        Observation {
            observed_at: "2026-08-06T00:00:00Z".into(),
            statement: statement.clone(),
        }
    }

    #[cfg(feature = "signing")]
    fn trust_verification(keys: &Keys) -> TrustRootVerification {
        verify_trust_root(&keys.trust_root, None, Some("2026-08-06T00:30:00Z")).unwrap()
    }

    #[cfg(feature = "signing")]
    fn incidents_of(report: &MonitorReport, kind: IncidentKind) -> Vec<&Incident> {
        report
            .channels
            .iter()
            .flat_map(|channel| channel.incidents.iter())
            .filter(|incident| incident.kind == kind)
            .collect()
    }

    #[cfg(feature = "signing")]
    #[test]
    fn a_clean_history_has_no_incidents() {
        let keys = keys();
        let mut s1 = base_statement(1, "a");
        sign(&mut s1, &keys.release_secret);
        let mut s2 = base_statement(2, "b");
        supersede(&mut s2, "a");
        sign(&mut s2, &keys.release_secret);

        let observations = vec![observation(&s1), observation(&s2)];
        let trust = trust_verification(&keys);
        let report = monitor(&observations, &keys.trust_root, &trust, None).unwrap();

        assert_eq!(report.total_incidents, 0, "{report:?}");
        assert_eq!(report.channels[0].highest_sequence, Some(2));
        assert_eq!(report.channels[0].distinct_sequences, 2);
    }

    #[cfg(feature = "signing")]
    #[test]
    fn two_different_statements_at_one_sequence_is_equivocation() {
        let keys = keys();
        let mut s1a = base_statement(1, "a");
        sign(&mut s1a, &keys.release_secret);
        let mut s1b = base_statement(1, "b");
        sign(&mut s1b, &keys.release_secret);

        let observations = vec![observation(&s1a), observation(&s1b)];
        let trust = trust_verification(&keys);
        let report = monitor(&observations, &keys.trust_root, &trust, None).unwrap();

        let found = incidents_of(&report, IncidentKind::Equivocation);
        assert_eq!(found.len(), 1, "{report:?}");
        assert_eq!(found[0].evidence.len(), 2);
    }

    #[cfg(feature = "signing")]
    #[test]
    fn two_unchained_current_roots_is_a_conflict() {
        let keys = keys();
        // s2 does not list s1's root in `superseded`: no chain between them.
        let mut s1 = base_statement(1, "a");
        sign(&mut s1, &keys.release_secret);
        let mut s2 = base_statement(2, "b");
        sign(&mut s2, &keys.release_secret);

        let observations = vec![observation(&s1), observation(&s2)];
        let trust = trust_verification(&keys);
        let report = monitor(&observations, &keys.trust_root, &trust, None).unwrap();

        let found = incidents_of(&report, IncidentKind::Conflict);
        assert_eq!(found.len(), 1, "{report:?}");
    }

    #[cfg(feature = "signing")]
    #[test]
    fn an_unauthorized_signer_is_an_authority_violation() {
        let keys = keys();
        let mut s1 = base_statement(1, "a");
        // Signed by a key no role authorises.
        sign(&mut s1, &keys.outsider_secret);

        let observations = vec![observation(&s1)];
        let trust = trust_verification(&keys);
        let report = monitor(&observations, &keys.trust_root, &trust, None).unwrap();

        let found = incidents_of(&report, IncidentKind::AuthorityViolation);
        assert_eq!(found.len(), 1, "{report:?}");
    }

    #[cfg(feature = "signing")]
    #[test]
    fn a_revocation_only_signature_is_not_an_authority_violation() {
        // Distinct from the unauthorized case: RevocationOnly is a real,
        // meaningful authority (it just doesn't cover `current`), and must
        // not be confused with SigningAuthority::None.
        let keys = keys();
        let mut s1 = base_statement(1, "a");
        sign(&mut s1, &keys.revocation_secret);

        let observations = vec![observation(&s1)];
        let trust = trust_verification(&keys);
        let report = monitor(&observations, &keys.trust_root, &trust, None).unwrap();

        assert!(incidents_of(&report, IncidentKind::AuthorityViolation).is_empty());
    }

    #[cfg(feature = "signing")]
    #[test]
    fn a_gap_between_observed_sequences_is_flagged() {
        let keys = keys();
        let mut s1 = base_statement(1, "a");
        sign(&mut s1, &keys.release_secret);
        let mut s5 = base_statement(5, "b");
        sign(&mut s5, &keys.release_secret);

        let observations = vec![observation(&s1), observation(&s5)];
        let trust = trust_verification(&keys);
        let report = monitor(&observations, &keys.trust_root, &trust, None).unwrap();

        let found = incidents_of(&report, IncidentKind::SequenceGap);
        assert_eq!(found.len(), 1, "{report:?}");
    }

    #[cfg(feature = "signing")]
    #[test]
    fn consecutive_sequences_have_no_gap() {
        let keys = keys();
        let mut s1 = base_statement(1, "a");
        sign(&mut s1, &keys.release_secret);
        let mut s2 = base_statement(2, "b");
        supersede(&mut s2, "a");
        sign(&mut s2, &keys.release_secret);

        let observations = vec![observation(&s1), observation(&s2)];
        let trust = trust_verification(&keys);
        let report = monitor(&observations, &keys.trust_root, &trust, None).unwrap();

        assert!(incidents_of(&report, IncidentKind::SequenceGap).is_empty());
    }

    #[cfg(feature = "signing")]
    #[test]
    fn a_revoked_root_still_advertised_is_a_security_incident() {
        let keys = keys();
        let mut revocation = base_statement(1, "c");
        revoke(&mut revocation, "a");
        sign(&mut revocation, &keys.revocation_secret);
        // A later statement (higher sequence) illegitimately re-advertises
        // the already-revoked root as current.
        let mut s2 = base_statement(2, "a");
        sign(&mut s2, &keys.release_secret);

        let observations = vec![observation(&revocation), observation(&s2)];
        let trust = trust_verification(&keys);
        let report = monitor(&observations, &keys.trust_root, &trust, None).unwrap();

        let found = incidents_of(&report, IncidentKind::RevokedRootAdvertised);
        assert!(!found.is_empty(), "{report:?}");
    }

    #[cfg(feature = "signing")]
    #[test]
    fn a_revocation_before_the_advertisement_is_not_flagged() {
        // The recommended semantic: only a later-or-equal sequence still
        // advertising a revoked root counts. An earlier statement (already
        // superseded history) predating the revocation is expected and not
        // an incident.
        let keys = keys();
        let mut early = base_statement(1, "a");
        sign(&mut early, &keys.release_secret);
        let mut revocation = base_statement(2, "c");
        revoke(&mut revocation, "z"); // revokes an unrelated root
        sign(&mut revocation, &keys.revocation_secret);

        let observations = vec![observation(&early), observation(&revocation)];
        let trust = trust_verification(&keys);
        let report = monitor(&observations, &keys.trust_root, &trust, None).unwrap();

        assert!(incidents_of(&report, IncidentKind::RevokedRootAdvertised).is_empty());
    }

    #[cfg(feature = "signing")]
    #[test]
    fn a_higher_authorized_sequence_than_retained_is_stale_local_state() {
        let keys = keys();
        let mut s2 = base_statement(2, "b");
        sign(&mut s2, &keys.release_secret);

        let observations = vec![observation(&s2)];
        let trust = trust_verification(&keys);
        let retained = RetainedState {
            publisher: "example.com".into(),
            corpus: "support-manual".into(),
            channel: "production".into(),
            highest_sequence: 1,
            statement_digest: "irrelevant".into(),
            artifact_root: "a".repeat(64),
            accepted_at: "2026-08-06T00:00:00Z".into(),
        };
        let report = monitor(&observations, &keys.trust_root, &trust, Some(&retained)).unwrap();

        let found = incidents_of(&report, IncidentKind::StaleLocalState);
        assert_eq!(found.len(), 1, "{report:?}");
    }

    #[cfg(feature = "signing")]
    #[test]
    fn retained_state_for_a_different_scope_is_not_consulted() {
        let keys = keys();
        let mut s2 = base_statement(2, "b");
        sign(&mut s2, &keys.release_secret);

        let observations = vec![observation(&s2)];
        let trust = trust_verification(&keys);
        let retained = RetainedState {
            publisher: "other.example".into(),
            corpus: "support-manual".into(),
            channel: "production".into(),
            highest_sequence: 1,
            statement_digest: "irrelevant".into(),
            artifact_root: "a".repeat(64),
            accepted_at: "2026-08-06T00:00:00Z".into(),
        };
        let report = monitor(&observations, &keys.trust_root, &trust, Some(&retained)).unwrap();

        assert!(incidents_of(&report, IncidentKind::StaleLocalState).is_empty());
    }

    #[test]
    fn append_and_parse_round_trip() {
        let s1 = base_statement(1, "a");
        let history = append_observation("", &s1, "2026-08-06T00:00:00Z").unwrap();
        let s2 = base_statement(2, "b");
        let history = append_observation(&history, &s2, "2026-08-06T01:00:00Z").unwrap();

        let observations = parse_observations(&history).unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].statement.sequence, 1);
        assert_eq!(observations[1].statement.sequence, 2);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let s1 = base_statement(1, "a");
        let mut history = append_observation("", &s1, "2026-08-06T00:00:00Z").unwrap();
        history.push('\n');
        history.push('\n');
        let observations = parse_observations(&history).unwrap();
        assert_eq!(observations.len(), 1);
    }

    #[test]
    fn a_malformed_line_is_reported_with_its_line_number() {
        let error = parse_observations("not json").unwrap_err();
        assert!(error.to_string().contains("line 1"));
    }

    #[test]
    fn oversized_history_is_rejected() {
        let oversized = "a".repeat(MAX_OBSERVATIONS_FILE_BYTES as usize + 1);
        let error = parse_observations(&oversized).unwrap_err();
        assert!(error.to_string().contains("exceeds size limit"));
    }
}
