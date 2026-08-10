//! Consumer trust policies: what a runtime requires before it will use an artifact.
//!
//! The five claims a complete verification establishes — integrity, publisher
//! authority, release currency, retrieval evidence, execution evidence — are
//! produced by separate stages and stay separate here. A policy decides which
//! of them must hold; it never merges them into one boolean, and a caller
//! reading only [`PolicyDecision::permitted`] can still recover exactly which
//! claim was missing.
//!
//! # No silent downgrade
//!
//! A stronger policy that cannot be satisfied fails. It never quietly behaves
//! like a weaker one. This is the rule most likely to be violated by
//! convenience: a runtime that cannot reach a freshness authority is tempted to
//! carry on with publisher authority alone, which converts a deliberate
//! requirement into an attacker-triggerable one — withholding the statement
//! becomes a way to weaken the consumer.
//!
//! [`TrustPolicy::AuthorizedCurrentWitnessed`] is the live demonstration.
//! Transparency evidence is not implemented yet, so that policy currently
//! always denies. That is the correct behaviour and it is asserted by tests: a
//! policy whose requirement cannot yet be met must refuse, not degrade.
//!
//! # Revocation outranks policy
//!
//! A known revocation denies under every policy, including
//! [`TrustPolicy::IntegrityOnly`]. The architecture contract lists "revoked
//! root: security failure" without qualification, and a consumer that has been
//! told an artifact was withdrawn should not use it merely because it asked a
//! weaker question. Revocation is a security event; supersession is policy.

use serde::{Deserialize, Serialize};

use crate::release::{ChannelStateVerification, Currency};
use crate::trust::{ROLE_ARTIFACT, TrustRootVerification};

/// Stage A: does the container hold together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactIntegrity {
    Valid,
    Invalid,
}

/// Stage B: is the signer allowed to publish for this publisher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublisherAuthority {
    Authorized,
    /// A trust root verified and did not authorise the artifact's signers.
    Unauthorized,
    /// No verified trust root was available. Distinct from `Unauthorized`: one
    /// is a negative answer, the other is no answer.
    Unknown,
}

/// Stage F: was the release statement publicly logged and witnessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransparencyEvidence {
    Verified,
    /// Present but below the configured witness quorum, or inconsistent.
    Insufficient,
    /// Not supplied. The reference runtime returns this today.
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustPolicy {
    /// The container holds together. Claims nothing about who published it or
    /// whether it is current.
    IntegrityOnly,
    /// Additionally, an authorised artifact key signed it.
    AuthorizedPublisher,
    /// Additionally, a verified channel-state statement reports it current.
    AuthorizedCurrent,
    /// Additionally, that statement is in a transparency log with a witness
    /// quorum. Not satisfiable in this release.
    AuthorizedCurrentWitnessed,
}

impl TrustPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IntegrityOnly => "integrity_only",
            Self::AuthorizedPublisher => "authorized_publisher",
            Self::AuthorizedCurrent => "authorized_current",
            Self::AuthorizedCurrentWitnessed => "authorized_current_witnessed",
        }
    }
}

/// Everything the stages produced, assembled for one decision.
#[derive(Debug, Clone)]
pub struct PolicyInputs<'a> {
    pub artifact_root: &'a str,
    pub artifact_integrity: ArtifactIntegrity,
    /// Key ids that produced a valid signature over the artifact root.
    pub artifact_signers: &'a [String],
    pub trust: Option<&'a TrustRootVerification>,
    pub channel_state: Option<&'a ChannelStateVerification>,
    pub currency: Currency,
    pub transparency: TransparencyEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub policy: String,
    pub artifact_root: String,
    pub permitted: bool,
    pub artifact_integrity: ArtifactIntegrity,
    pub publisher_authority: PublisherAuthority,
    pub currency: Currency,
    pub transparency: TransparencyEvidence,
    /// Named requirements of the requested policy that were not met. Empty when
    /// permitted. A caller must be able to say what was missing, not only that
    /// something was.
    pub unmet_requirements: Vec<String>,
    /// What the decision rests on that this evaluation did not itself check.
    pub assumptions: Vec<String>,
}

fn publisher_authority(inputs: &PolicyInputs<'_>) -> PublisherAuthority {
    let Some(trust) = inputs.trust else {
        return PublisherAuthority::Unknown;
    };
    if !trust.verified {
        return PublisherAuthority::Unknown;
    }
    let authorized = trust
        .authorized_roles
        .get(ROLE_ARTIFACT)
        .is_some_and(|keys| {
            inputs
                .artifact_signers
                .iter()
                .any(|signer| keys.iter().any(|key| key == signer))
        });
    if authorized {
        PublisherAuthority::Authorized
    } else {
        PublisherAuthority::Unauthorized
    }
}

/// Evaluate one policy against the evidence gathered for an artifact.
pub fn evaluate_policy(inputs: &PolicyInputs<'_>, policy: TrustPolicy) -> PolicyDecision {
    let mut unmet = Vec::new();
    let mut assumptions = Vec::new();

    let authority = publisher_authority(inputs);

    if inputs.artifact_integrity != ArtifactIntegrity::Valid {
        unmet.push("artifact integrity did not verify".into());
    }

    // Checked before the policy ladder and independent of it. Being told an
    // artifact was withdrawn is a reason not to use it whatever question the
    // caller asked.
    if inputs.currency == Currency::Revoked {
        unmet.push("the publisher revoked this artifact".into());
    }

    let needs_publisher = matches!(
        policy,
        TrustPolicy::AuthorizedPublisher
            | TrustPolicy::AuthorizedCurrent
            | TrustPolicy::AuthorizedCurrentWitnessed
    );
    if needs_publisher {
        match authority {
            PublisherAuthority::Authorized => {}
            PublisherAuthority::Unauthorized => {
                unmet.push("no authorised artifact key signed this artifact".into())
            }
            PublisherAuthority::Unknown => {
                unmet.push("no verified trust root was available".into())
            }
        }
    }

    let needs_currency = matches!(
        policy,
        TrustPolicy::AuthorizedCurrent | TrustPolicy::AuthorizedCurrentWitnessed
    );
    if needs_currency {
        match inputs.channel_state {
            None => unmet.push("no channel-state statement was supplied".into()),
            Some(state) if !state.verified => {
                unmet.push("the channel-state statement did not verify".into())
            }
            Some(state) => {
                if state.publisher != inputs.trust.map(|t| t.publisher.as_str()).unwrap_or("") {
                    unmet.push(
                        "the channel-state statement and the trust root name different publishers"
                            .into(),
                    );
                }
                assumptions.push(format!(
                    "currency taken from statement sequence {} for {}/{}",
                    state.sequence, state.corpus, state.channel
                ));
            }
        }
        match inputs.currency {
            Currency::Current => {}
            Currency::Superseded => unmet.push("a newer release supersedes this artifact".into()),
            Currency::Revoked => {}
            Currency::Unknown => unmet.push(
                "currency is unknown; no verified statement covers this artifact root".into(),
            ),
        }
    }

    if policy == TrustPolicy::AuthorizedCurrentWitnessed {
        match inputs.transparency {
            TransparencyEvidence::Verified => {}
            TransparencyEvidence::Insufficient => {
                unmet.push("transparency evidence did not meet the witness quorum".into())
            }
            TransparencyEvidence::Unavailable => unmet.push(
                "no transparency evidence was supplied; this policy is not satisfiable in this \
                 release and deliberately denies rather than degrading to authorized_current"
                    .into(),
            ),
        }
    }

    if !needs_currency {
        assumptions.push(
            "this policy does not consult release state, so nothing here says the artifact is \
             current"
                .into(),
        );
    }
    if !needs_publisher {
        assumptions.push(
            "this policy does not consult a trust root, so nothing here says an authorised \
             publisher produced the artifact"
                .into(),
        );
    }

    PolicyDecision {
        policy: policy.as_str().to_string(),
        artifact_root: inputs.artifact_root.to_string(),
        permitted: unmet.is_empty(),
        artifact_integrity: inputs.artifact_integrity,
        publisher_authority: authority,
        currency: inputs.currency,
        transparency: inputs.transparency,
        unmet_requirements: unmet,
        assumptions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn trust_verification(authorized_key: Option<&str>) -> TrustRootVerification {
        let mut authorized_roles = BTreeMap::new();
        if let Some(key) = authorized_key {
            authorized_roles.insert(ROLE_ARTIFACT.to_string(), vec![key.to_string()]);
        }
        TrustRootVerification {
            publisher: "example.com".into(),
            version: 1,
            payload_digest: "d".into(),
            schema_supported: true,
            structurally_valid: true,
            key_ids_match_keys: true,
            self_signed: true,
            signed_by_prior_root: None,
            version_advanced: None,
            publisher_unchanged: None,
            within_validity: Some(true),
            first_contact: true,
            authorized_roles,
            verified: true,
            assumptions: Vec::new(),
            issues: Vec::new(),
        }
    }

    fn channel_verification(verified: bool) -> ChannelStateVerification {
        ChannelStateVerification {
            publisher: "example.com".into(),
            corpus: "support-manual".into(),
            channel: "production".into(),
            sequence: 4,
            statement_digest: "d".into(),
            schema_supported: true,
            structurally_valid: true,
            trust_root_verified: true,
            authority: crate::release::SigningAuthority::Full,
            signers: Vec::new(),
            scope_matches: true,
            within_validity: Some(true),
            sequence_verdict: crate::release::SequenceVerdict::Advanced,
            verified,
            assumptions: Vec::new(),
            issues: Vec::new(),
        }
    }

    const SIGNER: &str = "signer-key-id";

    fn inputs<'a>(
        trust: Option<&'a TrustRootVerification>,
        state: Option<&'a ChannelStateVerification>,
        currency: Currency,
        signers: &'a [String],
    ) -> PolicyInputs<'a> {
        PolicyInputs {
            artifact_root: "root",
            artifact_integrity: ArtifactIntegrity::Valid,
            artifact_signers: signers,
            trust,
            channel_state: state,
            currency,
            transparency: TransparencyEvidence::Unavailable,
        }
    }

    #[test]
    fn integrity_only_permits_an_unsigned_artifact_and_claims_nothing_more() {
        let signers: Vec<String> = Vec::new();
        let decision = evaluate_policy(
            &inputs(None, None, Currency::Unknown, &signers),
            TrustPolicy::IntegrityOnly,
        );
        assert!(decision.permitted);
        assert_eq!(decision.publisher_authority, PublisherAuthority::Unknown);
        assert_eq!(decision.currency, Currency::Unknown);
        // The decision must say out loud what it did not establish, or a caller
        // reading only `permitted` will take it for more than it is.
        let notes = decision.assumptions.join(" ");
        assert!(
            notes.contains("nothing here says the artifact is current"),
            "assumptions did not disclaim currency: {notes}"
        );
        assert!(
            notes.contains("nothing here says an authorised"),
            "assumptions did not disclaim publisher authority: {notes}"
        );
    }

    #[test]
    fn integrity_only_still_refuses_a_revoked_artifact() {
        let signers: Vec<String> = Vec::new();
        let decision = evaluate_policy(
            &inputs(None, None, Currency::Revoked, &signers),
            TrustPolicy::IntegrityOnly,
        );
        assert!(!decision.permitted);
        assert!(
            decision
                .unmet_requirements
                .iter()
                .any(|reason| reason.contains("revoked"))
        );
    }

    #[test]
    fn an_unauthorized_signer_is_distinguished_from_no_trust_root() {
        let signers = vec!["some-other-key".to_string()];
        let trust = trust_verification(Some(SIGNER));
        let decision = evaluate_policy(
            &inputs(Some(&trust), None, Currency::Unknown, &signers),
            TrustPolicy::AuthorizedPublisher,
        );
        assert_eq!(
            decision.publisher_authority,
            PublisherAuthority::Unauthorized
        );
        assert!(!decision.permitted);

        let decision = evaluate_policy(
            &inputs(None, None, Currency::Unknown, &signers),
            TrustPolicy::AuthorizedPublisher,
        );
        assert_eq!(decision.publisher_authority, PublisherAuthority::Unknown);
        assert!(!decision.permitted);
    }

    #[test]
    fn authorized_current_requires_both_authority_and_currency() {
        let signers = vec![SIGNER.to_string()];
        let trust = trust_verification(Some(SIGNER));
        let state = channel_verification(true);

        let decision = evaluate_policy(
            &inputs(Some(&trust), Some(&state), Currency::Current, &signers),
            TrustPolicy::AuthorizedCurrent,
        );
        assert!(decision.permitted, "{:?}", decision.unmet_requirements);

        // Superseded is not a security failure, and it is also not permitted
        // under a policy that asked for current.
        let decision = evaluate_policy(
            &inputs(Some(&trust), Some(&state), Currency::Superseded, &signers),
            TrustPolicy::AuthorizedCurrent,
        );
        assert!(!decision.permitted);
        assert!(
            decision
                .unmet_requirements
                .iter()
                .any(|reason| reason.contains("supersedes"))
        );
    }

    #[test]
    fn a_missing_statement_denies_rather_than_downgrading() {
        // The attacker-triggerable case: withhold the statement and see whether
        // the consumer quietly accepts publisher authority alone.
        let signers = vec![SIGNER.to_string()];
        let trust = trust_verification(Some(SIGNER));

        let weaker = evaluate_policy(
            &inputs(Some(&trust), None, Currency::Unknown, &signers),
            TrustPolicy::AuthorizedPublisher,
        );
        assert!(weaker.permitted, "the weaker policy is satisfiable");

        let stronger = evaluate_policy(
            &inputs(Some(&trust), None, Currency::Unknown, &signers),
            TrustPolicy::AuthorizedCurrent,
        );
        assert!(!stronger.permitted, "a stronger policy must not degrade");
        assert!(
            stronger
                .unmet_requirements
                .iter()
                .any(|reason| reason.contains("no channel-state statement"))
        );
    }

    #[test]
    fn the_witnessed_policy_denies_rather_than_behaving_like_authorized_current() {
        let signers = vec![SIGNER.to_string()];
        let trust = trust_verification(Some(SIGNER));
        let state = channel_verification(true);
        let evidence = inputs(Some(&trust), Some(&state), Currency::Current, &signers);

        assert!(evaluate_policy(&evidence, TrustPolicy::AuthorizedCurrent).permitted);

        let witnessed = evaluate_policy(&evidence, TrustPolicy::AuthorizedCurrentWitnessed);
        assert!(
            !witnessed.permitted,
            "a policy whose requirement is unimplemented must refuse"
        );
        assert!(
            witnessed
                .unmet_requirements
                .iter()
                .any(|reason| reason.contains("transparency"))
        );
    }

    #[test]
    fn an_unverified_statement_does_not_satisfy_a_currency_requirement() {
        let signers = vec![SIGNER.to_string()];
        let trust = trust_verification(Some(SIGNER));
        let state = channel_verification(false);
        let decision = evaluate_policy(
            &inputs(Some(&trust), Some(&state), Currency::Current, &signers),
            TrustPolicy::AuthorizedCurrent,
        );
        assert!(!decision.permitted);
        assert!(
            decision
                .unmet_requirements
                .iter()
                .any(|reason| reason.contains("did not verify"))
        );
    }

    #[test]
    fn broken_integrity_denies_under_every_policy() {
        let signers = vec![SIGNER.to_string()];
        let trust = trust_verification(Some(SIGNER));
        let state = channel_verification(true);
        let mut evidence = inputs(Some(&trust), Some(&state), Currency::Current, &signers);
        evidence.artifact_integrity = ArtifactIntegrity::Invalid;
        evidence.transparency = TransparencyEvidence::Verified;

        for policy in [
            TrustPolicy::IntegrityOnly,
            TrustPolicy::AuthorizedPublisher,
            TrustPolicy::AuthorizedCurrent,
            TrustPolicy::AuthorizedCurrentWitnessed,
        ] {
            let decision = evaluate_policy(&evidence, policy);
            assert!(
                !decision.permitted,
                "{} permitted a broken artifact",
                decision.policy
            );
        }
    }
}
