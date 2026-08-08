//! Signed occurrence evidence for one application execution.
//!
//! A run bundle proves its receipts and merely carries application metadata.
//! This module leaves that format untouched and signs a separate DSSE-wrapped,
//! in-toto-compatible statement binding the exact receipt set, release evidence,
//! query/model/policy identifiers, and output bytes to a distinct workload.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bundle::{RunBundle, verify_run_bundle};
use crate::error::{AnnpackError, Result};
use crate::policy::{
    ArtifactIntegrity, PolicyInputs, TransparencyEvidence, TrustPolicy, evaluate_policy,
};
use crate::provenance::{
    DSSE_PAYLOAD_TYPE, DsseSignature, Envelope, Subject, SubjectDigest, b64_decode, b64_encode,
    check_signer, pae, sha256_hex,
};
use crate::release::{
    ChannelState, ChannelStateVerification, Currency, currency_for_root, statement_digest,
};
use crate::trust::{ROLE_ARTIFACT, TrustRootVerification, parse_utc_timestamp};

pub const RUN_ATTESTATION_PREDICATE_TYPE: &str = "https://annpack.dev/attestations/run/v1";
pub const IN_TOTO_STATEMENT_TYPE: &str = "https://in-toto.io/Statement/v1";
pub const RUN_ATTESTATION_SCHEMA_V1: &str = "annpack-run-attestation-v1";
pub const MAX_RUN_ATTESTATION_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_RUN_RECEIPTS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    Verified,
    Carried,
    Missing,
    Mismatched,
    Invalid,
    Untrusted,
    Unknown,
    NotEvaluated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OccurrenceStrength {
    WorkloadAttested,
    WorkloadAttestedWithTrustedTime,
    ExternallyAnchored,
    Unattested,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmptyReceiptPolicy {
    Deny,
    AllowExplicit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DigestValue {
    pub algorithm: String,
    pub value: String,
}

impl DigestValue {
    fn sha256(bytes: &[u8]) -> Self {
        Self {
            algorithm: "sha256".into(),
            value: sha256_hex(bytes),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptBinding {
    pub digest: DigestValue,
    pub artifact_root: String,
    pub passage_id: String,
    pub passage_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionClaim {
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    pub workload_identity: String,
    pub started_at: String,
    pub completed_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signing_time: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeClaim {
    pub run_bundle_digest: DigestValue,
    pub receipts: Vec<ReceiptBinding>,
    pub receipt_count: usize,
    pub no_passages_retrieved: bool,
    pub artifact_roots: Vec<String>,
    pub publisher: String,
    pub corpus: String,
    pub channel: String,
    pub channel_state_digest: DigestValue,
    pub channel_state_sequence: u64,
    pub observed_currency: Currency,
    pub trust_policy: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalClaim {
    pub query_digest: DigestValue,
    pub retrieval_policy_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_mode: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationClaim {
    pub application_identity: String,
    pub application_version: String,
    pub model_identifier: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,
    pub prompt_policy_digest: DigestValue,
    pub tool_policy_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deployment_identity: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunPredicate {
    pub schema: String,
    pub execution: ExecutionClaim,
    pub knowledge: KnowledgeClaim,
    pub retrieval: RetrievalClaim,
    pub application: ApplicationClaim,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunStatement {
    #[serde(rename = "_type")]
    pub statement_type: String,
    pub subject: Vec<Subject>,
    #[serde(rename = "predicateType")]
    pub predicate_type: String,
    pub predicate: RunPredicate,
}

#[derive(Debug, Clone)]
pub struct ExecutionMetadata {
    pub run_id: String,
    pub trace_id: Option<String>,
    pub workload_identity: String,
    pub started_at: String,
    pub completed_at: String,
    pub retrieval_policy_revision: String,
    pub retrieval_mode: Option<String>,
    pub application_identity: String,
    pub application_version: String,
    pub model_identifier: String,
    pub model_provider: Option<String>,
    pub prompt_policy_bytes: Vec<u8>,
    pub tool_policy_revision: String,
    pub deployment_identity: Option<String>,
}

pub struct CreateRunAttestationInput<'a> {
    pub run_bundle: &'a RunBundle,
    pub channel_state: &'a ChannelState,
    pub channel_verification: &'a ChannelStateVerification,
    pub publisher_trust: &'a TrustRootVerification,
    pub trust_policy: TrustPolicy,
    pub metadata: ExecutionMetadata,
    pub output: &'a [u8],
    pub empty_receipts: EmptyReceiptPolicy,
}

fn canonical_receipts(bundle: &RunBundle) -> Result<Vec<ReceiptBinding>> {
    if bundle.receipts.len() > MAX_RUN_RECEIPTS {
        return Err(AnnpackError::InvalidInput(format!(
            "run carries {} receipts, above the {MAX_RUN_RECEIPTS} limit",
            bundle.receipts.len()
        )));
    }
    let mut bindings = Vec::with_capacity(bundle.receipts.len());
    let mut seen = BTreeSet::new();
    for receipt in &bundle.receipts {
        let bytes = serde_json::to_vec(receipt)?;
        let digest = DigestValue::sha256(&bytes);
        if !seen.insert(digest.value.clone()) {
            return Err(AnnpackError::InvalidInput(
                "duplicate receipt digest is ambiguous and is refused".into(),
            ));
        }
        bindings.push(ReceiptBinding {
            digest,
            artifact_root: receipt.pack_root.to_lowercase(),
            passage_id: receipt.passage_id.clone(),
            passage_hash: receipt.passage_hash.to_lowercase(),
        });
    }
    bindings.sort_by(|left, right| left.digest.value.cmp(&right.digest.value));
    Ok(bindings)
}

#[derive(Serialize)]
struct CanonicalBundle<'a> {
    schema: &'a str,
    run_id: &'a str,
    created_at: &'a Option<String>,
    application: &'a Option<String>,
    model: &'a Option<String>,
    query: &'a str,
    answer: &'a Option<String>,
    answer_hash: &'a Option<String>,
    receipt_digests: Vec<&'a str>,
}

fn canonical_bundle_digest(bundle: &RunBundle, receipts: &[ReceiptBinding]) -> Result<DigestValue> {
    Ok(DigestValue::sha256(&serde_json::to_vec(
        &CanonicalBundle {
            schema: &bundle.schema,
            run_id: &bundle.run_id,
            created_at: &bundle.created_at,
            application: &bundle.application,
            model: &bundle.model,
            query: &bundle.query,
            answer: &bundle.answer,
            answer_hash: &bundle.answer_hash,
            receipt_digests: receipts
                .iter()
                .map(|receipt| receipt.digest.value.as_str())
                .collect(),
        },
    )?))
}

fn artifact_signers(bundle: &RunBundle) -> Vec<String> {
    let mut signers = BTreeSet::new();
    for receipt in &bundle.receipts {
        if let Some(signature) = &receipt.signature {
            signers.insert(signature.key_id.clone());
        }
    }
    signers.into_iter().collect()
}

fn policy_name(policy: TrustPolicy) -> &'static str {
    policy.as_str()
}

fn parse_policy(value: &str) -> Option<TrustPolicy> {
    match value {
        "integrity_only" => Some(TrustPolicy::IntegrityOnly),
        "authorized_publisher" => Some(TrustPolicy::AuthorizedPublisher),
        "authorized_current" => Some(TrustPolicy::AuthorizedCurrent),
        "authorized_current_witnessed" => Some(TrustPolicy::AuthorizedCurrentWitnessed),
        _ => None,
    }
}

fn validate_extensions(extensions: &BTreeMap<String, serde_json::Value>) -> bool {
    extensions.keys().all(|key| key.starts_with("x-"))
}

fn validate_time_order(started: &str, completed: &str) -> Result<bool> {
    Ok(parse_utc_timestamp(started)? <= parse_utc_timestamp(completed)?)
}

/// Create an unsigned run statement from verified evidence and raw output.
///
/// Digests, counts, roots, and release coordinates are derived here. A caller
/// cannot supply an output digest or receipt-set digest independently.
pub fn create_run_attestation(input: CreateRunAttestationInput<'_>) -> Result<RunStatement> {
    if input.metadata.run_id != input.run_bundle.run_id {
        return Err(AnnpackError::InvalidInput(
            "execution run_id contradicts the run bundle".into(),
        ));
    }
    if input.metadata.run_id.is_empty()
        || input.metadata.workload_identity.is_empty()
        || input.metadata.model_identifier.is_empty()
        || input.metadata.application_identity.is_empty()
        || input.metadata.application_version.is_empty()
        || input.metadata.retrieval_policy_revision.is_empty()
        || input.metadata.tool_policy_revision.is_empty()
    {
        return Err(AnnpackError::InvalidInput(
            "required execution, workload, model, application, or policy identity is empty".into(),
        ));
    }
    if !validate_time_order(&input.metadata.started_at, &input.metadata.completed_at)? {
        return Err(AnnpackError::InvalidInput(
            "execution completion precedes execution start".into(),
        ));
    }
    if input.run_bundle.receipts.is_empty() && input.empty_receipts == EmptyReceiptPolicy::Deny {
        return Err(AnnpackError::InvalidInput(
            "empty receipt set requires explicit allow-empty policy".into(),
        ));
    }
    let bundle_report = verify_run_bundle(input.run_bundle, None)?;
    if !input.run_bundle.receipts.is_empty() && !bundle_report.attested {
        return Err(AnnpackError::Integrity(
            "run bundle contains an invalid receipt".into(),
        ));
    }
    for (index, receipt) in input.run_bundle.receipts.iter().enumerate() {
        let verification = crate::evidence::verify_receipt(receipt, None)?;
        if !verification.verified || !verification.signature_valid {
            return Err(AnnpackError::Integrity(format!(
                "receipt {index} is not both valid and publisher-signed"
            )));
        }
    }
    let receipts = canonical_receipts(input.run_bundle)?;
    let artifact_roots: Vec<String> = receipts
        .iter()
        .map(|receipt| receipt.artifact_root.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if artifact_roots.len() > 1 {
        return Err(AnnpackError::InvalidInput(
            "this run policy disallows receipts from conflicting artifact roots".into(),
        ));
    }
    if !input.channel_verification.verified {
        return Err(AnnpackError::Integrity(
            "supplied channel-state verification is not valid".into(),
        ));
    }
    let release_digest = statement_digest(input.channel_state)?;
    if release_digest != input.channel_verification.statement_digest {
        return Err(AnnpackError::Integrity(
            "channel-state bytes contradict their verification report".into(),
        ));
    }
    let root = artifact_roots
        .first()
        .cloned()
        .unwrap_or_else(|| input.channel_state.current.artifact_root.to_lowercase());
    let currency = currency_for_root(input.channel_state, input.channel_verification, &root);
    let signers = artifact_signers(input.run_bundle);
    let policy = evaluate_policy(
        &PolicyInputs {
            artifact_root: &root,
            artifact_integrity: ArtifactIntegrity::Valid,
            artifact_signers: &signers,
            trust: Some(input.publisher_trust),
            channel_state: Some(input.channel_verification),
            currency,
            transparency: TransparencyEvidence::Unavailable,
        },
        input.trust_policy,
    );
    if !policy.permitted {
        return Err(AnnpackError::Integrity(format!(
            "runtime trust policy was not met: {}",
            policy.unmet_requirements.join("; ")
        )));
    }
    if input.publisher_trust.publisher != input.channel_verification.publisher {
        return Err(AnnpackError::Integrity(
            "publisher trust and channel state name different publishers".into(),
        ));
    }
    if let Some(bundle_model) = &input.run_bundle.model
        && bundle_model != &input.metadata.model_identifier
    {
        return Err(AnnpackError::InvalidInput(
            "model identifier contradicts the run bundle".into(),
        ));
    }
    let output_digest = sha256_hex(input.output);
    Ok(RunStatement {
        statement_type: IN_TOTO_STATEMENT_TYPE.into(),
        subject: vec![Subject {
            name: "agent-output".into(),
            digest: SubjectDigest {
                sha256: output_digest,
            },
        }],
        predicate_type: RUN_ATTESTATION_PREDICATE_TYPE.into(),
        predicate: RunPredicate {
            schema: RUN_ATTESTATION_SCHEMA_V1.into(),
            execution: ExecutionClaim {
                run_id: input.metadata.run_id,
                trace_id: input.metadata.trace_id,
                workload_identity: input.metadata.workload_identity,
                started_at: input.metadata.started_at,
                completed_at: input.metadata.completed_at,
                signing_time: None,
            },
            knowledge: KnowledgeClaim {
                run_bundle_digest: canonical_bundle_digest(input.run_bundle, &receipts)?,
                receipt_count: receipts.len(),
                no_passages_retrieved: receipts.is_empty(),
                receipts,
                artifact_roots,
                publisher: input.channel_verification.publisher.clone(),
                corpus: input.channel_verification.corpus.clone(),
                channel: input.channel_verification.channel.clone(),
                channel_state_digest: DigestValue {
                    algorithm: "blake3".into(),
                    value: release_digest,
                },
                channel_state_sequence: input.channel_verification.sequence,
                observed_currency: currency,
                trust_policy: policy_name(input.trust_policy).into(),
            },
            retrieval: RetrievalClaim {
                query_digest: DigestValue::sha256(input.run_bundle.query.as_bytes()),
                retrieval_policy_revision: input.metadata.retrieval_policy_revision,
                retrieval_mode: input.metadata.retrieval_mode,
            },
            application: ApplicationClaim {
                application_identity: input.metadata.application_identity,
                application_version: input.metadata.application_version,
                model_identifier: input.metadata.model_identifier,
                model_provider: input.metadata.model_provider,
                prompt_policy_digest: DigestValue::sha256(&input.metadata.prompt_policy_bytes),
                tool_policy_revision: input.metadata.tool_policy_revision,
                deployment_identity: input.metadata.deployment_identity,
            },
            extensions: BTreeMap::new(),
        },
    })
}

#[cfg(feature = "signing")]
pub fn sign_run_attestation(statement: &RunStatement, secret_key: &[u8; 32]) -> Result<Envelope> {
    use ed25519_dalek::{Signer, SigningKey};

    let payload = serde_json::to_vec(statement)?;
    let key = SigningKey::from_bytes(secret_key);
    let (keyid, _) = crate::trust::key_identity(secret_key);
    let signature = key.sign(&pae(DSSE_PAYLOAD_TYPE, &payload));
    Ok(Envelope {
        payload: b64_encode(&payload),
        payload_type: DSSE_PAYLOAD_TYPE.into(),
        signatures: vec![DsseSignature {
            keyid,
            sig: hex::encode(signature.to_bytes()),
        }],
    })
}

#[derive(Debug, Clone)]
pub struct WorkloadKey {
    pub public_key: String,
    pub identity: String,
    pub trusted: bool,
}

/// Result supplied by an external workload-authentication adapter (for example
/// Sigstore). It is accepted only for the exact in-toto payload digest and does
/// not inherit builder or publisher authority.
#[derive(Debug, Clone)]
pub struct ExternalWorkloadVerification {
    pub payload_sha256: String,
    pub envelope_signature_verified: bool,
    pub identity: String,
    pub trusted: bool,
    pub signer_key_ids: Vec<String>,
    pub trusted_signing_time: Option<String>,
    pub externally_anchored: bool,
}

#[derive(Debug, Clone)]
pub struct RunExpectations {
    pub run_id: String,
    pub trace_id: Option<String>,
    pub model_identifier: String,
    pub prompt_policy_sha256: String,
}

pub struct VerifyRunAttestationInput<'a> {
    pub envelope: &'a Envelope,
    pub run_bundle: &'a RunBundle,
    pub bound_channel_state: &'a ChannelState,
    pub bound_channel_verification: &'a ChannelStateVerification,
    pub publisher_trust: &'a TrustRootVerification,
    pub workload_keys: &'a [WorkloadKey],
    pub external_workload: Option<&'a ExternalWorkloadVerification>,
    pub expectations: &'a RunExpectations,
    pub output: Option<&'a [u8]>,
    pub require_output: bool,
    pub current_channel_state: Option<(&'a ChannelState, &'a ChannelStateVerification)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunAttestationVerification {
    pub statement_digest: String,
    pub envelope_signature: VerificationStatus,
    pub workload_identity: VerificationStatus,
    pub run_identity: VerificationStatus,
    pub receipt_set_binding: VerificationStatus,
    pub receipt_verification: VerificationStatus,
    pub artifact_root_binding: VerificationStatus,
    pub publisher_authority: VerificationStatus,
    pub channel_state_binding: VerificationStatus,
    pub currency: VerificationStatus,
    pub currency_at_evaluation: Currency,
    pub present_use_permitted: bool,
    pub runtime_policy: VerificationStatus,
    pub query_digest_binding: VerificationStatus,
    pub model_identity: VerificationStatus,
    pub prompt_policy_binding: VerificationStatus,
    pub output_digest_binding: VerificationStatus,
    pub execution_time: VerificationStatus,
    pub cryptographic_signing_time: VerificationStatus,
    pub overall_occurrence_evidence: bool,
    pub occurrence_strength: OccurrenceStrength,
    pub signer_key_ids: Vec<String>,
    pub issues: Vec<String>,
}

fn status(matches: bool) -> VerificationStatus {
    if matches {
        VerificationStatus::Verified
    } else {
        VerificationStatus::Mismatched
    }
}

pub fn verify_run_attestation(
    input: VerifyRunAttestationInput<'_>,
) -> Result<RunAttestationVerification> {
    let mut issues = Vec::new();
    if input.envelope.payload_type != DSSE_PAYLOAD_TYPE {
        return Err(AnnpackError::Unsupported(
            "run attestation uses an unsupported DSSE payload type".into(),
        ));
    }
    let payload = b64_decode(&input.envelope.payload, MAX_RUN_ATTESTATION_BYTES)?;
    let statement: RunStatement = serde_json::from_slice(&payload)?;
    let attestation_digest = sha256_hex(&payload);
    let predicate_supported = statement.statement_type == IN_TOTO_STATEMENT_TYPE
        && statement.predicate_type == RUN_ATTESTATION_PREDICATE_TYPE
        && statement.predicate.schema == RUN_ATTESTATION_SCHEMA_V1
        && validate_extensions(&statement.predicate.extensions);
    if !predicate_supported {
        issues.push("unsupported statement, predicate, schema, or extension key".into());
    }

    let mut valid_keys = Vec::new();
    let mut trusted_identity = false;
    for key in input.workload_keys {
        if let Some(keyid) = check_signer(input.envelope, &payload, &key.public_key) {
            valid_keys.push(keyid);
            if key.trusted && key.identity == statement.predicate.execution.workload_identity {
                trusted_identity = true;
            }
        }
    }
    let external_valid = input.external_workload.is_some_and(|external| {
        external.envelope_signature_verified && external.payload_sha256 == attestation_digest
    });
    if let Some(external) = input.external_workload.filter(|_| external_valid) {
        valid_keys.extend(external.signer_key_ids.iter().cloned());
        if external.trusted && external.identity == statement.predicate.execution.workload_identity
        {
            trusted_identity = true;
        }
    }
    valid_keys.sort();
    valid_keys.dedup();
    let envelope_signature = if valid_keys.is_empty() && !external_valid {
        issues.push("no DSSE signature validated against a supplied workload candidate".into());
        VerificationStatus::Invalid
    } else {
        VerificationStatus::Verified
    };
    let workload_identity = if trusted_identity {
        VerificationStatus::Verified
    } else if valid_keys.is_empty() && !external_valid {
        VerificationStatus::Unknown
    } else {
        issues.push("a valid workload signature exists, but its identity is not trusted".into());
        VerificationStatus::Untrusted
    };

    let run_identity = status(
        statement.predicate.execution.run_id == input.expectations.run_id
            && statement.predicate.execution.run_id == input.run_bundle.run_id
            && statement.predicate.execution.trace_id == input.expectations.trace_id,
    );
    if run_identity != VerificationStatus::Verified {
        issues.push("run_id or trace_id does not match the expected execution".into());
    }

    let canonical = canonical_receipts(input.run_bundle);
    let (receipt_set_binding, receipt_verification, artifact_roots, signers) = match canonical {
        Err(error) => {
            issues.push(error.to_string());
            (
                VerificationStatus::Invalid,
                VerificationStatus::Invalid,
                Vec::new(),
                Vec::new(),
            )
        }
        Ok(receipts) => {
            let bundle_digest = canonical_bundle_digest(input.run_bundle, &receipts)?;
            let exact = statement.predicate.knowledge.receipts == receipts
                && statement.predicate.knowledge.receipt_count == receipts.len()
                && statement.predicate.knowledge.no_passages_retrieved == receipts.is_empty()
                && statement.predicate.knowledge.run_bundle_digest == bundle_digest;
            if !exact {
                issues.push("the supplied run bundle is not the exact signed receipt set".into());
            }
            let receipt_set_status = if exact {
                VerificationStatus::Verified
            } else if receipts.len() < statement.predicate.knowledge.receipt_count {
                VerificationStatus::Missing
            } else {
                VerificationStatus::Mismatched
            };
            let mut every_receipt = true;
            for receipt in &input.run_bundle.receipts {
                match crate::evidence::verify_receipt(receipt, None) {
                    Ok(report) if report.verified && report.signature_valid => {}
                    Ok(_) | Err(_) => every_receipt = false,
                }
            }
            if !every_receipt {
                issues.push("one or more bound receipts failed independent verification".into());
            }
            let roots = receipts
                .iter()
                .map(|entry| entry.artifact_root.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            (
                receipt_set_status,
                status(every_receipt),
                roots,
                artifact_signers(input.run_bundle),
            )
        }
    };

    let artifact_root_binding = status(
        artifact_roots == statement.predicate.knowledge.artifact_roots && artifact_roots.len() <= 1,
    );
    if artifact_root_binding != VerificationStatus::Verified {
        issues.push("artifact-root set is inconsistent or does not match".into());
    }
    let authorized = input.publisher_trust.verified
        && input
            .publisher_trust
            .authorized_roles
            .get(ROLE_ARTIFACT)
            .is_some_and(|keys| signers.iter().any(|signer| keys.contains(signer)))
        && input.publisher_trust.publisher == statement.predicate.knowledge.publisher;
    let publisher_authority = if authorized {
        VerificationStatus::Verified
    } else if input.publisher_trust.verified {
        VerificationStatus::Untrusted
    } else {
        VerificationStatus::Unknown
    };
    if publisher_authority != VerificationStatus::Verified {
        issues.push("receipts were not signed by an authorized publisher role".into());
    }

    let bound_digest = statement_digest(input.bound_channel_state)?;
    let channel_state_binding = status(
        input.bound_channel_verification.verified
            && input.bound_channel_verification.statement_digest == bound_digest
            && statement.predicate.knowledge.channel_state_digest.algorithm == "blake3"
            && statement.predicate.knowledge.channel_state_digest.value == bound_digest
            && statement.predicate.knowledge.channel_state_sequence
                == input.bound_channel_verification.sequence
            && statement.predicate.knowledge.publisher
                == input.bound_channel_verification.publisher
            && statement.predicate.knowledge.corpus == input.bound_channel_verification.corpus
            && statement.predicate.knowledge.channel == input.bound_channel_verification.channel,
    );
    if channel_state_binding != VerificationStatus::Verified {
        issues.push("bound channel-state statement or scope does not match".into());
    }

    let root = artifact_roots.first().cloned().unwrap_or_else(|| {
        input
            .bound_channel_state
            .current
            .artifact_root
            .to_lowercase()
    });
    let bound_currency = currency_for_root(
        input.bound_channel_state,
        input.bound_channel_verification,
        &root,
    );
    let currency = status(bound_currency == statement.predicate.knowledge.observed_currency);
    if currency != VerificationStatus::Verified {
        issues.push("recorded currency contradicts the bound release statement".into());
    }
    let Some(policy) = parse_policy(&statement.predicate.knowledge.trust_policy) else {
        issues.push("unknown runtime trust policy".into());
        return Ok(RunAttestationVerification {
            statement_digest: attestation_digest,
            envelope_signature,
            workload_identity,
            run_identity,
            receipt_set_binding,
            receipt_verification,
            artifact_root_binding,
            publisher_authority,
            channel_state_binding,
            currency,
            currency_at_evaluation: Currency::Unknown,
            present_use_permitted: false,
            runtime_policy: VerificationStatus::Invalid,
            query_digest_binding: VerificationStatus::NotEvaluated,
            model_identity: VerificationStatus::NotEvaluated,
            prompt_policy_binding: VerificationStatus::NotEvaluated,
            output_digest_binding: VerificationStatus::NotEvaluated,
            execution_time: VerificationStatus::NotEvaluated,
            cryptographic_signing_time: VerificationStatus::Unknown,
            overall_occurrence_evidence: false,
            occurrence_strength: OccurrenceStrength::Invalid,
            signer_key_ids: valid_keys,
            issues,
        });
    };
    let decision = evaluate_policy(
        &PolicyInputs {
            artifact_root: &root,
            artifact_integrity: if receipt_verification == VerificationStatus::Verified {
                ArtifactIntegrity::Valid
            } else {
                ArtifactIntegrity::Invalid
            },
            artifact_signers: &signers,
            trust: Some(input.publisher_trust),
            channel_state: Some(input.bound_channel_verification),
            currency: bound_currency,
            transparency: TransparencyEvidence::Unavailable,
        },
        policy,
    );
    let runtime_policy = status(decision.permitted);
    if !decision.permitted {
        issues.push(format!(
            "recorded runtime policy is not met: {}",
            decision.unmet_requirements.join("; ")
        ));
    }

    let query_digest_binding = status(
        statement.predicate.retrieval.query_digest.algorithm == "sha256"
            && statement.predicate.retrieval.query_digest.value
                == sha256_hex(input.run_bundle.query.as_bytes()),
    );
    let model_identity = status(
        statement.predicate.application.model_identifier == input.expectations.model_identifier
            && input
                .run_bundle
                .model
                .as_ref()
                .is_none_or(|model| model == &input.expectations.model_identifier),
    );
    let prompt_policy_binding = status(
        statement
            .predicate
            .application
            .prompt_policy_digest
            .algorithm
            == "sha256"
            && statement.predicate.application.prompt_policy_digest.value
                == input.expectations.prompt_policy_sha256,
    );
    let output_digest_binding = match input.output {
        Some(output) => status(
            statement.subject.len() == 1
                && statement.subject[0].name == "agent-output"
                && statement.subject[0].digest.sha256 == sha256_hex(output),
        ),
        None if input.require_output => VerificationStatus::Missing,
        None => VerificationStatus::NotEvaluated,
    };
    if input.require_output && output_digest_binding != VerificationStatus::Verified {
        issues.push("required output bytes are missing or do not match the subject".into());
    }
    let execution_time = match validate_time_order(
        &statement.predicate.execution.started_at,
        &statement.predicate.execution.completed_at,
    ) {
        Ok(true) => VerificationStatus::Carried,
        Ok(false) | Err(_) => {
            issues.push("execution completion precedes start or time is malformed".into());
            VerificationStatus::Invalid
        }
    };
    let external_signing_time = input
        .external_workload
        .filter(|_| external_valid)
        .and_then(|external| external.trusted_signing_time.as_ref());
    let cryptographic_signing_time = match external_signing_time {
        Some(signing_time) => match (
            parse_utc_timestamp(&statement.predicate.execution.completed_at),
            parse_utc_timestamp(signing_time),
        ) {
            (Ok(completed), Ok(signed)) if completed <= signed => VerificationStatus::Verified,
            _ => {
                issues.push("trusted signing time precedes completion or is malformed".into());
                VerificationStatus::Invalid
            }
        },
        None => match &statement.predicate.execution.signing_time {
            None => VerificationStatus::Unknown,
            Some(signing_time) => match (
                parse_utc_timestamp(&statement.predicate.execution.completed_at),
                parse_utc_timestamp(signing_time),
            ) {
                (Ok(completed), Ok(signed)) if completed <= signed => VerificationStatus::Carried,
                _ => {
                    issues.push(
                    "claimed signing time precedes completion or is malformed; no trusted time was established"
                        .into(),
                );
                    VerificationStatus::Invalid
                }
            },
        },
    };

    let currency_at_evaluation = input
        .current_channel_state
        .map(|(state, verification)| currency_for_root(state, verification, &root))
        .unwrap_or(bound_currency);
    let present_use_permitted = currency_at_evaluation == Currency::Current;
    let output_ok = output_digest_binding == VerificationStatus::Verified
        || (!input.require_output && output_digest_binding == VerificationStatus::NotEvaluated);
    let overall_occurrence_evidence = predicate_supported
        && envelope_signature == VerificationStatus::Verified
        && workload_identity == VerificationStatus::Verified
        && run_identity == VerificationStatus::Verified
        && receipt_set_binding == VerificationStatus::Verified
        && receipt_verification == VerificationStatus::Verified
        && artifact_root_binding == VerificationStatus::Verified
        && publisher_authority == VerificationStatus::Verified
        && channel_state_binding == VerificationStatus::Verified
        && currency == VerificationStatus::Verified
        && runtime_policy == VerificationStatus::Verified
        && query_digest_binding == VerificationStatus::Verified
        && model_identity == VerificationStatus::Verified
        && prompt_policy_binding == VerificationStatus::Verified
        && output_ok
        && execution_time == VerificationStatus::Carried
        && cryptographic_signing_time != VerificationStatus::Invalid;
    let occurrence_strength = if overall_occurrence_evidence
        && input
            .external_workload
            .is_some_and(|external| external_valid && external.externally_anchored)
    {
        OccurrenceStrength::ExternallyAnchored
    } else if overall_occurrence_evidence
        && cryptographic_signing_time == VerificationStatus::Verified
    {
        OccurrenceStrength::WorkloadAttestedWithTrustedTime
    } else if overall_occurrence_evidence {
        OccurrenceStrength::WorkloadAttested
    } else if envelope_signature == VerificationStatus::Verified {
        OccurrenceStrength::Invalid
    } else {
        OccurrenceStrength::Unattested
    };
    Ok(RunAttestationVerification {
        statement_digest: attestation_digest,
        envelope_signature,
        workload_identity,
        run_identity,
        receipt_set_binding,
        receipt_verification,
        artifact_root_binding,
        publisher_authority,
        channel_state_binding,
        currency,
        currency_at_evaluation,
        present_use_permitted,
        runtime_policy,
        query_digest_binding,
        model_identity,
        prompt_policy_binding,
        output_digest_binding,
        execution_time,
        cryptographic_signing_time,
        overall_occurrence_evidence,
        occurrence_strength,
        signer_key_ids: valid_keys,
        issues,
    })
}

/// OTel attributes are locators for the separately retained attestation, never
/// the evidence store itself. No query, prompt, passage, or output plaintext is
/// emitted.
pub fn run_attestation_telemetry(
    report: &RunAttestationVerification,
    statement: &RunStatement,
    uri: Option<&str>,
) -> BTreeMap<String, serde_json::Value> {
    let mut values = BTreeMap::new();
    values.insert(
        "annpack.run.attestation_digest".into(),
        report.statement_digest.clone().into(),
    );
    if let Some(uri) = uri {
        values.insert("annpack.run.attestation_uri".into(), uri.into());
    }
    values.insert(
        "annpack.run.id".into(),
        statement.predicate.execution.run_id.clone().into(),
    );
    values.insert(
        "annpack.run.receipt_count".into(),
        (statement.predicate.knowledge.receipt_count as u64).into(),
    );
    values.insert(
        "annpack.release.statement_digest".into(),
        statement
            .predicate
            .knowledge
            .channel_state_digest
            .value
            .clone()
            .into(),
    );
    values.insert(
        "annpack.release.sequence".into(),
        statement.predicate.knowledge.channel_state_sequence.into(),
    );
    values.insert(
        "annpack.currency.status".into(),
        serde_json::to_value(report.currency_at_evaluation).unwrap_or_default(),
    );
    values.insert(
        "annpack.workload.identity".into(),
        statement
            .predicate
            .execution
            .workload_identity
            .clone()
            .into(),
    );
    values.insert(
        "annpack.output.digest".into(),
        statement.subject[0].digest.sha256.clone().into(),
    );
    values
}
