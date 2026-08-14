//! Run bundles: the retrieval evidence for one agent run, in one portable file.
//!
//! A bundle is an envelope, not a second proof system. It carries EVIDENCE-v1
//! receipts verbatim and adds no cryptography of its own: verifying a bundle is
//! exactly [`crate::evidence::verify_receipt`] applied to each receipt in turn.
//! Nothing here can succeed that the receipt verifier would reject, and nothing
//! here has to be re-implemented by a reader that already verifies receipts.
//!
//! The reason to draw the boundary that hard is that the interesting failure is
//! not a forged receipt — it is a reader concluding more than the receipts say.
//! A bundle carries a query, a model id and an answer because an incident
//! responder needs them to locate the run, but none of those are attested by
//! anything. So the verifier reports two separate facts:
//!
//! - **attested**: every receipt proved its passage existed unmodified in a
//!   named immutable artifact at a named source revision.
//! - **carried**: the query, application, model and answer travelled with the
//!   receipts and are attested by nothing.
//!
//! `answer_hash` is a digest of the carried answer bytes, present so a bundle
//! can be correlated with an application's own logs. It is checked only for
//! internal consistency. Anyone who can edit the answer can edit its digest, so
//! a match proves the bundle was not corrupted in transit and nothing more.

use serde::{Deserialize, Serialize};

use crate::error::{AdyarError, Result};
use crate::evidence::{EvidenceReceipt, ReceiptVerification, verify_receipt};

// FROZEN WIRE IDENTIFIER: serialized and matched by third parties. It names a
// format version, not a project, and changes only when that version does.
pub const RUN_BUNDLE_SCHEMA_V1: &str = "annpack-run-bundle-v1";

/// Receipts one bundle may carry.
///
/// Each receipt embeds its artifact's Documents section, so a bundle's size
/// grows roughly linearly in the receipt count with a large constant. The cap
/// bounds verification work for a file that arrived from an untrusted party.
pub const MAX_BUNDLE_RECEIPTS: usize = 256;

/// Maximum bundle file size the reference CLI reads.
///
/// Bounds allocation before parsing, the same way
/// [`crate::evidence::MAX_RECEIPT_FILE_BYTES`] does for a single receipt.
pub const MAX_BUNDLE_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// Maximum answer the reference CLI will carry into a bundle.
pub const MAX_BUNDLE_ANSWER_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBundle {
    pub schema: String,
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub application: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_hash: Option<String>,
    pub receipts: Vec<EvidenceReceipt>,
}

/// BLAKE3 of the carried answer bytes. Not domain-separated, because this is
/// not a proof primitive and should not be mistaken for one.
pub fn answer_hash(answer: &str) -> String {
    blake3::hash(answer.as_bytes()).to_hex().to_string()
}

/// Deterministic run identifier over the retrieval that produced the bundle.
///
/// Two bundles built from the same query against the same artifact get the same
/// id, which makes a bundle reproducible from its inputs — the same property
/// artifacts themselves have. That also means it does not identify an
/// *occurrence*: an application correlating a bundle with one specific run
/// should supply its own id instead.
pub fn derive_run_id(query: &str, receipts: &[EvidenceReceipt]) -> String {
    let mut hasher = blake3::Hasher::new();
    // Length-prefixed so that concatenation is unambiguous: without this,
    // ("ab", "c") and ("a", "bc") would hash identically.
    let mut field = |bytes: &[u8]| {
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    field(query.as_bytes());
    for receipt in receipts {
        field(receipt.pack_root.as_bytes());
        field(receipt.passage_id.as_bytes());
    }
    format!("retrieval:{}", &hasher.finalize().to_hex()[..32])
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptOutcome {
    pub index: usize,
    pub passage_id: String,
    pub pack: String,
    pub pack_root: String,
    pub verification: ReceiptVerification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBundleVerification {
    pub run_id: String,
    pub query: String,
    pub receipts_total: usize,
    pub receipts_verified: usize,
    /// Distinct artifact roots the receipts resolve to, in first-seen order. A
    /// run that read from more than one artifact yields more than one root.
    pub pack_roots: Vec<String>,
    /// Distinct source revisions the authenticated manifests declare.
    pub source_revisions: Vec<String>,
    /// Whether every receipt both verified and carried a valid signature.
    ///
    /// The conjunction is deliberate. A receipt's signature covers the artifact
    /// root, not the passage, so a receipt whose passage record has been
    /// rewritten still carries a perfectly valid signature. Reporting such a
    /// bundle as "all signed" would invite a reader to take authenticity from a
    /// file that attests nothing. Signature status is still reported per receipt
    /// for anyone who needs to distinguish the two.
    pub all_receipts_signed: bool,
    /// Whether every receipt verified and was signed by the supplied trusted
    /// key. False when no trusted key was supplied.
    pub all_signers_trusted: bool,
    /// `Some(true)`/`Some(false)` when both `answer` and `answer_hash` are
    /// present; `None` when the bundle carries neither or only one. Internal
    /// consistency only — see the module documentation.
    pub answer_hash_consistent: Option<bool>,
    pub receipts: Vec<ReceiptOutcome>,
    /// Every receipt verified, and there was at least one. A bundle carrying no
    /// receipts attests nothing, so it is never reported as attested.
    pub attested: bool,
    pub issues: Vec<String>,
}

/// A receipt the verifier could not even parse far enough to judge.
///
/// `verify_receipt` returns `Err` for a structurally broken receipt rather than
/// a report of failures. Inside a bundle that must not abort the other
/// receipts, so the error becomes a failed outcome for that entry alone.
fn rejected(reason: String) -> ReceiptVerification {
    ReceiptVerification {
        passage_hash_matches: false,
        inclusion_proof_valid: false,
        manifest_commits_merkle_root: false,
        manifest_matches_directory: false,
        directory_matches_pack_root: false,
        passage_metadata_matches: false,
        source_revision_matches: false,
        pack_matches: false,
        canonical_url_matches: false,
        signature_valid: false,
        identity_trusted: false,
        verified: false,
        issues: vec![reason],
    }
}

pub fn verify_run_bundle(
    bundle: &RunBundle,
    trusted_public_key: Option<&str>,
) -> Result<RunBundleVerification> {
    if bundle.schema != RUN_BUNDLE_SCHEMA_V1 {
        return Err(AdyarError::Unsupported(format!(
            "run bundle schema {:?}; this verifier supports {RUN_BUNDLE_SCHEMA_V1}",
            bundle.schema
        )));
    }
    if bundle.receipts.len() > MAX_BUNDLE_RECEIPTS {
        return Err(AdyarError::InvalidFormat(format!(
            "run bundle carries {} receipts, above the {MAX_BUNDLE_RECEIPTS} limit",
            bundle.receipts.len()
        )));
    }

    let mut issues = Vec::new();
    let mut outcomes = Vec::with_capacity(bundle.receipts.len());
    let mut pack_roots: Vec<String> = Vec::new();
    let mut source_revisions: Vec<String> = Vec::new();
    let mut receipts_verified = 0;
    let mut all_receipts_signed = true;
    let mut all_signers_trusted = true;

    for (index, receipt) in bundle.receipts.iter().enumerate() {
        let verification = match verify_receipt(receipt, trusted_public_key) {
            Ok(report) => report,
            Err(error) => rejected(error.to_string()),
        };
        if verification.verified {
            receipts_verified += 1;
            // Only an authenticated receipt may contribute to the artifact and
            // revision sets. A failed receipt's self-declared root is just a
            // string the sender chose.
            if !pack_roots.contains(&receipt.pack_root) {
                pack_roots.push(receipt.pack_root.clone());
            }
            if let Some(revision) = &receipt.source_revision
                && !source_revisions.contains(revision)
            {
                source_revisions.push(revision.clone());
            }
        } else {
            issues.push(format!(
                "receipt {index} for passage {} did not verify: {}",
                receipt.passage_id,
                verification.issues.join("; ")
            ));
        }
        // Both aggregates are conditioned on the receipt having verified; see
        // the field documentation for why a failed receipt's valid signature
        // must not count toward them.
        if !(verification.verified && verification.signature_valid) {
            all_receipts_signed = false;
        }
        if !(verification.verified && verification.identity_trusted) {
            all_signers_trusted = false;
        }
        outcomes.push(ReceiptOutcome {
            index,
            passage_id: receipt.passage_id.clone(),
            pack: receipt.pack.clone(),
            pack_root: receipt.pack_root.clone(),
            verification,
        });
    }

    let answer_hash_consistent = match (&bundle.answer, &bundle.answer_hash) {
        (Some(answer), Some(declared)) => {
            let matches = answer_hash(answer).eq_ignore_ascii_case(declared);
            if !matches {
                issues.push("answer_hash does not match the carried answer".into());
            }
            Some(matches)
        }
        (Some(_), None) | (None, Some(_)) => None,
        (None, None) => None,
    };

    if bundle.receipts.is_empty() {
        issues.push("run bundle carries no receipts, so it attests nothing".into());
    }
    let attested = !bundle.receipts.is_empty() && receipts_verified == bundle.receipts.len();

    Ok(RunBundleVerification {
        run_id: bundle.run_id.clone(),
        query: bundle.query.clone(),
        receipts_total: bundle.receipts.len(),
        receipts_verified,
        pack_roots,
        source_revisions,
        // An empty bundle would otherwise satisfy both `all()` conditions
        // vacuously and be reported as fully signed and fully trusted.
        all_receipts_signed: all_receipts_signed && !bundle.receipts.is_empty(),
        all_signers_trusted: all_signers_trusted
            && !bundle.receipts.is_empty()
            && trusted_public_key.is_some(),
        answer_hash_consistent,
        receipts: outcomes,
        attested,
        issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(pack_root: &str, passage_id: &str) -> EvidenceReceipt {
        EvidenceReceipt {
            schema: "annpack-receipt-v2".into(),
            pack: "demo@1.0.0".into(),
            pack_root: pack_root.into(),
            passage_merkle_root: String::new(),
            source_revision: None,
            passage_id: passage_id.into(),
            passage_hash: String::new(),
            passage_ordinal: 0,
            canonical_url: None,
            passage_record_b64: String::new(),
            inclusion_proof: Vec::new(),
            manifest_bytes_b64: String::new(),
            directory_b64: String::new(),
            manifest_section_id: 1,
            documents_section_id: None,
            documents_bytes_b64: None,
            signature: None,
        }
    }

    fn bundle(receipts: Vec<EvidenceReceipt>) -> RunBundle {
        RunBundle {
            schema: RUN_BUNDLE_SCHEMA_V1.into(),
            run_id: "test".into(),
            created_at: None,
            application: None,
            model: None,
            query: "query".into(),
            answer: None,
            answer_hash: None,
            receipts,
        }
    }

    #[test]
    fn an_unknown_schema_is_refused() {
        let mut input = bundle(Vec::new());
        input.schema = "annpack-run-bundle-v2".into();
        assert!(verify_run_bundle(&input, None).is_err());
    }

    #[test]
    fn a_bundle_with_no_receipts_attests_nothing() {
        let report = verify_run_bundle(&bundle(Vec::new()), None).unwrap();
        assert!(!report.attested);
        // The vacuous-truth cases: zero receipts must not report as fully
        // signed or fully trusted just because no receipt failed the check.
        assert!(!report.all_receipts_signed);
        assert!(!report.all_signers_trusted);
        assert_eq!(report.receipts_verified, 0);
    }

    #[test]
    fn a_broken_receipt_fails_only_itself() {
        let report = verify_run_bundle(&bundle(vec![receipt("00", "p1")]), None).unwrap();
        assert_eq!(report.receipts_total, 1);
        assert_eq!(report.receipts_verified, 0);
        assert!(!report.attested);
        assert!(!report.receipts[0].verification.verified);
        // A receipt that failed must not contribute its self-declared root.
        assert!(report.pack_roots.is_empty());
    }

    #[test]
    fn receipts_above_the_cap_are_refused_before_verification() {
        let receipts = (0..=MAX_BUNDLE_RECEIPTS)
            .map(|index| receipt("00", &format!("p{index}")))
            .collect();
        assert!(verify_run_bundle(&bundle(receipts), None).is_err());
    }

    #[test]
    fn answer_hash_is_checked_for_consistency_only() {
        let mut input = bundle(Vec::new());
        input.answer = Some("an answer".into());
        input.answer_hash = Some(answer_hash("an answer"));
        assert_eq!(
            verify_run_bundle(&input, None)
                .unwrap()
                .answer_hash_consistent,
            Some(true)
        );

        input.answer_hash = Some(answer_hash("a different answer"));
        let report = verify_run_bundle(&input, None).unwrap();
        assert_eq!(report.answer_hash_consistent, Some(false));
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("answer_hash"))
        );
    }

    #[test]
    fn an_answer_without_a_hash_is_neither_consistent_nor_inconsistent() {
        let mut input = bundle(Vec::new());
        input.answer = Some("an answer".into());
        assert_eq!(
            verify_run_bundle(&input, None)
                .unwrap()
                .answer_hash_consistent,
            None
        );
    }

    #[test]
    fn run_ids_separate_fields_unambiguously() {
        // Without length prefixing these two would hash identically.
        assert_ne!(
            derive_run_id("ab", &[receipt("c", "d")]),
            derive_run_id("a", &[receipt("bc", "d")])
        );
    }

    #[test]
    fn run_ids_are_reproducible_from_their_inputs() {
        let receipts = [receipt("root", "p1")];
        assert_eq!(
            derive_run_id("query", &receipts),
            derive_run_id("query", &receipts)
        );
        assert_ne!(
            derive_run_id("query", &receipts),
            derive_run_id("other", &receipts)
        );
    }
}
