#!/usr/bin/env python3
"""Prove the test suite fails when a security check is bypassed.

A passing suite says the code does what the tests describe. It does not say the
tests would notice if a check were removed -- and this repository has shipped
tests that would not have. `web/smoke-hybrid-parity.mjs` once passed against a
deliberately reverted implementation, and `every_corruption_artifact_is_rejected`
sat green under `--all-features` while the artifact root check was compiled out.

Each mutation below deletes exactly one security property. The suite named
alongside it must fail. A mutation that survives means the property is
unguarded, whatever the coverage report says.

Usage:
    python3 scripts/check-mutations.py [--filter SUBSTRING]

Mutations edit the working tree in place and are reverted afterwards. An earlier
version of this docstring claimed the revert survived interrupts; it did not --
a cancelled run left `trust.rs` with a security check replaced by `if false`,
and the next audit reported a stale anchor rather than a mutated tree. The run
now refuses to start unless the target files are clean in git, retains each
file's exact original contents for restoration, and handles SIGINT and SIGTERM.

The run also does not mutate the shared checkout at all. It edits and tests a
disposable `git worktree` checked out at `HEAD` instead, and removes that
worktree when it finishes, fails, or is interrupted. An earlier version
mutated files directly in this checkout; running it concurrently with an
unrelated `cargo clippy` invocation in the same directory caused this exact
process to observe one of the mutated-but-not-yet-restored files mid-run. The
mutation itself was working as designed -- the shared-checkout execution model
was not safe under concurrent use.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Mutation:
    name: str
    file: str
    find: str
    replace: str
    #: Test target that must fail. Kept narrow so a failure is attributable.
    tests: list[str]
    #: What real defect this mutation stands in for.
    stands_for: str
    features: list[str] = field(default_factory=list)


MUTATIONS = [
    Mutation(
        name="trust: any signature is treated as valid",
        file="rust/src/trust.rs",
        find="            .is_ok()\n        {",
        replace="            .is_ok()\n            || true\n        {",
        tests=["--test", "trust_root"],
        stands_for="a forged or tampered trust root being accepted",
    ),
    Mutation(
        name="trust: unauthorised keys may sign the root",
        file="rust/src/trust.rs",
        find="        if !permitted.contains(&(*key_id).to_string()) {",
        replace="        if false {",
        tests=["--test", "trust_root"],
        stands_for="role separation collapsing, so an artifact key can rotate the root",
    ),
    Mutation(
        name="trust: duplicate signatures count toward a threshold",
        file="rust/src/trust.rs",
        find="            signers.insert((*key_id).to_string());",
        replace="            signers.insert(format!(\"{}-{}\", key_id, signers.len()));",
        tests=["--test", "trust_root"],
        stands_for="one compromised key satisfying a multi-key threshold",
    ),
    Mutation(
        name="trust: an older root may replace a newer one",
        file="rust/src/trust.rs",
        find="            let advanced = root.version > prior.version;",
        replace="            let advanced = root.version >= prior.version;",
        tests=["--test", "trust_root"],
        stands_for="trust-root rollback to a revoked key set",
    ),
    Mutation(
        name="trust: unknown validity is treated as valid",
        file="rust/src/trust.rs",
        find="        && within_validity.unwrap_or(false)",
        replace="        && within_validity.unwrap_or(true)",
        tests=["--test", "trust_root", "--lib"],
        stands_for="an expired root accepted because no clock was available",
    ),
    Mutation(
        name="trust: the self-signature requirement is dropped",
        file="rust/src/trust.rs",
        find="        && self_signed\n",
        replace="",
        tests=["--test", "trust_root"],
        stands_for="an unsigned trust root verifying",
    ),
    Mutation(
        name="release: the revocation role gains promotion authority",
        file="rust/src/release.rs",
        find="            (SigningAuthority::RevocationOnly, signers)",
        replace="            (SigningAuthority::Full, signers)",
        tests=["--test", "channel_state"],
        stands_for="a compromised revocation key declaring any artifact current",
    ),
    Mutation(
        name="release: rollback and equivocation become acceptable",
        file="rust/src/release.rs",
        find="        SequenceVerdict::FirstContact | SequenceVerdict::Advanced | SequenceVerdict::Idempotent",
        replace="        SequenceVerdict::FirstContact\n            | SequenceVerdict::Advanced\n            | SequenceVerdict::Idempotent\n            | SequenceVerdict::Rollback\n            | SequenceVerdict::Equivocation",
        tests=["--test", "channel_state"],
        stands_for="replaying a superseded release, or accepting two conflicting statements",
    ),
    Mutation(
        name="release: an unverified statement yields a currency verdict",
        file="rust/src/release.rs",
        find="    if !verification.verified {\n        return Currency::Unknown;\n    }",
        replace="",
        tests=["--test", "channel_state"],
        stands_for="an unsigned or expired statement reporting an artifact as current",
    ),
    Mutation(
        name="release: unknown validity is treated as valid",
        file="rust/src/release.rs",
        find="        && within_validity.unwrap_or(false)",
        replace="        && within_validity.unwrap_or(true)",
        tests=["--test", "channel_state"],
        stands_for="an unbounded currency claim accepted with no clock",
    ),
    Mutation(
        name="release: scope mismatch stops blocking verification",
        file="rust/src/release.rs",
        # Mutating the computation, not one of the two terms that consume it.
        # Removing `&& scope_matches` alone survives, because a mismatch also
        # yields NotEvaluated and fails `sequence_acceptable` -- the property is
        # guarded twice on purpose, so only removing its source tests it.
        find="    let scope_matches = statement.publisher == publisher",
        replace="    let scope_matches = true || statement.publisher == publisher",
        tests=["--test", "channel_state", "--test", "release_lifecycle"],
        stands_for="cross-channel replay: a staging statement accepted for production",
    ),
    Mutation(
        name="release: retained state is consulted for a foreign scope",
        file="rust/src/release.rs",
        find="    let verdict = if scope_matches {",
        replace="    let verdict = if true {",
        tests=["--test", "release_lifecycle"],
        stands_for="a statement for another channel reading or overwriting this channel's state",
    ),
    Mutation(
        name="cli: policy denial stops being an exit failure",
        file="rust/src/main.rs",
        find="                return Err(policy_failure(&decision).with_details(value));",
        replace="                {}",
        tests=["--test", "release_lifecycle"],
        stands_for="a denied artifact reported only in stdout while the command exits zero",
    ),
    Mutation(
        name="policy: a stronger policy silently degrades to a weaker one",
        file="rust/src/policy.rs",
        find='            None => unmet.push("no channel-state statement was supplied".into()),',
        replace="            None => {}",
        tests=["--lib", "policy"],
        stands_for="withholding a statement weakening the consumer instead of denying",
    ),
    Mutation(
        name="policy: an unverified statement satisfies a currency requirement",
        file="rust/src/policy.rs",
        find="            Some(state) if !state.verified => {",
        replace="            Some(state) if false => {",
        tests=["--lib", "policy"],
        stands_for="an expired or unsigned statement being treated as authoritative",
    ),
    Mutation(
        name="policy: revocation stops being a security failure",
        file="rust/src/policy.rs",
        find="    if inputs.currency == Currency::Revoked {",
        replace="    if false {",
        tests=["--lib", "policy"],
        stands_for="a withdrawn artifact used because the caller asked a weaker question",
    ),
    Mutation(
        name="policy: the witnessed policy accepts missing transparency",
        file="rust/src/policy.rs",
        find="            TransparencyEvidence::Unavailable => unmet.push(",
        replace="            TransparencyEvidence::Unavailable => drop(",
        tests=["--lib", "policy"],
        stands_for="authorized_current_witnessed behaving as authorized_current",
    ),
    Mutation(
        name="policy: superseded releases are permitted",
        file="rust/src/policy.rs",
        find="            Currency::Superseded => unmet.push(\"a newer release supersedes this artifact\".into()),",
        replace="            Currency::Superseded => {}",
        tests=["--lib", "policy"],
        stands_for="an agent citing a release the publisher has already replaced",
    ),
    Mutation(
        name="build: a new artifact omits its source descriptor",
        file="rust/src/build.rs",
        find="        source: Some(crate::model::SourceDescriptor {",
        replace="        source: None.or(Some(crate::model::SourceDescriptor {",
        tests=["--test", "source_binding"],
        stands_for="provenance for a Markdown artifact reverting to a builder claim",
    ),
    Mutation(
        name="build: the authenticated digest is not the consumed-bytes digest",
        file="rust/src/build.rs",
        find="            digest: corpus.source_digest.clone(),",
        replace='            digest: "00".repeat(32),',
        tests=["--test", "source_binding"],
        stands_for="an artifact committing to a digest no source produced",
    ),
    Mutation(
        name="format: the source-descriptor requirement stops applying",
        file="rust/src/format.rs",
        find="    if format_version < 4 {",
        replace="    if format_version < 99 {",
        tests=["--test", "source_binding"],
        stands_for="a format 4 artifact with no source binding being accepted",
    ),
    Mutation(
        name="provenance: distributed file digest comparison is bypassed",
        file="rust/src/provenance.rs",
        find="Some(subject) if subject.digest.sha256 == actual_file_digest => BindingStatus::Verified,",
        replace="Some(_subject) => BindingStatus::Verified,",
        tests=["--test", "provenance"],
        stands_for="a distributed file that was tampered with after signing verifying anyway",
    ),
    Mutation(
        name="provenance: artifact root comparison is bypassed",
        file="rust/src/provenance.rs",
        find="let artifact_root_binding = if statement.predicate.annpack.artifact_root == actual_root {",
        replace="let artifact_root_binding = if true {",
        tests=["--test", "provenance"],
        stands_for="provenance naming the wrong artifact root verifying anyway",
    ),
    Mutation(
        name="provenance: source digest comparison is bypassed",
        file="rust/src/provenance.rs",
        find="Some(actual) if actual == statement.predicate.source.tree_digest => {",
        replace="Some(actual) if actual == actual => {",
        tests=["--test", "provenance"],
        stands_for="a format-4 artifact's own digest no longer being checked against the statement",
    ),
    Mutation(
        name="provenance: builder executable digest comparison is bypassed",
        file="rust/src/provenance.rs",
        find="Some(claimed) if *claimed == actual_digest => BindingStatus::Verified,",
        replace="Some(_claimed) => BindingStatus::Verified,",
        tests=["--test", "provenance"],
        stands_for="a swapped builder executable being reported as the one that built the artifact",
    ),
    Mutation(
        name="provenance: builder trust enforcement is bypassed",
        file="rust/src/provenance.rs",
        find="    } else if !valid_trusted.is_empty() {\n        BuilderIdentity::Trusted\n    } else {\n        BuilderIdentity::Untrusted\n    };",
        replace="    } else {\n        BuilderIdentity::Trusted\n    };",
        tests=["--test", "provenance"],
        stands_for="an untrusted builder's signature being treated as trusted",
    ),
    Mutation(
        name="provenance: unsupported predicate type is accepted",
        file="rust/src/provenance.rs",
        find="statement.statement_type == STATEMENT_TYPE && statement.predicate_type == PREDICATE_TYPE;",
        replace="true;",
        tests=["--test", "provenance"],
        stands_for="a statement under a future or foreign predicate being interpreted as a build claim",
    ),
    Mutation(
        name="provenance: creation skips the integrity gate",
        file="rust/src/provenance.rs",
        find="    reader.verify_all()?; // integrity gate before any claim is recorded",
        replace="    let _ = reader.verify_all();",
        tests=["--test", "provenance"],
        stands_for="signed provenance being issued for an artifact that is not even self-consistent",
    ),
    Mutation(
        name="attestation: repository policy check is bypassed",
        file="rust/src/attestation.rs",
        find="""    if !policy
        .allowed_repositories
        .iter()
        .any(|allowed| allowed == repository)
    {""",
        replace="""    if false {""",
        tests=["--test", "sigstore_fixture", "--features", "github-attestation"],
        stands_for="a certificate from an unlisted repository being trusted",
    ),
    Mutation(
        name="attestation: an untrusted policy verdict is reported as trusted",
        file="rust/src/attestation.rs",
        find="""        verdict: if issues.is_empty() {
            PolicyVerdict::Trusted
        } else {
            PolicyVerdict::Untrusted
        },""",
        replace="        verdict: PolicyVerdict::Trusted,",
        tests=["--test", "sigstore_fixture", "--features", "github-attestation"],
        stands_for="policy mismatches being silently ignored regardless of what was checked",
    ),
    Mutation(
        name="attestation: overall verification conjunction is bypassed",
        file="rust/src/attestation.rs",
        find="    report.verified = calculate_overall(&report);",
        replace="    report.verified = true;",
        tests=["--test", "sigstore_fixture", "--features", "github-attestation"],
        stands_for="a real fixture reporting verified despite a failed verification stage",
    ),
    Mutation(
        name="run attestation: workload signature verification is bypassed",
        file="rust/src/run_attestation.rs",
        find="        if let Some(keyid) = check_signer(input.envelope, &payload, &key.public_key) {",
        replace="        if let Some(keyid) = check_signer(input.envelope, &payload, &key.public_key).or_else(|| input.envelope.signatures.first().map(|signature| signature.keyid.clone())) {",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="tampered DSSE payload bytes being accepted as workload-signed",
    ),
    Mutation(
        name="run attestation: workload trust is bypassed",
        file="rust/src/run_attestation.rs",
        find="            if key.trusted && key.identity == statement.predicate.execution.workload_identity {",
        replace="            if key.identity == statement.predicate.execution.workload_identity {",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="a valid but untrusted workload key satisfying workload identity",
    ),
    Mutation(
        name="run attestation: receipt-set equality is bypassed",
        file="rust/src/run_attestation.rs",
        find="            let exact = statement.predicate.knowledge.receipts == receipts",
        replace="            let exact = true || statement.predicate.knowledge.receipts == receipts",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="an inserted, removed, or substituted receipt set matching the attestation",
    ),
    Mutation(
        name="run attestation: individual receipt verification is bypassed",
        file="rust/src/run_attestation.rs",
        find="                    Ok(report) if report.verified && report.signature_valid => {}",
        replace="                    Ok(_report) => {}",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="a receipt with invalid content, proof, or signature verifying",
    ),
    Mutation(
        name="run attestation: artifact-root binding is bypassed",
        file="rust/src/run_attestation.rs",
        find="        artifact_roots == statement.predicate.knowledge.artifact_roots && artifact_roots.len() <= 1,",
        replace="        true || artifact_roots == statement.predicate.knowledge.artifact_roots && artifact_roots.len() <= 1,",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="run evidence naming a different knowledge artifact",
    ),
    Mutation(
        name="run attestation: channel-state digest binding is bypassed",
        file="rust/src/run_attestation.rs",
        find="            && statement.predicate.knowledge.channel_state_digest.value == bound_digest",
        replace="            && (true || statement.predicate.knowledge.channel_state_digest.value == bound_digest)",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="a run claiming a different release-state snapshot",
    ),
    Mutation(
        name="run attestation: publisher authority is bypassed",
        file="rust/src/run_attestation.rs",
        find="    let authorized = input.publisher_trust.verified",
        replace="    let authorized = true || input.publisher_trust.verified",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="receipts from an unauthorized publisher role being trusted",
    ),
    Mutation(
        name="run attestation: runtime currency policy is bypassed",
        file="rust/src/run_attestation.rs",
        find="    let runtime_policy = status(decision.permitted);",
        replace="    let runtime_policy = status(true || decision.permitted);",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="a run satisfying a trust policy that its bound evidence denies",
    ),
    Mutation(
        name="run attestation: query digest binding is bypassed",
        file="rust/src/run_attestation.rs",
        find="        statement.predicate.retrieval.query_digest.algorithm == \"sha256\"",
        replace="        true || statement.predicate.retrieval.query_digest.algorithm == \"sha256\"",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="an attestation being transplanted onto a different query",
    ),
    Mutation(
        name="run attestation: output digest binding is bypassed",
        file="rust/src/run_attestation.rs",
        find="                && statement.subject[0].digest.sha256 == sha256_hex(output),",
        replace="                && (true || statement.subject[0].digest.sha256 == sha256_hex(output)),",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="different output bytes satisfying the run subject",
    ),
    Mutation(
        name="run attestation: execution time ordering is bypassed",
        file="rust/src/run_attestation.rs",
        find="        Ok(true) => VerificationStatus::Carried,",
        replace="        Ok(_) => VerificationStatus::Carried,",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="a run completing before it started",
    ),
    Mutation(
        name="run attestation: overall occurrence conjunction is bypassed",
        file="rust/src/run_attestation.rs",
        find="    let overall_occurrence_evidence = predicate_supported",
        replace="    let overall_occurrence_evidence = true || predicate_supported",
        tests=["--test", "run_attestation", "--all-features"],
        stands_for="partial or invalid evidence being reported as a verified occurrence",
    ),
    Mutation(
        name="transparency: an unverified proof is reported as verified",
        file="rust/src/transparency.rs",
        find="""        Err(error) => Ok(TransparencyReport {
            evidence: TransparencyEvidence::Insufficient,""",
        replace="""        Err(error) => Ok(TransparencyReport {
            evidence: TransparencyEvidence::Verified,""",
        tests=["--test", "release_lifecycle", "--features", "transparency-log"],
        stands_for="a Sigsum proof that failed cryptographic verification -- bad leaf signature, insufficient witness quorum, wrong log, tampered inclusion proof -- being reported as fully verified",
    ),
    Mutation(
        name="monitor: equivocation at one sequence goes unreported",
        file="rust/src/monitor.rs",
        find="        if entries.len() > 1 {",
        replace="        if entries.len() > 100 {",
        tests=["--lib", "monitor"],
        stands_for="a publisher signing two different statements at the same sequence number, undetected",
    ),
    Mutation(
        name="monitor: unchained current roots go unreported",
        file="rust/src/monitor.rs",
        find="    if distinct_roots.len() > 1 {",
        replace="    if distinct_roots.len() > 100 {",
        tests=["--lib", "monitor"],
        stands_for="two statements each claiming to be current, with no supersession chain between them, going unnoticed",
    ),
    Mutation(
        name="monitor: an unauthorized signer is not an authority violation",
        file="rust/src/monitor.rs",
        find="        if verification.authority == SigningAuthority::None {",
        replace="        if false {",
        tests=["--lib", "monitor"],
        stands_for="a statement whose signatures met no authorised role's threshold being treated as a real statement",
    ),
    Mutation(
        name="monitor: a revoked root still advertised as current goes unreported",
        file="rust/src/monitor.rs",
        find="                if advertiser.statement.sequence >= revoker.statement.sequence",
        replace="                if false && advertiser.statement.sequence >= revoker.statement.sequence",
        tests=["--lib", "monitor"],
        stands_for="a withdrawn root still being served as current after its revocation",
    ),
    Mutation(
        name="monitor: an authorized statement newer than retained state is not reported stale",
        file="rust/src/monitor.rs",
        find="        if verification.authority == SigningAuthority::Full {",
        replace="        if false {",
        tests=["--lib", "monitor"],
        stands_for="a consumer's retained state silently falling behind a real, authorised release it already has evidence of",
    ),
    Mutation(
        name="fleet: self-signature threshold check is bypassed",
        file="rust/src/fleet.rs",
        find="        let met = signers.len() >= policy.threshold as usize;",
        replace="        let met = true;",
        tests=["--lib", "fleet"],
        stands_for="a fleet policy with no valid signature being treated as self-signed",
    ),
    Mutation(
        name="fleet: rotation drops the prior-signature requirement",
        file="rust/src/fleet.rs",
        find="                let met = signers.len() >= prior.threshold as usize;",
        replace="                let met = true;",
        tests=["--lib", "fleet"],
        stands_for="an attacker minting a fleet-policy rotation with self-chosen keys nobody controls",
    ),
    Mutation(
        name="fleet: a rotation changing domain is not rejected",
        file="rust/src/fleet.rs",
        find="                domain_matches = false;",
        replace="",
        tests=["--lib", "fleet"],
        stands_for="one organization's fleet policy rotation being accepted as another's",
    ),
    Mutation(
        name="fleet: an equal or lower revision may replace a newer one",
        file="rust/src/fleet.rs",
        find="            let advanced = policy.revision > prior.revision;",
        replace="            let advanced = policy.revision >= prior.revision;",
        tests=["--lib", "fleet"],
        stands_for="fleet-policy rollback to a revoked configuration",
    ),
    Mutation(
        name="fleet: an unverified policy still reaches a compliance verdict",
        file="rust/src/fleet.rs",
        find="    let status = if !local_verification.verified || !required_verification.verified {",
        replace="    let status = if false {",
        tests=["--lib", "fleet"],
        stands_for="an unsigned or malformed fleet policy being compared as if it were trustworthy",
    ),
    Mutation(
        name="fleet: digest disagreement is not reported as drift",
        file="rust/src/fleet.rs",
        find="        && local_verification.policy_digest == required_verification.policy_digest",
        replace="        && true",
        tests=["--lib", "fleet"],
        stands_for="a locally configured fleet policy that genuinely differs from the required one being reported compliant",
    ),
    Mutation(
        name="container: the artifact root check is removed",
        file="rust/src/format.rs",
        find="        if compute_root_hash(&entries) != header.root_hash {",
        replace="        if false {",
        tests=["--test", "corruption", "--test", "conformance_vectors"],
        stands_for="the --all-features integrity hole that shipped in v0.5.1",
    ),
    Mutation(
        name="evidence: passage records need not match their hash",
        file="rust/src/evidence.rs",
        find="    let passage_hash_matches = computed_leaf == declared_leaf;",
        replace="    let passage_hash_matches = true;",
        tests=["--test", "receipt_tamper"],
        stands_for="a rewritten passage verifying inside a receipt",
    ),
    Mutation(
        name="bundle: a bundle attests regardless of its receipts",
        file="rust/src/bundle.rs",
        find="    let attested = !bundle.receipts.is_empty() && receipts_verified == bundle.receipts.len();",
        replace="    let attested = true;",
        tests=["--test", "run_bundle"],
        stands_for="an evidence bundle reporting success for unverified receipts",
    ),
]


def run(command: list[str], cwd: Path) -> int:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True).returncode


def git(*arguments: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=False
    )


def is_dirty(relative_path: str) -> bool:
    """Is this file dirty in the real checkout, not the disposable worktree?

    The audit tests `HEAD`, not whatever is currently on disk here -- so a
    dirty target file would mean the audit passes against code that is about
    to be replaced by an uncommitted edit, which proves nothing about what
    actually lands.
    """
    return bool(git("status", "--porcelain", "--", relative_path).stdout.strip())


def create_worktree() -> Path:
    """A disposable checkout at `HEAD`, so mutation and test execution never
    touch the shared checkout this script itself lives in."""
    directory = Path(tempfile.mkdtemp(prefix="annpack-mutation-audit-"))
    result = git("worktree", "add", "--detach", "--force", str(directory), "HEAD")
    if result.returncode != 0:
        shutil.rmtree(directory, ignore_errors=True)
        raise RuntimeError(f"could not create audit worktree: {result.stderr}")
    return directory


def remove_worktree(directory: Path) -> None:
    result = git("worktree", "remove", "--force", str(directory))
    if result.returncode != 0:
        # The worktree add may have partially failed, or something inside it
        # is still open; fall back to removing the directory by hand and let
        # git forget the now-missing entry.
        shutil.rmtree(directory, ignore_errors=True)
        git("worktree", "prune")


def restore(directory: Path, relative_path: str, original: str) -> None:
    """Restore the exact contents captured before applying the mutation."""
    (directory / relative_path).write_text(original, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default="")
    arguments = parser.parse_args()

    selected = [m for m in MUTATIONS if arguments.filter in m.name]
    if not selected:
        print(f"no mutation matches {arguments.filter!r}")
        return 1

    # Refuse to test a file that has uncommitted edits in the real checkout:
    # the audit runs against HEAD in a disposable worktree, and testing HEAD
    # while the real file differs would prove nothing about the code that is
    # actually about to land.
    dirty = sorted({m.file for m in selected if is_dirty(m.file)})
    if dirty:
        print("refusing to run: these files have uncommitted changes")
        for path in dirty:
            print(f"- {path}")
        print("commit or stash them first; the audit tests HEAD, not the working tree")
        return 1

    worktree = create_worktree()

    def emergency_cleanup(signum, _frame):
        remove_worktree(worktree)
        print(f"\ninterrupted ({signum}); audit worktree removed")
        sys.exit(130)

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, emergency_cleanup)

    survivors = []
    try:
        for mutation in selected:
            path = worktree / mutation.file
            original = path.read_text(encoding="utf-8")
            occurrences = original.count(mutation.find)
            if occurrences != 1:
                # An anchor matching zero or many places silently mutates the
                # wrong thing, or nothing, and the run would look clean either way.
                survivors.append(
                    f"{mutation.name}: anchor matched {occurrences} times, expected exactly 1"
                )
                print(f"  ANCHOR  {mutation.name}")
                continue

            path.write_text(original.replace(mutation.find, mutation.replace), encoding="utf-8")
            try:
                code = run(["cargo", "test", *mutation.tests], cwd=worktree)
            finally:
                restore(worktree, mutation.file, original)

            if code == 0:
                survivors.append(f"{mutation.name}: survived -- {mutation.stands_for}")
                print(f"  SURVIVED  {mutation.name}")
            else:
                print(f"  caught    {mutation.name}")
    finally:
        remove_worktree(worktree)

    if survivors:
        print("\nmutation audit failed: a removed security check went unnoticed")
        for survivor in survivors:
            print(f"- {survivor}")
        return 1

    print(f"\nmutation audit passed: {len(selected)} bypassed checks all caused test failure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
