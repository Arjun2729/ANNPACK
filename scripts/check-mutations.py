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
now refuses to start unless the target files are clean in git, restores through
git rather than an in-memory copy, and handles SIGINT and SIGTERM.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
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
        name="provenance: format-4 source binding requirement is bypassed",
        file="rust/src/format.rs",
        find="    if format_version < 4 {",
        replace="    if format_version < 99 {",
        # This gate has no test in provenance.rs of its own: the normal
        # builder never omits the descriptor, so exercising "format 4 with no
        # descriptor" needs a hand-constructed manifest, which is what
        # source_binding.rs already does. It is the same code path provenance
        # relies on, so that suite is what has to catch a regression here.
        tests=["--test", "source_binding"],
        stands_for="a format-4 artifact with no source binding being accepted",
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


def run(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True).returncode


def git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )


def is_dirty(relative_path: str) -> bool:
    return bool(git("status", "--porcelain", "--", relative_path).stdout.strip())


def restore(relative_path: str) -> None:
    """Revert a mutated file from the index, not from a remembered string.

    Restoring from an in-memory copy only works if this process survives to do
    it. Git holds the pristine content independently of whether it does.
    """
    git("checkout", "--", relative_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default="")
    arguments = parser.parse_args()

    selected = [m for m in MUTATIONS if arguments.filter in m.name]
    if not selected:
        print(f"no mutation matches {arguments.filter!r}")
        return 1

    # Refuse to mutate a file that already has uncommitted edits: the restore
    # below would discard them, and a mutation applied on top of unknown changes
    # tests something other than what it names.
    dirty = sorted({m.file for m in selected if is_dirty(m.file)})
    if dirty:
        print("refusing to run: these files have uncommitted changes")
        for path in dirty:
            print(f"- {path}")
        print("commit or stash them first; the audit restores files through git")
        return 1

    active: list[str] = []

    def emergency_restore(signum, _frame):
        for relative in active:
            restore(relative)
        print(f"\ninterrupted ({signum}); mutated files restored")
        sys.exit(130)

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, emergency_restore)

    survivors = []
    for mutation in selected:
        path = ROOT / mutation.file
        original = path.read_text(encoding="utf-8")
        occurrences = original.count(mutation.find)
        if occurrences != 1:
            # An anchor matching zero or many places silently mutates the wrong
            # thing, or nothing, and the run would look clean either way.
            survivors.append(
                f"{mutation.name}: anchor matched {occurrences} times, expected exactly 1"
            )
            print(f"  ANCHOR  {mutation.name}")
            continue

        path.write_text(original.replace(mutation.find, mutation.replace), encoding="utf-8")
        active.append(mutation.file)
        try:
            code = run(["cargo", "test", *mutation.tests])
        finally:
            restore(mutation.file)
            active.remove(mutation.file)

        if code == 0:
            survivors.append(f"{mutation.name}: survived -- {mutation.stands_for}")
            print(f"  SURVIVED  {mutation.name}")
        else:
            print(f"  caught    {mutation.name}")

    if survivors:
        print("\nmutation audit failed: a removed security check went unnoticed")
        for survivor in survivors:
            print(f"- {survivor}")
        return 1

    print(f"\nmutation audit passed: {len(selected)} bypassed checks all caused test failure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
