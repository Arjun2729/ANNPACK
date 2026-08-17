#!/usr/bin/env python3
"""Run CI's steps locally, read from the workflow rather than transcribed.

Every local verification pass before this one was a hand-kept list of the
interesting checks, and it was wrong three times in a row: `cargo fmt --check`
went unrun while a rename reflowed nine files, a conformance script was invoked
with no arguments and its traceback read as a failure, and five whole jobs'
worth of steps were never executed at all. A transcribed checklist drifts from
CI silently, because nothing compares the two.

So this does not restate the steps. It parses `.github/workflows/ci.yml` and
executes each job's `run:` blocks in order, with the same shell options CI uses.
A step added to the workflow is picked up here with no edit, and a step this
cannot run locally is reported loudly rather than skipped quietly.

    scripts/ci-local.py                     # every job in ci.yml
    scripts/ci-local.py native wasm         # named jobs only
    scripts/ci-local.py --workflow fuzz.yml # another workflow
    scripts/ci-local.py --list

It is not a sandbox: it runs in the working tree, as CI does in a clean
checkout. That difference is the point of --clean, which removes build outputs
first so a stale artifact cannot answer for a fresh one.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / ".github/workflows"
DEFAULT_WORKFLOW = "ci.yml"

# Values CI supplies from its own context. A step needing anything not listed
# here is reported as unrunnable rather than guessed at.
MATRIX_DEFAULTS = {"matrix.os": "local", "matrix.rust": "stable"}



def steps_of(job: dict) -> list[tuple[str, str, dict]]:
    out = []
    for step in job.get("steps", []):
        if "run" not in step:
            continue
        out.append((step.get("name", "(unnamed)"), step["run"], step.get("env", {})))
    return out


def resolve(script: str) -> tuple[str, list[str]]:
    """Substitute CI expressions, reporting any that cannot be resolved."""
    unresolved = []
    import re

    def sub(match):
        key = match.group(1).strip()
        if key in MATRIX_DEFAULTS:
            return MATRIX_DEFAULTS[key]
        unresolved.append(key)
        return match.group(0)

    return re.sub(r"\$\{\{([^}]+)\}\}", sub, script), unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="*", help="job names to run (default: all)")
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        help=f"workflow file under .github/workflows (default: {DEFAULT_WORKFLOW}). "
        "Scheduled workflows are not run by a push, so nothing else checks them "
        "until they fire on their own.",
    )
    parser.add_argument("--list", action="store_true", help="list jobs and exit")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete build outputs first, so a stale artifact cannot pass for a fresh one",
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="run every step even after a failure"
    )
    args = parser.parse_args()

    workflow_path = WORKFLOWS_DIR / args.workflow
    if not workflow_path.is_file():
        available = ", ".join(sorted(p.name for p in WORKFLOWS_DIR.glob("*.yml")))
        print(f"no such workflow: {args.workflow}", file=sys.stderr)
        print(f"available: {available}", file=sys.stderr)
        return 2
    workflow = yaml.safe_load(workflow_path.read_text())
    jobs = workflow["jobs"]

    if args.list:
        print(f"{args.workflow}:")
        for name, job in jobs.items():
            print(f"  {name}: {len(steps_of(job))} run-steps")
        others = sorted(p.name for p in WORKFLOWS_DIR.glob("*.yml") if p.name != args.workflow)
        if others:
            print(f"other workflows (--workflow): {', '.join(others)}")
        return 0

    selected = args.jobs or list(jobs)
    unknown = [j for j in selected if j not in jobs]
    if unknown:
        print(f"unknown job(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(jobs)}", file=sys.stderr)
        return 2

    if args.clean:
        # The failure this exists to prevent: a binary left over from a previous
        # name or version answering for the one that was just built.
        for path in (ROOT / "target/release", ROOT / "target/debug"):
            if path.exists():
                print(f"[clean] removing {path.relative_to(ROOT)}")
                shutil.rmtree(path)

    runner_temp = pathlib.Path(tempfile.mkdtemp(prefix="ci-local-"))
    env_base = {"RUNNER_TEMP": str(runner_temp), "CI": "true"}

    failures: list[str] = []
    for job_name in selected:
        steps = steps_of(jobs[job_name])
        print(f"\n=== job: {job_name} ({len(steps)} steps) ===", flush=True)
        for name, script, step_env in steps:
            script, unresolved = resolve(script)
            label = f"{job_name} :: {name}"
            if unresolved:
                # Loudly, and counted as a failure: a step silently skipped is
                # exactly the false green this script exists to remove.
                print(f"[UNRUNNABLE] {label}: needs {', '.join(sorted(set(unresolved)))}", flush=True)
                failures.append(f"{label} (unrunnable)")
                continue
            print(f"[run] {label}", flush=True)
            env = dict(env_base)
            for key, raw in step_env.items():
                value = str(raw)
                if value.startswith("/") and not pathlib.Path(value).exists():
                    # An absolute path that exists on the runner and not here
                    # names CI's filesystem, e.g. the wasm job's pinned
                    # wasm-bindgen. Defer to this host's value, or to the
                    # script's own default. Testing for existence rather than
                    # matching a runner prefix also covers macOS runners.
                    inherited = os.environ.get(key)
                    print(
                        f"       [host] {key}: ignoring CI path {value}"
                        + (f", using {inherited}" if inherited else ", using the script default"),
                        flush=True,
                    )
                    if inherited:
                        env[key] = inherited
                    continue
                env[key] = value
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", script],
                cwd=ROOT,
                env={**os.environ, **env},
            )
            if result.returncode != 0:
                print(f"[FAIL] {label} (exit {result.returncode})", flush=True)
                failures.append(label)
                if not args.keep_going:
                    break
        if failures and not args.keep_going:
            break

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"all steps passed for: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
