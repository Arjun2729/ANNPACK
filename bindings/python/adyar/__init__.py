"""Python binding that delegates all untrusted parsing to the Rust runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable
import warnings

#: Canonical variable naming the CLI to drive.
BINARY_ENV = "ADYAR_BINARY"
#: The name this variable carried when the project was called Adyar.
LEGACY_BINARY_ENV = "ANNPACK_BINARY"


class AdyarError(RuntimeError):
    """The native Adyar runtime rejected an operation."""


def _discover_binary() -> str | None:
    """Locate the CLI.

    Order: ``ADYAR_BINARY``, the legacy ``ANNPACK_BINARY``, ``adyar`` on PATH,
    then the legacy ``annpack`` on PATH. Either legacy hit warns once so a
    pinned old install is visible before it stops being published.
    """
    candidate = os.environ.get(BINARY_ENV)
    if candidate:
        return candidate

    candidate = os.environ.get(LEGACY_BINARY_ENV)
    if candidate:
        warnings.warn(
            f"{LEGACY_BINARY_ENV} is deprecated; use {BINARY_ENV}",
            DeprecationWarning,
            stacklevel=3,
        )
        return candidate

    found = shutil.which("adyar")
    if found:
        return found

    found = shutil.which("annpack")
    if found:
        warnings.warn(
            "the 'annpack' binary is deprecated; install the 'adyar' CLI",
            DeprecationWarning,
            stacklevel=3,
        )
    return found


class Client:
    def __init__(self, binary: str | os.PathLike[str] | None = None):
        # An explicitly supplied path is used exactly as given.
        self.binary = str(binary) if binary else _discover_binary()
        if not self.binary:
            raise AdyarError(
                f"adyar binary was not found; pass binary= or set {BINARY_ENV}"
            )

    def inspect(self, pack: str | os.PathLike[str]) -> dict[str, Any]:
        return self._json("inspect", str(pack), "--json")

    def verify(
        self,
        pack: str | os.PathLike[str],
        public_key: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        command = ["verify", str(pack), "--json"]
        if public_key:
            command.extend(["--public-key", str(public_key)])
        return self._json(*command)

    def search(
        self,
        pack: str | os.PathLike[str],
        query: str,
        *,
        limit: int = 10,
        mode: str = "hybrid",
        query_vector: str | os.PathLike[str] | None = None,
        vector_profile: str | None = None,
        vector_probes: int = 4,
        debug: bool = False,
    ) -> dict[str, Any]:
        command = [
            "search",
            str(pack),
            query,
            "--limit",
            str(limit),
            "--mode",
            mode,
            "--json",
            "--vector-probes",
            str(vector_probes),
        ]
        if query_vector:
            command.extend(["--query-vector", str(query_vector)])
        if vector_profile:
            command.extend(["--vector-profile", vector_profile])
        if debug:
            command.append("--debug")
        return self._json(*command)

    def build(
        self,
        source: str | os.PathLike[str],
        output: str | os.PathLike[str],
        *,
        name: str,
        version: str,
        extra: Iterable[str] = (),
    ) -> dict[str, Any]:
        return self._json(
            "build",
            str(source),
            "--output",
            str(output),
            "--name",
            name,
            "--version",
            version,
            *extra,
            "--json",
        )

    def push(
        self,
        pack: str | os.PathLike[str],
        reference: str,
        *,
        username: str | None = None,
    ) -> dict[str, Any]:
        command = ["push", str(pack), reference, "--json"]
        if username:
            command.extend(["--username", username])
        return self._json(*command)

    def pull(
        self,
        reference: str,
        output: str | os.PathLike[str],
        *,
        username: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        command = ["pull", reference, "--output", str(output), "--json"]
        if username:
            command.extend(["--username", username])
        if force:
            command.append("--force")
        return self._json(*command)

    def bundle(
        self,
        pack: str | os.PathLike[str],
        query: str,
        output: str | os.PathLike[str],
        *,
        limit: int = 5,
        mode: str = "lexical",
        run_id: str | None = None,
        application: str | None = None,
        model: str | None = None,
        answer: str | os.PathLike[str] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Collect one run's retrieval evidence into a portable bundle.

        The bundle carries a standalone receipt per retrieved passage. The
        query, application, model and answer travel with them and are attested
        by nothing.
        """
        command = [
            "bundle",
            str(pack),
            query,
            "--output",
            str(output),
            "--limit",
            str(limit),
            "--mode",
            mode,
        ]
        for flag, value in (
            ("--run-id", run_id),
            ("--application", application),
            ("--model", model),
            ("--answer", None if answer is None else str(answer)),
            ("--created-at", created_at),
        ):
            if value is not None:
                command.extend([flag, value])
        self._run(*command)
        with open(output, encoding="utf-8") as handle:
            return json.load(handle)

    def verify_run(
        self,
        bundle: str | os.PathLike[str],
        *,
        trusted_public_key: str | None = None,
    ) -> dict[str, Any]:
        """Verify every receipt in a run bundle.

        Returns the report whether or not it attests, because the useful
        information in a failure is which receipt failed and why. Callers must
        check ``attested`` rather than relying on an exception.
        """
        command = ["verify-run", str(bundle), "--json"]
        if trusted_public_key:
            command.extend(["--trusted-public-key", trusted_public_key])
        result = self._run(*command, check=False)
        return self._parse(result.stdout)

    def telemetry(
        self,
        pack: str | os.PathLike[str],
        query: str,
        *,
        limit: int = 10,
        mode: str = "lexical",
        receipt_uri_template: str | None = None,
    ) -> dict[str, Any]:
        """OpenTelemetry span and event attributes for one retrieval.

        ``receipt_uri_template`` must contain ``{passage_id}`` and may contain
        ``{root}``; Adyar does not define where receipts are served.
        """
        command = [
            "search",
            str(pack),
            query,
            "--limit",
            str(limit),
            "--mode",
            mode,
            "--otel",
        ]
        if receipt_uri_template:
            command.extend(["--otel-receipt-uri", receipt_uri_template])
        return self._json(*command)

    def mcp_command(self, pack: str | os.PathLike[str]) -> list[str]:
        return [self.binary, "mcp", str(pack)]

    def _run(
        self, *arguments: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [self.binary, *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        if check and result.returncode:
            raise AdyarError(result.stderr.strip() or result.stdout.strip())
        return result

    @staticmethod
    def _parse(stdout: str) -> dict[str, Any]:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as error:
            raise AdyarError(f"native runtime returned invalid JSON: {error}") from error

    def _json(self, *arguments: str) -> dict[str, Any]:
        return self._parse(self._run(*arguments).stdout)


__all__ = ["AdyarError", "Client"]
