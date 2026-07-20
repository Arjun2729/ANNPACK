"""Python binding that delegates all untrusted parsing to the Rust runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable


class ANNPackError(RuntimeError):
    """The native ANNPack runtime rejected an operation."""


class Client:
    def __init__(self, binary: str | os.PathLike[str] | None = None):
        candidate = str(binary) if binary else os.environ.get("ANNPACK_BINARY")
        self.binary = candidate or shutil.which("annpack")
        if not self.binary:
            raise ANNPackError(
                "annpack binary was not found; pass binary= or set ANNPACK_BINARY"
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

    def mcp_command(self, pack: str | os.PathLike[str]) -> list[str]:
        return [self.binary, "mcp", str(pack)]

    def _json(self, *arguments: str) -> dict[str, Any]:
        result = subprocess.run(
            [self.binary, *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ANNPackError(result.stderr.strip() or result.stdout.strip())
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ANNPackError(f"native runtime returned invalid JSON: {error}") from error


__all__ = ["ANNPackError", "Client"]
