"""ANNPack command-line interface."""

from __future__ import annotations

import argparse
import os
import json
import shutil
import socket
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from importlib import resources
from importlib.resources import as_file

from typing import Any, Dict, Optional, Sequence, Tuple

from .build import build_index, build_index_from_hf_wikipedia
from .logutil import log_event
from .verify import (
    diagnose_env,
    inspect_pack,
    sign_manifest,
    verify_pack,
    verify_manifest_signature,
)
from .packset import create_packset, promote_delta, rebase_packset, revert_packset, run_canary
from . import __version__


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_header(path: Path) -> Dict[str, int]:
    """Read ANNPack header fields from a file path."""
    import struct

    with open(path, "rb") as f:
        header = f.read(72)
    magic, version, endian, header_size, dim, metric, n_lists, n_vectors, offset_table_pos = (
        struct.unpack("<QIIIIIIIQ", header[:44])
    )
    return {
        "magic": magic,
        "version": version,
        "endian": endian,
        "header_size": header_size,
        "dim": dim,
        "metric": metric,
        "n_lists": n_lists,
        "n_vectors": n_vectors,
        "offset_table_pos": offset_table_pos,
    }


def _find_manifest(pack_dir: Path) -> Optional[Path]:
    """Return the first manifest in a pack directory, if any."""
    candidates = (
        list(pack_dir.glob("*.manifest.json"))
        + list(pack_dir.glob("manifest.json"))
        + list(pack_dir.glob("manifest.jsonl"))
    )
    return candidates[0] if candidates else None


def _materialize_ui_root() -> Path:
    """Copy packaged UI assets to a temp dir and return its path."""
    ui = resources.files("annpack.ui")
    tmp = Path(tempfile.mkdtemp(prefix="annpack_ui_"))
    with as_file(ui) as ui_path:
        if ui_path.is_dir():
            shutil.copytree(ui_path, tmp, dirs_exist_ok=True)
        else:
            shutil.copy(ui_path, tmp / "index.html")
    return tmp


def _write_manifest(prefix: Path, ann_path: Path, meta_path: Path) -> Path:
    """Write a simple shard manifest if missing."""
    info = _read_header(ann_path)
    manifest = {
        "schema_version": 2,
        "version": 1,
        "created_by": "annpack.cli",
        "dim": info["dim"],
        "n_lists": info["n_lists"],
        "n_vectors": info["n_vectors"],
        "shards": [
            {
                "name": prefix.name,
                "annpack": ann_path.name,
                "meta": meta_path.name,
                "n_vectors": info["n_vectors"],
            }
        ],
    }
    manifest_path = prefix.with_suffix(".manifest.json")
    if manifest_path.exists():
        print(f"[write] Manifest already exists, keeping: {manifest_path}")
        return manifest_path
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[write] Manifest: {manifest_path}")
    return manifest_path


def _port_in_use(host: str, port: int) -> bool:
    """Return True if host:port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _start_http_server(
    host: str, port: int, root_dir: Path, pack_dir: Path, quiet: bool = False
) -> ThreadingHTTPServer:
    """Start a threaded HTTP server with /pack mounted."""

    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            parsed = urlparse(path)
            clean = parsed.path
            base_root = Path(root_dir)
            base_pack = Path(pack_dir)
            if clean.startswith("/pack/"):
                rel = clean[len("/pack/") :]
                base = base_pack
            else:
                rel = clean.lstrip("/")
                base = base_root
            return str((base / rel).resolve())

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            if quiet:
                return
            super().log_message(fmt, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _health_check(url: str) -> Tuple[int, bytes]:
    """Fetch a URL and return (status, body)."""
    req = Request(url, headers={"User-Agent": "annpack-smoke"})
    with urlopen(req, timeout=5) as resp:
        return resp.status, resp.read()


def cmd_build(args: argparse.Namespace) -> None:
    """Handle `annpack build`."""
    output_prefix = Path(args.output).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    text_col = args.text_col or "text"
    device = args.device

    if args.hf_dataset:
        if args.hf_dataset not in ("wikimedia/wikipedia", "wikipedia"):
            raise SystemExit(
                f"HF dataset '{args.hf_dataset}' not supported. Try --hf-dataset wikimedia/wikipedia."
            )
        build_index_from_hf_wikipedia(
            output_prefix=str(output_prefix),
            dataset_name=args.hf_dataset,
            config=args.hf_config or "20231101.en",
            split=args.hf_split,
            max_rows=args.max_rows,
            model_name=args.model,
            n_lists=args.lists if args.lists else 4096,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        if not args.input:
            raise SystemExit("--input is required for local build")
        build_index(
            input_path=args.input,
            text_col=text_col,
            id_col=args.id_col,
            output_prefix=str(output_prefix),
            model_name=args.model,
            n_lists=args.lists,
            max_rows=args.max_rows,
            batch_size=args.batch_size,
            device=device,
        )

    ann_path = output_prefix.with_suffix(".annpack")
    meta_path = output_prefix.with_suffix(".meta.jsonl")
    _write_manifest(output_prefix, ann_path, meta_path)
    print("[done] Build complete.")


def cmd_serve(args: argparse.Namespace) -> None:
    """Handle `annpack serve`."""
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    if not pack_dir.exists():
        raise SystemExit(f"Pack dir not found: {pack_dir}")

    ui_root = _materialize_ui_root()
    root_dir = ui_root
    manifest = _find_manifest(pack_dir)
    manifest_hint = f"/pack/{manifest.name}" if manifest else "none found"

    try:
        server = _start_http_server(
            args.host, args.port, root_dir=root_dir, pack_dir=pack_dir, quiet=False
        )
    except OSError as e:
        raise SystemExit(f"Failed to start server on {args.host}:{args.port}: {e}")

    actual_port = server.server_address[1]
    log_event("serve_start", {"host": args.host, "port": actual_port})
    base = f"http://{args.host}:{actual_port}"
    print(f"[serve] Serving from {root_dir}")
    print(f"[serve] Pack mounted at /pack/ -> {pack_dir}")
    if manifest:
        print(f"[serve] Manifest candidate: {manifest_hint}")
    print(f"[serve] Open: {base}/")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[serve] Stopping server...")
        server.shutdown()


def _resolve_root_and_manifest(pack_dir: Path) -> Tuple[Path, Optional[Path]]:
    """Return UI root and manifest path for a pack dir."""
    root_dir = _materialize_ui_root()
    manifest = _find_manifest(pack_dir)
    return root_dir, manifest


def cmd_smoke(args: argparse.Namespace) -> None:
    """Handle `annpack smoke`."""
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    if not pack_dir.exists():
        raise SystemExit(f"Pack dir not found: {pack_dir}")
    root_dir, manifest = _resolve_root_and_manifest(pack_dir)
    if not manifest:
        raise SystemExit(
            f"No manifest found in {pack_dir} (looked for *.manifest.json / manifest.json)"
        )

    host, port = args.host, args.port
    if port != 0 and _port_in_use(host, port):
        raise SystemExit(f"Port {port} is already in use; start a fresh server first.")

    server = _start_http_server(host, port, root_dir=root_dir, pack_dir=pack_dir, quiet=True)
    actual_port = server.server_address[1]
    base = f"http://{host}:{actual_port}"
    time.sleep(0.2)

    index_url = base + "/index.html"
    try:
        status, _ = _health_check(index_url)
    except Exception:
        status, _ = _health_check(base + "/")
        index_url = base + "/"
    if status != 200:
        server.shutdown()
        raise SystemExit(f"Index page check failed ({status}): {index_url}")
    # Also ensure root path responds
    status_root, _ = _health_check(base + "/")
    if status_root != 200:
        server.shutdown()
        raise SystemExit(f"Root page check failed ({status_root}): {base}/")

    manifest_url = base + (
        f"/{manifest.name}" if root_dir == pack_dir else f"/pack/{manifest.name}"
    )
    try:
        status, manifest_body = _health_check(manifest_url)
    except Exception as e:
        server.shutdown()
        raise SystemExit(f"Manifest check failed: {e}")
    if status != 200:
        server.shutdown()
        raise SystemExit(f"Manifest check failed ({status}): {manifest_url}")
    data = json.loads(manifest_body)
    shards = data.get("shards") or []
    if not shards:
        server.shutdown()
        raise SystemExit("Manifest contains no shards.")

    manifest_base = manifest_url.rsplit("/", 1)[0] + "/"
    for shard in shards:
        ann_url = urljoin(manifest_base, shard.get("annpack"))
        meta_url = urljoin(manifest_base, shard.get("meta"))
        for label, url in (("annpack", ann_url), ("meta", meta_url)):
            try:
                s, _ = _health_check(url)
            except Exception as e:
                server.shutdown()
                raise SystemExit(f"{label} check failed: {e}")
            if s != 200:
                server.shutdown()
                raise SystemExit(f"{label} check failed ({s}): {url}")
    server.shutdown()
    print("PASS smoke")


def cmd_verify(args: argparse.Namespace) -> None:
    """Handle `annpack verify`."""
    result = verify_pack(args.pack_dir, deep=args.deep)
    if args.pubkey:
        verify_manifest_signature(args.pack_dir, args.pubkey, sig_path=args.sig)
        result["signature"] = "ok"
    print(json.dumps(result, indent=2))


def cmd_inspect(args: argparse.Namespace) -> None:
    """Handle `annpack inspect`."""
    info = inspect_pack(args.pack_dir)
    print(json.dumps(info, indent=2))


def cmd_sign(args: argparse.Namespace) -> None:
    """Handle `annpack sign`."""
    sig_path = sign_manifest(args.pack_dir, args.key, out_path=args.out)
    print(json.dumps({"signature": sig_path}, indent=2))


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Handle `annpack diagnose`."""
    info = diagnose_env()
    print(json.dumps(info, indent=2))


def cmd_packset_create(args: argparse.Namespace) -> None:
    """Handle `annpack packset create`."""
    manifest = create_packset(args.base, args.delta, args.out)
    print(json.dumps({"manifest": str(manifest)}, indent=2))


def cmd_packset_promote(args: argparse.Namespace) -> None:
    """Handle `annpack packset promote-delta`."""
    manifest = promote_delta(args.packset, args.delta)
    print(json.dumps({"manifest": str(manifest)}, indent=2))


def cmd_packset_revert(args: argparse.Namespace) -> None:
    """Handle `annpack packset revert`."""
    manifest = revert_packset(args.packset, args.to)
    print(json.dumps({"manifest": str(manifest)}, indent=2))


def cmd_packset_rebase(args: argparse.Namespace) -> None:
    """Handle `annpack packset rebase`."""
    manifest = rebase_packset(
        args.packset,
        args.out,
        text_col=args.text_col,
        id_col=args.id_col,
        lists=args.lists,
        seed=args.seed,
        offline=os.environ.get("ANNPACK_OFFLINE") == "1",
    )
    print(json.dumps({"manifest": str(manifest)}, indent=2))


def cmd_canary(args: argparse.Namespace) -> None:
    """Handle `annpack canary`."""
    result = run_canary(
        args.base,
        args.candidate,
        args.queries,
        top_k=args.top_k,
        min_overlap=args.min_overlap,
        avg_overlap=args.avg_overlap,
    )
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""
    p = argparse.ArgumentParser(prog="annpack", description="ANNPack tools (build, serve, smoke)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build an ANNPack from CSV/Parquet or HF Wikipedia")
    b.add_argument("--input", help="Path to input CSV/Parquet")
    b.add_argument("--text-col", help="Text column name (default: text)")
    b.add_argument("--id-col", help="Optional ID column (int64)")
    b.add_argument("--output", required=True, help="Output prefix (e.g., ./out/tiny)")
    b.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    b.add_argument("--lists", type=int, default=1024, help="Number of IVF lists/clusters")
    b.add_argument("--max-rows", type=int, default=100000, help="Maximum rows to index")
    b.add_argument("--batch-size", type=int, default=512, help="Embedding batch size")
    b.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Force embedding device (default: auto or ANNPACK_DEVICE)",
    )
    b.add_argument("--hf-dataset", help="HuggingFace dataset name (optional)")
    b.add_argument("--hf-config", help="HF dataset config")
    b.add_argument("--hf-split", default="train", help="HF dataset split (default: train)")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("serve", help="Serve the UI with a pack mounted at /pack/")
    s.add_argument("pack_dir", help="Directory containing .annpack/.meta/.manifest")
    s.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    s.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    s.set_defaults(func=cmd_serve)

    demo_alias = sub.add_parser("demo", help=argparse.SUPPRESS)
    demo_alias.add_argument("pack_dir", help=argparse.SUPPRESS)
    demo_alias.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    demo_alias.add_argument("--port", type=int, default=8000, help=argparse.SUPPRESS)
    demo_alias.set_defaults(func=cmd_serve)

    sm = sub.add_parser("smoke", help="Start demo server (if needed) and verify assets + manifest")
    sm.add_argument("pack_dir", help="Directory containing .annpack/.meta/.manifest")
    sm.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    sm.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    sm.set_defaults(func=cmd_smoke)

    v = sub.add_parser("verify", help="Verify pack structure + manifest checks")
    v.add_argument("pack_dir", help="Directory containing .annpack/.meta/.manifest")
    v.add_argument("--deep", action="store_true", help="Validate list payload sizes")
    v.add_argument("--pubkey", help="Ed25519 public key (PEM) to verify signature")
    v.add_argument("--sig", help="Signature path (optional)")
    v.set_defaults(func=cmd_verify)

    i = sub.add_parser("inspect", help="Print pack header + manifest info")
    i.add_argument("pack_dir", help="Directory containing .annpack/.meta/.manifest")
    i.set_defaults(func=cmd_inspect)

    sg = sub.add_parser("sign", help="Sign a manifest with an Ed25519 key")
    sg.add_argument("pack_dir", help="Directory containing .annpack/.meta/.manifest")
    sg.add_argument("--key", required=True, help="Path to Ed25519 private key (PEM)")
    sg.add_argument("--out", help="Signature output path (optional)")
    sg.set_defaults(func=cmd_sign)

    dg = sub.add_parser("diagnose", help="Print environment + dependency versions")
    dg.set_defaults(func=cmd_diagnose)

    ps = sub.add_parser("packset", help="PackSet management")
    ps_sub = ps.add_subparsers(dest="packset_cmd", required=True)

    psc = ps_sub.add_parser("create", help="Create a packset from base + deltas")
    psc.add_argument("--base", required=True, help="Base pack directory")
    psc.add_argument(
        "--delta", action="append", default=[], help="Delta pack directories (repeatable)"
    )
    psc.add_argument("--out", required=True, help="Output packset directory")
    psc.set_defaults(func=cmd_packset_create)

    psp = ps_sub.add_parser("promote-delta", help="Promote a delta into an existing packset")
    psp.add_argument("--packset", required=True, help="Packset directory")
    psp.add_argument("--delta", required=True, help="Delta pack directory")
    psp.set_defaults(func=cmd_packset_promote)

    psr = ps_sub.add_parser("revert", help="Revert a packset to a delta sequence")
    psr.add_argument("--packset", required=True, help="Packset directory")
    psr.add_argument("--to", type=int, required=True, help="Delta sequence to keep (inclusive)")
    psr.set_defaults(func=cmd_packset_revert)

    psb = ps_sub.add_parser("rebase", help="Compact deltas into a new base packset")
    psb.add_argument("--packset", required=True, help="Packset directory")
    psb.add_argument("--out", required=True, help="Output packset directory")
    psb.add_argument("--text-col", default="text", help="Text column name")
    psb.add_argument("--id-col", default="id", help="ID column name")
    psb.add_argument("--lists", type=int, default=1024, help="Number of IVF lists")
    psb.add_argument("--seed", type=int, default=0, help="Seed for deterministic builds")
    psb.set_defaults(func=cmd_packset_rebase)

    cn = sub.add_parser("canary", help="Compare base vs candidate packs with queries")
    cn.add_argument("--base", required=True, help="Base pack directory")
    cn.add_argument("--candidate", required=True, help="Candidate pack directory")
    cn.add_argument("--queries", required=True, help="JSONL queries (text or vector)")
    cn.add_argument("--top-k", type=int, default=5, help="Top K results")
    cn.add_argument("--min-overlap", type=float, default=0.7, help="Min overlap threshold")
    cn.add_argument("--avg-overlap", type=float, default=0.8, help="Avg overlap threshold")
    cn.set_defaults(func=cmd_canary)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
