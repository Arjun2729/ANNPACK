"""ANNPack command-line interface."""

from __future__ import annotations

import argparse
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

from typing import Optional

from .build import build_index, build_index_from_hf_wikipedia
from . import __version__


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_header(path: Path) -> dict:
    """Read ANNPack header fields from a file path."""
    import struct

    with open(path, "rb") as f:
        header = f.read(72)
    magic, version, endian, header_size, dim, metric, n_lists, n_vectors, offset_table_pos = struct.unpack(
        "<QIIIIIIIQ", header[:44]
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
    candidates = list(pack_dir.glob("*.manifest.json")) + list(pack_dir.glob("manifest.json")) + list(
        pack_dir.glob("manifest.jsonl")
    )
    return candidates[0] if candidates else None


def _materialize_ui_root() -> Path:
    """Copy packaged UI assets to a temp dir and return its path."""
    ui = resources.files("annpack.ui")
    tmp = Path(tempfile.mkdtemp(prefix="annpack_ui_"))
    if ui.is_dir():
        shutil.copytree(ui, tmp, dirs_exist_ok=True)
    else:
        shutil.copy(ui, tmp / "index.html")
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


def _start_http_server(host: str, port: int, root_dir: Path, pack_dir: Path, quiet: bool = False):
    """Start a threaded HTTP server with /pack mounted."""
    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
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

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt, *args):
            if quiet:
                return
            super().log_message(fmt, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _health_check(url: str):
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
            raise SystemExit(f"HF dataset '{args.hf_dataset}' not supported. Try --hf-dataset wikimedia/wikipedia.")
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
        server = _start_http_server(args.host, args.port, root_dir=root_dir, pack_dir=pack_dir, quiet=False)
    except OSError as e:
        raise SystemExit(f"Failed to start server on {args.host}:{args.port}: {e}")

    actual_port = server.server_address[1]
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


def _resolve_root_and_manifest(pack_dir: Path):
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
        raise SystemExit(f"No manifest found in {pack_dir} (looked for *.manifest.json / manifest.json)")

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

    manifest_url = base + (f"/{manifest.name}" if root_dir == pack_dir else f"/pack/{manifest.name}")
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
    b.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Force embedding device (default: auto or ANNPACK_DEVICE)")
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

    return p


def main(argv=None) -> None:
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
