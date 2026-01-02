#!/usr/bin/env python3
import argparse
import json
import re
import sys
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def strip_query(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def fetch(url: str):
    req = Request(url, headers={"User-Agent": "annpack-demo-smoke"})
    try:
        with urlopen(req) as resp:
            return resp.status, resp.read()
    except HTTPError as e:
        return e.code, e.read()
    except URLError as e:
        return None, str(e).encode("utf-8")


def require_ok(url: str, label: str):
    status, body = fetch(url)
    if status != 200:
        msg = body.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FAIL {label}: {url} -> {status} {msg[:120]}")
    print(f"OK {label}: {url}")
    return body


def parse_index_html(html: str):
    script_re = re.compile(r'<script[^>]+src="([^"]+)"')
    client_re = re.compile(r'["\']([^"\']*annpack-client\\.js[^"\']*)["\']')
    wasm_re = re.compile(r'["\']([^"\']+\\.wasm(?:\\?[^"\']*)?)["\']')
    default_re = re.compile(r'data-default-manifest="([^"]+)"')

    scripts = script_re.findall(html)
    client = client_re.findall(html)
    wasm = wasm_re.findall(html)
    default_manifest = None
    m = default_re.search(html)
    if m:
        default_manifest = m.group(1)
    return {
        "scripts": scripts,
        "client": client,
        "wasm": wasm,
        "default_manifest": default_manifest,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000", help="Base URL for the demo root")
    args = ap.parse_args()

    base = args.base.rstrip("/") + "/"
    index_url = urljoin(base, "index.html")
    index_body = require_ok(index_url, "index.html").decode("utf-8", errors="ignore")

    parsed = parse_index_html(index_body)
    scripts = parsed["scripts"]
    wasm_paths = parsed["wasm"] or []
    client_paths = parsed["client"] or ["js/annpack-client.js"]

    # Layer 1: critical assets
    for path in scripts:
        require_ok(urljoin(base, strip_query(path)), "script")
    for path in wasm_paths:
        require_ok(urljoin(base, strip_query(path)), "wasm")
    for path in client_paths:
        require_ok(urljoin(base, strip_query(path)), "client")

    # Layer 2: manifest discovery from index.html
    default_manifest = parsed["default_manifest"]
    if not default_manifest:
        raise RuntimeError("FAIL default manifest: data-default-manifest attribute missing")

    default_manifest_url = urljoin(base, strip_query(default_manifest))
    manifest_body = require_ok(default_manifest_url, "manifest-default").decode("utf-8", errors="ignore")
    try:
        manifest = json.loads(manifest_body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"FAIL manifest parse: {default_manifest_url} -> {e}")

    shards = manifest.get("shards") or []
    if not shards:
        raise RuntimeError(f"FAIL manifest shards: {default_manifest_url} -> missing shards")
    for sh in shards:
        annpack = sh.get("annpack")
        meta = sh.get("meta")
        if annpack:
            require_ok(urljoin(base, strip_query(annpack)), "shard-annpack")
        if meta:
            require_ok(urljoin(base, strip_query(meta)), "shard-meta")

    print("PASS demo smoke")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
