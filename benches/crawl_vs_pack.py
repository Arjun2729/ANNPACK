#!/usr/bin/env python3
"""Measured ANNPack transfer versus an explicit rendered-page crawl model."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading

from benchmark import generate_corpus


class RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.server.pack_bytes)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", '"crawl-benchmark"')
        self.end_headers()

    def do_GET(self):
        value = self.headers.get("Range")
        if not value or not value.startswith("bytes="):
            self.send_error(400, "strict benchmark requires a byte range")
            return
        start, end = value.removeprefix("bytes=").split("-", 1)
        start, end = int(start), int(end)
        if start < 0 or end < start or end >= len(self.server.pack_bytes):
            self.send_error(416)
            return
        body = self.server.pack_bytes[start : end + 1]
        with self.server.counter_lock:
            self.server.range_requests += 1
            self.server.transferred_bytes += len(body)
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Range", f"bytes {start}-{end}/{len(self.server.pack_bytes)}"
        )
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", '"crawl-benchmark"')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="target/release/annpack")
    parser.add_argument("--documents", type=int, default=1000)
    parser.add_argument("--crawl-pages", type=int, default=50)
    parser.add_argument("--rendered-page-bytes", type=int, default=300_000)
    parser.add_argument("--min-reduction", type=float, default=0.95)
    parser.add_argument("--max-range-requests", type=int, default=8)
    parser.add_argument("--output")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())

    with tempfile.TemporaryDirectory(prefix="annpack-crawl-bench-") as directory:
        root = Path(directory)
        corpus = root / "docs"
        corpus.mkdir()
        generate_corpus(corpus, args.documents)
        pack = root / "benchmark.annpack"
        subprocess.run(
            [
                binary,
                "build",
                str(corpus),
                "--output",
                str(pack),
                "--name",
                "crawl-benchmark",
                "--version",
                "3.0.0",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        server.pack_bytes = pack.read_bytes()
        server.counter_lock = threading.Lock()
        server.range_requests = 0
        server.transferred_bytes = 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/knowledge.annpack"
            result = subprocess.run(
                [
                    binary,
                    "search",
                    url,
                    "AP-0100",
                    "--mode",
                    "lexical",
                    "--limit",
                    "1",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            response = json.loads(result.stdout)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        crawl_bytes = args.crawl_pages * args.rendered_page_bytes
        reduction = 1 - server.transferred_bytes / crawl_bytes
        report = {
            "corpus_documents": args.documents,
            "query": "AP-0100",
            "result_correct": "rejected a request" in response["results"][0]["text"],
            "annpack_bytes": len(server.pack_bytes),
            "annpack_query_range_requests": server.range_requests,
            "annpack_query_transferred_bytes": server.transferred_bytes,
            "crawl_model": {
                "pages": args.crawl_pages,
                "rendered_bytes_per_page": args.rendered_page_bytes,
                "total_bytes": crawl_bytes,
                "note": "The crawl baseline is an explicit model, not a measured remote website. ANNPack transfer is measured from the strict local range server.",
            },
            "modeled_transfer_reduction": reduction,
            "gates": {
                "correct": response["results"][0]["text"].find("rejected a request") >= 0,
                "range_requests": server.range_requests <= args.max_range_requests,
                "reduction": reduction >= args.min_reduction,
            },
        }
        encoded = json.dumps(report, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        if args.enforce and not all(report["gates"].values()):
            raise SystemExit("crawl-vs-pack gates failed")


if __name__ == "__main__":
    main()
