#!/usr/bin/env python3
"""Strict local range server for the ANNPack browser conformance demo."""

import argparse
import http.server
import os
import re
import socketserver


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            path = os.path.join(path, "index.html")
        try:
            handle = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(handle.fileno()).st_size
        etag = f'"{size:x}-{int(os.path.getmtime(path)):x}"'
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        self._range = None
        if match:
            start, end = map(int, match.groups())
            if start > end or end >= size:
                handle.close()
                self.send_error(416)
                return None
            self._range = (start, end)
            handle.seek(start)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(size))
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        return handle

    def copyfile(self, source, outputfile):
        if not self._range:
            return super().copyfile(source, outputfile)
        remaining = self._range[1] - self._range[0] + 1
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), RangeHandler) as server:
        print(f"Serving http://127.0.0.1:{server.server_address[1]}", flush=True)
        server.serve_forever()
