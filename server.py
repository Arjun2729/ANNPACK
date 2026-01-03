import os
import http.server


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def send_head(self):
        # If no Range header, let the default handler serve the file (HTML/JS/WASM, etc.)
        if "Range" not in self.headers:
            return super().send_head()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return None

        # Use getsize so we don't open a transient fd that can close early
        file_size = os.path.getsize(path)

        try:
            # Parse Range header: bytes=START-END
            _, r = self.headers["Range"].strip().split("=", 1)
            start, end = r.split("-")
            start = int(start)
            end = int(end) if end else file_size - 1

            if start >= file_size:
                raise ValueError()
            end = min(end, file_size - 1)
            length = end - start + 1
        except ValueError:
            self.send_error(416, "Invalid Range")
            return None

        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        # Stream the requested byte range
        try:
            with open(path, "rb") as f:
                f.seek(start)
                while length > 0:
                    chunk = f.read(min(length, 64 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    length -= len(chunk)
        except BrokenPipeError:
            # Client disconnected mid-stream; safe to ignore
            pass
        return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"✅ Robust Range Server running on port {port}...")
    http.server.HTTPServer(("0.0.0.0", port), RangeRequestHandler).serve_forever()
