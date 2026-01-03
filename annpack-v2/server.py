import http.server
import os
import socketserver

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8080))


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        fs = os.fstat(f.fileno())
        size = fs.st_size
        range_header = self.headers.get("Range")
        if range_header:
            try:
                _, rng = range_header.split("=")
                start_s, end_s = rng.split("-")
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
                if start >= size:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    return None
                print(f"[srv] 206 {self.path} start={start} end={end} size={size}")
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", end - start + 1)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
                self.send_header(
                    "Access-Control-Expose-Headers", "Content-Range, Accept-Ranges, Content-Length"
                )
                self.end_headers()
                f.seek(start)
                self.copyfile(f, self.wfile, length=end - start + 1)
                f.close()
                return None
            except Exception:
                pass
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header(
            "Access-Control-Expose-Headers", "Content-Range, Accept-Ranges, Content-Length"
        )
        self.end_headers()
        return f

    def copyfile(self, source, outputfile, length=None):
        if length is None:
            return super().copyfile(source, outputfile)
        bufsize = 64 * 1024
        remaining = length
        while remaining > 0:
            chunk = source.read(min(bufsize, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


if __name__ == "__main__":

    class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True
        allow_reuse_address = True

    with ThreadingTCPServer((HOST, PORT), RangeHandler) as httpd:
        print(f"Serving on http://{HOST}:{PORT}")
        httpd.serve_forever()
