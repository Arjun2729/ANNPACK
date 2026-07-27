#!/usr/bin/env python3
import http.server
import os
import socketserver

CHUNK = 64 * 1024


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    range = None

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            for index in ("index.html", "index.htm"):
                index_path = os.path.join(path, index)
                if os.path.exists(index_path):
                    path = index_path
                    break
            else:
                return super().send_head()

        ctype = self.guess_type(path)
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        size = fs.st_size
        range_header = self.headers.get('Range')
        self.range = None

        if range_header and range_header.startswith('bytes='):
            try:
                start_str, end_str = range_header.replace('bytes=', '').split('-')
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else size - 1
                start = max(0, start)
                end = min(size - 1, end)
                if start > end:
                    start, end = 0, size - 1
                self.send_response(206)
                self.send_header('Content-type', ctype)
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.send_header('Content-Length', str(end - start + 1))
                self.end_headers()
                f.seek(start)
                self.range = (start, end)
                return f
            except Exception:
                pass

        self.send_response(200)
        self.send_header('Content-type', ctype)
        self.send_header('Content-Length', str(size))
        self.end_headers()
        return f

    def do_GET(self):
        f = self.send_head()
        if f:
            try:
                if self.range:
                    start, end = self.range
                    remaining = end - start + 1
                    while remaining > 0:
                        data = f.read(min(CHUNK, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
                else:
                    self.copyfile(f, self.wfile)
            finally:
                f.close()


if __name__ == "__main__":
    with socketserver.TCPServer(("", 8080), RangeRequestHandler) as httpd:
        print("Serving on http://localhost:8080")
        httpd.serve_forever()
