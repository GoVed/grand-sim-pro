import http.server

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

if __name__ == '__main__':
    print("Serving on http://localhost:8000 with multithreading headers...")
    http.server.HTTPServer(("", 8000), Handler).serve_forever()