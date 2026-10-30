"""Minimal FastAPI-style service (stdlib http.server) for the Course Assistant."""
import json, os, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from agent import Agent
from rag import CourseRAG

_rag = None; _agent = None
def get_agent():
    global _rag, _agent
    if _agent is None:
        _rag = CourseRAG(use_lora=False)
        _agent = Agent(_rag)
    return _agent

class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return self._json(200, {"status": "ok", "service": "course-assistant"})
        if path == "/metrics":
            p = os.path.join(os.path.dirname(__file__), "out", "model_card.json")
            if os.path.exists(p):
                card = json.load(open(p))
                return self._json(200, {"eval": card.get("eval_summary", {}), "version": card.get("version")})
            return self._json(200, {"eval": {}})
        return self._json(404, {"error": "not found"})
    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try: data = json.loads(raw.decode() or "{}")
        except Exception: return self._json(400, {"error": "bad json"})
        if path == "/ask":
            q = (data.get("question") or "").strip()
            if not q: return self._json(400, {"error": "question required"})
            r = get_agent().run(q)
            return self._json(200, r)
        return self._json(404, {"error": "not found"})
    def log_message(self, fmt, *args): pass

def serve(host="127.0.0.1", port=8765):
    httpd = HTTPServer((host, port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd
