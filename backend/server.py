"""Localhost frontend backend for BOT-Agency. Stdlib only, no new deps.

Run:
  py backend/server.py            # http://127.0.0.1:8000
  py backend/server.py --port 8080

API (same-origin, JSON):
  GET  /api/health   -> {ok, boss_model, agents, memory_docs}
  GET  /api/agents   -> {agents: [{role, model, description}]}
  GET  /api/memory   -> {docs}
  POST /api/memory/clear -> {docs: 0}
  POST /api/chat {message, history?, mode?} -> {mode, final, tasks, results, trace, errors}
    mode: auto (default) | direct | agency | pipeline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agency import registry

_BOSS = None
_BOSS_LOCK = threading.Lock()
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


def get_boss():
    global _BOSS
    with _BOSS_LOCK:
        if _BOSS is None:
            from agency.main_agent import MainAgent

            _BOSS = MainAgent(use_memory=True)
        return _BOSS


def run_auto(goal, history):
    """Mirror main.py run_auto without rich UI. Returns dict."""
    from agency.worker import WorkerAgent

    boss = get_boss()
    decision = boss.decide(goal, history=history, stream_output=False)
    if decision["type"] == "chat":
        boss.remember_exchange(goal, decision["reply"], "chat")
        return {"mode": "chat", "final": decision["reply"], "tasks": [],
                "results": [], "trace": ["router: chat"], "errors": []}
    tasks = decision["tasks"]
    results: list[dict] = []
    trace = [f"router: task x{len(tasks)}"]
    errors: list[str] = []
    for t in tasks:
        try:
            w = WorkerAgent(t["agent"])
            out = w.run(t["instruction"], context=f"Overall goal: {goal}", stream_output=False)
            out["id"] = t["id"]
            results.append(out)
            trace.append(f"worker: {t['agent']}#{t['id']} ok")
        except Exception as e:  # noqa: BLE001 - isolate like pipeline
            results.append({"role": t["agent"], "id": t["id"], "model": "", "output": "", "error": str(e)[:300]})
            errors.append(f"{t['agent']}#{t['id']}: {e}")
            trace.append(f"worker: {t['agent']}#{t['id']} FAILED isolated")
    good = [r for r in results if r.get("output")]
    try:
        final = boss.synthesize(goal, good or [{"role": "none", "id": 0, "output": "no worker output"}],
                                stream_output=False)["content"]
    except Exception as e:  # noqa: BLE001 - easy pipeline: never return empty on synth failure
        fallback = "\n\n".join(f"[{r.get('role')}]: {r.get('output', '')}" for r in good).strip()
        final = fallback or f"Workers finished but synthesis failed: {e}"
        errors.append(f"synthesizer fallback: {e}")
    boss.remember_task(goal, tasks, final)
    return {"mode": "task", "final": final, "tasks": tasks, "results": results, "trace": trace, "errors": errors}


def run_direct(goal, history):
    boss = get_boss()
    res = boss.direct_ask(goal, history=history, stream_output=False)
    boss.remember_exchange(goal, res["content"], "direct")
    return {"mode": "chat", "final": res["content"], "tasks": [],
            "results": [], "trace": ["direct: 1 NIM call"], "errors": []}


def run_agency(goal, history):
    from agency.worker import WorkerAgent

    boss = get_boss()
    tasks = boss.plan(goal, stream_output=False)
    results: list[dict] = []
    trace = [f"plan: {len(tasks)} sub-task(s)"]
    errors: list[str] = []
    for t in tasks:
        try:
            w = WorkerAgent(t["agent"])
            out = w.run(t["instruction"], context=f"Overall goal: {goal}", stream_output=False)
            out["id"] = t["id"]
            results.append(out)
            trace.append(f"worker: {t['agent']}#{t['id']} ok")
        except Exception as e:  # noqa: BLE001
            results.append({"role": t["agent"], "id": t["id"], "model": "", "output": "", "error": str(e)[:300]})
            errors.append(f"{t['agent']}#{t['id']}: {e}")
            trace.append(f"worker: {t['agent']}#{t['id']} FAILED isolated")
    final = None
    try:
        final = boss.synthesize(goal, [r for r in results if r.get("output")] or
                                [{"role": "none", "id": 0, "output": "no worker output"}],
                                stream_output=False)["content"]
    except Exception as e:  # noqa: BLE001 - easy pipeline: never return empty on synth failure
        good = [r for r in results if r.get("output")]
        fallback = "\n\n".join(f"[{r.get('role')}]: {r.get('output', '')}" for r in good).strip()
        final = fallback or f"Workers finished but synthesis failed: {e}"
        errors.append(f"synthesizer fallback: {e}")
    boss.remember_task(goal, tasks, final)
    return {"mode": "task", "final": final, "tasks": tasks, "results": results, "trace": trace, "errors": errors}


def run_pipeline_mode(goal, history):
    from agency.pipeline import run_pipeline

    return run_pipeline(goal, history=history, boss=get_boss())


MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8", ".json": "application/json",
        ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    server_version = "BOT-Agency/1.0"

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p = parsed.path
        if p == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if p == "/api/health":
            boss = get_boss()
            try:
                docs = boss.memory_stats()
            except Exception:
                docs = 0
            has_key = bool(config.NVIDIA_API_KEY) and not config.NVIDIA_API_KEY.startswith("PASTE_")
            return self._send_json({"ok": True, "boss_model": boss.model,
                                    "agents": len(registry.AGENTS), "memory_docs": docs,
                                    "has_key": has_key})
        if p == "/api/agents":
            agents = [{"role": r, "model": registry.model_for(r), "description": i["description"]}
                      for r, i in registry.AGENTS.items()]
            return self._send_json({"agents": agents})
        if p == "/api/memory":
            try:
                return self._send_json({"docs": get_boss().memory_stats()})
            except Exception as e:  # noqa: BLE001
                return self._send_json({"error": str(e)}, status=500)
        # static frontend
        if p in ("/", "/index.html"):
            return self._send_file(os.path.join(FRONTEND_DIR, "index.html"))
        rel = urllib.parse.unquote(p.lstrip("/"))
        # block path traversal
        target = os.path.normpath(os.path.join(FRONTEND_DIR, rel))
        if not target.startswith(os.path.normpath(FRONTEND_DIR)):
            return self.send_error(403, "Forbidden")
        if os.path.isfile(target):
            return self._send_file(target)
        return self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        p = parsed.path
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode() or "{}")
        except Exception:
            return self._send_json({"error": "Invalid JSON body."}, status=400)

        if p == "/api/memory/clear":
            try:
                get_boss().memory_clear()
            except Exception:
                pass
            return self._send_json({"docs": 0})

        if p == "/api/chat":
            message = str(data.get("message", "")).strip()
            history = data.get("history") or []
            mode = str(data.get("mode", "auto")).lower()
            if not message:
                return self._send_json({"error": "Message must not be empty."}, status=400)
            if len(message) > 4000:
                return self._send_json({"error": "Message too long (max 4000 chars)."}, status=400)
            if not isinstance(history, list):
                return self._send_json({"error": "History must be a list."}, status=400)
            history = [h for h in history if isinstance(h, dict)][:20]
            try:
                if mode == "direct":
                    out = run_direct(message, history)
                elif mode == "agency":
                    out = run_agency(message, history)
                elif mode == "pipeline":
                    out = run_pipeline_mode(message, history)
                else:
                    out = run_auto(message, history)
            except RuntimeError as e:
                return self._send_json({"mode": "error", "final": str(e),
                                        "tasks": [], "results": [], "trace": [],
                                        "errors": [str(e)]}, status=502)
            except Exception as e:  # noqa: BLE001
                return self._send_json({"mode": "error", "final": f"Backend failed: {e}",
                                        "tasks": [], "results": [], "trace": [],
                                        "errors": [str(e)]}, status=500)
            return self._send_json(out)

        return self.send_error(404, "Not found")

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="BOT-Agency localhost server (backend + frontend)")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="Open the frontend in a browser")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if not os.path.isdir(FRONTEND_DIR):
        print(f"ERROR: frontend dir missing: {FRONTEND_DIR}", file=sys.stderr)
        return 1
    # Port fallback: try --port .. --port+5 so a stale server gives a clear URL, not a traceback.
    srv = None
    port = args.port
    for _ in range(6):
        try:
            srv = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError as e:
            print(f"Port {port} busy ({e}), trying {port + 1}…", file=sys.stderr)
            port += 1
    if srv is None:
        print(f"ERROR: no free port {args.port}-{port}. Stop the old server first.", file=sys.stderr)
        return 1
    has_key = bool(config.NVIDIA_API_KEY) and not config.NVIDIA_API_KEY.startswith("PASTE_")
    url = f"http://{args.host}:{port}"
    print(f"BOT-Agency frontend: {url}  (Ctrl+C to stop)")
    print(f"Boss: {config.MAIN_MODEL} | agents: {len(registry.AGENTS)} | key: {'ok' if has_key else 'MISSING - put NVIDIA_API_KEY in .env and restart'}")
    print("Open the URL above in a browser (do NOT double-click frontend/index.html).")
    if not has_key:
        print("WARNING: chat calls will fail until a real key is set.", file=sys.stderr)
    if args.open:
        try:
            webbrowser.open(url)
        except Exception as e:  # noqa: BLE001
            print(f"Could not open browser: {e}", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    main()
