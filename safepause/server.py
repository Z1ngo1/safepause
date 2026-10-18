"""Scoring API + dashboard (stdlib http.server).

POST /api/score  (one payment or a list)
GET  /api/stats /api/feed /api/cases /api/case?tx_id= /api/health
POST /api/reset
GET  /  dashboard

python -m safepause.server --port 8080
"""
import argparse
import calendar
import json
import os
import platform
import queue
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .alerts import EmailAlerter
from .config import ENGINE_STATE, MODELS
from .features import FeatureEngine
from .genai import Explainer, template_customer, template_review
from .scorer import Scorer
from .store import Store, pseudonym

STATIC = Path(__file__).parent / "static"
API_KEY = os.environ.get("SAFEPAUSE_API_KEY", "")
MAX_BODY = 2_000_000
REQUIRED = ("user", "ts", "amount", "merchant")


def to_epoch(ts):
    if isinstance(ts, (int, float)):
        return int(ts)
    dt = datetime.fromisoformat(str(ts).replace("T", " ").replace("Z", ""))
    return calendar.timegm(dt.timetuple())


def nnpa_available():
    try:
        with open("/proc/cpuinfo") as fh:
            txt = fh.read()
        return " nnpa" in txt or "nnpa " in txt
    except Exception:
        return False


class SafePauseService:
    def __init__(self, use_genai=True, use_seq=True):
        t0 = time.perf_counter()
        self.engine = FeatureEngine.load(ENGINE_STATE) if ENGINE_STATE.exists() else FeatureEngine()
        self.scorer = Scorer(MODELS, use_seq=use_seq)
        self.store = Store()
        self.alerter = EmailAlerter()
        self.explainer = Explainer(enabled=use_genai)
        self.lock = threading.Lock()
        self.jobs = queue.Queue()
        self.started = time.time()
        threading.Thread(target=self._worker, daemon=True).start()
        print(f"[safepause] ready in {time.perf_counter() - t0:.1f}s | customers in state: "
              f"{len(self.engine.users):,} | seq model: {self.scorer.seq_model is not None} | "
              f"email live: {self.alerter.live}")

    def score_one(self, tx):
        missing = [k for k in REQUIRED if k not in tx]
        if missing:
            return {"error": f"missing fields: {missing}"}
        try:
            amount = float(tx["amount"])
            ts = to_epoch(tx["ts"])
        except Exception:
            return {"error": "invalid amount or ts"}
        if abs(amount) > 1e8:
            return {"error": "amount out of range"}
        tx_id = str(tx.get("tx_id", f"{tx['user']}-{ts}"))
        t0 = time.perf_counter()
        with self.lock:
            f, seq = self.engine.compute(int(tx["user"]), ts, amount, str(tx["merchant"]),
                                         str(tx.get("mcc", "")), str(tx.get("city", "NA")),
                                         str(tx.get("use_chip", "")), str(tx.get("errors", "")),
                                         str(tx.get("state", "NA")))
            res = self.scorer.score(f, seq)
        latency = (time.perf_counter() - t0) * 1000
        ev = {"tx_id": tx_id, "ts": str(tx["ts"]), "customer": pseudonym(tx["user"], "c"),
              "merchant": pseudonym(tx["merchant"], "m"), "mcc": str(tx.get("mcc", "")),
              "amount": amount, "use_chip": str(tx.get("use_chip", "")),
              "latency_ms": round(latency, 3), "label": tx.get("is_fraud"), **res}
        self.store.insert(ev)
        if res["decision"] == "pause":
            self.jobs.put(ev)
        elif res["decision"] == "review":
            self.store.update(tx_id, customer_msg=template_review(ev), genai_src="template")
        return {"tx_id": tx_id, "decision": res["decision"], "risk": res["risk"],
                "reasons": res["reasons"], "latency_ms": ev["latency_ms"]}

    # background thread: granite text + email
    def _worker(self):
        self.explainer.load()
        while True:
            ev = self.jobs.get()
            try:
                backlog = self.jobs.qsize()
                if backlog > 20:  # too many in queue, use templates
                    cm, src, ms = template_customer(ev), "template", 0.0
                    am = None
                else:
                    cm, src, ms = self.explainer.customer_message(ev)
                    am, _, ms2 = self.explainer.analyst_summary(ev)
                    ms += ms2
                self.store.update(ev["tx_id"], customer_msg=cm, analyst_msg=am, genai_src=src,
                                  genai_ms=ms)
                status = self.alerter.send(ev, cm, am)
                self.store.update(ev["tx_id"], alert=status)
            except Exception as e:
                self.store.update(ev["tx_id"], alert=f"error: {e}")

    def health(self):
        return {"status": "ok", "uptime_s": round(time.time() - self.started),
                "arch": platform.machine(), "python": platform.python_version(),
                "nnpa": nnpa_available(), "seq_model": self.scorer.seq_model is not None,
                "llm": self.explainer.model_path if self.explainer.llm else None,
                "llm_error": self.explainer.load_error, "email_live": self.alerter.live,
                "queue": self.jobs.qsize(), "customers_in_state": len(self.engine.users),
                "thresholds": {"pause": self.scorer.t_pause, "review": self.scorer.t_review}}


def make_handler(svc):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SafePause/1.0"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            lim = int(q.get("limit", ["40"])[0])
            if u.path in ("/", "/index.html", "/dashboard"):
                return self._send(200, (STATIC / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/health":
                return self._send(200, svc.health())
            if u.path == "/api/stats":
                return self._send(200, svc.store.stats())
            if u.path == "/api/feed":
                return self._send(200, svc.store.feed(min(lim, 200)))
            if u.path == "/api/cases":
                return self._send(200, svc.store.cases(min(lim, 50)))
            if u.path == "/api/case":
                c = svc.store.case(q.get("tx_id", [""])[0])
                return self._send(200 if c else 404, c or {"error": "not found"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if API_KEY and self.headers.get("X-API-Key") != API_KEY:
                return self._send(401, {"error": "unauthorized"})
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._send(413, {"error": "payload too large"})
            u = urlparse(self.path)
            if u.path == "/api/reset":
                svc.store.reset()
                return self._send(200, {"status": "cleared"})
            if u.path != "/api/score":
                return self._send(404, {"error": "not found"})
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON"})
            if isinstance(payload, list):
                return self._send(200, [svc.score_one(t) for t in payload[:5000]])
            return self._send(200, svc.score_one(payload))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-genai", action="store_true")
    ap.add_argument("--no-seq", action="store_true")
    a = ap.parse_args()
    svc = SafePauseService(use_genai=not a.no_genai, use_seq=not a.no_seq)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(svc))
    print(f"[safepause] dashboard: http://{a.host}:{a.port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
