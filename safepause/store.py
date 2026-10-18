"""SQLite log of decisions. Customer/merchant ids are stored as HMAC-SHA256 pseudonyms."""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time

from .config import DB_PATH, SECRET_FILE


def _secret():
    env = os.environ.get("SAFEPAUSE_SECRET")
    if env:
        return env.encode()
    if not SECRET_FILE.exists():
        SECRET_FILE.write_text(secrets.token_hex(32))
        os.chmod(SECRET_FILE, 0o600)
    return SECRET_FILE.read_text().strip().encode()


_KEY = None


def pseudonym(value, prefix="c"):
    global _KEY
    if _KEY is None:
        _KEY = _secret()
    return prefix + hmac.new(_KEY, str(value).encode(), hashlib.sha256).hexdigest()[:10]


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  tx_id TEXT PRIMARY KEY, ts TEXT, received REAL, customer TEXT, merchant TEXT, mcc TEXT,
  amount REAL, use_chip TEXT, risk REAL, p_xgb REAL, p_seq REAL, decision TEXT,
  rules TEXT, reasons TEXT, latency_ms REAL, label INTEGER,
  customer_msg TEXT, analyst_msg TEXT, genai_src TEXT, genai_ms REAL, alert TEXT
);
CREATE INDEX IF NOT EXISTS ix_received ON events(received);
CREATE INDEX IF NOT EXISTS ix_decision ON events(decision);
"""


class Store:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)

    def insert(self, ev):
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO events (tx_id, ts, received, customer, merchant, mcc, amount,"
                " use_chip, risk, p_xgb, p_seq, decision, rules, reasons, latency_ms, label)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(ev["tx_id"]), ev["ts"], time.time(), ev["customer"], ev["merchant"],
                 ev["mcc"], ev["amount"], ev["use_chip"], ev["risk"], ev["p_xgb"], ev["p_seq"],
                 ev["decision"], json.dumps(ev["rules"]), json.dumps(ev["reasons"]),
                 ev["latency_ms"], ev.get("label")))
            self.db.commit()

    def update(self, tx_id, **fields):
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.lock:
            self.db.execute(f"UPDATE events SET {cols} WHERE tx_id=?", (*fields.values(), str(tx_id)))
            self.db.commit()

    def _rows(self, sql, args=()):
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("rules", "reasons"):
                if k in d and d[k]:
                    d[k] = json.loads(d[k])
            out.append(d)
        return out

    def feed(self, limit=40):
        return self._rows("SELECT tx_id, ts, received, customer, mcc, amount, use_chip, risk,"
                          " decision, latency_ms, label FROM events ORDER BY received DESC LIMIT ?",
                          (limit,))

    def cases(self, limit=12):
        return self._rows("SELECT * FROM events WHERE decision IN ('pause','review')"
                          " ORDER BY decision='pause' DESC, received DESC LIMIT ?", (limit,))

    def case(self, tx_id):
        r = self._rows("SELECT * FROM events WHERE tx_id=?", (str(tx_id),))
        return r[0] if r else None

    def stats(self):
        with self.lock:
            c = self.db.execute(
                "SELECT COUNT(*), SUM(decision='pause'), SUM(decision='review'), AVG(latency_ms),"
                " SUM(label=1), SUM(label=1 AND decision IN ('pause','review')),"
                " SUM(decision IN ('pause','review') AND label=0), SUM(amount*(label=1 AND decision='pause'))"
                " FROM events").fetchone()
            now = time.time()
            last10 = self.db.execute("SELECT COUNT(*) FROM events WHERE received>=?", (now - 10,)).fetchone()[0]
            lat = [r[0] for r in self.db.execute(
                "SELECT latency_ms FROM events ORDER BY received DESC LIMIT 2000").fetchall()]
            series = self.db.execute(
                "SELECT CAST(received AS INTEGER) AS s, COUNT(*), AVG(latency_ms),"
                " SUM(decision='pause') FROM events WHERE received>=? GROUP BY s ORDER BY s",
                (now - 120,)).fetchall()
        lat.sort()
        p = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 3) if lat else 0
        total = c[0] or 0
        return {
            "processed": total, "paused": c[1] or 0, "review": c[2] or 0,
            "avg_latency_ms": round(c[3] or 0, 3), "p50_ms": p(0.5), "p99_ms": p(0.99),
            "tps_10s": round(last10 / 10, 1),
            "fraud_seen": c[4] or 0, "fraud_caught": c[5] or 0, "false_alarms": c[6] or 0,
            "money_protected": round(c[7] or 0, 2),
            "series": [{"t": s, "tps": n, "lat": round(l or 0, 3), "paused": pz or 0}
                       for s, n, l, pz in series],
            "latency_hist": lat[:2000],
        }

    def reset(self):
        with self.lock:
            self.db.execute("DELETE FROM events")
            self.db.commit()
