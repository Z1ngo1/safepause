"""Sends test-period transactions to the API like a live stream.

python scripts/simulate_stream.py --rate 20
python scripts/simulate_stream.py --mode all --rate 0 --batch 500   (load test)
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from safepause.config import STREAM_CSV  # noqa: E402


def post(url, payload, key=""):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def load_stream(mode, limit, seed=7):
    df = pd.read_csv(STREAM_CSV, dtype={"merchant": str, "mcc": str, "city": str, "state": str,
                                        "use_chip": str, "errors": str})
    df["errors"] = df["errors"].fillna("")
    df["use_chip"] = df["use_chip"].fillna("")
    if "state" in df:
        df["state"] = df["state"].fillna("NA")
    if mode == "demo":
        # users with fraud in test period + some random others, time order kept
        fraud_users = df.loc[df.is_fraud == 1, "user"].unique()
        others = pd.Series(df["user"].unique()).sample(frac=0.15, random_state=seed).to_numpy()
        df = df[df["user"].isin(set(fraud_users) | set(others))]
        # start a bit before the first fraud
        first = df.index[df.is_fraud == 1]
        if len(first):
            df = df.loc[max(df.index[0], first[0] - 300):]
    return df.head(limit) if limit else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("SAFEPAUSE_URL", "http://localhost:8080"))
    ap.add_argument("--mode", choices=["demo", "all"], default="demo")
    ap.add_argument("--rate", type=float, default=20, help="payments per second (0 = max speed)")
    ap.add_argument("--batch", type=int, default=1, help="micro-batch size per request")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reset", action="store_true", help="clear the dashboard log first")
    a = ap.parse_args()
    key = os.environ.get("SAFEPAUSE_API_KEY", "")
    if a.reset:
        post(a.url + "/api/reset", {}, key)

    df = load_stream(a.mode, a.limit)
    cols = ["tx_id", "user", "ts", "amount", "merchant", "mcc", "city", "state", "use_chip", "errors",
            "is_fraud"]
    records = df[cols].to_dict("records")
    print(f"streaming {len(records):,} payments ({int(df.is_fraud.sum())} labelled fraud) to {a.url}")

    t_start = time.perf_counter()
    sent = paused = 0
    lat = []
    for i in range(0, len(records), a.batch):
        chunk = records[i:i + a.batch]
        for r in chunk:
            r["tx_id"] = str(r["tx_id"])
            r["is_fraud"] = int(r["is_fraud"])
        t0 = time.perf_counter()
        res = post(a.url + "/api/score", chunk if a.batch > 1 else chunk[0], key)
        res = res if isinstance(res, list) else [res]
        for r, src in zip(res, chunk):
            lat.append(r.get("latency_ms", 0))
            if r.get("decision") == "pause":
                paused += 1
                tag = "FRAUD" if src["is_fraud"] else "legit"
                print(f"  PAUSE  ${abs(src['amount']):>9,.2f}  risk {r['risk']:.2f}  [{tag}]  "
                      f"{'; '.join(r.get('reasons', []))[:110]}")
        sent += len(chunk)
        if a.rate > 0:
            wait = sent / a.rate - (time.perf_counter() - t_start)
            if wait > 0:
                time.sleep(wait)
        if sent % 1000 < a.batch:
            el = time.perf_counter() - t_start
            print(f"... {sent:,} sent | {sent / el:,.0f} payments/s | paused {paused}")
    el = time.perf_counter() - t_start
    lat.sort()
    if lat:
        print(f"\nDONE {sent:,} payments in {el:.1f}s = {sent / el:,.0f}/s end-to-end over HTTP")
        print(f"server-side scoring latency: p50 {lat[len(lat) // 2]:.2f} ms, "
              f"p99 {lat[int(len(lat) * .99)]:.2f} ms | paused {paused}")


if __name__ == "__main__":
    main()
