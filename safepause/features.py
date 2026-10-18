"""Per-customer behaviour features.

Same code is used for training (replaying history) and in the API, so features match.
Only past payments are used for each feature.
"""
import math
import pickle
from collections import deque

import numpy as np

from .config import SEQ_LEN

FEATURES = [
    "log_amount",
    "refund",              # negative amount
    "amount_ratio",        # amount / customer avg
    "amount_z",
    "spend_24h_ratio",     # spent in 24h / avg
    "new_merchant",        # first time paying this merchant
    "new_mcc",
    "new_city",
    "online",
    "has_error",           # bad PIN, bad CVV etc
    "hour",
    "night",               # 0-5h
    "tx_1h",
    "tx_24h",
    "log_secs_since_prev",
    "log_user_history",    # log(number of past payments)
    "mcc_code",            # merchant category as a number
    "foreign",             # merchant state is not a US state (abroad)
    "new_state",
    "new_merch_24h",       # payments to new merchants in last 24h, incl. this one
    "new_frac_recent",     # share of new merchants in the previous payments
]
F = len(FEATURES)
IDX = {name: i for i, name in enumerate(FEATURES)}
NO_PREV_SECS = 30 * 24 * 3600


class FeatureEngine:
    # keeps a small state per customer

    def __init__(self, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.users = {}

    def _new_state(self):
        return {"n": 0, "s": 0.0, "ss": 0.0, "last": None, "recent": deque(),
                "merchants": set(), "mccs": set(), "cities": set(), "states": set(),
                "hist": deque(maxlen=self.seq_len - 1)}

    def compute(self, user, ts_sec, amount, merchant, mcc, city, use_chip, errors, state="NA",
                update=True):
        """Features for one payment -> (f[F], seq[seq_len, F]). ts_sec is unix time."""
        st = self.users.get(user)
        if st is None:
            st = self._new_state()
            if update:
                self.users[user] = st

        a = abs(float(amount))
        n = st["n"]
        mean = st["s"] / n if n else a
        var = max(st["ss"] / n - mean * mean, 0.0) if n else 0.0
        std = math.sqrt(var)

        recent = st["recent"]
        while recent and recent[0][0] < ts_sec - 86400:
            recent.popleft()
        tx_24h = len(recent)
        tx_1h = 0
        spend_24h = a
        is_new = merchant not in st["merchants"]
        new_24h = int(is_new)
        for t, amt, nw in recent:
            spend_24h += amt
            new_24h += nw
            if t >= ts_sec - 3600:
                tx_1h += 1
        secs = ts_sec - st["last"] if st["last"] is not None else NO_PREV_SECS
        hour = int((ts_sec // 3600) % 24)

        f = np.empty(F, dtype=np.float32)
        f[0] = math.copysign(math.log1p(a), float(amount))
        f[1] = 1.0 if float(amount) < 0 else 0.0
        f[2] = min(a / max(mean, 1.0), 50.0)
        f[3] = max(min((a - mean) / (std + 1.0), 20.0), -20.0)
        f[4] = min(spend_24h / max(mean, 1.0), 100.0)
        f[5] = 0.0 if merchant in st["merchants"] else 1.0
        f[6] = 0.0 if mcc in st["mccs"] else 1.0
        f[7] = 0.0 if city in st["cities"] else 1.0
        f[8] = 1.0 if "Online" in use_chip else 0.0
        f[9] = 1.0 if errors else 0.0
        f[10] = hour
        f[11] = 1.0 if hour < 6 else 0.0
        f[12] = tx_1h
        f[13] = tx_24h
        f[14] = math.log1p(max(secs, 0))
        f[15] = math.log1p(n)
        try:
            f[16] = float(mcc)
        except (TypeError, ValueError):
            f[16] = -1.0
        f[17] = 1.0 if len(state) > 2 and state not in ("NA", "nan") else 0.0
        f[18] = 0.0 if state in st["states"] else 1.0
        f[19] = new_24h
        h = st["hist"]
        f[20] = float(np.mean([x[5] for x in h])) if h else 1.0

        seq = np.zeros((self.seq_len, F), dtype=np.float32)
        if h:
            seq[self.seq_len - 1 - len(h):self.seq_len - 1] = np.stack(h)
        seq[-1] = f

        if update:
            st["n"] = n + 1
            st["s"] += a
            st["ss"] += a * a
            st["last"] = ts_sec
            recent.append((ts_sec, a, int(is_new)))
            st["merchants"].add(merchant)
            st["mccs"].add(mcc)
            st["cities"].add(city)
            st["states"].add(state)
            h.append(f)
        return f, seq

    def dumps(self):
        return pickle.dumps(self.users, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self, path):
        with open(path, "wb") as fh:
            fh.write(self.dumps())

    @classmethod
    def load(cls, path, seq_len=SEQ_LEN):
        eng = cls(seq_len)
        with open(path, "rb") as fh:
            eng.users = pickle.load(fh)
        return eng


def build_features(df, snapshot_at=None, verbose=True):
    """Run all rows (sorted by time) through the engine.

    X: features, prev_idx: rows of the customer's previous payments (-1 = none).
    snapshot: engine state right before row snapshot_at, the API starts from it.
    """
    eng = FeatureEngine()
    n = len(df)
    X = np.empty((n, F), dtype=np.float32)
    prev_idx = np.full((n, SEQ_LEN - 1), -1, dtype=np.int32)
    rows_by_user = {}
    snapshot = None

    users = df["user"].to_numpy()
    secs = df["ts"].to_numpy().astype("datetime64[s]").astype(np.int64)
    amounts = df["amount"].to_numpy()
    merchants = df["merchant"].to_numpy()
    mccs = df["mcc"].to_numpy()
    cities = df["city"].to_numpy()
    chips = df["use_chip"].to_numpy()
    errs = df["errors"].to_numpy()
    states = df["state"].to_numpy() if "state" in df else np.full(n, "NA", dtype=object)

    for i in range(n):
        if i == snapshot_at:
            snapshot = eng.dumps()
        u = int(users[i])
        f, _ = eng.compute(u, int(secs[i]), amounts[i], merchants[i], mccs[i], cities[i],
                           chips[i], errs[i], states[i])
        X[i] = f
        dq = rows_by_user.get(u)
        if dq is None:
            dq = rows_by_user[u] = deque(maxlen=SEQ_LEN - 1)
        if dq:
            prev_idx[i, SEQ_LEN - 1 - len(dq):] = dq
        dq.append(i)
        if verbose and i % 200_000 == 0:
            print(f"\rfeatures: {i:,}/{n:,}", end="")
    if verbose:
        print(f"\rfeatures: {n:,}/{n:,} done")
    if snapshot_at is not None and snapshot is None:
        snapshot = eng.dumps()
    return X, prev_idx, snapshot


def gather_sequences(X, prev_idx, rows):
    # [n, SEQ_LEN, F], oldest first, zero padded
    rows = np.asarray(rows)
    seq = np.zeros((len(rows), SEQ_LEN, X.shape[1]), dtype=np.float32)
    seq[:, -1] = X[rows]
    for k in range(SEQ_LEN - 1):
        idx = prev_idx[rows, k]
        ok = idx >= 0
        seq[ok, k] = X[idx[ok]]
    return seq
