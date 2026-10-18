"""Loading and cleaning the IBM TabFormer credit-card transaction dataset."""
import glob
import os

import numpy as np
import pandas as pd

from .config import DATASET_CANDIDATES, LAB_HOMES

RAW_COLS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
            "Merchant Name", "Merchant City", "Merchant State", "MCC", "Errors?", "Is Fraud?"]

STR_DTYPES = {c: "string" for c in ["Time", "Amount", "Use Chip", "Merchant Name", "Merchant City",
                                     "Merchant State", "MCC", "Errors?", "Is Fraud?"]}


def find_dataset():
    # look for the TabFormer csv in the lab container / repo
    for p in DATASET_CANDIDATES:
        if p and os.path.exists(p):
            return p
    hits = [h for home in LAB_HOMES for h in glob.glob(f"{home}/**/*transaction*.csv", recursive=True)]
    hits = [h for h in hits if "test_" not in os.path.basename(h)]
    if hits:
        return max(hits, key=os.path.getsize)
    raise FileNotFoundError(
        "TabFormer CSV not found. Set SAFEPAUSE_DATA=/path/to/transactions_history.csv")


def load_tabformer(path=None, user_mod=None, min_year=None, chunksize=1_000_000, verbose=True):
    """Read the csv in chunks.
    Small lab file -> everything. Full 24M file -> every 4th user, 2015+.
    """
    path = path or find_dataset()
    big = os.path.getsize(path) > 500e6
    if user_mod is None:
        user_mod = 4 if big else 1
    if min_year is None:
        min_year = 2015 if big else 0
    parts, total = [], 0
    for chunk in pd.read_csv(path, usecols=lambda c: c in RAW_COLS, dtype=STR_DTYPES,
                             chunksize=chunksize):
        total += len(chunk)
        mask = (chunk["Year"] >= min_year) & (chunk["User"] % user_mod == 0)
        if mask.any():
            parts.append(chunk.loc[mask])
        if verbose:
            print(f"\rread {total:,} rows, kept {sum(len(p) for p in parts):,}", end="")
    if verbose:
        print()
    df = pd.concat(parts, ignore_index=True)
    return prepare(df)


def prepare(raw):
    # raw TabFormer columns -> our columns
    hm = raw["Time"].astype(str).str.split(":", expand=True)
    out = pd.DataFrame({
        "user": raw["User"].astype("int32").to_numpy(),
        "card": raw["Card"].astype("int16").to_numpy(),
        "ts": pd.to_datetime(dict(year=raw["Year"], month=raw["Month"], day=raw["Day"],
                                  hour=hm[0].astype(int), minute=hm[1].astype(int))).to_numpy(),
        "amount": raw["Amount"].astype(str).str.replace("$", "", regex=False).astype(float).to_numpy(),
        "merchant": raw["Merchant Name"].astype(str).to_numpy(),
        "mcc": raw["MCC"].astype(str).to_numpy(),
        "city": raw["Merchant City"].fillna("NA").astype(str).to_numpy(),
        "state": raw["Merchant State"].fillna("NA").astype(str).to_numpy(),
        "use_chip": raw["Use Chip"].fillna("").astype(str).to_numpy(),
        "errors": raw["Errors?"].fillna("").astype(str).to_numpy(),
        "is_fraud": (raw["Is Fraud?"].astype(str) == "Yes").astype("int8").to_numpy(),
    })
    out = out.sort_values(["ts", "user"], kind="stable").reset_index(drop=True)
    out.insert(0, "tx_id", np.arange(len(out), dtype=np.int64))
    return out


def time_split(df, val_frac=0.1, test_frac=0.2):
    """Split by time: train | val | test.
    Fraud is rare and comes in bursts, so cut points are chosen by fraud count
    (otherwise val/test can end up with no fraud at all).
    """
    fraud_rows = np.flatnonzero(df["is_fraud"].to_numpy() == 1)
    n = len(df)
    if len(fraud_rows) >= 30:
        k_val = fraud_rows[int(len(fraud_rows) * (1 - test_frac - val_frac))]
        k_test = fraud_rows[int(len(fraud_rows) * (1 - test_frac))]
        val_start = max(1, k_val - 1)
        test_start = max(val_start + 1, k_test - 1)
    else:
        test_start = int(n * (1 - test_frac))
        val_start = int(n * (1 - test_frac - val_frac))
    return val_start, test_start
