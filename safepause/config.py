"""Paths and constants."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SAFEPAUSE_HOME", Path(__file__).resolve().parents[1]))
ARTIFACTS = Path(os.environ.get("SAFEPAUSE_ARTIFACTS", ROOT / "artifacts"))
MODELS = ARTIFACTS / "models"
FIGURES = ARTIFACTS / "figures"
OUTBOX = ARTIFACTS / "outbox"
DB_PATH = ARTIFACTS / "safepause.db"
ENGINE_STATE = ARTIFACTS / "engine_state.pkl"
DATASET_NPZ = ARTIFACTS / "features.npz"
STREAM_CSV = ARTIFACTS / "stream_test.csv"
SECRET_FILE = ARTIFACTS / ".secret"

# 6 previous payments + current one
SEQ_LEN = 7

for _d in (ARTIFACTS, MODELS, FIGURES, OUTBOX):
    _d.mkdir(parents=True, exist_ok=True)

# home dir in the lab image is /home/ibm-user (older images used jovyan)
LAB_HOMES = [str(Path.home()), "/home/ibm-user", "/home/jovyan"]

DATASET_CANDIDATES = [os.environ.get("SAFEPAUSE_DATA", "")] + [
    f"{h}/{sub}{name}" for h in LAB_HOMES for sub in ("", "shared/")
    for name in ("card_transaction.v1.csv", "transactions_history.csv")
] + [str(ROOT / "data" / "card_transaction.v1.csv"), str(ROOT / "data" / "transactions_history.csv")]
