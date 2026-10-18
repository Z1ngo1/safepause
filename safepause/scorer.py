"""Risk score = XGBoost + LSTM (runs on NNPA) + scam rules. Decision: allow / review / pause."""
import json
import os

import numpy as np

from .config import MODELS, SEQ_LEN
from .features import F, FEATURES, IDX

# scam rules
RULES = [
    ("new_payee_big", "First payment to this recipient and {amount_ratio:.0f}x the usual amount"),
    ("burst", "{tx_1h:.0f} payments in the previous hour"),
    ("night_big", "Large payment at night ({hour:02.0f}:00)"),
    ("online_new", "Online payment to a recipient never used before"),
    ("errors", "Errors on this attempt (wrong PIN / card details)"),
    ("spend_spike", "Spent {spend_24h_ratio:.0f}x the usual amount within 24 hours"),
]
RULE_CODES = [r[0] for r in RULES]


def rule_matrix(X):
    # bool matrix [N, n_rules]
    X = np.atleast_2d(X)
    c = lambda name: X[:, IDX[name]]
    return np.stack([
        (c("new_merchant") > 0) & (c("amount_ratio") >= 3) & (c("log_user_history") > 1.5),
        c("tx_1h") >= 3,
        (c("night") > 0) & (c("amount_ratio") >= 2),
        (c("online") > 0) & (c("new_merchant") > 0) & (c("log_user_history") > 1.5),
        c("has_error") > 0,
        c("spend_24h_ratio") >= 8,
    ], axis=1)


FEATURE_TEXT = {
    "amount_ratio": "Amount is {amount_ratio:.1f}x this customer's average",
    "amount_z": "Amount is far outside this customer's normal range",
    "spend_24h_ratio": "Unusual total spending in the last 24 hours",
    "new_merchant": "Recipient never paid before",
    "new_mcc": "New type of merchant for this customer",
    "new_city": "Payment in a city never used before",
    "online": "Online (card-not-present) payment",
    "has_error": "Errors on this payment attempt",
    "hour": "Unusual time of day ({hour:02.0f}:00)",
    "night": "Night-time payment",
    "tx_1h": "Several payments within one hour",
    "tx_24h": "Many payments within 24 hours",
    "log_secs_since_prev": "Unusual time gap since previous payment",
    "log_amount": "Unusual payment amount",
    "refund": "Refund / reversal pattern",
    "log_user_history": "Little payment history for this customer",
    "mcc_code": "Merchant type often used in fraud",
    "foreign": "Payment to a merchant abroad",
    "new_state": "Payment in a state / country never used before",
    "new_merch_24h": "{new_merch_24h:.0f} new recipients within 24 hours",
    "new_frac_recent": "Recent payments mostly to new recipients",
}


def combine(p_xgb, p_seq, n_rules, meta):
    # used in training (threshold search) and in the API
    w = meta.get("w_xgb", 0.6)
    risk = w * np.asarray(p_xgb) + (1 - w) * np.asarray(p_seq)
    risk = risk + meta.get("rule_boost", 0.05) * np.asarray(n_rules)
    return np.clip(risk, 0.0, 1.0)


def build_seq_model(norm_data=None, units=64):
    # LSTM over last SEQ_LEN payments, LSTM/Dense ops go to NNPA through ibm-zdnn-plugin
    import tensorflow as tf
    inp = tf.keras.Input(shape=(SEQ_LEN, F), name="payments")
    norm = tf.keras.layers.Normalization(axis=-1, name="norm")
    if norm_data is not None:
        norm.adapt(norm_data)
    x = norm(inp)
    x = tf.keras.layers.LSTM(units, name="lstm")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="risk")(x)
    model = tf.keras.Model(inp, out, name="safepause_seq")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model


class Scorer:
    def __init__(self, model_dir=MODELS, use_seq=True):
        import xgboost as xgb
        with open(os.path.join(model_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.booster = xgb.Booster()
        self.booster.load_model(os.path.join(model_dir, "xgb.json"))
        self.booster.set_param({"nthread": 1})
        self._xgb = xgb
        self.seq_model = None
        if use_seq and os.path.exists(os.path.join(model_dir, "seq_model.keras")):
            import tensorflow as tf
            self.seq_model = tf.keras.models.load_model(os.path.join(model_dir, "seq_model.keras"))
            self._seq_fn = tf.function(
                lambda x: self.seq_model(x, training=False),
                input_signature=[tf.TensorSpec([None, SEQ_LEN, F], tf.float32)])
            self._seq_fn(np.zeros((1, SEQ_LEN, F), np.float32))  # warm up
        self.t_pause = self.meta["t_pause"]
        self.t_review = self.meta["t_review"]

    def predict_xgb(self, X):
        return self.booster.inplace_predict(np.ascontiguousarray(X, dtype=np.float32))

    def predict_seq(self, SEQ):
        if self.seq_model is None:
            return np.zeros(len(SEQ), dtype=np.float32)
        return self._seq_fn(np.asarray(SEQ, dtype=np.float32)).numpy().ravel()

    def score_batch(self, X, SEQ):
        p_x = self.predict_xgb(X)
        p_s = self.predict_seq(SEQ)
        rules = rule_matrix(X)
        risk = combine(p_x, p_s, rules.sum(1), self.meta)
        dec = np.where(risk >= self.t_pause, "pause", np.where(risk >= self.t_review, "review", "allow"))
        return risk, dec, p_x, p_s, rules

    def score(self, f, seq, explain=True):
        f2 = f.reshape(1, -1)
        p_x = float(self.predict_xgb(f2)[0])
        p_s = float(self.predict_seq(seq.reshape(1, SEQ_LEN, F))[0])
        fired = rule_matrix(f2)[0]
        risk = float(combine(p_x, p_s, int(fired.sum()), self.meta))
        decision = "pause" if risk >= self.t_pause else "review" if risk >= self.t_review else "allow"
        out = {"risk": round(risk, 4), "p_xgb": round(p_x, 4), "p_seq": round(p_s, 4),
               "decision": decision,
               "rules": [RULE_CODES[i] for i in np.flatnonzero(fired)]}
        if explain and decision != "allow":
            out["reasons"] = self.reasons(f, fired)
        else:
            out["reasons"] = []
        return out

    def reasons(self, f, fired=None, k=3):
        vals = {name: float(f[i]) for i, name in enumerate(FEATURES)}
        texts = []
        if fired is None:
            fired = rule_matrix(f.reshape(1, -1))[0]
        for i in np.flatnonzero(fired):
            texts.append(RULES[i][1].format(**vals))
        # top xgboost contributions (shap values from pred_contribs)
        dm = self._xgb.DMatrix(f.reshape(1, -1), feature_names=FEATURES)
        contrib = self.booster.predict(dm, pred_contribs=True)[0][:-1]
        for j in np.argsort(-contrib):
            if contrib[j] <= 0 or len(texts) >= k:
                break
            name = FEATURES[j]
            if name in ("new_merchant", "refund", "online", "has_error", "night", "new_mcc", "new_city",
                        "foreign", "new_state") and vals[name] == 0:
                continue
            t = FEATURE_TEXT[name].format(**vals)
            if t not in texts:
                texts.append(t)
        return texts[:k]
