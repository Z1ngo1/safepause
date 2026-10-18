"""Granite (gguf via llama.cpp, runs locally) writes the text for the customer and the analyst.
If the model can't load we fall back to templates."""
import glob
import os
import re
import sys
import time

CUSTOMER_SYSTEM = (
    "You are the fraud-protection assistant of a bank. A payment was automatically paused. "
    "Write a short, calm, friendly message to the customer in plain English (2-3 sentences, "
    "under 60 words). Explain why the payment was paused using ONLY the facts given. "
    "Ask if someone contacted them and asked them to send money urgently. "
    "Never invent numbers, names or facts. Never ask for passwords, PINs or codes."
)
ANALYST_SYSTEM = (
    "You are an assistant for a bank fraud analyst. Write a 2-3 sentence case summary "
    "(under 70 words): risk level, the main warning signals, and one recommended next action "
    "(call the customer, keep on hold, or release). Use ONLY the facts given."
)


def find_granite_model():
    # granite gguf from the lab container
    env = os.environ.get("SAFEPAUSE_GGUF")
    if env and os.path.exists(env):
        return env
    from .config import LAB_HOMES
    for base in LAB_HOMES + [os.getcwd()]:
        if base not in sys.path:
            sys.path.append(base)
    try:  # same path as the loan approval lab
        from loan_demo.rag_utils import MODEL_PATH  # type: ignore
        if os.path.exists(MODEL_PATH):
            return MODEL_PATH
    except Exception:
        pass
    pats = [f"{h}/**/*granite*.gguf" for h in LAB_HOMES] + [f"{h}/**/*.gguf" for h in LAB_HOMES] + \
        ["/models/**/*.gguf", "/opt/**/*granite*.gguf"]
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return sorted(hits, key=os.path.getsize)[0]
    return None


def facts_text(case):
    lines = [
        f"Payment amount: ${abs(case['amount']):,.2f}",
        f"Merchant category code: {case.get('mcc', 'unknown')}",
        f"Payment type: {'online' if 'Online' in str(case.get('use_chip', '')) else 'in person'}",
        f"Risk score: {case['risk']:.2f} (0 = safe, 1 = almost certainly fraud)",
        f"Decision: {case['decision']}",
        "Warning signals:",
    ]
    lines += [f"- {r}" for r in case.get("reasons", [])] or ["- unusual behaviour"]
    return "\n".join(lines)


def template_customer(case):
    reasons = case.get("reasons") or ["this payment looks unusual for your account"]
    why = "; ".join(r[0].lower() + r[1:] for r in reasons[:2])
    return (f"We paused your payment of ${abs(case['amount']):,.2f} to protect you: {why}. "
            "If someone called or messaged you and asked you to send money urgently, please stop "
            "and contact us. If it was really you, just confirm the payment.")


def template_review(case):
    reasons = case.get("reasons") or ["this payment looks unusual for your account"]
    why = "; ".join(r[0].lower() + r[1:] for r in reasons[:2])
    return (f"Please confirm your payment of ${abs(case['amount']):,.2f}. It looks unusual: {why}. "
            "If someone is pressuring you to pay, cancel it and call us.")


def template_analyst(case):
    level = "HIGH" if case["risk"] >= 0.8 else "ELEVATED"
    return (f"{level} risk ({case['risk']:.2f}) payment of ${abs(case['amount']):,.2f}. "
            f"Signals: {'; '.join(case.get('reasons', [])) or 'model anomaly'}. "
            "Recommended action: keep on hold and call the customer to confirm.")


# the lab image has the granite base model (no chat template), so few-shot completion
CUSTOMER_FEWSHOT = """You write short, calm messages from a bank to a customer whose payment was paused because it may be a scam. Use only the facts. Ask if someone contacted them and pressured them to pay. Never ask for passwords, PINs or codes.

Facts:
Amount: $2,400.00
Payment type: online
Signals: First payment to this recipient and 9x the usual amount; Large payment at night (01:00)
Message: We paused your online payment of $2,400.00 because it is your first payment to this recipient and much larger than usual, made late at night. Did someone call or message you and ask you to send money quickly? If so, please stop and call us. If it really was you, just confirm the payment.

Facts:
Amount: $640.00
Payment type: in person
Signals: 4 payments in the previous hour; Errors on this attempt (wrong PIN / card details)
Message: We paused your payment of $640.00 because there were several payments within an hour and a wrong PIN was entered. If anyone is asking you to pay or is using your card, please stop and contact us right away. If it was you, simply confirm the payment.

Facts:
Amount: {amount}
Payment type: {ptype}
Signals: {signals}
Message:"""

ANALYST_FEWSHOT = """Write a short case summary for a bank fraud analyst: risk level, the main signals and one recommended action. Use only the facts.

Facts:
Amount: $2,400.00 online
Risk: 0.97
Signals: First payment to this recipient and 9x the usual amount; Large payment at night (01:00)
Summary: HIGH risk (0.97). First payment to a new recipient at about 9x the customer's usual amount, made online at night. Pattern matches an authorised push payment scam. Action: keep on hold and call the customer before release.

Facts:
Amount: {amount} {ptype}
Risk: {risk}
Signals: {signals}
Summary: {level} risk ({risk})."""

STOPS = ["\n\n", "\nFacts:", "\nMessage:", "\nSummary:"]
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text):
    out = set()
    for n in NUM_RE.findall(text or ""):
        try:
            out.add(f"{float(n.replace(',', '').rstrip('.')):g}")
        except ValueError:
            pass
    return out


class Explainer:
    def __init__(self, model_path=None, n_threads=None, n_ctx=2048, enabled=True):
        self.model_path = model_path
        self.n_threads = n_threads or max(1, (os.cpu_count() or 2))
        self.n_ctx = n_ctx
        self.enabled = enabled
        self.llm = None
        self.load_error = None
        self.is_chat = False
        self.rejected = 0

    def load(self):
        if self.llm is not None or not self.enabled:
            return self.llm
        try:
            from llama_cpp import Llama
            path = self.model_path or find_granite_model()
            if not path:
                raise FileNotFoundError("Granite GGUF model not found")
            t0 = time.perf_counter()
            self.llm = Llama(model_path=path, n_ctx=self.n_ctx, n_threads=self.n_threads,
                             verbose=False)
            self.model_path = path
            self.is_chat = "instruct" in os.path.basename(path).lower()
            print(f"[genai] Granite loaded from {path} in {time.perf_counter() - t0:.1f}s "
                  f"({'chat' if self.is_chat else 'base model, few-shot completion'})")
        except Exception as e:
            self.load_error = repr(e)
            self.enabled = False
            print(f"[genai] LLM unavailable, using templates: {e}")
        return self.llm

    def _generate(self, fewshot, system, case, max_tokens=110):
        llm = self.load()
        if llm is None:
            return None
        fields = {
            "amount": f"${abs(case['amount']):,.2f}",
            "ptype": "online" if "Online" in str(case.get("use_chip", "")) else "in person",
            "signals": "; ".join(case.get("reasons", [])) or "unusual behaviour for this customer",
            "risk": f"{case['risk']:.2f}",
            "level": "HIGH" if case["risk"] >= 0.8 else "ELEVATED",
        }
        try:
            if self.is_chat:
                r = llm.create_chat_completion(
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": facts_text(case)}],
                    max_tokens=max_tokens, temperature=0)
                text = r["choices"][0]["message"]["content"]
            else:
                r = llm.create_completion(prompt=fewshot.format(**fields), max_tokens=max_tokens,
                                          temperature=0, stop=STOPS)
                text = r["choices"][0]["text"]
        except Exception as e:
            print(f"[genai] generation failed: {e}")
            return None
        if not self.is_chat and fewshot is ANALYST_FEWSHOT:
            # risk level is decided by the model score, not by the LLM
            text = f"{fields['level']} risk ({fields['risk']}). " + (text or "").strip()
        text = self._clean(text)
        # drop the answer if it has numbers that are not in the facts
        allowed = _numbers(" ".join(fields.values()) + " " + facts_text(case)) | {"1", "2", "3", "24"}
        if text and not _numbers(text) <= allowed:
            self.rejected += 1
            return None
        return text

    @staticmethod
    def _clean(text):
        text = re.sub(r"\s+", " ", (text or "")).strip().strip('"')
        if len(text) < 20:
            return None
        if len(text) > 600:
            text = text[:600].rsplit(".", 1)[0] + "."
        return text

    def customer_message(self, case):
        t0 = time.perf_counter()
        text = self._generate(CUSTOMER_FEWSHOT, CUSTOMER_SYSTEM, case)
        src = "granite"
        if not text:
            text, src = template_customer(case), "template"
        return text, src, round((time.perf_counter() - t0) * 1000, 1)

    def analyst_summary(self, case):
        t0 = time.perf_counter()
        text = self._generate(ANALYST_FEWSHOT, ANALYST_SYSTEM, case, max_tokens=90)
        src = "granite"
        if not text:
            text, src = template_analyst(case), "template"
        return text, src, round((time.perf_counter() - t0) * 1000, 1)
