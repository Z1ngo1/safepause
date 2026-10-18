"""Email to the trusted contact.

Set SMTP_USER, SMTP_APP_PASSWORD (gmail app password) and ALERT_EMAIL_TO.
Without them the mail is saved to artifacts/outbox as .eml.
"""
import os
import smtplib
import ssl
import time
from email.message import EmailMessage

from .config import OUTBOX


class EmailAlerter:
    def __init__(self):
        self.user = os.environ.get("SMTP_USER", "")
        self.password = os.environ.get("SMTP_APP_PASSWORD", "")
        self.to = os.environ.get("ALERT_EMAIL_TO", self.user)
        self.host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.port = int(os.environ.get("SMTP_PORT", "465"))
        self.dashboard_url = os.environ.get("DASHBOARD_URL", "")

    @property
    def live(self):
        return bool(self.user and self.password and self.to)

    def build(self, case, customer_msg, analyst_msg):
        msg = EmailMessage()
        msg["Subject"] = f"SafePause: a ${abs(case['amount']):,.2f} payment by your family member was paused"
        msg["From"] = self.user or "safepause@localhost"
        msg["To"] = self.to or "trusted.contact@example.com"
        reasons = "\n".join(f"  - {r}" for r in case.get("reasons", []))
        body = (
            "Hello,\n\nYou are the trusted contact for a bank customer. SafePause has just paused "
            f"a payment of ${abs(case['amount']):,.2f} because it looks like a possible scam.\n\n"
            f"Why it was paused:\n{reasons}\n\n"
            f"What the customer sees:\n  \"{customer_msg}\"\n\n"
            "Please call them now and check that nobody is pressuring them to send money. "
            "Scammers often pretend to be the bank, the police or a relative in trouble.\n\n"
            f"Case reference: {case['tx_id']} (risk {case['risk']:.2f})\n"
            + (f"Dashboard: {self.dashboard_url}\n" if self.dashboard_url else "")
            + "\nThis alert contains no account numbers or personal identifiers.\nSafePause"
        )
        msg.set_content(body)
        return msg

    def send(self, case, customer_msg, analyst_msg):
        msg = self.build(case, customer_msg, analyst_msg)
        if self.live:
            try:
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.host, self.port, context=ctx, timeout=15) as s:
                    s.login(self.user, self.password)
                    s.send_message(msg)
                return "sent"
            except Exception as e:
                status = f"smtp-failed ({type(e).__name__}), saved"
        else:
            status = "saved (dry-run)"
        path = OUTBOX / f"alert_{case['tx_id']}_{int(time.time())}.eml"
        path.write_bytes(bytes(msg))
        return status
