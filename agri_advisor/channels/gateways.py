"""Outbound messaging / payment gateways for Kenyan channels.

=====================  ======================================================
gateway                use
=====================  ======================================================
Africa's Talking       SMS, USSD, Voice on Safaricom, Airtel and Telkom via one
                       account + shortcode (the usual route for Kenyan NGOs and
                       research institutes)
Generic HTTP SMS       direct bulk-SMS contracts with Safaricom (SDP), Airtel
                       or Telkom, or any local aggregator with a JSON/form API
WhatsApp Cloud API     WhatsApp Business messaging
Safaricom Daraja       M-Pesa STK push (optional premium services / payments)
=====================  ======================================================

All gateways honour ``CHANNELS_DRY_RUN`` (default **true**): messages are
logged and returned, not sent, until you configure real credentials and set
``CHANNELS_DRY_RUN=false``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from agri_advisor.channels.kenya import detect_operator, normalise_msisdn

logger = logging.getLogger(__name__)
TIMEOUT = 30


def dry_run() -> bool:
    return os.environ.get("CHANNELS_DRY_RUN", "true").lower() in {"1", "true", "yes"}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class SendResult:
    ok: bool
    gateway: str
    to: list[str]
    dry_run: bool = False
    response: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Gateway:
    name = "base"

    def configured(self) -> bool:
        return False

    def send_sms(self, to: list[str], message: str) -> SendResult:
        raise NotImplementedError

    def _dry(self, to: list[str], message: str, **meta: Any) -> SendResult:
        logger.info("[dry-run %s] to=%s msg=%r", self.name, to, message[:160])
        return SendResult(True, self.name, to, True, {"message": message}, meta=meta)


# ---------------------------------------------------------- Africa's Talking
class AfricasTalking(Gateway):
    """https://developers.africastalking.com - SMS ``/version1/messaging``."""

    name = "africastalking"

    def __init__(self) -> None:
        self.username = _env("AT_USERNAME", "sandbox")
        self.api_key = _env("AT_API_KEY")
        self.sender = _env("AT_SENDER_ID")  # alphanumeric sender ID or shortcode
        sandbox = self.username == "sandbox" or _env("AT_ENV", "sandbox") == "sandbox"
        host = "api.sandbox.africastalking.com" if sandbox else "api.africastalking.com"
        self.sms_url = f"https://{host}/version1/messaging"

    def configured(self) -> bool:
        return bool(self.api_key)

    def send_sms(self, to: list[str], message: str) -> SendResult:
        if dry_run() or not self.configured():
            return self._dry(to, message, reason="dry_run" if dry_run() else "not_configured")
        data = {"username": self.username, "to": ",".join(to), "message": message}
        if self.sender:
            data["from"] = self.sender
        try:
            r = requests.post(self.sms_url, data=data, timeout=TIMEOUT, headers={
                "apiKey": self.api_key, "Accept": "application/json"})
            r.raise_for_status()
            body = r.json()
            recips = body.get("SMSMessageData", {}).get("Recipients", [])
            ok = bool(recips) and all(x.get("statusCode") in (100, 101, 102) for x in recips)
            return SendResult(ok, self.name, to, response=body)
        except Exception as exc:  # noqa: BLE001
            return SendResult(False, self.name, to, error=f"{type(exc).__name__}: {exc}"[:300])


# ---------------------------------------------------------- Generic HTTP SMS
class GenericHTTPSMS(Gateway):
    """Configurable adapter for operator bulk-SMS / SDP contracts.

    Environment (``<P>`` = ``SAFARICOM``, ``AIRTEL`` or ``TELKOM``)::

        SMS_<P>_URL            endpoint
        SMS_<P>_AUTH           "bearer:<token>" | "basic:<user>:<pass>" | "header:<Name>:<value>"
        SMS_<P>_BODY           JSON template using {to}, {message}, {sender}
        SMS_<P>_SENDER         sender ID / shortcode
        SMS_<P>_FORM=true      send the template as form fields instead of JSON
    """

    def __init__(self, operator: str) -> None:
        self.operator = operator
        self.name = f"http_{operator}"
        p = f"SMS_{operator.upper()}_"
        self.url = _env(p + "URL")
        self.auth = _env(p + "AUTH")
        self.template = _env(p + "BODY", '{"to": "{to}", "message": "{message}", "from": "{sender}"}')
        self.sender = _env(p + "SENDER")
        self.form = _env(p + "FORM", "false").lower() == "true"

    def configured(self) -> bool:
        return bool(self.url)

    def _headers(self) -> dict[str, str]:
        kind, _, rest = self.auth.partition(":")
        if kind == "bearer":
            return {"Authorization": f"Bearer {rest}"}
        if kind == "basic":
            return {"Authorization": "Basic " + base64.b64encode(rest.encode()).decode()}
        if kind == "header":
            name, _, value = rest.partition(":")
            return {name: value}
        return {}

    def build_body(self, to: str, message: str) -> dict[str, Any]:
        esc = json.dumps(message)[1:-1]
        return json.loads(self.template.replace("{to}", to).replace("{message}", esc)
                          .replace("{sender}", self.sender))

    def send_sms(self, to: list[str], message: str) -> SendResult:
        if dry_run() or not self.configured():
            return self._dry(to, message, reason="dry_run" if dry_run() else "not_configured")
        responses, ok = [], True
        for n in to:
            body = self.build_body(n.lstrip("+"), message)
            try:
                kw = {"data": body} if self.form else {"json": body}
                r = requests.post(self.url, headers=self._headers(), timeout=TIMEOUT, **kw)
                responses.append({"to": n, "status": r.status_code, "body": r.text[:300]})
                ok &= r.ok
            except Exception as exc:  # noqa: BLE001
                responses.append({"to": n, "error": str(exc)[:200]})
                ok = False
        return SendResult(ok, self.name, to, response=responses)


# ----------------------------------------------------------- WhatsApp Cloud
class WhatsAppCloud(Gateway):
    name = "whatsapp"

    def __init__(self) -> None:
        self.token = _env("WHATSAPP_TOKEN")
        self.phone_id = _env("WHATSAPP_PHONE_NUMBER_ID")
        self.version = _env("WHATSAPP_API_VERSION", "v21.0")

    def configured(self) -> bool:
        return bool(self.token and self.phone_id)

    def send_sms(self, to: list[str], message: str) -> SendResult:
        if dry_run() or not self.configured():
            return self._dry(to, message, reason="dry_run" if dry_run() else "not_configured")
        url = f"https://graph.facebook.com/{self.version}/{self.phone_id}/messages"
        out, ok = [], True
        for n in to:
            try:
                r = requests.post(url, timeout=TIMEOUT, headers={
                    "Authorization": f"Bearer {self.token}"}, json={
                    "messaging_product": "whatsapp", "to": n.lstrip("+"), "type": "text",
                    "text": {"body": message[:4096]}})
                out.append(r.json() if r.content else {"status": r.status_code})
                ok &= r.ok
            except Exception as exc:  # noqa: BLE001
                out.append({"error": str(exc)[:200]})
                ok = False
        return SendResult(ok, self.name, to, response=out)


# ------------------------------------------------------------- M-Pesa Daraja
class Daraja:
    """Safaricom Daraja: OAuth + Lipa na M-Pesa Online (STK push)."""

    def __init__(self) -> None:
        self.key = _env("DARAJA_CONSUMER_KEY")
        self.secret = _env("DARAJA_CONSUMER_SECRET")
        self.shortcode = _env("DARAJA_SHORTCODE", "174379")  # Safaricom sandbox paybill
        self.passkey = _env("DARAJA_PASSKEY")
        self.callback = _env("DARAJA_CALLBACK_URL")
        prod = _env("DARAJA_ENV", "sandbox") == "production"
        self.base = "https://api.safaricom.co.ke" if prod else "https://sandbox.safaricom.co.ke"
        self._token: tuple[str, float] | None = None

    def configured(self) -> bool:
        return bool(self.key and self.secret and self.passkey and self.callback)

    def token(self) -> str:
        if self._token and self._token[1] > time.time():
            return self._token[0]
        r = requests.get(f"{self.base}/oauth/v1/generate",
                         params={"grant_type": "client_credentials"},
                         auth=(self.key, self.secret), timeout=TIMEOUT)
        r.raise_for_status()
        body = r.json()
        self._token = (body["access_token"], time.time() + int(body.get("expires_in", 3599)) - 60)
        return self._token[0]

    def stk_payload(self, phone: str, amount: int, reference: str, description: str,
                    ts: str | None = None) -> dict[str, Any]:
        ts = ts or time.strftime("%Y%m%d%H%M%S")
        msisdn = (normalise_msisdn(phone) or phone).lstrip("+")
        pw = base64.b64encode(f"{self.shortcode}{self.passkey}{ts}".encode()).decode()
        return {
            "BusinessShortCode": self.shortcode, "Password": pw, "Timestamp": ts,
            "TransactionType": "CustomerPayBillOnline", "Amount": int(amount),
            "PartyA": msisdn, "PartyB": self.shortcode, "PhoneNumber": msisdn,
            "CallBackURL": self.callback, "AccountReference": reference[:12],
            "TransactionDesc": description[:13],
        }

    def stk_push(self, phone: str, amount: int, reference: str = "AgriAdvice",
                 description: str = "Advisory") -> dict[str, Any]:
        payload = self.stk_payload(phone, amount, reference, description)
        if dry_run() or not self.configured():
            return {"ok": True, "dry_run": True, "payload": {**payload, "Password": "****"}}
        r = requests.post(f"{self.base}/mpesa/stkpush/v1/processrequest", json=payload,
                          headers={"Authorization": f"Bearer {self.token()}"}, timeout=TIMEOUT)
        return {"ok": r.ok, "status": r.status_code, "response": r.json() if r.content else None}


# ------------------------------------------------------------------- router
def gateway_for(operator: str | None, channel: str = "sms") -> Gateway:
    """Pick the gateway for an operator.

    ``SMS_ROUTE_<OPERATOR>`` = ``africastalking`` (default) | ``http``.
    """
    if channel == "whatsapp":
        return WhatsAppCloud()
    if operator in {"safaricom", "airtel", "telkom"}:
        route = _env(f"SMS_ROUTE_{operator.upper()}", "africastalking")
        if route == "http":
            return GenericHTTPSMS(operator)
    return AfricasTalking()


def send_message(to: list[str] | str, message: str, channel: str = "sms") -> list[dict]:
    """Normalise numbers, group by operator and send through the routed gateway."""
    numbers = [to] if isinstance(to, str) else to
    groups: dict[str, list[str]] = {}
    invalid = []
    for raw in numbers:
        n = normalise_msisdn(raw)
        if not n:
            invalid.append(raw)
            continue
        op = detect_operator(n)["operator"] or "other"
        groups.setdefault(op, []).append(n)
    results = []
    for op, nums in groups.items():
        res = gateway_for(op, channel).send_sms(nums, message)
        res.meta["operator"] = op
        results.append(res.to_dict())
    if invalid:
        results.append(SendResult(False, "none", invalid, error="invalid MSISDN").to_dict())
    return results


def gateway_status() -> dict[str, Any]:
    return {
        "dry_run": dry_run(),
        "africastalking": AfricasTalking().configured(),
        "whatsapp": WhatsAppCloud().configured(),
        "daraja_mpesa": Daraja().configured(),
        "routes": {op: _env(f"SMS_ROUTE_{op.upper()}", "africastalking")
                   for op in ("safaricom", "airtel", "telkom")},
        "http_sms": {op: GenericHTTPSMS(op).configured() for op in ("safaricom", "airtel", "telkom")},
    }
