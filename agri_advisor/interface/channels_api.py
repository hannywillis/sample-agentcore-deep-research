"""Kenyan communication channels: Safaricom, Airtel, Telkom (via Africa's
Talking or direct operator contracts), WhatsApp Business and M-Pesa.

Configure the callback URLs in the Africa's Talking dashboard:

* USSD   -> ``POST /v1/channels/at/ussd``   (service code e.g. ``*384*123#``)
* SMS    -> ``POST /v1/channels/at/sms``    (shortcode / keyword inbox)
* Voice  -> ``POST /v1/channels/at/voice``  (virtual number)
* Delivery reports -> ``POST /v1/channels/at/delivery``

WhatsApp Cloud API webhook -> ``GET|POST /v1/channels/whatsapp``.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any
from xml.sax.saxutils import escape

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from agri_advisor.accounts.store import User
from agri_advisor.channels.gateways import Daraja, gateway_status, send_message
from agri_advisor.channels.kenya import (
    default_language,
    detect_operator,
    normalise_msisdn,
)
from agri_advisor.interface.research_api import current_user

router = APIRouter(prefix="/v1/channels", tags=["kenyan channels"])
logger = logging.getLogger(__name__)
SENDER_ROLES = {"admin", "extension", "consortium", "data_owner"}


def _engine():  # type: ignore[no-untyped-def]
    from agri_advisor.interface.api import get_engine

    return get_engine()


def _webhook_guard(request: Request) -> None:
    """Optional shared-secret check (?key=...) - AT does not sign callbacks."""
    secret = os.environ.get("CHANNELS_WEBHOOK_SECRET")
    if secret and not hmac.compare_digest(request.query_params.get("key", ""), secret):
        raise HTTPException(403, "bad webhook key")


@router.get("/status")
def status() -> dict[str, Any]:
    return gateway_status()


@router.get("/operator")
def operator(msisdn: str, network_code: str | None = None) -> dict[str, Any]:
    return {"msisdn": normalise_msisdn(msisdn), **detect_operator(msisdn, network_code),
            "default_language": default_language(msisdn)}


# ------------------------------------------------------------------- USSD
@router.post("/at/ussd", response_class=PlainTextResponse)
def at_ussd(
    request: Request,
    sessionId: str = Form(""),  # noqa: N803 - Africa's Talking field names
    serviceCode: str = Form(""),  # noqa: N803
    phoneNumber: str = Form(""),  # noqa: N803
    text: str = Form(""),
    networkCode: str | None = Form(None),  # noqa: N803
) -> str:
    _webhook_guard(request)
    from agri_advisor.interface.api import handle_ussd

    op = detect_operator(phoneNumber, networkCode)
    return handle_ussd(
        sessionId, phoneNumber, text, None, default_language(phoneNumber),
        extra={"operator": op["operator"], "service_code": serviceCode},
    )


# -------------------------------------------------------------------- SMS
@router.post("/at/sms")
def at_sms(
    request: Request,
    from_: str = Form(..., alias="from"),
    to: str = Form(""),
    text: str = Form(""),
    date: str | None = Form(None),
    id: str | None = Form(None),  # noqa: A002
    linkId: str | None = Form(None),  # noqa: N803 - needed for premium replies
    networkCode: str | None = Form(None),  # noqa: N803
) -> dict[str, Any]:
    """Inbound SMS: answer through the engine and reply via the routed gateway."""
    _webhook_guard(request)
    from agri_advisor.interface.formatters import format_sms

    if not text.strip():
        return {"ok": False, "reason": "empty message"}
    rec = _engine().recommend(text, None, None, "sms", "public", from_ or None)
    reply = format_sms(rec)
    sent = send_message(from_, reply)
    return {"ok": all(s["ok"] for s in sent), "reply": reply,
            "operator": detect_operator(from_, networkCode), "delivery": sent}


@router.post("/at/delivery")
async def at_delivery(request: Request) -> dict[str, str]:
    _webhook_guard(request)
    form = dict(await request.form())
    logger.info("AT delivery report: %s", {k: form.get(k) for k in ("id", "status", "phoneNumber")})
    return {"ok": "received"}


# ------------------------------------------------------------------ Voice
def voice_xml(say: str, gather_url: str | None = None, lang: str = "en") -> str:
    # Kiswahili text is read by the gateway's default TTS; for natural Kiswahili
    # audio, pre-record prompts and use <Play url=...> instead.
    say_el = f'<Say voice="woman">{escape(say)}</Say>'
    if gather_url:
        return ('<?xml version="1.0" encoding="UTF-8"?><Response>'
                f'<GetDigits timeout="20" finishOnKey="#" callbackUrl="{escape(gather_url)}">'
                f"{say_el}</GetDigits></Response>")
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{say_el}</Response>'


VOICE_MENU = {
    "en": "Welcome to farm advice. For maize press 1, wheat 2, sorghum 3, rice 4, millet 5, "
          "then the hash key. To request an extension officer call back press 9.",
    "sw": "Karibu kwa ushauri wa kilimo. Kwa mahindi bonyeza 1, ngano 2, mtama 3, mpunga 4, "
          "uwele 5, kisha alama ya reli. Kupigiwa simu na afisa ugani bonyeza 9.",
}


@router.post("/at/voice")
def at_voice(
    request: Request,
    isActive: str = Form("1"),  # noqa: N803
    sessionId: str = Form(""),  # noqa: N803
    callerNumber: str = Form(""),  # noqa: N803
    dtmfDigits: str | None = Form(None),  # noqa: N803
) -> Response:
    """IVR: DTMF crop menu -> spoken, evidence-grounded risk advice."""
    _webhook_guard(request)
    if isActive == "0":  # call ended notification
        return Response("", media_type="text/plain")
    from agri_advisor.config import FOCUS_CROPS
    from agri_advisor.interface.formatters import format_voice

    lang = default_language(callerNumber) or "en"
    lang = lang if lang in VOICE_MENU else "en"
    if not dtmfDigits:
        return Response(voice_xml(VOICE_MENU[lang], str(request.url), lang),
                        media_type="application/xml")
    digit = dtmfDigits.strip("#")
    if digit == "9":
        _engine().protocol.enqueue({"type": "callback_request", "session": sessionId,
                                    "phone": callerNumber, "channel": "voice"})
        from agri_advisor.nlp.phrases import phrase

        return Response(voice_xml(phrase("ussd_expert", lang), None, lang),
                        media_type="application/xml")
    try:
        crop = FOCUS_CROPS[int(digit) - 1]
    except (ValueError, IndexError):
        return Response(voice_xml(VOICE_MENU[lang], str(request.url), lang),
                        media_type="application/xml")
    rec = _engine().recommend(
        f"main pests diseases and risks for {crop} and how to manage them",
        None, lang, "voice", "public", callerNumber or None)
    return Response(voice_xml(format_voice(rec)["text"], None, lang), media_type="application/xml")


# --------------------------------------------------------------- WhatsApp
@router.get("/whatsapp", response_class=PlainTextResponse)
def whatsapp_verify(
    mode: str = Query("", alias="hub.mode"),
    token: str = Query("", alias="hub.verify_token"),
    challenge: str = Query("", alias="hub.challenge"),
) -> str:
    expected = os.environ.get("WHATSAPP_VERIFY_TOKEN")
    if mode == "subscribe" and expected and hmac.compare_digest(token, expected):
        return challenge
    raise HTTPException(403, "verification failed")


@router.post("/whatsapp")
async def whatsapp_inbound(request: Request) -> dict[str, Any]:
    raw = await request.body()
    app_secret = os.environ.get("WHATSAPP_APP_SECRET")
    if app_secret:
        sig = request.headers.get("X-Hub-Signature-256", "")
        good = "sha256=" + hmac.new(app_secret.encode(), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            raise HTTPException(403, "bad signature")
    payload = await request.json()
    from agri_advisor.interface.formatters import format_sms

    replies = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for msg in change.get("value", {}).get("messages", []) or []:
                if msg.get("type") != "text":
                    continue
                sender = "+" + msg["from"].lstrip("+")
                rec = _engine().recommend(msg["text"]["body"], None, None, "sms", "public", sender)
                body = format_sms(rec)
                replies.append({"to": sender, "reply": body,
                                "delivery": send_message(sender, body, "whatsapp")})
    return {"ok": True, "handled": len(replies), "replies": replies}


# --------------------------------------------------------- outbound / M-Pesa
class SendRequest(BaseModel):
    to: list[str] = Field(..., min_length=1, max_length=1000)
    message: str = Field(..., min_length=1, max_length=918)
    channel: str = Field("sms", pattern="^(sms|whatsapp)$")


@router.post("/send")
def send(req: SendRequest, user: User = Depends(current_user)) -> dict[str, Any]:
    """Bulk advisory / alert broadcast (extension officers, curators, admins)."""
    if user.role not in SENDER_ROLES:
        raise HTTPException(403, f"sending requires one of {sorted(SENDER_ROLES)}")
    return {"results": send_message(req.to, req.message, req.channel)}


class STKRequest(BaseModel):
    phone: str
    amount: int = Field(..., ge=1, le=150000)
    reference: str = "AgriAdvice"
    description: str = "Advisory"


@router.post("/mpesa/stk")
def mpesa_stk(req: STKRequest, user: User = Depends(current_user)) -> dict[str, Any]:
    if user.role != "admin":
        raise HTTPException(403, "admin only")
    if not (normalise_msisdn(req.phone) or "").startswith("+254"):
        raise HTTPException(422, "Kenyan M-Pesa number required")
    return Daraja().stk_push(req.phone, req.amount, req.reference, req.description)


@router.post("/mpesa/callback")
async def mpesa_callback(request: Request) -> dict[str, Any]:
    _webhook_guard(request)
    body = await request.json()
    cb = body.get("Body", {}).get("stkCallback", {})
    logger.info("M-Pesa callback %s result=%s", cb.get("CheckoutRequestID"), cb.get("ResultCode"))
    return {"ResultCode": 0, "ResultDesc": "Accepted"}
