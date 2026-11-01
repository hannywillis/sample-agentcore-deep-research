"""Kenyan numbering plan helpers: MSISDN normalisation and operator detection.

Prefix allocations follow the Communications Authority of Kenya numbering
plan as published (2026). Because of mobile number portability the prefix
shows the *issuing* network, not necessarily the current one - when a gateway
supplies a ``networkCode`` (Africa's Talking USSD/SMS callbacks do) that is
used instead and is authoritative.
"""

from __future__ import annotations

import os
import re

# MCC-MNC codes reported by gateways (Africa's Talking ``networkCode``).
NETWORK_CODES = {
    "63902": "safaricom",
    "63903": "airtel",
    "63907": "telkom",
    "99999": "sandbox",  # Africa's Talking simulator
}


def _r(a: int, b: int) -> list[str]:
    return [f"{n:03d}" for n in range(a, b + 1)]


# 3-digit national prefix after the leading 0 / +254 (e.g. "722", "110").
PREFIXES: dict[str, list[str]] = {
    "safaricom": _r(700, 729) + ["740", "741", "742", "743", "745", "746", "748"]
    + _r(757, 759) + ["768", "769"] + _r(790, 799) + _r(110, 119),
    "airtel": _r(730, 739) + _r(750, 756) + _r(780, 789) + _r(100, 109),
    "telkom": _r(770, 779) + _r(120, 129),
    "faiba": ["747"],
    "equitel": _r(763, 766),
}
_PREFIX_INDEX = {p: op for op, lst in PREFIXES.items() for p in lst}


def normalise_msisdn(number: str, default_cc: str = "254") -> str | None:
    """Return E.164 (``+2547XXXXXXXX``) or ``None`` if not a valid mobile number."""
    digits = re.sub(r"\D", "", number or "")
    if digits.startswith("00"):
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 10:
        digits = default_cc + digits[1:]
    elif len(digits) == 9 and digits[0] in "71":
        digits = default_cc + digits
    if digits.startswith("254") and len(digits) != 12:
        return None
    return f"+{digits}" if 10 <= len(digits) <= 15 else None


def is_kenyan(number: str) -> bool:
    n = normalise_msisdn(number)
    return bool(n and n.startswith("+254"))


def detect_operator(number: str, network_code: str | None = None) -> dict[str, str | None]:
    """Identify the operator; ``basis`` says how reliable the answer is."""
    if network_code and str(network_code) in NETWORK_CODES:
        return {"operator": NETWORK_CODES[str(network_code)], "basis": "network_code"}
    n = normalise_msisdn(number)
    if not n or not n.startswith("+254"):
        return {"operator": None, "basis": "not_kenyan"}
    op = _PREFIX_INDEX.get(n[4:7])
    return {"operator": op, "basis": "prefix (may be ported)" if op else "unknown_prefix"}


def default_language(number: str) -> str | None:
    """Kiswahili default for Kenyan numbers (override with KENYA_DEFAULT_LANGUAGE)."""
    if is_kenyan(number):
        return os.environ.get("KENYA_DEFAULT_LANGUAGE", "sw") or None
    return None
