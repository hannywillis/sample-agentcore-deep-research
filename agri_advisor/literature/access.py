"""Full-text access resolution: open access first, then the user's or
organisation's institutional / personal journal entitlements.

Supported entitlement types (stored encrypted in the credential vault):

======================  =====================================================
provider                fields
======================  =====================================================
``ezproxy``             ``prefix`` (e.g. ``https://ezproxy.uni.ac.ke/login?url=``),
                        optional ``username`` / ``password`` for server-side
                        sessions (only if ``ALLOW_PROXY_SESSION_LOGIN=true``)
``openathens``          ``domain`` (e.g. ``kalro.org``) - redirector links
``research4life``       ``prefix`` from your AGORA/Hinari/OARE portal (EZproxy
                        style), optional ``username`` / ``password``
``link_template``       ``template`` with ``{doi}`` / ``{url}`` placeholders
                        (Shibboleth WAYFless URLs, library link resolvers)
``elsevier``            ``api_key`` and optional ``insttoken`` (TDM API)
``springer``            ``api_key`` (Springer Nature Open Access / TDM API)
``wiley``               ``tdm_token`` (Wiley TDM API)
``unpaywall``           ``email``
``openalex`` / ``core`` / ``semantic_scholar``  ``api_key`` (search quotas)
======================  =====================================================

Licence note: subscription content retrieved with institutional credentials
may only be used under the subscriber's licence terms (most permit personal
research use / TDM, not redistribution). Licensed full texts are therefore
stored in the *owner's private research library*, never in the public farmer
knowledge base (see governance ``closed`` level).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from agri_advisor.literature.models import normalise_doi

logger = logging.getLogger(__name__)
TIMEOUT = 40
UA = "agri-advisor-literature/1.0"

ELSEVIER_PREFIXES = ("10.1016/",)
SPRINGER_PREFIXES = ("10.1007/", "10.1186/", "10.1038/", "10.1140/")
WILEY_PREFIXES = ("10.1002/", "10.1111/")


@dataclass
class AccessOption:
    method: str  # open_access | publisher_api | proxy_link | openathens | template
    url: str
    via: str  # credential label / "public"
    downloadable_by_server: bool
    requires_browser_login: bool
    licence_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _proxy_session_allowed() -> bool:
    return os.environ.get("ALLOW_PROXY_SESSION_LOGIN", "false").lower() in {"1", "true", "yes"}


def unpaywall_lookup(doi: str, email: str) -> dict[str, Any] | None:
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{quote(doi)}",
            params={"email": email}, timeout=TIMEOUT, headers={"User-Agent": UA},
        )
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def resolve_access(
    doi: str | None,
    landing_url: str | None = None,
    credentials: list[dict[str, Any]] | None = None,
    known_pdf_url: str | None = None,
) -> list[AccessOption]:
    """Return every way the caller can reach the full text, best first.

    ``credentials`` = decrypted entries ``{"provider", "label", "data": {...}}``
    from the vault (personal entries first, then organisation entries).
    """
    doi = normalise_doi(doi)
    creds = credentials or []
    options: list[AccessOption] = []
    target = f"https://doi.org/{doi}" if doi else (landing_url or "")

    # 1. Open access ---------------------------------------------------------
    if known_pdf_url:
        options.append(AccessOption("open_access", known_pdf_url, "public", True, False))
    email = next(
        (c["data"].get("email") for c in creds if c["provider"] == "unpaywall"),
        os.environ.get("UNPAYWALL_EMAIL") or os.environ.get("LITERATURE_CONTACT_EMAIL"),
    )
    if doi and email:
        up = unpaywall_lookup(doi, email)
        for loc in (up or {}).get("oa_locations", []) or []:
            url = loc.get("url_for_pdf") or loc.get("url")
            if url and all(o.url != url for o in options):
                options.append(
                    AccessOption(
                        "open_access", url, f"unpaywall:{loc.get('host_type')}",
                        bool(loc.get("url_for_pdf")), False,
                        f"licence: {loc.get('license') or 'unspecified'}",
                    )
                )

    # 2. Publisher text-and-data-mining APIs ---------------------------------
    for c in creds:
        d, label = c["data"], c.get("label") or c["provider"]
        if not doi:
            break
        if c["provider"] == "elsevier" and doi.startswith(ELSEVIER_PREFIXES) and d.get("api_key"):
            options.append(AccessOption(
                "publisher_api", f"https://api.elsevier.com/content/article/doi/{doi}",
                label, True, False, "Elsevier TDM licence terms apply",
            ))
        if c["provider"] == "springer" and doi.startswith(SPRINGER_PREFIXES) and d.get("api_key"):
            options.append(AccessOption(
                "publisher_api",
                f"https://api.springernature.com/openaccess/jats?q=doi:{doi}",
                label, True, False, "Springer Nature API terms apply",
            ))
        if c["provider"] == "wiley" and doi.startswith(WILEY_PREFIXES) and d.get("tdm_token"):
            options.append(AccessOption(
                "publisher_api",
                f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{quote(doi, safe='')}",
                label, True, False, "Wiley TDM licence terms apply",
            ))

    # 3. Institutional proxies / single sign-on --------------------------------
    for c in creds:
        d, label, prov = c["data"], c.get("label") or c["provider"], c["provider"]
        if not target:
            break
        if prov in {"ezproxy", "research4life"} and d.get("prefix"):
            server = bool(d.get("username") and d.get("password") and _proxy_session_allowed())
            options.append(AccessOption(
                "proxy_link", f"{d['prefix']}{target}", label, server, not server,
                "Use within your institution's licence",
            ))
        elif prov == "openathens" and d.get("domain"):
            options.append(AccessOption(
                "openathens",
                f"https://go.openathens.net/redirector/{d['domain']}?url={quote(target, safe='')}",
                label, False, True, "Sign in with your OpenAthens account",
            ))
        elif prov == "link_template" and d.get("template"):
            url = d["template"].replace("{doi}", doi or "").replace(
                "{url}", quote(target, safe="")
            )
            options.append(AccessOption("template", url, label, False, True))

    if target and not options:
        options.append(AccessOption("landing_page", target, "public", False, False,
                                    "No open copy or entitlement found"))
    return options


# ------------------------------------------------------------------- download
def _publisher_headers(cred: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    d, prov = cred["data"], cred["provider"]
    if prov == "elsevier":
        h = {"X-ELS-APIKey": d["api_key"], "Accept": "application/pdf"}
        if d.get("insttoken"):
            h["X-ELS-Insttoken"] = d["insttoken"]
        return h, {}
    if prov == "springer":
        return {}, {"api_key": d["api_key"]}
    if prov == "wiley":
        return {"Wiley-TDM-Client-Token": d["tdm_token"]}, {}
    return {}, {}


def _ezproxy_session(cred: dict[str, Any]) -> requests.Session:
    """Log in to an EZproxy-style server (standard ``user``/``pass`` form)."""
    d = cred["data"]
    base = re.sub(r"/login.*$", "", d["prefix"])
    s = requests.Session()
    s.headers["User-Agent"] = UA
    r = s.post(f"{base}/login", data={"user": d["username"], "pass": d["password"]},
               timeout=TIMEOUT, allow_redirects=True)
    if r.status_code >= 400 or ("login" in r.url.lower() and "invalid" in r.text.lower()):
        raise PermissionError(f"Proxy login failed for {cred.get('label')}")
    return s


def download_fulltext(
    options: list[AccessOption],
    credentials: list[dict[str, Any]],
    dest_dir: Path,
    stem: str,
) -> dict[str, Any]:
    """Try server-downloadable options in order; save the first that works."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    by_label = {(c.get("label") or c["provider"]): c for c in credentials}
    errors = []
    for opt in options:
        if not opt.downloadable_by_server:
            continue
        try:
            headers, params = {"User-Agent": UA}, {}
            session: Any = requests
            cred = by_label.get(opt.via)
            if opt.method == "publisher_api" and cred:
                h, params = _publisher_headers(cred)
                headers.update(h)
            elif opt.method == "proxy_link" and cred:
                session = _ezproxy_session(cred)
            r = session.get(opt.url, headers=headers, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            ext = ".pdf" if "pdf" in ctype or r.content[:4] == b"%PDF" else (
                ".xml" if "xml" in ctype else ".html"
            )
            if ext == ".html" and opt.method == "open_access" and len(r.content) < 5000:
                raise ValueError("landing page, not full text")
            path = dest_dir / f"{stem}{ext}"
            path.write_bytes(r.content)
            return {"ok": True, "path": str(path), "via": opt.via, "method": opt.method,
                    "bytes": len(r.content), "licence_note": opt.licence_note}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{opt.method}/{opt.via}: {type(exc).__name__}: {exc}"[:200])
    return {"ok": False, "errors": errors,
            "browser_links": [o.to_dict() for o in options if o.requires_browser_login]}
