"""Researcher-facing API: accounts, journal entitlements, literature search,
full-text access and systematic-review workspace.

Auth: ``POST /v1/auth/login`` returns a bearer token; send it as
``Authorization: Bearer <token>``. Journal credentials are stored encrypted and
only ever returned masked.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from agri_advisor.accounts.store import CREDENTIAL_PROVIDERS, AccountStore, User
from agri_advisor.config import REPO_ROOT
from agri_advisor.literature.access import download_fulltext, resolve_access
from agri_advisor.literature.journals import catalogue
from agri_advisor.literature.library import ReviewLibrary
from agri_advisor.literature.models import Paper
from agri_advisor.literature.search import federated_search
from agri_advisor.literature.sources import DEFAULT_SOURCES, SOURCES, openalex_related

router = APIRouter()

PUBLIC_SIGNUP_ROLES = ("researcher", "extension", "public")


# ------------------------------------------------------------- singletons
def _db_path() -> str:
    return os.environ.get("PLATFORM_DB", ".agri_store/platform.sqlite3")


@lru_cache(maxsize=4)
def _store(path: str) -> AccountStore:
    p = Path(path)
    return AccountStore(p if p.is_absolute() else REPO_ROOT / p)


@lru_cache(maxsize=4)
def _library(path: str) -> ReviewLibrary:
    p = Path(path)
    return ReviewLibrary(p if p.is_absolute() else REPO_ROOT / p)


def get_store() -> AccountStore:
    return _store(_db_path())


def get_library() -> ReviewLibrary:
    return _library(_db_path())


def private_dir(user: User) -> Path:
    base = Path(os.environ.get("PRIVATE_LIBRARY_DIR", ".agri_store/library"))
    base = base if base.is_absolute() else REPO_ROOT / base
    return base / str(user.id)


# ------------------------------------------------------------------ auth
def optional_user(authorization: str | None = Header(None)) -> User | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return get_store().verify_token(authorization.split(" ", 1)[1].strip())


def current_user(user: User | None = Depends(optional_user)) -> User:
    if user is None:
        raise HTTPException(401, "login required (Authorization: Bearer <token>)",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "platform admin only")
    return user


class RegisterRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=8)
    name: str | None = None
    org: str | None = Field(None, description="organisation slug, e.g. kalro")
    role: str = Field("researcher", description=f"one of {PUBLIC_SIGNUP_ROLES}")


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/v1/auth/register", tags=["auth"])
def register(req: RegisterRequest) -> dict[str, Any]:
    if req.role not in PUBLIC_SIGNUP_ROLES:
        raise HTTPException(422, f"self-registration roles: {PUBLIC_SIGNUP_ROLES}")
    try:
        user = get_store().register(req.email, req.password, req.name, req.org, req.role)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"user": user.to_dict(), "token": get_store().issue_token(user)}


@router.post("/v1/auth/login", tags=["auth"])
def login(req: LoginRequest) -> dict[str, Any]:
    user = get_store().authenticate(req.email, req.password)
    if not user:
        raise HTTPException(401, "invalid email or password")
    return {"user": user.to_dict(), "token": get_store().issue_token(user),
            "token_type": "bearer"}


@router.get("/v1/auth/me", tags=["auth"])
def me(user: User = Depends(current_user)) -> dict[str, Any]:
    org = get_store().get_org(user.org_id) if user.org_id else None
    return {"user": user.to_dict(), "organisation": org,
            "entitlements": get_store().list_credentials(user)}


# ------------------------------------------------------------------ orgs
class OrgRequest(BaseModel):
    name: str
    slug: str = Field(..., pattern=r"^[a-z0-9-]{2,40}$")
    email_domain: str | None = None


class OrgAdminRequest(BaseModel):
    email: str
    org_admin: bool = True


@router.post("/v1/orgs", tags=["organisations"])
def create_org(req: OrgRequest, _: User = Depends(admin_user)) -> dict[str, Any]:
    try:
        return get_store().create_org(req.name, req.slug, req.email_domain)
    except Exception as exc:  # noqa: BLE001 - sqlite integrity etc.
        raise HTTPException(422, str(exc)) from exc


@router.post("/v1/orgs/{slug}/admins", tags=["organisations"])
def set_org_admin(slug: str, req: OrgAdminRequest,
                  _: User = Depends(admin_user)) -> dict[str, Any]:
    try:
        return get_store().set_org_admin(slug, req.email, req.org_admin)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


# ----------------------------------------------------------- credentials
class CredentialRequest(BaseModel):
    provider: str = Field(..., description=f"one of {CREDENTIAL_PROVIDERS}")
    data: dict[str, str] = Field(..., description="e.g. {'prefix': 'https://ezproxy.../login?url='}")
    label: str | None = None
    for_org: bool = Field(False, description="org admins: share with all organisation members")


@router.get("/v1/credentials", tags=["journal access"])
def list_credentials(user: User = Depends(current_user)) -> list[dict]:
    return get_store().list_credentials(user)


@router.post("/v1/credentials", tags=["journal access"])
def add_credential(req: CredentialRequest, user: User = Depends(current_user)) -> dict:
    try:
        return get_store().add_credential(user, req.provider, req.data, req.label, req.for_org)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.delete("/v1/credentials/{cred_id}", tags=["journal access"])
def delete_credential(cred_id: int, user: User = Depends(current_user)) -> dict:
    try:
        ok = get_store().delete_credential(user, cred_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not ok:
        raise HTTPException(404, "not found")
    return {"deleted": cred_id}


# ------------------------------------------------------------ literature
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    sources: list[str] | None = Field(None, description=f"subset of {list(SOURCES)}")
    limit_per_source: int = Field(25, ge=1, le=200)
    year_from: int | None = None
    year_to: int | None = None
    oa_only: bool = False
    max_results: int | None = Field(100, ge=1, le=1000)
    review_id: int | None = Field(None, description="save results into this review")


@router.get("/v1/literature/sources", tags=["literature"])
def list_sources() -> dict[str, Any]:
    return {"available": list(SOURCES), "default": DEFAULT_SOURCES}


@router.post("/v1/literature/search", tags=["literature"])
def literature_search(req: SearchRequest,
                      user: User | None = Depends(optional_user)) -> dict[str, Any]:
    keys = get_store().search_keys_for(user)
    res = federated_search(req.query, req.sources, req.limit_per_source, keys,
                           req.year_from, req.year_to, req.oa_only, req.max_results)
    out = res.to_dict()
    if req.review_id is not None:
        if user is None:
            raise HTTPException(401, "login required to save into a review")
        _review_or_404(req.review_id, user)
        out["saved"] = get_library().add_search_result(req.review_id, res)
    return out


@router.get("/v1/literature/journals", tags=["literature"])
def journals(subject: list[str] | None = Query(None), live: bool = True,
             per_subject: int = Query(25, ge=1, le=100), region: str | None = None) -> dict:
    return catalogue(subject, live, per_subject, region)


class AccessRequest(BaseModel):
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None


@router.post("/v1/literature/access", tags=["journal access"])
def literature_access(req: AccessRequest,
                      user: User | None = Depends(optional_user)) -> dict[str, Any]:
    """All the ways *this* user can reach the full text (OA + their entitlements)."""
    if not (req.doi or req.url):
        raise HTTPException(422, "doi or url required")
    creds = get_store().credentials_for(user)
    opts = resolve_access(req.doi, req.url, creds, req.pdf_url)
    return {"doi": req.doi, "options": [o.to_dict() for o in opts],
            "entitlements_used": sorted({c["label"] for c in creds})}


class FulltextRequest(AccessRequest):
    review_id: int | None = None
    paper_key: str | None = None


@router.post("/v1/literature/fulltext", tags=["journal access"])
def literature_fulltext(req: FulltextRequest,
                        user: User = Depends(current_user)) -> dict[str, Any]:
    """Download the full text into the user's *private* library (licence-safe)."""
    if not (req.doi or req.url):
        raise HTTPException(422, "doi or url required")
    store = get_store()
    creds = store.credentials_for(user)
    opts = resolve_access(req.doi, req.url, creds, req.pdf_url)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", req.doi or req.url or "paper")[:120]
    res = download_fulltext(opts, creds, private_dir(user), stem)
    store.audit(user.id, "fulltext", f"{req.doi or req.url} ok={res['ok']} via={res.get('via')}")
    if res["ok"]:
        res["file"] = Path(res.pop("path")).name
        if req.review_id is not None and req.paper_key:
            _review_or_404(req.review_id, user)
            get_library().set_fulltext(req.review_id, req.paper_key, res["file"])
    return res


@router.get("/v1/library/files", tags=["journal access"])
def library_files(user: User = Depends(current_user)) -> list[dict]:
    d = private_dir(user)
    return [{"file": p.name, "bytes": p.stat().st_size} for p in sorted(d.glob("*"))] if d.exists() else []


@router.get("/v1/library/files/{name}", tags=["journal access"])
def library_file(name: str, user: User = Depends(current_user)) -> FileResponse:
    path = (private_dir(user) / name).resolve()
    if path.parent != private_dir(user).resolve() or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path)


class SnowballRequest(BaseModel):
    openalex_id: str = Field(..., description="OpenAlex work id, e.g. W2741809807")
    direction: str = Field("citing", pattern="^(citing|references)$")
    limit: int = Field(50, ge=1, le=200)
    review_id: int | None = None


@router.post("/v1/literature/snowball", tags=["literature"])
def snowball(req: SnowballRequest, user: User | None = Depends(optional_user)) -> dict:
    papers = openalex_related(req.openalex_id, req.direction, req.limit,
                              get_store().search_keys_for(user))
    out: dict[str, Any] = {"n": len(papers), "papers": [p.to_dict() for p in papers]}
    if req.review_id is not None:
        if user is None:
            raise HTTPException(401, "login required to save into a review")
        _review_or_404(req.review_id, user)
        out["saved"] = get_library().add_papers(
            req.review_id, papers, f"{req.direction}:{req.openalex_id}", "snowball", ["openalex"])
    return out


# --------------------------------------------------------------- reviews
class ReviewRequest(BaseModel):
    title: str = Field(..., min_length=3)
    question: str | None = None
    pico: dict[str, str] | None = Field(None, description="population/intervention/comparator/outcome")
    inclusion: list[str] = []
    exclusion: list[str] = []
    shared_with_org: bool = False


class ScreenRequest(BaseModel):
    key: str
    decision: str = Field(..., pattern="^(include|exclude|maybe|unscreened)$")
    stage: str = Field("title_abstract", pattern="^(title_abstract|full_text)$")
    reason: str | None = None
    notes: str | None = None


class PapersRequest(BaseModel):
    papers: list[dict[str, Any]]
    origin: str = "manual"


class PrioritiseRequest(BaseModel):
    include_terms: list[str]
    exclude_terms: list[str] = []


def _review_or_404(review_id: int, user: User) -> dict:
    rv = get_library().get_review(review_id, user.id, user.org_id)
    if not rv:
        raise HTTPException(404, "review not found")
    return rv


@router.post("/v1/reviews", tags=["reviews"])
def create_review(req: ReviewRequest, user: User = Depends(current_user)) -> dict:
    protocol = req.model_dump(exclude={"title", "shared_with_org"})
    return get_library().create_review(user.id, req.title, protocol, user.org_id,
                                       req.shared_with_org)


@router.get("/v1/reviews", tags=["reviews"])
def list_reviews(user: User = Depends(current_user)) -> list[dict]:
    return get_library().list_reviews(user.id, user.org_id)


@router.get("/v1/reviews/{review_id}", tags=["reviews"])
def get_review(review_id: int, stage: str | None = None, decision: str | None = None,
               user: User = Depends(current_user)) -> dict:
    rv = _review_or_404(review_id, user)
    rv["papers"] = get_library().papers(review_id, stage, decision)
    return rv


@router.delete("/v1/reviews/{review_id}", tags=["reviews"])
def delete_review(review_id: int, user: User = Depends(current_user)) -> dict:
    _review_or_404(review_id, user)
    try:
        get_library().delete_review(review_id, user.id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    return {"deleted": review_id}


@router.post("/v1/reviews/{review_id}/papers", tags=["reviews"])
def add_review_papers(review_id: int, req: PapersRequest,
                      user: User = Depends(current_user)) -> dict:
    _review_or_404(review_id, user)
    papers = [Paper.from_dict(d) for d in req.papers if d.get("title")]
    return get_library().add_papers(review_id, papers, "", req.origin)


@router.post("/v1/reviews/{review_id}/screen", tags=["reviews"])
def screen(review_id: int, req: ScreenRequest, user: User = Depends(current_user)) -> dict:
    _review_or_404(review_id, user)
    try:
        return get_library().screen(review_id, req.key, req.decision, req.stage, req.reason,
                                    user.email, req.notes)
    except KeyError as exc:
        raise HTTPException(404, f"paper {exc} not in review") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/v1/reviews/{review_id}/prioritise", tags=["reviews"])
def prioritise(review_id: int, req: PrioritiseRequest,
               user: User = Depends(current_user)) -> list[dict]:
    _review_or_404(review_id, user)
    return get_library().prioritise(review_id, req.include_terms, req.exclude_terms)


@router.get("/v1/reviews/{review_id}/prisma", tags=["reviews"])
def prisma(review_id: int, user: User = Depends(current_user)) -> dict:
    _review_or_404(review_id, user)
    return get_library().prisma(review_id)


@router.get("/v1/reviews/{review_id}/export", tags=["reviews"],
            response_class=PlainTextResponse)
def export(review_id: int, fmt: str = Query("bibtex", pattern="^(bibtex|ris|csv)$"),
           which: str = Query("included", pattern="^(included|all)$"),
           user: User = Depends(current_user)) -> PlainTextResponse:
    _review_or_404(review_id, user)
    ext = {"bibtex": "bib", "ris": "ris", "csv": "csv"}[fmt]
    return PlainTextResponse(
        get_library().export(review_id, fmt, which),
        headers={"Content-Disposition": f'attachment; filename="review_{review_id}.{ext}"'},
    )


@router.get("/v1/reviews/{review_id}/synthesis", tags=["reviews"])
def synthesis(review_id: int, use_llm: bool = True,
              user: User = Depends(current_user)) -> dict:
    _review_or_404(review_id, user)
    llm = None
    if use_llm:
        from agri_advisor.nlp.llm import get_llm

        llm = get_llm()
    return get_library().synthesis(review_id, llm=llm)


@router.post("/v1/reviews/{review_id}/ingest", tags=["reviews"])
def ingest(review_id: int, user: User = Depends(current_user)) -> dict:
    """Push open-licence included studies into the knowledge base (needs re-index)."""
    if user.role not in {"admin", "data_owner", "consortium"}:
        raise HTTPException(403, "curator role required to modify the knowledge base")
    _review_or_404(review_id, user)
    out = REPO_ROOT / "data" / "knowledge_base" / "literature" / f"review_{review_id}"
    res = get_library().ingest_to_kb(review_id, out)
    res["next_step"] = "python -m agri_advisor.rag.ingest --rebuild"
    return res
