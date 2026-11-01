"""Platform accounts, organisations and the encrypted journal-credential vault.

* Users log in to the platform (email + password, scrypt-hashed) and receive a
  signed bearer token.
* Users may belong to an organisation (e.g. KALRO, UFRN). Organisation admins
  register institutional entitlements (EZproxy, OpenAthens, Research4Life,
  publisher TDM keys) once; every member then gets that access automatically.
* Users may also add personal credentials; personal entries take precedence.
* Credential payloads are encrypted at rest with Fernet (AES-128-CBC + HMAC)
  using ``AGRI_VAULT_KEY``. Secrets are never returned by the API - only masked.

Storage: SQLite (``PLATFORM_DB``). Swap for Postgres/DynamoDB in production.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from agri_advisor.config import REPO_ROOT

logger = logging.getLogger(__name__)

PLATFORM_ROLES = ("public", "extension", "researcher", "consortium", "data_owner", "admin")
CREDENTIAL_PROVIDERS = (
    "ezproxy", "openathens", "research4life", "link_template", "elsevier",
    "springer", "wiley", "unpaywall", "openalex", "core", "semantic_scholar",
)
SECRET_FIELDS = {"password", "api_key", "insttoken", "tdm_token", "token"}
TOKEN_TTL_S = 12 * 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orgs (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, slug TEXT UNIQUE NOT NULL,
  email_domain TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT,
  pw_hash TEXT NOT NULL, org_id INTEGER REFERENCES orgs(id),
  role TEXT NOT NULL DEFAULT 'researcher', org_admin INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS credentials (
  id INTEGER PRIMARY KEY, owner_type TEXT NOT NULL CHECK(owner_type IN ('user','org')),
  owner_id INTEGER NOT NULL, provider TEXT NOT NULL, label TEXT NOT NULL,
  blob BLOB NOT NULL, created REAL NOT NULL, last_used REAL);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, user_id INTEGER, action TEXT, detail TEXT);
"""


def _env_path(name: str, default: str) -> Path:
    p = Path(os.environ.get(name, default))
    return p if p.is_absolute() else REPO_ROOT / p


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, _ = stored.split("$")
    except ValueError:
        return False
    return hmac.compare_digest(_hash_password(password, base64.b64decode(salt_b64)), stored)


def mask(value: str) -> str:
    return "****" if len(value) <= 6 else f"{value[:2]}****{value[-2:]}"


@dataclass
class User:
    id: int
    email: str
    name: str | None
    org_id: int | None
    role: str
    org_admin: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class AccountStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _env_path("PLATFORM_DB", ".agri_store/platform.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)
        self._fernet = Fernet(self._vault_key())
        self._secret = self._signing_key()

    # ---------------------------------------------------------------- keys
    def _key_file(self, name: str, gen: Any) -> bytes:
        path = self.db_path.parent / name
        if path.exists():
            return path.read_bytes().strip()
        key = gen()
        path.write_bytes(key)
        os.chmod(path, 0o600)
        logger.warning("Generated development key %s - set the env var in production", path)
        return key

    def _vault_key(self) -> bytes:
        env = os.environ.get("AGRI_VAULT_KEY")
        return env.encode() if env else self._key_file(".vault.key", Fernet.generate_key)

    def _signing_key(self) -> bytes:
        env = os.environ.get("AGRI_SECRET_KEY")
        return env.encode() if env else self._key_file(
            ".signing.key", lambda: secrets.token_hex(32).encode()
        )

    @contextmanager
    def _conn(self):  # type: ignore[no-untyped-def]
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def audit(self, user_id: int | None, action: str, detail: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT INTO audit(ts,user_id,action,detail) VALUES(?,?,?,?)",
                      (time.time(), user_id, action, detail[:500]))

    # --------------------------------------------------------------- orgs
    def create_org(self, name: str, slug: str, email_domain: str | None = None) -> dict:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO orgs(name,slug,email_domain,created) VALUES(?,?,?,?)",
                (name, slug.lower(), (email_domain or "").lower() or None, time.time()),
            )
            return {"id": cur.lastrowid, "name": name, "slug": slug.lower()}

    def get_org(self, slug_or_id: str | int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM orgs WHERE slug=? OR id=?",
                            (str(slug_or_id).lower(), slug_or_id)).fetchone()
            return dict(row) if row else None

    # -------------------------------------------------------------- users
    def register(self, email: str, password: str, name: str | None = None,
                 org_slug: str | None = None, role: str = "researcher",
                 org_admin: bool = False) -> User:
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        if role not in PLATFORM_ROLES:
            raise ValueError(f"role must be one of {PLATFORM_ROLES}")
        org = self.get_org(org_slug) if org_slug else None
        if org_slug and not org:
            raise ValueError(f"unknown organisation {org_slug}")
        if org and org.get("email_domain") and not email.lower().endswith("@" + org["email_domain"]):
            raise ValueError("email domain does not match the organisation")
        with self._conn() as c:
            try:
                cur = c.execute(
                    "INSERT INTO users(email,name,pw_hash,org_id,role,org_admin,created) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (email.lower(), name, _hash_password(password),
                     org["id"] if org else None, role, int(org_admin), time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("email already registered") from exc
        self.audit(cur.lastrowid, "register", email.lower())
        return self.get_user(int(cur.lastrowid))  # type: ignore[return-value]

    def get_user(self, user_id: int) -> User | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not r:
            return None
        return User(r["id"], r["email"], r["name"], r["org_id"], r["role"], bool(r["org_admin"]))

    def set_org_admin(self, org_slug: str, email: str, org_admin: bool = True) -> dict:
        org = self.get_org(org_slug)
        if not org:
            raise ValueError(f"unknown organisation {org_slug}")
        with self._conn() as c:
            cur = c.execute("UPDATE users SET org_id=?, org_admin=? WHERE email=?",
                            (org["id"], int(org_admin), email.lower()))
            if not cur.rowcount:
                raise ValueError(f"unknown user {email}")
        self.audit(None, "org_admin", f"{org_slug}:{email}:{org_admin}")
        return {"org": org["slug"], "email": email.lower(), "org_admin": org_admin}

    def set_role(self, email: str, role: str) -> None:
        if role not in PLATFORM_ROLES:
            raise ValueError(f"role must be one of {PLATFORM_ROLES}")
        with self._conn() as c:
            if not c.execute("UPDATE users SET role=? WHERE email=?",
                             (role, email.lower())).rowcount:
                raise ValueError(f"unknown user {email}")
        self.audit(None, "set_role", f"{email}:{role}")

    def authenticate(self, email: str, password: str) -> User | None:
        with self._conn() as c:
            r = c.execute("SELECT id,pw_hash FROM users WHERE email=?", (email.lower(),)).fetchone()
        if not r or not _verify_password(password, r["pw_hash"]):
            self.audit(None, "login_failed", email.lower())
            return None
        self.audit(r["id"], "login")
        return self.get_user(r["id"])

    # ------------------------------------------------------------- tokens
    def issue_token(self, user: User, ttl_s: int = TOKEN_TTL_S) -> str:
        payload = base64.urlsafe_b64encode(
            json.dumps({"uid": user.id, "exp": int(time.time()) + ttl_s}).encode()
        ).decode().rstrip("=")
        sig = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    def verify_token(self, token: str) -> User | None:
        try:
            payload, sig = token.rsplit(".", 1)
        except ValueError:
            return None
        good = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            return None
        try:
            data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except (ValueError, TypeError):
            return None
        if data.get("exp", 0) < time.time():
            return None
        return self.get_user(int(data["uid"]))

    # -------------------------------------------------------- credentials
    def add_credential(self, user: User, provider: str, data: dict[str, str],
                       label: str | None = None, for_org: bool = False) -> dict:
        if provider not in CREDENTIAL_PROVIDERS:
            raise ValueError(f"provider must be one of {CREDENTIAL_PROVIDERS}")
        if for_org:
            if not (user.org_id and user.org_admin):
                raise PermissionError("only organisation admins can add org credentials")
            owner = ("org", user.org_id)
        else:
            owner = ("user", user.id)
        blob = self._fernet.encrypt(json.dumps(data).encode())
        label = label or f"{owner[0]}:{provider}"
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO credentials(owner_type,owner_id,provider,label,blob,created) "
                "VALUES(?,?,?,?,?,?)", (*owner, provider, label, blob, time.time()),
            )
        self.audit(user.id, "credential_add", f"{owner[0]}:{provider}:{label}")
        return {"id": cur.lastrowid, "provider": provider, "label": label, "scope": owner[0]}

    def _rows_for(self, user: User) -> list[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM credentials WHERE (owner_type='user' AND owner_id=?) "
                "OR (owner_type='org' AND owner_id=?) "
                "ORDER BY CASE owner_type WHEN 'user' THEN 0 ELSE 1 END, id",
                (user.id, user.org_id or -1),
            ).fetchall()

    def list_credentials(self, user: User) -> list[dict]:
        out = []
        for r in self._rows_for(user):
            try:
                data = json.loads(self._fernet.decrypt(r["blob"]))
            except InvalidToken:
                data = {"error": "cannot decrypt (vault key changed?)"}
            out.append({
                "id": r["id"], "provider": r["provider"], "label": r["label"],
                "scope": r["owner_type"],
                "fields": {k: (mask(str(v)) if k in SECRET_FIELDS else v) for k, v in data.items()},
            })
        return out

    def credentials_for(self, user: User | None) -> list[dict[str, Any]]:
        """Decrypted entitlements (server-side use only - never return to clients)."""
        if user is None:
            return []
        out = []
        for r in self._rows_for(user):
            try:
                out.append({"provider": r["provider"], "label": r["label"],
                            "scope": r["owner_type"],
                            "data": json.loads(self._fernet.decrypt(r["blob"]))})
            except InvalidToken:
                logger.error("credential %s cannot be decrypted", r["id"])
        return out

    def search_keys_for(self, user: User | None) -> dict[str, str]:
        """API keys for search connectors (openalex/core/semantic_scholar/email)."""
        keys: dict[str, str] = {}
        for c in reversed(self.credentials_for(user)):  # personal overrides org
            d = c["data"]
            if c["provider"] in {"openalex", "core", "semantic_scholar"} and d.get("api_key"):
                keys[f"{c['provider']}_api_key"] = d["api_key"]
            if c["provider"] == "unpaywall" and d.get("email"):
                keys["contact_email"] = d["email"]
        return keys

    def delete_credential(self, user: User, cred_id: int) -> bool:
        with self._conn() as c:
            r = c.execute("SELECT owner_type,owner_id FROM credentials WHERE id=?",
                          (cred_id,)).fetchone()
            if not r:
                return False
            allowed = (r["owner_type"] == "user" and r["owner_id"] == user.id) or (
                r["owner_type"] == "org" and r["owner_id"] == user.org_id and user.org_admin
            )
            if not allowed:
                raise PermissionError("not your credential")
            c.execute("DELETE FROM credentials WHERE id=?", (cred_id,))
        self.audit(user.id, "credential_delete", str(cred_id))
        return True
