"""Data-governance access tiers (proposal Section 25 / Table 8, deliverable D1.8).

Every document ingested into the knowledge base carries an ``access_level``.
Retrieval filters out chunks the caller is not entitled to see, so restricted
or closed material can never leak into a farmer-facing answer.
"""

from __future__ import annotations

from datetime import date
from enum import Enum


class AccessLevel(str, Enum):
    OPEN = "open"
    EMBARGOED = "embargoed"
    RESTRICTED = "restricted"
    CLOSED = "closed"


class Role(str, Enum):
    PUBLIC = "public"  # farmers, anonymous SMS/USSD/voice users
    EXTENSION = "extension"  # registered extension workers
    RESEARCHER = "researcher"  # registered researchers (literature & review tools)
    CONSORTIUM = "consortium"  # project beneficiaries (confidential repository)
    DATA_OWNER = "data_owner"  # owner/curator of a closed dataset


# Which access levels each role may read.
_ROLE_PERMISSIONS: dict[Role, set[AccessLevel]] = {
    Role.PUBLIC: {AccessLevel.OPEN},
    Role.EXTENSION: {AccessLevel.OPEN},
    Role.RESEARCHER: {AccessLevel.OPEN},
    Role.CONSORTIUM: {AccessLevel.OPEN, AccessLevel.EMBARGOED, AccessLevel.RESTRICTED},
    Role.DATA_OWNER: set(AccessLevel),
}


def parse_role(value: str | Role | None) -> Role:
    if isinstance(value, Role):
        return value
    if (value or "").lower() == "admin":  # platform admins curate all tiers
        return Role.DATA_OWNER
    try:
        return Role((value or "public").lower())
    except ValueError:
        return Role.PUBLIC


def parse_access_level(value: str | AccessLevel | None) -> AccessLevel:
    if isinstance(value, AccessLevel):
        return value
    try:
        return AccessLevel((value or "open").lower())
    except ValueError:
        # Unknown labels are treated as the most protective level.
        return AccessLevel.CLOSED


def can_access(
    role: str | Role,
    access_level: str | AccessLevel,
    embargo_until: str | None = None,
    today: date | None = None,
) -> bool:
    """Return True if ``role`` may read content at ``access_level``.

    Embargoed content becomes OPEN once ``embargo_until`` (ISO date) has passed.
    """
    r = parse_role(role)
    level = parse_access_level(access_level)
    if level is AccessLevel.EMBARGOED and embargo_until:
        try:
            if (today or date.today()) >= date.fromisoformat(embargo_until):
                level = AccessLevel.OPEN
        except ValueError:
            pass
    return level in _ROLE_PERMISSIONS[r]
