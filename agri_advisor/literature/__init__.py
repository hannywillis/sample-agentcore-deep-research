"""Research literature: federated open-journal search, entitlement-aware
full-text access, and a systematic-review workspace."""

from agri_advisor.literature.models import Paper
from agri_advisor.literature.search import SearchResult, federated_search

__all__ = ["Paper", "SearchResult", "federated_search"]
