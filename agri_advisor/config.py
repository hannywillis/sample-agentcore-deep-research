"""Central configuration for the agricultural decision-support platform.

All settings are read from environment variables (see ``.env.example`` at the
repository root) so the same code runs locally, in Docker, and inside
AgentCore Runtime without modification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ISO 639 codes of the languages named in the proposal (Task 1.6).
# "dje" = Zarma-Songhay. en/sw/fr/ha/pt have curated UI phrases; dje falls back
# to French UI phrases until native-speaker translations are validated.
SUPPORTED_LANGUAGES = ("en", "sw", "fr", "ha", "pt", "dje")

# Focus cereal crops shared by all target countries.
FOCUS_CROPS = ("maize", "wheat", "sorghum", "rice", "millet")

TARGET_COUNTRIES = ("KE", "UG", "NE", "NG", "GH", "BR")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class Settings:
    """Runtime settings. Instantiate via :func:`get_settings`."""

    # --- LLM (knowledge translation) ---
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none"))
    llm_model_id: str = field(
        default_factory=lambda: _env("LLM_MODEL_ID", "global.anthropic.claude-sonnet-5")
    )
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2)
    )
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 900))
    aws_region: str = field(
        default_factory=lambda: _env("AWS_DEFAULT_REGION", "us-east-1")
    )

    # --- Embeddings / vector store (RAG) ---
    embedding_provider: str = field(
        default_factory=lambda: _env("EMBEDDING_PROVIDER", "hashing")
    )
    embedding_model_id: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL_ID",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 768))
    vector_store_path: Path = field(
        default_factory=lambda: _env_path("VECTOR_STORE_PATH", Path(".agri_store"))
    )
    knowledge_base_dir: Path = field(
        default_factory=lambda: _env_path(
            "KNOWLEDGE_BASE_DIR", Path("data/knowledge_base")
        )
    )
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 900))
    chunk_overlap: int = field(
        default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 150)
    )
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    min_relevance: float = field(
        default_factory=lambda: _env_float("RAG_MIN_RELEVANCE", 0.12)
    )

    # --- Predictive models ---
    model_dir: Path = field(
        default_factory=lambda: _env_path("MODEL_DIR", Path(".agri_models"))
    )
    auto_train_scaffold: bool = field(
        default_factory=lambda: _env_bool("AUTO_TRAIN_SCAFFOLD_MODELS", True)
    )
    conformal_alpha: float = field(
        default_factory=lambda: _env_float("CONFORMAL_ALPHA", 0.1)
    )

    # --- Expert validation / governance ---
    expert_validation_config: Path = field(
        default_factory=lambda: _env_path(
            "EXPERT_VALIDATION_CONFIG", Path("config/expert_validation.yaml")
        )
    )
    review_queue_path: Path = field(
        default_factory=lambda: _env_path(
            "EXPERT_REVIEW_QUEUE", Path(".agri_store/expert_review_queue.jsonl")
        )
    )
    default_access_role: str = field(
        default_factory=lambda: _env("DEFAULT_ACCESS_ROLE", "public")
    )

    # --- Language / interface ---
    default_language: str = field(
        default_factory=lambda: _env("DEFAULT_LANGUAGE", "en")
    )
    sms_max_chars: int = field(default_factory=lambda: _env_int("SMS_MAX_CHARS", 459))
    ussd_max_chars: int = field(default_factory=lambda: _env_int("USSD_MAX_CHARS", 182))
    voice_max_items: int = field(default_factory=lambda: _env_int("VOICE_MAX_ITEMS", 3))


_SETTINGS: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return the process-wide settings (re-read from env when ``refresh``)."""
    global _SETTINGS
    if _SETTINGS is None or refresh:
        _SETTINGS = Settings()
    return _SETTINGS
