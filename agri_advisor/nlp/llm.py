"""Minimal LLM client abstraction (Bedrock, OpenAI-compatible, or none)."""

from __future__ import annotations

from typing import Protocol

import requests

from agri_advisor.config import Settings, get_settings


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class BedrockLLM:  # pragma: no cover - requires AWS credentials
    def __init__(self, s: Settings) -> None:
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=s.aws_region)
        self._s = s
        self.name = f"bedrock:{s.llm_model_id}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.converse(
            modelId=self._s.llm_model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={
                "temperature": self._s.llm_temperature,
                "maxTokens": self._s.llm_max_tokens,
            },
        )
        parts = resp["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts)


class OpenAICompatibleLLM:  # pragma: no cover - requires network/API key
    """Any /chat/completions endpoint (OpenAI, Azure, vLLM, Ollama, Abacus...)."""

    def __init__(self, s: Settings) -> None:
        if not s.llm_api_key:
            raise ValueError("LLM_API_KEY is required for LLM_PROVIDER=openai")
        self._s = s
        self.name = f"openai:{s.llm_model_id}"

    def complete(self, system: str, user: str) -> str:
        resp = requests.post(
            f"{self._s.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self._s.llm_api_key}"},
            json={
                "model": self._s.llm_model_id,
                "temperature": self._s.llm_temperature,
                "max_tokens": self._s.llm_max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])


def get_llm(settings: Settings | None = None) -> LLMClient | None:
    """Return the configured client, or ``None`` for offline/extractive mode."""
    s = settings or get_settings()
    provider = s.llm_provider.lower()
    if provider == "bedrock":
        return BedrockLLM(s)
    if provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleLLM(s)
    return None
