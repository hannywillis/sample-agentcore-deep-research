"""Pluggable embedding back-ends.

* ``hashing`` (default) - dependency-free, deterministic feature hashing of word
  and character n-grams. Works offline and in CI; good enough for a demo corpus
  but NOT a substitute for a semantic model in production.
* ``sentence-transformers`` - local multilingual semantic model (recommended for
  Kiswahili/Hausa/French/Portuguese corpora).
* ``bedrock`` - Amazon Titan Text Embeddings v2 via Bedrock (matches the
  AgentCore deployment of this repository).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Protocol

import numpy as np

from agri_advisor.config import Settings, get_settings

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEmbedder:
    """Signed feature hashing over word unigrams/bigrams and char 4-grams."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(
            hashlib.blake2b(token.encode(), digest_size=8).digest(), "little"
        )
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = [w for w in _TOKEN.findall(_normalise(text)) if len(w) > 1]
        feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(words, words[1:], strict=False)]
        for w in words:
            padded = f"#{w}#"
            feats += [(f"c:{padded[i : i + 4]}", 0.25) for i in range(len(padded) - 3)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token, weight in self._features(text):
                idx, sign = self._bucket(token)
                out[row, idx] += sign * weight
        # Sub-linear scaling dampens very frequent tokens.
        out = np.sign(out) * np.log1p(np.abs(out))
        return _l2(out)


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    def __init__(self, model_id: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        self.name = f"st:{model_id}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class BedrockTitanEmbedder:  # pragma: no cover - requires AWS credentials
    def __init__(
        self, model_id: str = "amazon.titan-embed-text-v2:0", region: str = ""
    ) -> None:
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=region or None)
        self._model_id = model_id
        self.dim = 1024
        self.name = f"bedrock:{model_id}"

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            resp = self._client.invoke_model(
                modelId=self._model_id,
                body=json.dumps({"inputText": text[:8000], "normalize": True}),
            )
            rows.append(json.loads(resp["body"].read())["embedding"])
        return np.asarray(rows, dtype=np.float32)


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    provider = s.embedding_provider.lower()
    if provider in {"sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(s.embedding_model_id)
    if provider == "bedrock":
        model = s.embedding_model_id
        if model.startswith("sentence-transformers/"):
            model = "amazon.titan-embed-text-v2:0"
        return BedrockTitanEmbedder(model, s.aws_region)
    return HashingEmbedder(s.embedding_dim)
