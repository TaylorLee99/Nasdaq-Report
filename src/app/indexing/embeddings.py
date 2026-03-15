"""Embedding abstractions for retrieval indexing."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Minimal embedding provider contract used by the indexing pipeline."""

    @property
    def model_name(self) -> str:
        """Return the provider/model identifier stored with chunks."""

    @property
    def dimensions(self) -> int:
        """Return the embedding width."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text inputs into fixed-width vectors."""


class FakeEmbeddingProvider:
    """Deterministic local embedding stub for tests and offline development."""

    def __init__(self, *, dimensions: int = 12, model_name: str = "fake-local-v1") -> None:
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower()) or [text.lower()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self._dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            weight = max(len(token), 1)
            vector[index] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
