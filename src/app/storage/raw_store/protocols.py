"""Protocols for raw document persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StoredDocumentPayload:
    """Binary payload returned by the archive downloader."""

    content: bytes
    content_type: str | None
    source_url: str
    headers: Mapping[str, str]


class RawDocumentStore(Protocol):
    """Storage backend abstraction for raw filing documents."""

    def exists(self, relative_path: str) -> bool:
        """Return whether a stored object already exists."""

    def save(self, relative_path: str, content: bytes) -> Path:
        """Persist bytes to the given relative path."""

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative object path to an absolute filesystem path."""
