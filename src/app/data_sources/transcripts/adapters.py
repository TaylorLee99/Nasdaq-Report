"""Transcript source adapter abstractions."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.data_sources.sec.client import UrllibBinaryHttpClient
from app.domain import ConfidenceLabel, TranscriptRawPayload, TranscriptSourceType


class TranscriptSourceAdapter(Protocol):
    """Abstract transcript source adapter."""

    def load(self, source: str | Path) -> TranscriptRawPayload:
        """Load transcript text from the given source."""


class FileTranscriptSourceAdapter:
    """Load transcript text from a local file."""

    def load(self, source: str | Path) -> TranscriptRawPayload:
        """Read transcript text from the local filesystem."""

        path = Path(source)
        return TranscriptRawPayload(
            text=path.read_text(encoding="utf-8"),
            source_url=str(path.resolve()),
            source_reliability=ConfidenceLabel.HIGH,
            source_type=TranscriptSourceType.FILE,
        )


class GenericHttpTranscriptSourceAdapter:
    """Generic HTTP adapter skeleton for future transcript providers."""

    def __init__(self, http_client: UrllibBinaryHttpClient | None = None) -> None:
        self._http_client = http_client or UrllibBinaryHttpClient()

    def load(self, source: str | Path) -> TranscriptRawPayload:
        """Fetch transcript text over HTTP without provider-specific parsing."""

        response = self._http_client.get_binary(str(source))
        return TranscriptRawPayload(
            text=response.content.decode("utf-8"),
            content_type=response.content_type,
            source_url=str(source),
            source_reliability=ConfidenceLabel.MEDIUM,
            source_type=TranscriptSourceType.GENERIC_HTTP,
        )
