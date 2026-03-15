"""Transcript raw store path helpers."""

from __future__ import annotations

from pathlib import Path

from app.domain import TranscriptMetadata


def build_transcript_store_path(metadata: TranscriptMetadata, source_name: str) -> str:
    """Build a deterministic raw transcript storage path."""

    source_suffix = Path(source_name).suffix or ".txt"
    return (
        "transcripts/"
        f"{metadata.company.ticker}/"
        f"{metadata.call_date.isoformat()}/"
        f"{metadata.fiscal_quarter or 'unknown-quarter'}/"
        f"{metadata.transcript_id}{source_suffix}"
    )
