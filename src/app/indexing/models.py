"""Typed schemas for indexing and retrieval workflows."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from app.domain import (
    CanonicalModel,
    ChunkRecord,
    ConfidenceLabel,
    FilingType,
    TranscriptSegmentType,
)


class ChunkSearchFilters(CanonicalModel):
    """Metadata filters applied during chunk listing or search."""

    ticker: str | None = None
    filing_types: list[FilingType] = Field(default_factory=list)
    accession_number: str | None = None
    transcript_id: str | None = None
    filing_date_on_or_after: date | None = None
    report_period_on_or_after: date | None = None
    section_name: str | None = None
    item_number: str | None = None
    speaker: str | None = None
    transcript_segment_type: TranscriptSegmentType | None = None
    parse_confidence: ConfidenceLabel | None = None
    source_reliability: ConfidenceLabel | None = None


class ChunkSearchResult(CanonicalModel):
    """Ranked retrieval result for a chunk search."""

    chunk: ChunkRecord
    score: float


class IndexingRunResult(CanonicalModel):
    """Summary for an indexing run over filings or transcripts."""

    ticker: str
    source_label: str
    attempted_documents: int = 0
    indexed_documents: int = 0
    indexed_chunks: int = 0
    skipped_documents: int = 0
    completed_at: datetime
