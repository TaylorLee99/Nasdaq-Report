"""Transcript ingestion and availability schemas."""

from datetime import UTC, datetime

from pydantic import Field

from app.domain.enums import ConfidenceLabel, TranscriptSourceType
from app.domain.models import CanonicalModel, Company, CoverageStatus, TranscriptMetadata


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class TranscriptRawPayload(CanonicalModel):
    """Raw transcript text returned by a source adapter."""

    text: str
    content_type: str | None = "text/plain; charset=utf-8"
    source_url: str | None = None
    speaker_separated: bool = False
    source_reliability: ConfidenceLabel = ConfidenceLabel.MEDIUM
    source_type: TranscriptSourceType


class TranscriptAvailabilityGap(CanonicalModel):
    """Recorded absence of a transcript for reduced-mode processing."""

    gap_id: str
    company: Company
    coverage_status: CoverageStatus
    reason: str
    source_url: str | None = None
    observed_at: datetime = Field(default_factory=utc_now)


class TranscriptImportResult(CanonicalModel):
    """Summary of transcript import and availability state."""

    ticker: str
    imported_count: int = 0
    missing_count: int = 0
    transcripts: list[TranscriptMetadata] = Field(default_factory=list)
    availability_gaps: list[TranscriptAvailabilityGap] = Field(default_factory=list)
