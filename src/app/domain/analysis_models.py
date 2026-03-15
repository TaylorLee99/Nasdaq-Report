"""Analysis job schemas shared by CLI, API, and storage layers."""

from __future__ import annotations

from datetime import UTC, date, datetime

from pydantic import Field

from app.domain.enums import AnalysisJobStatus
from app.domain.graph_models import DocumentSignal
from app.domain.models import CanonicalModel, FinalReport


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class AnalysisRunRequest(CanonicalModel):
    """Input payload for synchronous company analysis."""

    ticker: str
    question: str
    include_transcript: bool = False
    as_of_date: date | None = None


class AnalysisJobRecord(CanonicalModel):
    """Persisted analysis job plus final report payloads."""

    analysis_id: str
    ticker: str
    cik: str | None = None
    question: str
    status: AnalysisJobStatus
    document_signals: list[DocumentSignal] = Field(default_factory=list)
    final_report: FinalReport | None = None
    final_report_markdown: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
