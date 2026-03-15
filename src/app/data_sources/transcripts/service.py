"""Transcript ingestion service."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from app.data_sources.transcripts.paths import build_transcript_store_path
from app.domain import (
    Company,
    ConfidenceLabel,
    CoverageLabel,
    CoverageStatus,
    TranscriptAvailabilityGap,
    TranscriptImportResult,
    TranscriptMetadata,
    TranscriptSourceType,
)
from app.storage.raw_store import RawDocumentStore
from app.storage.repositories import TranscriptRepository, UniverseRepository


class TranscriptIngestionService:
    """Import optional transcript text or record reduced-mode gaps."""

    def __init__(
        self,
        transcript_repository: TranscriptRepository,
        universe_repository: UniverseRepository,
        raw_document_store: RawDocumentStore,
        *,
        skip_existing: bool = True,
    ) -> None:
        self._transcript_repository = transcript_repository
        self._universe_repository = universe_repository
        self._raw_document_store = raw_document_store
        self._skip_existing = skip_existing

    def import_transcript(
        self,
        *,
        ticker: str,
        file_path: Path,
        call_date: date,
        fiscal_quarter: str | None = None,
        source_url: str | None = None,
        speaker_separated: bool = False,
        source_reliability: ConfidenceLabel = ConfidenceLabel.HIGH,
    ) -> TranscriptImportResult:
        """Import a transcript file into metadata and raw storage."""

        company = self._require_company(ticker)
        transcript_id = self._build_transcript_id(company, call_date, fiscal_quarter)
        existing = self._transcript_repository.get_transcript_by_id(transcript_id)
        if existing is not None and self._skip_existing:
            return TranscriptImportResult(
                ticker=company.ticker,
                imported_count=0,
                transcripts=[existing],
            )

        raw_text = file_path.read_text(encoding="utf-8")
        raw_bytes = raw_text.encode("utf-8")
        metadata = TranscriptMetadata(
            transcript_id=transcript_id,
            company=company,
            call_date=call_date,
            fiscal_quarter=fiscal_quarter,
            source_url=source_url or str(file_path.resolve()),
            speaker_separated=speaker_separated,
            source_reliability=source_reliability,
            source_type=TranscriptSourceType.FILE,
            availability_coverage=CoverageStatus(
                label=CoverageLabel.COMPLETE,
                covered_topics=["transcript"],
            ),
        )
        relative_path = build_transcript_store_path(metadata, file_path.name)
        absolute_path = self._raw_document_store.save(relative_path, raw_bytes)
        stored_metadata = metadata.model_copy(
            update={
                "storage_path": str(absolute_path),
                "checksum_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "content_type": "text/plain; charset=utf-8",
                "size_bytes": len(raw_bytes),
            }
        )
        self._transcript_repository.upsert_transcript(stored_metadata)
        return TranscriptImportResult(
            ticker=company.ticker,
            imported_count=1,
            transcripts=[stored_metadata],
        )

    def record_unavailable(
        self,
        *,
        ticker: str,
        reason: str,
        source_url: str | None = None,
    ) -> TranscriptImportResult:
        """Record transcript unavailability and reduced-mode coverage."""

        company = self._require_company(ticker)
        gap = TranscriptAvailabilityGap(
            gap_id=self._build_gap_id(company),
            company=company,
            coverage_status=CoverageStatus(
                label=CoverageLabel.PARTIAL,
                covered_topics=["filings_only"],
                missing_topics=["transcript"],
                notes="Transcript unavailable; reduced mode active.",
            ),
            reason=reason,
            source_url=source_url,
        )
        self._transcript_repository.add_availability_gap(gap)
        return TranscriptImportResult(
            ticker=company.ticker,
            missing_count=1,
            availability_gaps=[gap],
        )

    def list_transcripts(self, *, ticker: str) -> TranscriptImportResult:
        """Return stored transcripts and availability gaps for a ticker."""

        company = self._require_company(ticker)
        transcripts = self._transcript_repository.list_transcripts(ticker=company.ticker)
        gaps = self._transcript_repository.list_availability_gaps(ticker=company.ticker)
        return TranscriptImportResult(
            ticker=company.ticker,
            imported_count=len(transcripts),
            missing_count=len(gaps),
            transcripts=transcripts,
            availability_gaps=gaps,
        )

    def _require_company(self, ticker: str) -> Company:
        company = self._universe_repository.get_company_by_ticker(ticker)
        if company is None:
            msg = f"Ticker {ticker.upper()} was not found in company master"
            raise ValueError(msg)
        return company

    @staticmethod
    def _build_transcript_id(
        company: Company,
        call_date: date,
        fiscal_quarter: str | None,
    ) -> str:
        quarter = fiscal_quarter or "unknown-quarter"
        return f"{company.ticker}-{call_date.isoformat()}-{quarter}"

    @staticmethod
    def _build_gap_id(company: Company) -> str:
        return f"{company.ticker}-transcript-gap"
