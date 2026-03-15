"""Repository for transcript metadata and availability gaps."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain import (
    Company,
    ConfidenceLabel,
    CoverageLabel,
    CoverageStatus,
    TranscriptAvailabilityGap,
    TranscriptMetadata,
    TranscriptSourceType,
)
from app.storage.models import (
    CompanyMasterModel,
    TranscriptAvailabilityGapModel,
    TranscriptMetadataModel,
)


class TranscriptRepository:
    """Persist and query transcript metadata and missing-coverage gaps."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def get_transcript_by_id(self, transcript_id: str) -> TranscriptMetadata | None:
        """Return a transcript metadata row by transcript id."""

        with self._session_factory() as session:
            row = session.scalar(
                select(TranscriptMetadataModel).where(
                    TranscriptMetadataModel.transcript_id == transcript_id
                )
            )
            if row is None:
                return None
            return self._to_transcript_domain(row, self._get_company(session, row.cik, row.ticker))

    def upsert_transcript(self, metadata: TranscriptMetadata) -> None:
        """Insert or update transcript metadata."""

        with self._session_factory() as session:
            existing = session.scalar(
                select(TranscriptMetadataModel).where(
                    TranscriptMetadataModel.transcript_id == metadata.transcript_id
                )
            )
            if existing is None:
                session.add(self._to_transcript_model(metadata))
            else:
                existing.cik = metadata.company.cik
                existing.ticker = metadata.company.ticker
                existing.call_date = metadata.call_date
                existing.fiscal_quarter = metadata.fiscal_quarter
                existing.fiscal_year = metadata.fiscal_year
                existing.source_url = metadata.source_url
                existing.speaker_separated = metadata.speaker_separated
                existing.source_reliability = metadata.source_reliability.value
                existing.source_type = metadata.source_type.value
                existing.coverage_label = metadata.availability_coverage.label.value
                existing.covered_topics = metadata.availability_coverage.covered_topics
                existing.missing_topics = metadata.availability_coverage.missing_topics
                existing.coverage_notes = metadata.availability_coverage.notes
                existing.storage_path = metadata.storage_path
                existing.checksum_sha256 = metadata.checksum_sha256
                existing.content_type = metadata.content_type
                existing.size_bytes = metadata.size_bytes
                existing.imported_at = metadata.imported_at
            session.commit()

    def list_transcripts(self, *, ticker: str) -> list[TranscriptMetadata]:
        """Return transcript metadata rows for a ticker."""

        with self._session_factory() as session:
            rows = session.scalars(
                select(TranscriptMetadataModel)
                .where(TranscriptMetadataModel.ticker == ticker.upper())
                .order_by(TranscriptMetadataModel.call_date.desc())
            ).all()
            return [
                self._to_transcript_domain(row, self._get_company(session, row.cik, row.ticker))
                for row in rows
            ]

    def add_availability_gap(self, gap: TranscriptAvailabilityGap) -> None:
        """Insert or update an availability gap record."""

        with self._session_factory() as session:
            existing = session.scalar(
                select(TranscriptAvailabilityGapModel).where(
                    TranscriptAvailabilityGapModel.gap_id == gap.gap_id
                )
            )
            if existing is None:
                session.add(self._to_gap_model(gap))
            else:
                existing.reason = gap.reason
                existing.source_url = gap.source_url
                existing.coverage_label = gap.coverage_status.label.value
                existing.covered_topics = gap.coverage_status.covered_topics
                existing.missing_topics = gap.coverage_status.missing_topics
                existing.coverage_notes = gap.coverage_status.notes
                existing.observed_at = gap.observed_at
            session.commit()

    def list_availability_gaps(self, *, ticker: str) -> list[TranscriptAvailabilityGap]:
        """Return missing transcript coverage rows for a ticker."""

        with self._session_factory() as session:
            rows = session.scalars(
                select(TranscriptAvailabilityGapModel)
                .where(TranscriptAvailabilityGapModel.ticker == ticker.upper())
                .order_by(TranscriptAvailabilityGapModel.observed_at.desc())
            ).all()
            return [
                self._to_gap_domain(row, self._get_company(session, row.cik, row.ticker))
                for row in rows
            ]

    @staticmethod
    def _get_company(session: Session, cik: str, ticker: str) -> Company:
        company_row = session.get(CompanyMasterModel, cik)
        if company_row is None:
            return Company(cik=cik, ticker=ticker, company_name=ticker)
        return Company(
            cik=company_row.cik,
            ticker=company_row.ticker,
            company_name=company_row.company_name,
            exchange=company_row.exchange,
            is_domestic_filer=company_row.is_domestic_filer,
            latest_snapshot_date=company_row.latest_snapshot_date,
        )

    @staticmethod
    def _to_transcript_model(metadata: TranscriptMetadata) -> TranscriptMetadataModel:
        return TranscriptMetadataModel(
            transcript_id=metadata.transcript_id,
            cik=metadata.company.cik,
            ticker=metadata.company.ticker,
            call_date=metadata.call_date,
            fiscal_quarter=metadata.fiscal_quarter,
            fiscal_year=metadata.fiscal_year,
            source_url=metadata.source_url,
            speaker_separated=metadata.speaker_separated,
            source_reliability=metadata.source_reliability.value,
            source_type=metadata.source_type.value,
            coverage_label=metadata.availability_coverage.label.value,
            covered_topics=metadata.availability_coverage.covered_topics,
            missing_topics=metadata.availability_coverage.missing_topics,
            coverage_notes=metadata.availability_coverage.notes,
            storage_path=metadata.storage_path,
            checksum_sha256=metadata.checksum_sha256,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
            imported_at=metadata.imported_at,
        )

    @staticmethod
    def _to_gap_model(gap: TranscriptAvailabilityGap) -> TranscriptAvailabilityGapModel:
        return TranscriptAvailabilityGapModel(
            gap_id=gap.gap_id,
            cik=gap.company.cik,
            ticker=gap.company.ticker,
            reason=gap.reason,
            source_url=gap.source_url,
            coverage_label=gap.coverage_status.label.value,
            covered_topics=gap.coverage_status.covered_topics,
            missing_topics=gap.coverage_status.missing_topics,
            coverage_notes=gap.coverage_status.notes,
            observed_at=gap.observed_at,
        )

    @staticmethod
    def _to_transcript_domain(row: TranscriptMetadataModel, company: Company) -> TranscriptMetadata:
        return TranscriptMetadata(
            transcript_id=row.transcript_id,
            company=company,
            call_date=row.call_date,
            fiscal_quarter=row.fiscal_quarter,
            fiscal_year=row.fiscal_year,
            source_url=row.source_url,
            speaker_separated=row.speaker_separated,
            source_reliability=ConfidenceLabel(row.source_reliability),
            source_type=TranscriptSourceType(row.source_type),
            availability_coverage=CoverageStatus(
                label=CoverageLabel(row.coverage_label),
                covered_topics=list(row.covered_topics or []),
                missing_topics=list(row.missing_topics or []),
                notes=row.coverage_notes,
            ),
            storage_path=row.storage_path,
            checksum_sha256=row.checksum_sha256,
            content_type=row.content_type,
            size_bytes=row.size_bytes,
            imported_at=row.imported_at,
        )

    @staticmethod
    def _to_gap_domain(
        row: TranscriptAvailabilityGapModel,
        company: Company,
    ) -> TranscriptAvailabilityGap:
        return TranscriptAvailabilityGap(
            gap_id=row.gap_id,
            company=company,
            coverage_status=CoverageStatus(
                label=CoverageLabel(row.coverage_label),
                covered_topics=list(row.covered_topics or []),
                missing_topics=list(row.missing_topics or []),
                notes=row.coverage_notes,
            ),
            reason=row.reason,
            source_url=row.source_url,
            observed_at=row.observed_at,
        )
