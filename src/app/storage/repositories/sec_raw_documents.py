"""Repository for raw SEC filing download state."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain import DownloadStatusLabel, FilingType, SecRawDocumentRecord
from app.storage.models import SecRawDocumentModel


class SecRawDocumentRepository:
    """Persist and query raw SEC filing download state."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def get_download_record(self, accession_number: str) -> SecRawDocumentRecord | None:
        """Return the current download record for an accession number."""

        with self._session_factory() as session:
            row = session.scalar(
                select(SecRawDocumentModel).where(
                    SecRawDocumentModel.accession_number == accession_number
                )
            )
            if row is None:
                return None
            return self._to_domain(row)

    def upsert_download_record(self, record: SecRawDocumentRecord) -> None:
        """Insert or update a raw document download state row."""

        with self._session_factory() as session:
            existing = session.scalar(
                select(SecRawDocumentModel).where(
                    SecRawDocumentModel.accession_number == record.accession_number
                )
            )
            if existing is None:
                session.add(self._to_model(record))
            else:
                existing.cik = record.cik
                existing.ticker = record.ticker
                existing.form_type = record.form_type.value
                existing.source_url = record.source_url
                existing.storage_path = record.storage_path
                existing.checksum_sha256 = record.checksum_sha256
                existing.content_type = record.content_type
                existing.size_bytes = record.size_bytes
                existing.status = record.status.value
                existing.downloaded_at = record.downloaded_at
                existing.retry_count = record.retry_count
                existing.last_error = record.last_error
            session.commit()

    def query_download_records(
        self,
        *,
        ticker: str,
        forms: list[FilingType] | None = None,
        status: DownloadStatusLabel | None = None,
    ) -> list[SecRawDocumentRecord]:
        """Query raw document download records by ticker, form, and status."""

        with self._session_factory() as session:
            statement = select(SecRawDocumentModel).where(
                SecRawDocumentModel.ticker == ticker.upper()
            )
            if forms:
                statement = statement.where(
                    SecRawDocumentModel.form_type.in_([form.value for form in forms])
                )
            if status is not None:
                statement = statement.where(SecRawDocumentModel.status == status.value)
            rows = session.scalars(statement).all()
            return [self._to_domain(row) for row in rows]

    @staticmethod
    def _to_model(record: SecRawDocumentRecord) -> SecRawDocumentModel:
        return SecRawDocumentModel(
            accession_number=record.accession_number,
            cik=record.cik,
            ticker=record.ticker,
            form_type=record.form_type.value,
            source_url=record.source_url,
            storage_path=record.storage_path,
            checksum_sha256=record.checksum_sha256,
            content_type=record.content_type,
            size_bytes=record.size_bytes,
            status=record.status.value,
            downloaded_at=record.downloaded_at,
            retry_count=record.retry_count,
            last_error=record.last_error,
        )

    @staticmethod
    def _to_domain(row: SecRawDocumentModel) -> SecRawDocumentRecord:
        return SecRawDocumentRecord(
            accession_number=row.accession_number,
            cik=row.cik,
            ticker=row.ticker,
            form_type=FilingType(row.form_type),
            source_url=row.source_url,
            storage_path=row.storage_path,
            checksum_sha256=row.checksum_sha256,
            content_type=row.content_type,
            size_bytes=row.size_bytes,
            status=DownloadStatusLabel(row.status),
            downloaded_at=row.downloaded_at,
            retry_count=row.retry_count,
            last_error=row.last_error,
        )
