"""Raw SEC filing download service."""

from __future__ import annotations

import hashlib
import mimetypes
from datetime import UTC, datetime

from app.data_sources.sec.archive import build_archive_url, build_raw_store_path
from app.data_sources.sec.client import SecArchiveClient
from app.domain import (
    DownloadStatusLabel,
    FilingType,
    SecRawDocumentRecord,
    SecRawDownloadResult,
)
from app.storage.raw_store import RawDocumentStore
from app.storage.repositories import SecMetadataRepository, SecRawDocumentRepository


class SecRawFilingDownloadService:
    """Download and persist raw SEC filing documents."""

    def __init__(
        self,
        archive_client: SecArchiveClient,
        metadata_repository: SecMetadataRepository,
        raw_document_repository: SecRawDocumentRepository,
        raw_document_store: RawDocumentStore,
        *,
        skip_existing: bool = True,
    ) -> None:
        self._archive_client = archive_client
        self._metadata_repository = metadata_repository
        self._raw_document_repository = raw_document_repository
        self._raw_document_store = raw_document_store
        self._skip_existing = skip_existing

    def download_filings(
        self,
        *,
        ticker: str,
        forms: list[FilingType] | None = None,
    ) -> SecRawDownloadResult:
        """Download raw archive documents for stored filing metadata."""

        filings = self._metadata_repository.query_filings(
            ticker=ticker.upper(),
            forms=forms,
            limit=None,
        )
        requested_forms = forms or [FilingType.FORM_10K, FilingType.FORM_10Q, FilingType.FORM_8K]
        downloaded_count = 0
        skipped_count = 0
        failed_count = 0
        attempted_count = 0
        for filing in filings:
            attempted_count += 1
            relative_path = build_raw_store_path(filing)
            existing = self._raw_document_repository.get_download_record(filing.accession_number)
            if (
                self._skip_existing
                and existing is not None
                and existing.status == DownloadStatusLabel.DOWNLOADED
                and existing.storage_path is not None
                and self._raw_document_store.exists(existing.storage_path)
            ):
                skipped_count += 1
                continue
            source_url = build_archive_url(
                filing.cik,
                filing.accession_number,
                filing.primary_document,
            )
            try:
                response = self._archive_client.download(source_url)
                self._raw_document_store.save(relative_path, response.content)
                content_type = (
                    response.content_type or mimetypes.guess_type(filing.primary_document)[0]
                )
                record = SecRawDocumentRecord(
                    accession_number=filing.accession_number,
                    cik=filing.cik,
                    ticker=filing.ticker,
                    form_type=filing.form_type,
                    source_url=source_url,
                    storage_path=relative_path,
                    checksum_sha256=hashlib.sha256(response.content).hexdigest(),
                    content_type=content_type,
                    size_bytes=len(response.content),
                    status=DownloadStatusLabel.DOWNLOADED,
                    downloaded_at=datetime.now(UTC),
                    retry_count=(existing.retry_count if existing is not None else 0) + 1,
                )
                self._raw_document_repository.upsert_download_record(record)
                downloaded_count += 1
            except Exception as exc:
                record = SecRawDocumentRecord(
                    accession_number=filing.accession_number,
                    cik=filing.cik,
                    ticker=filing.ticker,
                    form_type=filing.form_type,
                    source_url=source_url,
                    status=DownloadStatusLabel.FAILED,
                    retry_count=(existing.retry_count if existing is not None else 0) + 1,
                    last_error=str(exc),
                )
                self._raw_document_repository.upsert_download_record(record)
                failed_count += 1
        return SecRawDownloadResult(
            ticker=ticker.upper(),
            requested_forms=requested_forms,
            attempted_count=attempted_count,
            downloaded_count=downloaded_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
        )
