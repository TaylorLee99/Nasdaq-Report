from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pytest

from app.config import get_settings
from app.data_sources.sec import (
    HttpBinaryResponse,
    RateLimiter,
    RetryableHttpError,
    SecArchiveClient,
)
from app.data_sources.sec.archive import build_archive_url
from app.data_sources.sec.download_service import SecRawFilingDownloadService
from app.domain import (
    DownloadStatusLabel,
    FilingType,
    SecSubmissionMetadata,
    UniverseSnapshot,
    UniverseSnapshotConstituent,
)
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.raw_store import LocalRawDocumentStore
from app.storage.repositories import (
    SecMetadataRepository,
    SecRawDocumentRepository,
    UniverseRepository,
)

HTML_FIXTURE = Path("tests/fixtures/nvda-20241027x10q.html")


class FakeBinaryClient:
    def __init__(
        self,
        content: bytes,
        *,
        fail_first: bool = False,
        always_fail: bool = False,
    ) -> None:
        self._content = content
        self._fail_first = fail_first
        self._always_fail = always_fail
        self.calls = 0

    def get_binary(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> HttpBinaryResponse:
        del url, headers, timeout_seconds
        self.calls += 1
        if self._always_fail:
            raise RetryableHttpError("archive unavailable")
        if self._fail_first and self.calls == 1:
            raise RetryableHttpError("temporary archive error")
        return HttpBinaryResponse(
            content=self._content,
            content_type="text/html; charset=utf-8",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )


def configure_database_and_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "downloads.db"
    raw_store_path = tmp_path / "raw-store"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    monkeypatch.setenv("APP_STORAGE__RAW_STORE_DIR", str(raw_store_path))
    get_settings.cache_clear()


def build_repositories() -> (
    tuple[UniverseRepository, SecMetadataRepository, SecRawDocumentRepository]
):
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    return (
        UniverseRepository(session_factory),
        SecMetadataRepository(session_factory),
        SecRawDocumentRepository(session_factory),
    )


def seed_metadata(
    universe_repository: UniverseRepository,
    metadata_repository: SecMetadataRepository,
) -> None:
    universe_repository.replace_snapshot(
        UniverseSnapshot(
            snapshot_date=date(2024, 12, 31),
            constituents=[
                UniverseSnapshotConstituent(
                    snapshot_date=date(2024, 12, 31),
                    ticker="NVDA",
                    cik="0001045810",
                    company_name="NVIDIA Corp.",
                    exchange="NASDAQ",
                    is_domestic_filer=True,
                )
            ],
        )
    )
    metadata_repository.upsert_filings(
        filings=[
            SecSubmissionMetadata(
                cik="0001045810",
                ticker="NVDA",
                accession_number="0001045810-24-000101",
                filing_date=date(2024, 11, 20),
                form_type=FilingType.FORM_10Q,
                primary_document="nvda-20241027x10q.html",
                report_period=date(2024, 10, 27),
                source_url=build_archive_url(
                    "0001045810",
                    "0001045810-24-000101",
                    "nvda-20241027x10q.html",
                ),
            )
        ],
        cik="0001045810",
    )


def test_raw_filing_download_persists_file_and_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database_and_store(monkeypatch, tmp_path)
    universe_repository, metadata_repository, raw_repository = build_repositories()
    seed_metadata(universe_repository, metadata_repository)
    archive_client = SecArchiveClient(
        http_client=FakeBinaryClient(HTML_FIXTURE.read_bytes(), fail_first=True),
        user_agent="test-agent",
        timeout_seconds=5,
        max_retries=1,
        backoff_seconds=0,
        rate_limiter=RateLimiter(1000, sleep_func=lambda _: None),
        sleep_func=lambda _: None,
    )
    service = SecRawFilingDownloadService(
        archive_client=archive_client,
        metadata_repository=metadata_repository,
        raw_document_repository=raw_repository,
        raw_document_store=LocalRawDocumentStore(get_settings().storage.raw_store_dir),
        skip_existing=True,
    )

    result = service.download_filings(ticker="NVDA", forms=[FilingType.FORM_10Q])
    record = raw_repository.get_download_record("0001045810-24-000101")
    raw_store = LocalRawDocumentStore(get_settings().storage.raw_store_dir)

    assert result.downloaded_count == 1
    assert result.failed_count == 0
    assert record is not None
    assert record.status == DownloadStatusLabel.DOWNLOADED
    assert record.storage_path is not None
    assert raw_store.resolve(record.storage_path).exists()
    assert record.content_type == "text/html; charset=utf-8"
    assert record.size_bytes == len(HTML_FIXTURE.read_bytes())
    assert record.downloaded_at is not None
    get_settings.cache_clear()


def test_raw_filing_download_stores_failure_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database_and_store(monkeypatch, tmp_path)
    universe_repository, metadata_repository, raw_repository = build_repositories()
    seed_metadata(universe_repository, metadata_repository)
    archive_client = SecArchiveClient(
        http_client=FakeBinaryClient(b"", always_fail=True),
        user_agent="test-agent",
        timeout_seconds=5,
        max_retries=1,
        backoff_seconds=0,
        rate_limiter=RateLimiter(1000, sleep_func=lambda _: None),
        sleep_func=lambda _: None,
    )
    service = SecRawFilingDownloadService(
        archive_client=archive_client,
        metadata_repository=metadata_repository,
        raw_document_repository=raw_repository,
        raw_document_store=LocalRawDocumentStore(get_settings().storage.raw_store_dir),
        skip_existing=True,
    )

    result = service.download_filings(ticker="NVDA", forms=[FilingType.FORM_10Q])
    record = raw_repository.get_download_record("0001045810-24-000101")

    assert result.downloaded_count == 0
    assert result.failed_count == 1
    assert record is not None
    assert record.status == DownloadStatusLabel.FAILED
    assert record.last_error == "archive unavailable"
    assert record.downloaded_at is None
    get_settings.cache_clear()
