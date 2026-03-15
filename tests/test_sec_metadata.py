import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pytest

from app.config import get_settings
from app.data_sources.sec import (
    RateLimiter,
    RetryableHttpError,
    SecMetadataService,
    SecSubmissionsClient,
)
from app.domain import FilingType, UniverseSnapshot, UniverseSnapshotConstituent
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.repositories import SecMetadataRepository, UniverseRepository

FIXTURE_FILE = Path("tests/fixtures/sec_submissions_nvda.json")


class FakeHttpClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.calls = 0

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, object]:
        del url, headers, timeout_seconds
        self.calls += 1
        if self.calls == 1:
            raise RetryableHttpError("temporary failure")
        return self._payload


def configure_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "sec.db"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    get_settings.cache_clear()


def build_repositories() -> tuple[UniverseRepository, SecMetadataRepository]:
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    return UniverseRepository(session_factory), SecMetadataRepository(session_factory)


def seed_company_master(universe_repository: UniverseRepository) -> None:
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


def test_sec_metadata_service_retries_and_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database(monkeypatch, tmp_path)
    universe_repository, metadata_repository = build_repositories()
    seed_company_master(universe_repository)
    payload = json.loads(FIXTURE_FILE.read_text(encoding="utf-8"))
    client = SecSubmissionsClient(
        http_client=FakeHttpClient(payload),
        base_url="https://data.sec.gov/submissions",
        user_agent="test-agent",
        timeout_seconds=5,
        max_retries=1,
        backoff_seconds=0,
        rate_limiter=RateLimiter(1000, sleep_func=lambda _: None),
        sleep_func=lambda _: None,
    )
    service = SecMetadataService(
        submissions_client=client,
        metadata_repository=metadata_repository,
        universe_repository=universe_repository,
        save_raw_json=True,
    )

    result = service.fetch_submissions("NVDA")
    recent = metadata_repository.recent_filing_window(
        ticker="NVDA",
        forms=[FilingType.FORM_10K, FilingType.FORM_10Q],
        days=365,
        as_of=date(2024, 12, 31),
    )

    assert result.fetched_rows == 3
    assert result.stored_rows == 3
    assert metadata_repository.raw_payload_count(cik="0001045810") == 1
    assert [filing.form_type for filing in recent] == [FilingType.FORM_10Q, FilingType.FORM_10K]
    get_settings.cache_clear()


def test_sec_metadata_repository_filters_form_and_date(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database(monkeypatch, tmp_path)
    universe_repository, metadata_repository = build_repositories()
    seed_company_master(universe_repository)
    payload = json.loads(FIXTURE_FILE.read_text(encoding="utf-8"))
    client = SecSubmissionsClient(
        http_client=FakeHttpClient(payload),
        base_url="https://data.sec.gov/submissions",
        user_agent="test-agent",
        timeout_seconds=5,
        max_retries=1,
        backoff_seconds=0,
        rate_limiter=RateLimiter(1000, sleep_func=lambda _: None),
        sleep_func=lambda _: None,
    )
    service = SecMetadataService(
        submissions_client=client,
        metadata_repository=metadata_repository,
        universe_repository=universe_repository,
        save_raw_json=False,
    )

    service.fetch_submissions("NVDA", forms=[FilingType.FORM_8K])
    filings = metadata_repository.query_filings(
        ticker="NVDA",
        forms=[FilingType.FORM_8K],
        filed_on_or_after=date(2024, 8, 1),
    )

    assert len(filings) == 1
    assert filings[0].primary_document == "nvda-ex991.htm"
    assert metadata_repository.raw_payload_count(cik="0001045810") == 0
    get_settings.cache_clear()
