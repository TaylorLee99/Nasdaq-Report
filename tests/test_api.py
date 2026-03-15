from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import get_settings
from app.domain import (
    FilingType,
    SecSubmissionMetadata,
    UniverseSnapshot,
    UniverseSnapshotConstituent,
)
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.repositories import SecMetadataRepository, UniverseRepository


def test_health_check_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analysis_run_and_get_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "api-analysis.db"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    monkeypatch.setenv("APP_STORAGE__RAW_STORE_DIR", str(tmp_path / "raw"))
    get_settings.cache_clear()
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    metadata_repository = SecMetadataRepository(session_factory)
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
                primary_document="nvda-20241027x10q.htm",
                report_period=date(2024, 10, 27),
                source_url="https://www.sec.gov/Archives/example-10q.htm",
            )
        ],
        cik="0001045810",
    )
    client = TestClient(app)

    response = client.post(
        "/analysis/run",
        json={
            "ticker": "NVDA",
            "question": "Build a disclosure-grounded thesis.",
            "include_transcript": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["final_report"] is not None
    assert payload["final_report_markdown"].startswith("# Evidence-Grounded Report")

    fetched = client.get(f"/analysis/{payload['analysis_id']}")

    assert fetched.status_code == 200
    fetched_payload = fetched.json()
    assert fetched_payload["analysis_id"] == payload["analysis_id"]
    assert fetched_payload["document_signals"]
