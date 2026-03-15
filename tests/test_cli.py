import json
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.config import get_settings
from app.domain import (
    ChunkRecord,
    Company,
    ConfidenceLabel,
    FilingType,
    SecSubmissionMetadata,
    UniverseSnapshot,
    UniverseSnapshotConstituent,
)
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.repositories import (
    PgVectorChunkRepository,
    SecMetadataRepository,
    UniverseRepository,
)


def test_info_command_runs() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert "app_name" in result.stdout


def test_analyze_company_command_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cli-analysis.db"
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
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "analyze",
            "company",
            "--ticker",
            "NVDA",
            "--question",
            "Build a disclosure-grounded thesis.",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["final_report"] is not None


def test_index_inspect_8k_command_lists_latest_item_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cli-inspect.db"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    get_settings.cache_clear()
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    vector_repository = PgVectorChunkRepository(session_factory)
    company = Company(
        cik="0001045810",
        ticker="NVDA",
        company_name="NVIDIA Corp.",
        exchange="NASDAQ",
        is_domestic_filer=True,
    )
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
    vector_repository.replace_document_chunks(
        document_id="doc-8k-latest",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-1-01",
                document_id="doc-8k-latest",
                section_id="section-1-01",
                company=company,
                filing_type=FilingType.FORM_8K,
                accession_number="0001045810-24-000210",
                filing_date=date(2024, 12, 5),
                section_name="Item 1.01 Entry into a Material Definitive Agreement",
                item_number="1.01",
                text="The company entered into a material definitive agreement.",
                token_count=8,
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            ChunkRecord(
                chunk_id="chunk-5-02",
                document_id="doc-8k-latest",
                section_id="section-5-02",
                company=company,
                filing_type=FilingType.FORM_8K,
                accession_number="0001045810-24-000210",
                filing_date=date(2024, 12, 5),
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text="The board approved a leadership compensation update.",
                token_count=8,
                parse_confidence=ConfidenceLabel.HIGH,
            ),
        ],
    )
    runner = CliRunner()

    result = runner.invoke(app, ["index", "inspect-8k", "--ticker", "NVDA"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["chunk_count"] == 2
    assert payload["item_counts"] == {"1.01": 1, "5.02": 1}
