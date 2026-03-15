from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.config import get_settings
from app.data_sources.transcripts.service import TranscriptIngestionService
from app.domain import CoverageLabel, UniverseSnapshot, UniverseSnapshotConstituent
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.raw_store import LocalRawDocumentStore
from app.storage.repositories import TranscriptRepository, UniverseRepository

TRANSCRIPT_FIXTURE = Path("tests/fixtures/nvda_q3_call.txt")


def configure_database_and_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "transcripts.db"
    raw_store_path = tmp_path / "raw-store"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    monkeypatch.setenv("APP_STORAGE__RAW_STORE_DIR", str(raw_store_path))
    get_settings.cache_clear()


def build_repositories() -> tuple[UniverseRepository, TranscriptRepository]:
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    return UniverseRepository(session_factory), TranscriptRepository(session_factory)


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


def test_transcript_import_persists_metadata_and_raw_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database_and_store(monkeypatch, tmp_path)
    universe_repository, transcript_repository = build_repositories()
    seed_company_master(universe_repository)
    service = TranscriptIngestionService(
        transcript_repository=transcript_repository,
        universe_repository=universe_repository,
        raw_document_store=LocalRawDocumentStore(get_settings().storage.raw_store_dir),
    )

    result = service.import_transcript(
        ticker="NVDA",
        file_path=TRANSCRIPT_FIXTURE,
        call_date=date(2024, 11, 20),
        fiscal_quarter="Q3FY2025",
        speaker_separated=True,
    )
    stored = transcript_repository.list_transcripts(ticker="NVDA")

    assert result.imported_count == 1
    assert len(stored) == 1
    assert stored[0].speaker_separated is True
    assert stored[0].storage_path is not None
    assert Path(stored[0].storage_path).exists()
    assert stored[0].source_reliability.value == "high"
    assert stored[0].availability_coverage.label == CoverageLabel.COMPLETE
    get_settings.cache_clear()


def test_transcript_unavailable_records_missing_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database_and_store(monkeypatch, tmp_path)
    universe_repository, transcript_repository = build_repositories()
    seed_company_master(universe_repository)
    service = TranscriptIngestionService(
        transcript_repository=transcript_repository,
        universe_repository=universe_repository,
        raw_document_store=LocalRawDocumentStore(get_settings().storage.raw_store_dir),
    )

    result = service.record_unavailable(
        ticker="NVDA",
        reason="No transcript provider coverage for this quarter.",
    )
    listed = service.list_transcripts(ticker="NVDA")

    assert result.missing_count == 1
    assert len(listed.availability_gaps) == 1
    assert listed.availability_gaps[0].coverage_status.label == CoverageLabel.PARTIAL
    assert listed.availability_gaps[0].coverage_status.missing_topics == ["transcript"]
    get_settings.cache_clear()


def test_transcript_cli_import_and_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database_and_store(monkeypatch, tmp_path)
    universe_repository, _ = build_repositories()
    seed_company_master(universe_repository)
    runner = CliRunner()

    import_result = runner.invoke(
        app,
        [
            "transcript",
            "import",
            "--ticker",
            "NVDA",
            "--file",
            str(TRANSCRIPT_FIXTURE),
            "--call-date",
            "2024-11-20",
            "--fiscal-quarter",
            "Q3FY2025",
            "--speaker-separated",
        ],
    )
    list_result = runner.invoke(app, ["transcript", "list", "--ticker", "NVDA"])

    assert import_result.exit_code == 0
    assert '"imported_count": 1' in import_result.stdout
    assert list_result.exit_code == 0
    assert "Q3FY2025" in list_result.stdout
    assert "NVIDIA Corp." in list_result.stdout
    get_settings.cache_clear()
