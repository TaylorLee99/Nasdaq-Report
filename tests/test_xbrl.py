from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.config import get_settings
from app.data_sources.xbrl.fixtures import FileXbrlFixtureLoader
from app.data_sources.xbrl.service import XbrlIngestionService
from app.domain import (
    FilingType,
    SecSubmissionMetadata,
    UniverseSnapshot,
    UniverseSnapshotConstituent,
    XbrlCanonicalFact,
)
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.repositories import SecMetadataRepository, UniverseRepository, XbrlRepository

XBRL_FIXTURE = Path("tests/fixtures/nvda_xbrl_fixture.json")


def configure_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "xbrl.db"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    get_settings.cache_clear()


def build_repositories() -> tuple[UniverseRepository, SecMetadataRepository, XbrlRepository]:
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    return (
        UniverseRepository(session_factory),
        SecMetadataRepository(session_factory),
        XbrlRepository(session_factory),
    )


def seed_metadata(
    universe_repository: UniverseRepository,
    sec_metadata_repository: SecMetadataRepository,
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
    sec_metadata_repository.upsert_filings(
        filings=[
            SecSubmissionMetadata(
                cik="0001045810",
                ticker="NVDA",
                accession_number="0001045810-24-000101",
                filing_date=date(2024, 11, 20),
                form_type=FilingType.FORM_10Q,
                primary_document="nvda-20241027x10q.html",
                report_period=date(2024, 10, 27),
                source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000101/nvda-20241027x10q.html",
            )
        ],
        cik="0001045810",
    )


def test_xbrl_fixture_import_and_typed_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database(monkeypatch, tmp_path)
    universe_repository, sec_metadata_repository, xbrl_repository = build_repositories()
    seed_metadata(universe_repository, sec_metadata_repository)
    service = XbrlIngestionService(
        fixture_loader=FileXbrlFixtureLoader(),
        xbrl_repository=xbrl_repository,
        universe_repository=universe_repository,
        sec_metadata_repository=sec_metadata_repository,
    )

    imported = service.import_fixture(ticker="NVDA", file_path=XBRL_FIXTURE)
    revenue_facts = service.query_facts(
        ticker="NVDA",
        canonical_facts=[XbrlCanonicalFact.REVENUE],
    )

    assert len(imported) == 3
    assert len(revenue_facts) == 1
    assert revenue_facts[0].canonical_fact == XbrlCanonicalFact.REVENUE
    assert revenue_facts[0].unit == "USDm"
    assert revenue_facts[0].fiscal_period == "Q3FY2025"
    get_settings.cache_clear()


def test_xbrl_cli_import_and_facts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_database(monkeypatch, tmp_path)
    universe_repository, sec_metadata_repository, _ = build_repositories()
    seed_metadata(universe_repository, sec_metadata_repository)
    runner = CliRunner()

    import_result = runner.invoke(
        app,
        ["xbrl", "import-fixture", "--ticker", "NVDA", "--file", str(XBRL_FIXTURE)],
    )
    facts_result = runner.invoke(
        app,
        ["xbrl", "facts", "--ticker", "NVDA", "--facts", "revenue,net_income"],
    )

    assert import_result.exit_code == 0
    assert '"canonical_fact": "revenue"' in import_result.stdout
    assert facts_result.exit_code == 0
    assert '"canonical_fact": "revenue"' in facts_result.stdout
    assert '"canonical_fact": "net_income"' in facts_result.stdout
    get_settings.cache_clear()
