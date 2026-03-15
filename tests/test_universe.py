from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.config import get_settings
from app.ingestion import (
    CsvUniverseConstituentLoader,
    DomesticFilerOnlyFilter,
    UniverseIngestionService,
)
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.repositories import UniverseRepository

FIXTURE_FILE = Path("tests/fixtures/universe_snapshot.csv")


def configure_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "universe.db"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    get_settings.cache_clear()


def build_repository() -> UniverseRepository:
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    return UniverseRepository(make_session_factory(settings))


def test_universe_service_loads_domestic_filers_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database(monkeypatch, tmp_path)
    repository = build_repository()
    service = UniverseIngestionService(
        loader=CsvUniverseConstituentLoader(),
        filter_policy=DomesticFilerOnlyFilter(),
        repository=repository,
    )

    result = service.load_snapshot(FIXTURE_FILE)
    snapshot = repository.list_snapshot()
    company = repository.get_company_by_ticker("MSFT")

    assert result.total_rows == 3
    assert result.loaded_rows == 2
    assert result.filtered_out_rows == 1
    assert [constituent.ticker for constituent in snapshot.constituents] == ["MSFT", "NVDA"]
    assert company is not None
    assert company.company_name == "Microsoft Corp."
    assert company.is_domestic_filer is True
    get_settings.cache_clear()


def test_universe_cli_load_and_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_database(monkeypatch, tmp_path)
    runner = CliRunner()

    load_result = runner.invoke(app, ["universe", "load", "--file", str(FIXTURE_FILE)])
    list_result = runner.invoke(app, ["universe", "list"])

    assert load_result.exit_code == 0
    assert '"loaded_rows": 2' in load_result.stdout
    assert list_result.exit_code == 0
    assert "MSFT" in list_result.stdout
    assert "NVDA" in list_result.stdout
    assert "ASML" not in list_result.stdout
    get_settings.cache_clear()
