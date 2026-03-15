"""Service for fixture-based XBRL ingestion and typed queries."""

from __future__ import annotations

from pathlib import Path

from app.data_sources.xbrl.fixtures import XbrlFixtureLoader
from app.data_sources.xbrl.normalizer import normalize_xbrl_fixture
from app.domain import FilingType, XbrlCanonicalFact, XbrlFact
from app.storage.repositories import SecMetadataRepository, UniverseRepository, XbrlRepository


class XbrlIngestionService:
    """Normalize and persist canonical XBRL facts."""

    def __init__(
        self,
        fixture_loader: XbrlFixtureLoader,
        xbrl_repository: XbrlRepository,
        universe_repository: UniverseRepository,
        sec_metadata_repository: SecMetadataRepository,
    ) -> None:
        self._fixture_loader = fixture_loader
        self._xbrl_repository = xbrl_repository
        self._universe_repository = universe_repository
        self._sec_metadata_repository = sec_metadata_repository

    def import_fixture(self, *, ticker: str, file_path: Path) -> list[XbrlFact]:
        """Import canonical XBRL facts from a fixture file."""

        company = self._universe_repository.get_company_by_ticker(ticker)
        if company is None:
            msg = f"Ticker {ticker.upper()} was not found in company master"
            raise ValueError(msg)

        payload = self._fixture_loader.load(file_path)
        accession_number = str(payload["accession_number"])
        filing = self._sec_metadata_repository.get_filing_by_accession(accession_number)
        if filing is None or filing.ticker != company.ticker:
            msg = f"SEC filing metadata not found for accession {accession_number}"
            raise ValueError(msg)

        facts = normalize_xbrl_fixture(
            payload,
            company=company,
            filing_type=filing.form_type,
        )
        self._xbrl_repository.upsert_facts(facts)
        return facts

    def query_facts(
        self,
        *,
        ticker: str,
        canonical_facts: list[XbrlCanonicalFact] | None = None,
        filing_types: list[FilingType] | None = None,
        limit: int | None = None,
    ) -> list[XbrlFact]:
        """Query stored XBRL facts using typed filters."""

        return self._xbrl_repository.query_facts(
            ticker=ticker.upper(),
            canonical_facts=canonical_facts,
            filing_types=filing_types,
            limit=limit,
        )
