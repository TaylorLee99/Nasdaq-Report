"""Service for SEC submissions metadata fetch and sync."""

from __future__ import annotations

import json
from datetime import date

from app.data_sources.sec.client import SecSubmissionsClient
from app.data_sources.sec.normalizer import filter_supported_forms, normalize_submissions_payload
from app.domain import FilingType, SecSubmissionsSyncResult
from app.storage.repositories import SecMetadataRepository, UniverseRepository


class SecMetadataService:
    """Fetch, normalize, and persist SEC submissions metadata."""

    def __init__(
        self,
        submissions_client: SecSubmissionsClient,
        metadata_repository: SecMetadataRepository,
        universe_repository: UniverseRepository,
        *,
        save_raw_json: bool = False,
    ) -> None:
        self._submissions_client = submissions_client
        self._metadata_repository = metadata_repository
        self._universe_repository = universe_repository
        self._save_raw_json = save_raw_json

    def fetch_submissions(
        self,
        ticker: str,
        *,
        forms: list[FilingType] | None = None,
        save_raw_json: bool | None = None,
    ) -> SecSubmissionsSyncResult:
        """Fetch and persist SEC submissions metadata for a company master ticker."""

        company = self._universe_repository.get_company_by_ticker(ticker)
        if company is None:
            msg = f"Ticker {ticker.upper()} was not found in company master"
            raise ValueError(msg)

        payload = self._submissions_client.fetch_submissions(company.cik)
        normalized = normalize_submissions_payload(payload, cik=company.cik, ticker=company.ticker)
        filtered = filter_supported_forms(normalized, forms)
        should_save_raw = self._save_raw_json if save_raw_json is None else save_raw_json
        self._metadata_repository.upsert_filings(
            filings=filtered,
            raw_payload_json=json.dumps(payload, sort_keys=True) if should_save_raw else None,
            cik=company.cik,
        )
        requested_forms = forms or [FilingType.FORM_10K, FilingType.FORM_10Q, FilingType.FORM_8K]
        return SecSubmissionsSyncResult(
            cik=company.cik,
            ticker=company.ticker,
            requested_forms=requested_forms,
            fetched_rows=len(normalized),
            stored_rows=len(filtered),
            save_raw_json=should_save_raw,
        )

    def recent_filing_window(
        self,
        ticker: str,
        *,
        forms: list[FilingType] | None = None,
        filed_on_or_after: date | None = None,
        limit: int = 10,
    ) -> list:
        """Query the recent stored filing window for a ticker."""

        return self._metadata_repository.query_filings(
            ticker=ticker.upper(),
            forms=forms,
            filed_on_or_after=filed_on_or_after,
            limit=limit,
        )
