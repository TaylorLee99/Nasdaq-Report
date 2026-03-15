"""Repository for SEC submissions metadata persistence and queries."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.domain import FilingType, SecSubmissionMetadata
from app.storage.models import SecSubmissionMetadataModel, SecSubmissionsRawModel


class SecMetadataRepository:
    """Persist and query SEC submissions metadata."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def upsert_filings(
        self,
        *,
        filings: list[SecSubmissionMetadata],
        cik: str,
        raw_payload_json: str | None = None,
    ) -> None:
        """Insert or update filing metadata rows and optionally save the raw payload."""

        with self._session_factory() as session:
            for filing in filings:
                existing = session.scalar(
                    select(SecSubmissionMetadataModel).where(
                        SecSubmissionMetadataModel.accession_number == filing.accession_number
                    )
                )
                if existing is None:
                    session.add(self._to_model(filing))
                else:
                    existing.ticker = filing.ticker
                    existing.filing_date = filing.filing_date
                    existing.form_type = filing.form_type.value
                    existing.primary_document = filing.primary_document
                    existing.report_period = filing.report_period
                    existing.source_url = filing.source_url
            if raw_payload_json is not None:
                session.add(SecSubmissionsRawModel(cik=cik, payload_json=raw_payload_json))
            session.commit()

    def query_filings(
        self,
        *,
        ticker: str,
        forms: list[FilingType] | None = None,
        filed_on_or_after: date | None = None,
        limit: int | None = None,
    ) -> list[SecSubmissionMetadata]:
        """Query stored filing metadata by ticker, form, and filing date."""

        with self._session_factory() as session:
            statement = select(SecSubmissionMetadataModel).where(
                SecSubmissionMetadataModel.ticker == ticker.upper()
            )
            if forms:
                statement = statement.where(
                    SecSubmissionMetadataModel.form_type.in_([form.value for form in forms])
                )
            if filed_on_or_after is not None:
                statement = statement.where(
                    SecSubmissionMetadataModel.filing_date >= filed_on_or_after
                )
            statement = statement.order_by(desc(SecSubmissionMetadataModel.filing_date))
            if limit is not None:
                statement = statement.limit(limit)
            rows = session.scalars(statement).all()
            return [self._to_domain(row) for row in rows]

    def get_filing_by_accession(self, accession_number: str) -> SecSubmissionMetadata | None:
        """Return a filing metadata row by accession number."""

        with self._session_factory() as session:
            row = session.scalar(
                select(SecSubmissionMetadataModel).where(
                    SecSubmissionMetadataModel.accession_number == accession_number
                )
            )
            if row is None:
                return None
            return self._to_domain(row)

    def recent_filing_window(
        self,
        *,
        ticker: str,
        forms: list[FilingType] | None = None,
        days: int = 365,
        as_of: date | None = None,
        limit: int | None = None,
    ) -> list[SecSubmissionMetadata]:
        """Return filings within a recent rolling window."""

        end_date = as_of or date.today()
        start_date = end_date - timedelta(days=days)
        return self.query_filings(
            ticker=ticker,
            forms=forms,
            filed_on_or_after=start_date,
            limit=limit,
        )

    def raw_payload_count(self, *, cik: str) -> int:
        """Return the number of stored raw payload snapshots for a CIK."""

        with self._session_factory() as session:
            rows = session.scalars(
                select(SecSubmissionsRawModel).where(SecSubmissionsRawModel.cik == cik)
            ).all()
            return len(rows)

    @staticmethod
    def _to_model(filing: SecSubmissionMetadata) -> SecSubmissionMetadataModel:
        return SecSubmissionMetadataModel(
            cik=filing.cik,
            ticker=filing.ticker,
            accession_number=filing.accession_number,
            filing_date=filing.filing_date,
            form_type=filing.form_type.value,
            primary_document=filing.primary_document,
            report_period=filing.report_period,
            source_url=filing.source_url,
        )

    @staticmethod
    def _to_domain(row: SecSubmissionMetadataModel) -> SecSubmissionMetadata:
        return SecSubmissionMetadata(
            cik=row.cik,
            ticker=row.ticker,
            accession_number=row.accession_number,
            filing_date=row.filing_date,
            form_type=FilingType(row.form_type),
            primary_document=row.primary_document,
            report_period=row.report_period,
            source_url=row.source_url,
        )
