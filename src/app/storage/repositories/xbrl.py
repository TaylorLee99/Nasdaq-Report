"""Repository for canonical XBRL fact storage and typed queries."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.domain import Company, FilingType, XbrlCanonicalFact, XbrlFact
from app.storage.models import CompanyMasterModel, XbrlFactModel


class XbrlRepository:
    """Persist and query canonical XBRL facts."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def upsert_facts(self, facts: list[XbrlFact]) -> None:
        """Insert or update XBRL facts by stable fact id."""

        with self._session_factory() as session:
            for fact in facts:
                existing = session.scalar(
                    select(XbrlFactModel).where(XbrlFactModel.fact_id == fact.fact_id)
                )
                if existing is None:
                    session.add(self._to_model(fact))
                else:
                    existing.accession_number = fact.accession_number
                    existing.cik = fact.company.cik
                    existing.ticker = fact.company.ticker
                    existing.filing_type = fact.filing_type.value
                    existing.canonical_fact = fact.canonical_fact.value
                    existing.original_concept = fact.original_concept
                    existing.numeric_value = fact.numeric_value
                    existing.unit = fact.unit
                    existing.period_start = fact.period_start
                    existing.period_end = fact.period_end
                    existing.frame = fact.frame
                    existing.context_ref = fact.context_ref
                    existing.decimals = fact.decimals
                    existing.filing_date = fact.filing_date
                    existing.fiscal_period = fact.fiscal_period
            session.commit()

    def query_facts(
        self,
        *,
        ticker: str,
        canonical_facts: list[XbrlCanonicalFact] | None = None,
        filing_types: list[FilingType] | None = None,
        limit: int | None = None,
    ) -> list[XbrlFact]:
        """Query XBRL facts by ticker with typed filters."""

        with self._session_factory() as session:
            statement = select(XbrlFactModel).where(XbrlFactModel.ticker == ticker.upper())
            if canonical_facts:
                statement = statement.where(
                    XbrlFactModel.canonical_fact.in_([fact.value for fact in canonical_facts])
                )
            if filing_types:
                statement = statement.where(
                    XbrlFactModel.filing_type.in_(
                        [filing_type.value for filing_type in filing_types]
                    )
                )
            statement = statement.order_by(
                desc(XbrlFactModel.period_end),
                XbrlFactModel.canonical_fact.asc(),
            )
            if limit is not None:
                statement = statement.limit(limit)
            rows = session.scalars(statement).all()
            return [
                self._to_domain(row, self._get_company(session, row.cik, row.ticker))
                for row in rows
            ]

    @staticmethod
    def _get_company(session: Session, cik: str, ticker: str) -> Company:
        company_row = session.get(CompanyMasterModel, cik)
        if company_row is None:
            return Company(cik=cik, ticker=ticker, company_name=ticker)
        return Company(
            cik=company_row.cik,
            ticker=company_row.ticker,
            company_name=company_row.company_name,
            exchange=company_row.exchange,
            is_domestic_filer=company_row.is_domestic_filer,
            latest_snapshot_date=company_row.latest_snapshot_date,
        )

    @staticmethod
    def _to_model(fact: XbrlFact) -> XbrlFactModel:
        return XbrlFactModel(
            fact_id=fact.fact_id,
            accession_number=fact.accession_number,
            cik=fact.company.cik,
            ticker=fact.company.ticker,
            filing_type=fact.filing_type.value,
            canonical_fact=fact.canonical_fact.value,
            original_concept=fact.original_concept,
            numeric_value=fact.numeric_value,
            unit=fact.unit,
            period_start=fact.period_start,
            period_end=fact.period_end,
            frame=fact.frame,
            context_ref=fact.context_ref,
            decimals=fact.decimals,
            filing_date=fact.filing_date,
            fiscal_period=fact.fiscal_period,
        )

    @staticmethod
    def _to_domain(row: XbrlFactModel, company: Company) -> XbrlFact:
        return XbrlFact(
            fact_id=row.fact_id,
            accession_number=row.accession_number,
            company=company,
            filing_type=FilingType(row.filing_type),
            canonical_fact=XbrlCanonicalFact(row.canonical_fact),
            original_concept=row.original_concept,
            numeric_value=row.numeric_value,
            unit=row.unit,
            period_start=row.period_start,
            period_end=row.period_end,
            frame=row.frame,
            context_ref=row.context_ref,
            decimals=row.decimals,
            filing_date=row.filing_date,
            fiscal_period=row.fiscal_period,
        )
