"""Repositories for universe snapshot persistence."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session

from app.domain import Company, UniverseSnapshot, UniverseSnapshotConstituent
from app.storage.models import CompanyMasterModel, UniverseSnapshotConstituentModel


class UniverseRepository:
    """Repository for company master and universe snapshot tables."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def replace_snapshot(self, snapshot: UniverseSnapshot) -> None:
        """Replace the stored rows for a snapshot date with the provided constituents."""

        with self._session_factory() as session:
            session.execute(
                delete(UniverseSnapshotConstituentModel).where(
                    UniverseSnapshotConstituentModel.snapshot_date == snapshot.snapshot_date
                )
            )
            for constituent in snapshot.constituents:
                self._upsert_company(session, constituent)
                session.add(self._to_snapshot_model(constituent))
            session.commit()

    def list_snapshot(self, snapshot_date: date | None = None) -> UniverseSnapshot:
        """Return a snapshot by date, or the latest snapshot if no date is provided."""

        with self._session_factory() as session:
            effective_date = snapshot_date or self._latest_snapshot_date(session)
            if effective_date is None:
                return UniverseSnapshot(snapshot_date=date.min, constituents=[])
            rows = session.scalars(
                select(UniverseSnapshotConstituentModel)
                .where(UniverseSnapshotConstituentModel.snapshot_date == effective_date)
                .order_by(UniverseSnapshotConstituentModel.ticker.asc())
            ).all()
            return UniverseSnapshot(
                snapshot_date=effective_date,
                constituents=[self._to_snapshot_constituent(row) for row in rows],
            )

    def list_companies(self) -> list[Company]:
        """Return all company master rows ordered by ticker."""

        with self._session_factory() as session:
            rows = session.scalars(
                select(CompanyMasterModel).order_by(CompanyMasterModel.ticker.asc())
            ).all()
            return [self._to_company(row) for row in rows]

    def get_company_by_ticker(self, ticker: str) -> Company | None:
        """Return a company master record by ticker."""

        with self._session_factory() as session:
            row = session.scalar(
                select(CompanyMasterModel).where(CompanyMasterModel.ticker == ticker.upper())
            )
            if row is None:
                return None
            return self._to_company(row)

    @staticmethod
    def _latest_snapshot_date(session: Session) -> date | None:
        return session.scalar(
            select(UniverseSnapshotConstituentModel.snapshot_date)
            .order_by(desc(UniverseSnapshotConstituentModel.snapshot_date))
            .limit(1)
        )

    @staticmethod
    def _upsert_company(session: Session, constituent: UniverseSnapshotConstituent) -> None:
        company_row = session.get(CompanyMasterModel, constituent.cik)
        if company_row is None:
            session.add(
                CompanyMasterModel(
                    cik=constituent.cik,
                    ticker=constituent.ticker,
                    company_name=constituent.company_name,
                    exchange=constituent.exchange,
                    is_domestic_filer=constituent.is_domestic_filer,
                    latest_snapshot_date=constituent.snapshot_date,
                )
            )
            return

        company_row.ticker = constituent.ticker
        company_row.company_name = constituent.company_name
        company_row.exchange = constituent.exchange
        company_row.is_domestic_filer = constituent.is_domestic_filer
        if (
            company_row.latest_snapshot_date is None
            or company_row.latest_snapshot_date < constituent.snapshot_date
        ):
            company_row.latest_snapshot_date = constituent.snapshot_date

    @staticmethod
    def _to_snapshot_model(
        constituent: UniverseSnapshotConstituent,
    ) -> UniverseSnapshotConstituentModel:
        return UniverseSnapshotConstituentModel(
            snapshot_date=constituent.snapshot_date,
            cik=constituent.cik,
            ticker=constituent.ticker,
            company_name=constituent.company_name,
            exchange=constituent.exchange,
            is_domestic_filer=constituent.is_domestic_filer,
        )

    @staticmethod
    def _to_snapshot_constituent(
        row: UniverseSnapshotConstituentModel,
    ) -> UniverseSnapshotConstituent:
        return UniverseSnapshotConstituent(
            snapshot_date=row.snapshot_date,
            ticker=row.ticker,
            cik=row.cik,
            company_name=row.company_name,
            exchange=row.exchange,
            is_domestic_filer=row.is_domestic_filer,
        )

    @staticmethod
    def _to_company(row: CompanyMasterModel) -> Company:
        return Company(
            cik=row.cik,
            ticker=row.ticker,
            company_name=row.company_name,
            exchange=row.exchange,
            is_domestic_filer=row.is_domestic_filer,
            latest_snapshot_date=row.latest_snapshot_date,
        )
