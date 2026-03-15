"""Repository for synchronous analysis job persistence."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.domain import AnalysisJobRecord, AnalysisJobStatus, DocumentSignal, FinalReport
from app.storage.models import AnalysisJobModel


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class AnalysisJobRepository:
    """Persist analysis job state for CLI and API retrieval."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def create_job(
        self,
        *,
        analysis_id: str,
        ticker: str,
        question: str,
        cik: str | None = None,
        request_payload: dict[str, object] | None = None,
    ) -> AnalysisJobRecord:
        """Insert an initial pending analysis job."""

        record = AnalysisJobModel(
            analysis_id=analysis_id,
            ticker=ticker.upper(),
            cik=cik,
            question=question,
            status=AnalysisJobStatus.PENDING.value,
            request_payload_json=request_payload or {},
            document_signals_json=[],
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return self._to_domain(record)

    def mark_running(
        self,
        *,
        analysis_id: str,
        document_signals: list[DocumentSignal],
    ) -> AnalysisJobRecord:
        """Mark a job as running and persist document availability signals."""

        with self._session_factory() as session:
            row = self._require_row(session, analysis_id)
            row.status = AnalysisJobStatus.RUNNING.value
            row.started_at = utc_now()
            row.document_signals_json = [
                signal.model_dump(mode="json") for signal in document_signals
            ]
            session.commit()
            session.refresh(row)
            return self._to_domain(row)

    def mark_completed(
        self,
        *,
        analysis_id: str,
        final_report: FinalReport,
        final_report_markdown: str,
    ) -> AnalysisJobRecord:
        """Mark a job as completed and store the final report payloads."""

        with self._session_factory() as session:
            row = self._require_row(session, analysis_id)
            row.status = AnalysisJobStatus.COMPLETED.value
            row.final_report_json = final_report.model_dump(mode="json")
            row.final_report_markdown = final_report_markdown
            row.error_message = None
            row.completed_at = utc_now()
            session.commit()
            session.refresh(row)
            return self._to_domain(row)

    def mark_failed(self, *, analysis_id: str, error_message: str) -> AnalysisJobRecord:
        """Mark a job as failed and store the error payload."""

        with self._session_factory() as session:
            row = self._require_row(session, analysis_id)
            row.status = AnalysisJobStatus.FAILED.value
            row.error_message = error_message
            row.completed_at = utc_now()
            session.commit()
            session.refresh(row)
            return self._to_domain(row)

    def get_job(self, analysis_id: str) -> AnalysisJobRecord | None:
        """Return one persisted analysis job."""

        with self._session_factory() as session:
            row = session.get(AnalysisJobModel, analysis_id)
            if row is None:
                return None
            return self._to_domain(row)

    @staticmethod
    def _require_row(session: Session, analysis_id: str) -> AnalysisJobModel:
        row = session.get(AnalysisJobModel, analysis_id)
        if row is None:
            msg = f"Analysis job {analysis_id} was not found"
            raise ValueError(msg)
        return row

    @staticmethod
    def _to_domain(row: AnalysisJobModel) -> AnalysisJobRecord:
        final_report = (
            FinalReport.model_validate(row.final_report_json)
            if row.final_report_json is not None
            else None
        )
        return AnalysisJobRecord(
            analysis_id=row.analysis_id,
            ticker=row.ticker,
            cik=row.cik,
            question=row.question,
            status=AnalysisJobStatus(row.status),
            document_signals=[
                DocumentSignal.model_validate(signal)
                for signal in list(row.document_signals_json or [])
            ],
            final_report=final_report,
            final_report_markdown=row.final_report_markdown,
            error_message=row.error_message,
            started_at=row.started_at,
            completed_at=row.completed_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
