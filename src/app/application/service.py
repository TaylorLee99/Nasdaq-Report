"""Synchronous application service for end-to-end analysis execution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Any, Protocol, cast
from uuid import uuid4

from app.domain import (
    AnalysisJobRecord,
    AnalysisRequest,
    AnalysisRunRequest,
    ConfidenceLabel,
    DocumentSignal,
    FilingType,
    FinalReport,
)
from app.graph import build_graph_invoke_config, build_initial_state, build_research_graph
from app.reporting import render_report_markdown
from app.storage.repositories import (
    AnalysisJobRepository,
    SecMetadataRepository,
    TranscriptRepository,
    UniverseRepository,
)

logger = logging.getLogger(__name__)


class GraphRunner(Protocol):
    """Compiled graph protocol used by the application service."""

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> Any:
        """Invoke the graph and return the terminal state payload."""


def default_graph_factory() -> GraphRunner:
    """Return the default compiled research graph."""

    return build_research_graph()


class AnalysisApplicationService:
    """Run one company analysis synchronously and persist the result."""

    def __init__(
        self,
        *,
        universe_repository: UniverseRepository,
        sec_metadata_repository: SecMetadataRepository,
        transcript_repository: TranscriptRepository,
        analysis_job_repository: AnalysisJobRepository,
        graph_factory: Callable[[], GraphRunner] = default_graph_factory,
    ) -> None:
        self._universe_repository = universe_repository
        self._sec_metadata_repository = sec_metadata_repository
        self._transcript_repository = transcript_repository
        self._analysis_job_repository = analysis_job_repository
        self._graph_factory = graph_factory

    def run_analysis(self, request: AnalysisRunRequest) -> AnalysisJobRecord:
        """Load company context, invoke the graph, persist the final report, and return the job."""

        company = self._universe_repository.get_company_by_ticker(request.ticker)
        if company is None:
            msg = f"Ticker {request.ticker.upper()} was not found in company master"
            raise ValueError(msg)

        analysis_id = f"analysis:{uuid4().hex}"
        logger.info("Starting analysis job %s for ticker=%s", analysis_id, company.ticker)
        self._analysis_job_repository.create_job(
            analysis_id=analysis_id,
            ticker=company.ticker,
            cik=company.cik,
            question=request.question,
            request_payload=request.model_dump(mode="json"),
        )

        try:
            document_signals = self._build_document_signals(
                ticker=company.ticker,
                include_transcript=request.include_transcript,
                as_of_date=request.as_of_date,
            )
            self._analysis_job_repository.mark_running(
                analysis_id=analysis_id,
                document_signals=document_signals,
            )
            graph_request = AnalysisRequest(
                request_id=analysis_id,
                company=company,
                question=request.question,
                as_of_date=request.as_of_date,
                include_transcript=request.include_transcript,
                document_signals=document_signals,
            )
            graph = self._graph_factory()
            result = graph.invoke(
                build_initial_state(graph_request),
                config=build_graph_invoke_config(thread_id=analysis_id),
            )
            final_report = cast(FinalReport | None, result.get("final_report"))
            if final_report is None:
                msg = f"Graph invocation for {analysis_id} did not produce a final report"
                raise ValueError(msg)
            markdown = render_report_markdown(final_report)
            job = self._analysis_job_repository.mark_completed(
                analysis_id=analysis_id,
                final_report=final_report,
                final_report_markdown=markdown,
            )
            logger.info("Completed analysis job %s for ticker=%s", analysis_id, company.ticker)
            return job
        except Exception as exc:
            logger.exception(
                "Analysis job %s failed for ticker=%s",
                analysis_id,
                company.ticker,
            )
            self._analysis_job_repository.mark_failed(
                analysis_id=analysis_id,
                error_message=str(exc),
            )
            raise

    def get_analysis(self, analysis_id: str) -> AnalysisJobRecord | None:
        """Return one persisted analysis job."""

        return self._analysis_job_repository.get_job(analysis_id)

    def _build_document_signals(
        self,
        *,
        ticker: str,
        include_transcript: bool,
        as_of_date: date | None,
    ) -> list[DocumentSignal]:
        signals: list[DocumentSignal] = []
        for filing_type in (
            FilingType.FORM_10K,
            FilingType.FORM_10Q,
            FilingType.FORM_8K,
        ):
            filings = self._sec_metadata_repository.recent_filing_window(
                ticker=ticker,
                forms=[filing_type],
                as_of=as_of_date,
                limit=1,
            )
            latest = filings[0] if filings else None
            signals.append(
                DocumentSignal(
                    filing_type=filing_type,
                    available=latest is not None,
                    document_date=latest.filing_date if latest is not None else None,
                    report_period=latest.report_period if latest is not None else None,
                    parse_confidence=(
                        ConfidenceLabel.HIGH if latest is not None else ConfidenceLabel.LOW
                    ),
                    notes=(
                        f"Latest {filing_type.value} metadata available."
                        if latest is not None
                        else f"No stored {filing_type.value} metadata."
                    ),
                )
            )

        transcripts = self._transcript_repository.list_transcripts(ticker=ticker)
        gaps = self._transcript_repository.list_availability_gaps(ticker=ticker)
        latest_transcript = transcripts[0] if transcripts else None
        transcript_available = include_transcript and latest_transcript is not None
        transcript_notes = "Transcript channel not requested."
        if include_transcript:
            if latest_transcript is not None:
                transcript_notes = "Transcript metadata available."
            elif gaps:
                transcript_notes = gaps[0].reason
            else:
                transcript_notes = "No transcript metadata is stored."
        signals.append(
            DocumentSignal(
                filing_type=FilingType.EARNINGS_CALL,
                available=transcript_available,
                document_date=(
                    latest_transcript.call_date if latest_transcript is not None else None
                ),
                parse_confidence=(
                    latest_transcript.source_reliability
                    if latest_transcript is not None
                    else ConfidenceLabel.LOW
                ),
                notes=transcript_notes,
            )
        )
        return signals
