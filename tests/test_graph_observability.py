from __future__ import annotations

from datetime import date
from pathlib import Path

from app.config import GraphRuntimeSettings, Settings
from app.domain import AnalysisRequest, Company, DocumentSignal, FilingType, ManualReviewRequest
from app.graph import (
    build_graph_invoke_config,
    build_initial_state,
    build_research_graph,
    build_resume_config,
    get_latest_checkpoint,
)
from app.graph.observability import instrument_node
from app.graph.review_hooks import ManualReviewHook
from app.graph.state import ResearchState


class StaticReviewHook(ManualReviewHook):
    def evaluate(
        self,
        *,
        node_name: str,
        thread_id: str,
        state: ResearchState,
        update: dict[str, object],
    ) -> list[ManualReviewRequest]:
        del state, update
        return [
            ManualReviewRequest(
                review_id=f"{thread_id}:{node_name}:review",
                thread_id=thread_id,
                node_name=node_name,
                reason="Manual review placeholder requested in test.",
            )
        ]


def build_request() -> AnalysisRequest:
    return AnalysisRequest(
        request_id="obs-thread-1",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Observe graph execution.",
        include_transcript=False,
        as_of_date=date(2024, 12, 31),
        document_signals=[
            DocumentSignal(
                filing_type=FilingType.FORM_10K,
                available=True,
                document_date=date(2024, 2, 21),
            ),
            DocumentSignal(
                filing_type=FilingType.FORM_10Q,
                available=True,
                document_date=date(2024, 11, 20),
            ),
            DocumentSignal(
                filing_type=FilingType.FORM_8K,
                available=True,
                document_date=date(2024, 8, 28),
            ),
            DocumentSignal(
                filing_type=FilingType.EARNINGS_CALL,
                available=False,
                notes="Transcript unavailable in test.",
            ),
        ],
    )


def test_graph_checkpoint_resume_and_snapshot_logging(tmp_path: Path) -> None:
    settings = Settings(
        graph=GraphRuntimeSettings(
            checkpoint_namespace="observability-test",
            trace_dir=str(tmp_path / "graph_traces"),
            enable_state_snapshot_logging=True,
        )
    )
    graph = build_research_graph(settings)
    request = build_request()
    thread_id = request.request_id

    graph.invoke(
        build_initial_state(request),
        config=build_graph_invoke_config(thread_id=thread_id, settings=settings),
    )

    latest = get_latest_checkpoint(thread_id=thread_id, settings=settings)
    resume_config = build_resume_config(thread_id=thread_id, settings=settings)
    trace_files = list((tmp_path / "graph_traces" / thread_id).glob("*.json"))

    assert latest is not None
    assert resume_config["configurable"]["thread_id"] == thread_id
    assert resume_config["configurable"].get("checkpoint_id") is not None
    assert trace_files


def test_instrument_node_attaches_manual_review_requests(tmp_path: Path) -> None:
    settings = Settings(
        graph=GraphRuntimeSettings(
            checkpoint_namespace="observability-hook-test",
            trace_dir=str(tmp_path / "graph_traces"),
            enable_state_snapshot_logging=False,
        )
    )
    request = build_request()

    wrapped = instrument_node(
        node_name="unit_test_node",
        node_fn=lambda state: {"completed_steps": ["unit_test_node"]},
        settings=settings,
        review_hook=StaticReviewHook(),
    )
    update = wrapped(
        build_initial_state(request),
        config=build_graph_invoke_config(thread_id=request.request_id, settings=settings),
    )

    manual_review_requests = update["manual_review_requests"]

    assert isinstance(manual_review_requests, list)
    assert len(manual_review_requests) == 1
    assert manual_review_requests[0].node_name == "unit_test_node"
