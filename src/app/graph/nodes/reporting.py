"""Final reporting node."""

from __future__ import annotations

from app.graph.nodes.common import state_update
from app.graph.state import ResearchState
from app.reporting import build_verified_report


def final_reporter(state: ResearchState) -> dict[str, object]:
    """Create the final valuation-free report from verified findings only."""

    request = state["request"]
    report = build_verified_report(
        report_id=f"report:{request.request_id}",
        company=request.company,
        question=request.question,
        packets=state.get("agent_packets", []),
        verification_results=state.get("verification_results", []),
        coverage_status=state.get("coverage_status"),
        include_transcript=request.include_transcript,
        as_of_date=request.as_of_date,
        max_reretrievals=request.max_reretrievals,
        conflicts=state.get("conflicts", []),
    )
    return state_update(
        step_name="final_reporter",
        final_report=report,
        coverage_status=report.coverage_status,
    )
