"""LangGraph runtime state and reducers for the research workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from operator import add
from typing import Annotated, TypedDict, TypeVar

from app.domain import (
    AgentOutputPacket,
    AnalysisRequest,
    ClaimVerificationResult,
    ConflictCandidate,
    CoverageStatus,
    ExecutionPlan,
    FinalReport,
    ManualReviewRequest,
    RoutedTask,
    SharedMemoryEntry,
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


ValueT = TypeVar("ValueT")


def replace_with_latest(current: ValueT, new_value: ValueT | None) -> ValueT:
    """Reducer that keeps the newest non-None value."""

    return current if new_value is None else new_value


class ResearchState(TypedDict, total=False):
    """Shared LangGraph state centered on append-only shared memory channels."""

    request: AnalysisRequest
    shared_memory: Annotated[list[SharedMemoryEntry], add]
    agent_packets: Annotated[list[AgentOutputPacket], add]
    verification_results: Annotated[list[ClaimVerificationResult], add]
    manual_review_requests: Annotated[list[ManualReviewRequest], add]
    execution_plan: Annotated[ExecutionPlan | None, replace_with_latest]
    routed_tasks: Annotated[list[RoutedTask], replace_with_latest]
    conflicts: Annotated[list[ConflictCandidate], add]
    completed_steps: Annotated[list[str], add]
    coverage_status: Annotated[CoverageStatus, replace_with_latest]
    final_report: Annotated[FinalReport | None, replace_with_latest]
    route_plan: Annotated[list[str], replace_with_latest]
    verifier_notes: Annotated[list[str], add]
    reretrieval_count: Annotated[int, replace_with_latest]
    max_reretrievals: Annotated[int, replace_with_latest]
    updated_at: Annotated[datetime, replace_with_latest]


def build_initial_state(request: AnalysisRequest) -> ResearchState:
    """Construct the initial graph state for a new analysis request."""

    return {
        "request": request,
        "shared_memory": [],
        "agent_packets": [],
        "verification_results": [],
        "manual_review_requests": [],
        "execution_plan": None,
        "routed_tasks": [],
        "conflicts": [],
        "completed_steps": [],
        "coverage_status": CoverageStatus.not_started(),
        "final_report": None,
        "route_plan": [],
        "verifier_notes": [],
        "reretrieval_count": 0,
        "max_reretrievals": request.max_reretrievals,
        "updated_at": utc_now(),
    }
