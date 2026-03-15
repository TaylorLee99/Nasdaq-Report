"""Shared helpers for LangGraph node stubs."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain import (
    AgentOutputPacket,
    ClaimVerificationResult,
    ConflictCandidate,
    CoverageLabel,
    CoverageStatus,
    ExecutionPlan,
    ManualReviewRequest,
    RoutedTask,
    SharedMemoryEntry,
)
from app.graph.state import ResearchState


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def make_memory_entry(
    *, state: ResearchState, source_agent: str, key: str, value: str
) -> SharedMemoryEntry:
    """Create a shared-memory update emitted by a node."""

    request = state["request"]
    return SharedMemoryEntry(
        memory_id=f"{request.request_id}:{source_agent}:{key}",
        key=key,
        value=value,
        source_agent=source_agent,
    )


def make_packet(
    *,
    state: ResearchState,
    agent_name: str,
    summary: str,
    coverage_label: CoverageLabel = CoverageLabel.PARTIAL,
    missing_topics: list[str] | None = None,
    extra_memory_updates: list[SharedMemoryEntry] | None = None,
) -> AgentOutputPacket:
    """Create a stubbed structured agent packet with shared-memory provenance."""

    memory_update = make_memory_entry(
        state=state,
        source_agent=agent_name,
        key=f"{agent_name}.summary",
        value=summary,
    )
    shared_memory_updates = [memory_update, *(extra_memory_updates or [])]
    return AgentOutputPacket(
        agent_name=agent_name,
        shared_memory_updates=shared_memory_updates,
        coverage_status=CoverageStatus(
            label=coverage_label,
            covered_topics=[agent_name],
            missing_topics=missing_topics or [],
            notes=summary,
        ),
    )


def state_update(
    *,
    step_name: str,
    packet: AgentOutputPacket | None = None,
    route_plan: list[str] | None = None,
    conflicts: list[ConflictCandidate] | None = None,
    verification_results: list[ClaimVerificationResult] | None = None,
    manual_review_requests: list[ManualReviewRequest] | None = None,
    execution_plan: ExecutionPlan | None = None,
    routed_tasks: list[RoutedTask] | None = None,
    coverage_status: CoverageStatus | None = None,
    verifier_notes: list[str] | None = None,
    final_report: object | None = None,
) -> dict[str, object]:
    """Build a consistent LangGraph state delta."""

    update: dict[str, object] = {
        "completed_steps": [step_name],
        "updated_at": utc_now(),
    }
    if packet is not None:
        update["agent_packets"] = [packet]
        update["shared_memory"] = packet.shared_memory_updates
    if route_plan is not None:
        update["route_plan"] = route_plan
    if conflicts is not None:
        update["conflicts"] = conflicts
    if verification_results is not None:
        update["verification_results"] = verification_results
    if manual_review_requests is not None:
        update["manual_review_requests"] = manual_review_requests
    if execution_plan is not None:
        update["execution_plan"] = execution_plan
    if routed_tasks is not None:
        update["routed_tasks"] = routed_tasks
    if coverage_status is not None:
        update["coverage_status"] = coverage_status
    if verifier_notes:
        update["verifier_notes"] = verifier_notes
    if final_report is not None:
        update["final_report"] = final_report
    return update
