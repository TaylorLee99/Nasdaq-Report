"""Deterministic planning and routing nodes."""

from __future__ import annotations

from datetime import date

from app.domain import (
    AgentOutputPacket,
    AnalysisRequest,
    AnalysisTaskType,
    ConfidenceLabel,
    CoverageLabel,
    CoverageStatus,
    DocumentSignal,
    ExecutionPlan,
    FilingType,
    PlannedTask,
    RoutedTask,
    TaskRoutingStatus,
)
from app.graph.nodes.common import make_memory_entry, make_packet, state_update
from app.graph.state import ResearchState

TASK_TO_CHANNEL: dict[AnalysisTaskType, tuple[FilingType | None, str]] = {
    AnalysisTaskType.LONG_TERM_STRUCTURE: (FilingType.FORM_10K, "run_10k_agent"),
    AnalysisTaskType.RECENT_QUARTER_CHANGE: (FilingType.FORM_10Q, "run_10q_agent"),
    AnalysisTaskType.MATERIAL_EVENTS: (FilingType.FORM_8K, "run_8k_agent"),
    AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE: (FilingType.EARNINGS_CALL, "run_call_agent"),
    AnalysisTaskType.FINAL_THESIS: (None, "synthetic_agent"),
}

TASK_DESCRIPTIONS: dict[AnalysisTaskType, str] = {
    AnalysisTaskType.LONG_TERM_STRUCTURE: (
        "Assess durable business structure and long-term disclosure baseline."
    ),
    AnalysisTaskType.RECENT_QUARTER_CHANGE: (
        "Assess quarter-over-quarter changes and short-term operating shifts."
    ),
    AnalysisTaskType.MATERIAL_EVENTS: "Review current-report disclosures for material events.",
    AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE: (
        "Assess management tone and guidance from the optional transcript channel."
    ),
    AnalysisTaskType.FINAL_THESIS: "Assemble the final valuation-free disclosure-grounded thesis.",
}

RECENCY_THRESHOLDS_DAYS: dict[AnalysisTaskType, int] = {
    AnalysisTaskType.LONG_TERM_STRUCTURE: 540,
    AnalysisTaskType.RECENT_QUARTER_CHANGE: 150,
    AnalysisTaskType.MATERIAL_EVENTS: 120,
    AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE: 150,
}


def task_planner(state: ResearchState) -> dict[str, object]:
    """Decompose the request into deterministic analysis tasks."""

    request = state["request"]
    tasks = [
        PlannedTask(
            task_type=task_type,
            description=TASK_DESCRIPTIONS[task_type],
            preferred_filing_type=TASK_TO_CHANNEL[task_type][0],
            preferred_agent=TASK_TO_CHANNEL[task_type][1],
        )
        for task_type in request.requested_tasks
    ]
    execution_plan = ExecutionPlan(
        plan_id=f"plan:{request.request_id}",
        request_id=request.request_id,
        tasks=tasks,
    )
    packet = make_packet(
        state=state,
        agent_name="task_planner",
        summary=(f"Planned {len(tasks)} deterministic tasks for {request.company.ticker}."),
        coverage_label=CoverageLabel.PARTIAL,
        extra_memory_updates=[
            make_memory_entry(
                state=state,
                source_agent="task_planner",
                key="execution_plan",
                value=execution_plan.model_dump_json(),
            )
        ],
    )
    return state_update(
        step_name="task_planner",
        packet=packet,
        execution_plan=execution_plan,
        coverage_status=CoverageStatus(
            label=CoverageLabel.PARTIAL,
            covered_topics=[task.task_type.value for task in tasks],
            notes="Execution plan established via deterministic task decomposition.",
        ),
    )


def document_router(state: ResearchState) -> dict[str, object]:
    """Route planned tasks to specialized agents using deterministic rules."""

    request = state["request"]
    plan = state.get("execution_plan") or ExecutionPlan(
        plan_id=f"plan:{request.request_id}",
        request_id=request.request_id,
        tasks=[],
    )
    signals = {signal.filing_type: signal for signal in request.document_signals}
    routed_tasks: list[RoutedTask] = []
    route_plan: list[str] = []
    coverage_gaps: list[str] = []
    missing_topics: list[str] = []

    for planned_task in plan.tasks:
        routed_task = _route_task(
            planned_task=planned_task,
            request=request,
            signal=(
                signals.get(planned_task.preferred_filing_type)
                if planned_task.preferred_filing_type is not None
                else None
            ),
        )
        routed_tasks.append(routed_task)
        if (
            routed_task.status == TaskRoutingStatus.ROUTED
            and routed_task.agent_name not in route_plan
        ):
            route_plan.append(routed_task.agent_name)
        if routed_task.status == TaskRoutingStatus.MISSING_COVERAGE:
            coverage_gaps.append(routed_task.reason)
            missing_topics.extend(routed_task.missing_coverage_topics)
        elif routed_task.parse_confidence == ConfidenceLabel.LOW:
            coverage_gaps.append(
                f"{routed_task.task_type.value}: low parse confidence for {routed_task.agent_name}."
            )
        if routed_task.recency_days is not None:
            threshold = RECENCY_THRESHOLDS_DAYS.get(routed_task.task_type)
            if threshold is not None and routed_task.recency_days > threshold:
                coverage_gaps.append(
                    f"{routed_task.task_type.value}: source recency is "
                    f"{routed_task.recency_days} days, exceeding {threshold} days."
                )

    updated_plan = plan.model_copy(update={"routed_tasks": routed_tasks})
    packet = _build_router_packet(
        state=state,
        execution_plan=updated_plan,
        routed_tasks=routed_tasks,
        coverage_gaps=coverage_gaps,
        missing_topics=list(dict.fromkeys(missing_topics)),
    )
    coverage_label = CoverageLabel.PARTIAL if coverage_gaps else CoverageLabel.COMPLETE
    return state_update(
        step_name="document_router",
        packet=packet,
        route_plan=route_plan,
        execution_plan=updated_plan,
        routed_tasks=routed_tasks,
        coverage_status=CoverageStatus(
            label=coverage_label,
            covered_topics=[
                task.task_type.value
                for task in routed_tasks
                if task.status == TaskRoutingStatus.ROUTED
            ],
            missing_topics=list(dict.fromkeys(missing_topics)),
            notes=(
                "Deterministic routing completed using document availability, "
                "recency, and parse confidence."
            ),
        ),
    )


def _route_task(
    *,
    planned_task: PlannedTask,
    request: AnalysisRequest,
    signal: DocumentSignal | None,
) -> RoutedTask:
    filing_type, agent_name = TASK_TO_CHANNEL[planned_task.task_type]
    if filing_type is None:
        return RoutedTask(
            task_type=planned_task.task_type,
            agent_name=agent_name,
            filing_type=None,
            status=TaskRoutingStatus.ROUTED,
            reason="Final synthesis task routed to synthetic agent.",
            document_available=True,
        )

    is_transcript_task = planned_task.task_type == AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE
    requested_channels = request.requested_filing_types
    channel_requested = filing_type in requested_channels
    document_available = signal.available if signal is not None else channel_requested
    if is_transcript_task and not request.include_transcript:
        document_available = False

    if not document_available:
        missing_topics = [planned_task.task_type.value]
        if is_transcript_task:
            missing_topics.append("transcript")
        return RoutedTask(
            task_type=planned_task.task_type,
            agent_name=agent_name,
            filing_type=filing_type,
            status=TaskRoutingStatus.MISSING_COVERAGE,
            reason=(
                "Transcript unavailable; reduced mode active for management tone/guidance."
                if is_transcript_task
                else f"{filing_type.value} document unavailable for routing."
            ),
            document_available=False,
            parse_confidence=signal.parse_confidence if signal is not None else None,
            missing_coverage_topics=missing_topics,
        )

    recency_days = _compute_recency_days(
        as_of_date=request.as_of_date,
        document_date=signal.document_date if signal is not None else None,
    )
    parse_confidence = signal.parse_confidence if signal is not None else ConfidenceLabel.MEDIUM
    reason_parts = [
        f"Routed {planned_task.task_type.value} to {agent_name} via {filing_type.value}"
    ]
    if recency_days is not None:
        reason_parts.append(f"recency={recency_days}d")
    reason_parts.append(f"parse_confidence={parse_confidence.value}")
    return RoutedTask(
        task_type=planned_task.task_type,
        agent_name=agent_name,
        filing_type=filing_type,
        status=TaskRoutingStatus.ROUTED,
        reason=", ".join(reason_parts) + ".",
        document_available=True,
        recency_days=recency_days,
        parse_confidence=parse_confidence,
    )


def _build_router_packet(
    *,
    state: ResearchState,
    execution_plan: ExecutionPlan,
    routed_tasks: list[RoutedTask],
    coverage_gaps: list[str],
    missing_topics: list[str],
) -> AgentOutputPacket:
    summary = (
        f"Routed {sum(task.status == TaskRoutingStatus.ROUTED for task in routed_tasks)} tasks; "
        f"coverage gaps={len(coverage_gaps)}."
    )
    memory_updates = [
        make_memory_entry(
            state=state,
            source_agent="document_router",
            key="routed_tasks",
            value=execution_plan.model_dump_json(),
        )
    ]
    memory_updates.extend(
        make_memory_entry(
            state=state,
            source_agent="document_router",
            key=f"coverage_gap.{index}",
            value=gap,
        )
        for index, gap in enumerate(coverage_gaps, start=1)
    )
    return make_packet(
        state=state,
        agent_name="document_router",
        summary=summary,
        coverage_label=CoverageLabel.PARTIAL if coverage_gaps else CoverageLabel.COMPLETE,
        missing_topics=missing_topics,
        extra_memory_updates=memory_updates,
    )


def _compute_recency_days(*, as_of_date: date | None, document_date: date | None) -> int | None:
    if as_of_date is None or document_date is None:
        return None
    return max((as_of_date - document_date).days, 0)
