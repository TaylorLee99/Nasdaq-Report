from datetime import date
from typing import cast

from app.domain import (
    AgentOutputPacket,
    AnalysisRequest,
    AnalysisTaskType,
    Company,
    ConfidenceLabel,
    CoverageLabel,
    CoverageStatus,
    DocumentSignal,
    ExecutionPlan,
    FilingType,
    RoutedTask,
    TaskRoutingStatus,
)
from app.graph import (
    build_graph_invoke_config,
    build_initial_state,
    build_research_graph,
    describe_graph,
    describe_graph_structure,
)
from app.graph.nodes.planning import document_router, task_planner
from app.graph.state import ResearchState


def apply_update(state: ResearchState, update: dict[str, object]) -> ResearchState:
    merged = dict(state)
    merged.update(update)
    return cast(ResearchState, merged)


def test_task_planner_and_router_emit_execution_plan_and_routed_tasks() -> None:
    request = AnalysisRequest(
        request_id="req-plan-1",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Build a disclosure-grounded thesis.",
        as_of_date=date(2024, 12, 31),
        include_transcript=True,
        document_signals=[
            DocumentSignal(
                filing_type=FilingType.FORM_10K,
                available=True,
                document_date=date(2024, 2, 21),
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            DocumentSignal(
                filing_type=FilingType.FORM_10Q,
                available=True,
                document_date=date(2024, 11, 20),
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            DocumentSignal(
                filing_type=FilingType.FORM_8K,
                available=True,
                document_date=date(2024, 8, 28),
                parse_confidence=ConfidenceLabel.MEDIUM,
            ),
            DocumentSignal(
                filing_type=FilingType.EARNINGS_CALL,
                available=True,
                document_date=date(2024, 11, 20),
                parse_confidence=ConfidenceLabel.MEDIUM,
            ),
        ],
    )

    planner_state = apply_update(
        build_initial_state(request), task_planner(build_initial_state(request))
    )
    router_update = document_router(planner_state)
    routed_tasks = cast(list[RoutedTask], router_update["routed_tasks"])
    execution_plan = cast(ExecutionPlan, router_update["execution_plan"])
    planner_packets = planner_state["agent_packets"]
    router_packets = cast(list[AgentOutputPacket], router_update["agent_packets"])
    route_plan = cast(list[str], router_update["route_plan"])

    assert execution_plan is not None
    assert len(execution_plan.tasks) == 5
    assert [task.task_type for task in execution_plan.tasks] == [
        AnalysisTaskType.LONG_TERM_STRUCTURE,
        AnalysisTaskType.RECENT_QUARTER_CHANGE,
        AnalysisTaskType.MATERIAL_EVENTS,
        AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE,
        AnalysisTaskType.FINAL_THESIS,
    ]
    assert [task.status for task in routed_tasks] == [
        TaskRoutingStatus.ROUTED,
        TaskRoutingStatus.ROUTED,
        TaskRoutingStatus.ROUTED,
        TaskRoutingStatus.ROUTED,
        TaskRoutingStatus.ROUTED,
    ]
    assert route_plan == [
        "run_10k_agent",
        "run_10q_agent",
        "run_8k_agent",
        "run_call_agent",
        "synthetic_agent",
    ]
    assert any(entry.key == "execution_plan" for entry in planner_packets[0].shared_memory_updates)
    assert any(entry.key == "routed_tasks" for entry in router_packets[0].shared_memory_updates)


def test_document_router_records_transcript_gap_in_reduced_mode() -> None:
    request = AnalysisRequest(
        request_id="req-plan-2",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Assess management tone with fallback behavior.",
        as_of_date=date(2024, 12, 31),
        include_transcript=False,
        document_signals=[
            DocumentSignal(
                filing_type=FilingType.FORM_10K,
                available=True,
                document_date=date(2024, 2, 21),
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            DocumentSignal(
                filing_type=FilingType.FORM_10Q,
                available=True,
                document_date=date(2024, 11, 20),
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            DocumentSignal(
                filing_type=FilingType.FORM_8K,
                available=True,
                document_date=date(2024, 8, 28),
                parse_confidence=ConfidenceLabel.MEDIUM,
            ),
            DocumentSignal(
                filing_type=FilingType.EARNINGS_CALL,
                available=False,
                parse_confidence=ConfidenceLabel.LOW,
                notes="Transcript provider had no coverage.",
            ),
        ],
    )

    planner_state = apply_update(
        build_initial_state(request), task_planner(build_initial_state(request))
    )
    router_update = document_router(planner_state)
    routed_tasks = cast(list[RoutedTask], router_update["routed_tasks"])
    router_packets = cast(list[AgentOutputPacket], router_update["agent_packets"])
    coverage_status = cast(CoverageStatus, router_update["coverage_status"])
    route_plan = cast(list[str], router_update["route_plan"])
    tone_task = next(
        task for task in routed_tasks if task.task_type == AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE
    )

    assert tone_task.status == TaskRoutingStatus.MISSING_COVERAGE
    assert tone_task.missing_coverage_topics == [
        AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE.value,
        "transcript",
    ]
    assert "run_call_agent" not in route_plan
    assert coverage_status.missing_topics == [
        AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE.value,
        "transcript",
    ]
    assert any(
        entry.key.startswith("coverage_gap.") and "Transcript unavailable" in entry.value
        for entry in router_packets[0].shared_memory_updates
    )


def test_research_graph_compiles_and_invokes() -> None:
    graph = build_research_graph()
    request = AnalysisRequest(
        request_id="req-1",
        company=Company(cik="0000789019", ticker="MSFT", company_name="Microsoft Corp."),
        question="Summarize filing-grounded risks.",
        include_transcript=True,
        as_of_date=date(2024, 12, 31),
        document_signals=[
            DocumentSignal(
                filing_type=FilingType.FORM_10K,
                available=True,
                document_date=date(2024, 1, 30),
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            DocumentSignal(
                filing_type=FilingType.FORM_10Q,
                available=True,
                document_date=date(2024, 10, 24),
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            DocumentSignal(
                filing_type=FilingType.FORM_8K,
                available=True,
                document_date=date(2024, 11, 1),
                parse_confidence=ConfidenceLabel.MEDIUM,
            ),
            DocumentSignal(
                filing_type=FilingType.EARNINGS_CALL,
                available=True,
                document_date=date(2024, 10, 24),
                parse_confidence=ConfidenceLabel.MEDIUM,
            ),
        ],
    )

    result = graph.invoke(
        build_initial_state(request),
        config=build_graph_invoke_config(thread_id=request.request_id),
    )

    assert result["completed_steps"] == [
        "task_planner",
        "document_router",
        "run_10k_agent",
        "run_10q_agent",
        "run_8k_agent",
        "run_call_agent",
        "conflict_checker",
        "evidence_verifier",
        "manual_review_gate",
        "synthetic_agent",
        "final_reporter",
    ]
    assert result["route_plan"] == [
        "run_10k_agent",
        "run_10q_agent",
        "run_8k_agent",
        "run_call_agent",
        "synthetic_agent",
    ]
    assert result["execution_plan"] is not None
    assert len(result["routed_tasks"]) == 5
    assert result["verification_results"]
    assert result["final_report"] is not None
    assert result["final_report"].coverage_summary.coverage_label in {
        CoverageLabel.COMPLETE,
        CoverageLabel.PARTIAL,
    }
    assert result["final_report"].verification_summary.total_claims >= 1
    assert result["final_report"].executive_summary.summary
    assert len(result["agent_packets"]) == 10
    assert any(entry.key == "manual_review.pending_count" for entry in result["shared_memory"])
    assert any(entry.key == "synthesis.verification_count" for entry in result["shared_memory"])
    assert any(entry.key == "synthesis.verified_finding_count" for entry in result["shared_memory"])
    assert result["shared_memory"]


def test_graph_description_exposes_all_nodes() -> None:
    mermaid = describe_graph_structure()

    assert describe_graph() == (
        "task_planner",
        "document_router",
        "run_10k_agent",
        "run_10q_agent",
        "run_8k_agent",
        "run_call_agent",
        "conflict_checker",
        "evidence_verifier",
        "manual_review_gate",
        "synthetic_agent",
        "final_reporter",
    )
    assert "task_planner" in mermaid
    assert "final_reporter" in mermaid
