"""Form-specific agent nodes backed by reusable specialized-agent subgraphs."""

from __future__ import annotations

from app.config import get_settings
from app.domain import CoverageLabel, FilingType, RoutedTask, TaskRoutingStatus
from app.graph.nodes.common import make_packet, state_update
from app.graph.retrieval import build_specialized_agent_retriever
from app.graph.state import ResearchState
from app.graph.subgraphs.specialized_agents import (
    CALL_PROFILE,
    EIGHT_K_PROFILE,
    TEN_K_PROFILE,
    TEN_Q_PROFILE,
    AgentProfile,
    execute_specialized_agent,
)
from app.llm.factory import build_specialized_agent_model


def run_10k_agent(state: ResearchState) -> dict[str, object]:
    """Run the 10-K specialist subgraph if the router activated it."""

    return _run_specialized_agent(
        state=state, profile=TEN_K_PROFILE, filing_type=FilingType.FORM_10K
    )


def run_10q_agent(state: ResearchState) -> dict[str, object]:
    """Run the 10-Q specialist subgraph if the router activated it."""

    return _run_specialized_agent(
        state=state, profile=TEN_Q_PROFILE, filing_type=FilingType.FORM_10Q
    )


def run_8k_agent(state: ResearchState) -> dict[str, object]:
    """Run the 8-K specialist subgraph if the router activated it."""

    return _run_specialized_agent(
        state=state, profile=EIGHT_K_PROFILE, filing_type=FilingType.FORM_8K
    )


def run_call_agent(state: ResearchState) -> dict[str, object]:
    """Run the optional call specialist subgraph if the router activated it."""

    return _run_specialized_agent(
        state=state,
        profile=CALL_PROFILE,
        filing_type=FilingType.EARNINGS_CALL,
        missing_topics=["transcript"] if not state["request"].include_transcript else [],
    )


def _run_specialized_agent(
    *,
    state: ResearchState,
    profile: AgentProfile,
    filing_type: FilingType,
    missing_topics: list[str] | None = None,
) -> dict[str, object]:
    tasks = _select_routed_tasks(state=state, agent_name=profile.agent_name)
    active = filing_type in state["request"].requested_filing_types
    routed = profile.agent_name in state.get("route_plan", [])
    if not active or not routed or not tasks:
        packet = make_packet(
            state=state,
            agent_name=profile.agent_name,
            summary=(
                f"{profile.agent_name} skipped because the task " "was not routed to this channel."
            ),
            coverage_label=CoverageLabel.NOT_STARTED,
            missing_topics=missing_topics if active and not routed else [],
        )
        return state_update(step_name=profile.agent_name, packet=packet)

    packet = execute_specialized_agent(
        profile=profile,
        request=state["request"],
        tasks=tasks,
        retriever=build_specialized_agent_retriever(get_settings()),
        model=build_specialized_agent_model(get_settings()),
    )
    return state_update(
        step_name=profile.agent_name,
        packet=packet,
        coverage_status=packet.coverage_status,
    )


def _select_routed_tasks(*, state: ResearchState, agent_name: str) -> list[RoutedTask]:
    return [
        task
        for task in state.get("routed_tasks", [])
        if task.agent_name == agent_name and task.status == TaskRoutingStatus.ROUTED
    ]
