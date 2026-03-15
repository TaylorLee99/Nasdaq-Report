"""Shared-memory-centered LangGraph workflow skeleton."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings, get_settings
from app.graph.checkpoint import build_checkpoint_config, build_checkpointer
from app.graph.nodes import (
    conflict_checker,
    document_router,
    evidence_verifier,
    final_reporter,
    manual_review_gate,
    run_8k_agent,
    run_10k_agent,
    run_10q_agent,
    run_call_agent,
    synthetic_agent,
    task_planner,
)
from app.graph.observability import instrument_node
from app.graph.state import ResearchState, build_initial_state
from app.graph.visualization import render_mermaid

GRAPH_NODE_NAMES: tuple[str, ...] = (
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


def build_research_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Compile the research workflow with checkpointer wiring enabled."""

    current_settings = settings or get_settings()
    workflow = StateGraph(ResearchState)
    _add_instrumented_node(workflow, "task_planner", task_planner, current_settings)
    _add_instrumented_node(workflow, "document_router", document_router, current_settings)
    _add_instrumented_node(workflow, "run_10k_agent", run_10k_agent, current_settings)
    _add_instrumented_node(workflow, "run_10q_agent", run_10q_agent, current_settings)
    _add_instrumented_node(workflow, "run_8k_agent", run_8k_agent, current_settings)
    _add_instrumented_node(workflow, "run_call_agent", run_call_agent, current_settings)
    _add_instrumented_node(workflow, "conflict_checker", conflict_checker, current_settings)
    _add_instrumented_node(workflow, "evidence_verifier", evidence_verifier, current_settings)
    _add_instrumented_node(
        workflow,
        "manual_review_gate",
        manual_review_gate,
        current_settings,
    )
    _add_instrumented_node(workflow, "synthetic_agent", synthetic_agent, current_settings)
    _add_instrumented_node(workflow, "final_reporter", final_reporter, current_settings)

    workflow.add_edge(START, "task_planner")
    workflow.add_edge("task_planner", "document_router")
    workflow.add_edge("document_router", "run_10k_agent")
    workflow.add_edge("run_10k_agent", "run_10q_agent")
    workflow.add_edge("run_10q_agent", "run_8k_agent")
    workflow.add_edge("run_8k_agent", "run_call_agent")
    workflow.add_edge("run_call_agent", "conflict_checker")
    workflow.add_edge("conflict_checker", "evidence_verifier")
    workflow.add_edge("evidence_verifier", "manual_review_gate")
    workflow.add_edge("manual_review_gate", "synthetic_agent")
    workflow.add_edge("synthetic_agent", "final_reporter")
    workflow.add_edge("final_reporter", END)

    return workflow.compile(checkpointer=build_checkpointer(current_settings))


def build_graph_invoke_config(
    *,
    thread_id: str,
    settings: Settings | None = None,
    checkpoint_id: str | None = None,
) -> RunnableConfig:
    """Build the invoke config required by the compiled graph checkpointer."""

    return build_checkpoint_config(
        thread_id=thread_id,
        settings=settings,
        checkpoint_id=checkpoint_id,
    )


def describe_graph() -> tuple[str, ...]:
    """Return stable node names for CLI and documentation."""

    return GRAPH_NODE_NAMES


def describe_graph_structure(settings: Settings | None = None) -> str:
    """Return a Mermaid description of the compiled graph."""

    return render_mermaid(build_research_graph(settings))


def _add_instrumented_node(
    workflow: StateGraph,
    node_name: str,
    node_fn: Callable[[ResearchState], dict[str, object]],
    settings: Settings,
) -> None:
    """Attach one instrumented node to the graph builder."""

    workflow.add_node(
        node_name,
        cast(Any, instrument_node(node_name=node_name, node_fn=node_fn, settings=settings)),
    )


__all__ = [
    "build_graph_invoke_config",
    "build_initial_state",
    "build_research_graph",
    "describe_graph",
    "describe_graph_structure",
]
