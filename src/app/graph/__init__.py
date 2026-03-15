"""LangGraph orchestration helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.graph.checkpoint import (
    build_checkpoint_config,
    build_checkpointer,
    build_resume_config,
    get_latest_checkpoint,
)
from app.graph.state import ResearchState, build_initial_state


def build_graph_invoke_config(*args: Any, **kwargs: Any) -> Any:
    from app.graph.workflow import build_graph_invoke_config as _build_graph_invoke_config

    return _build_graph_invoke_config(*args, **kwargs)


def build_research_graph(*args: Any, **kwargs: Any) -> Any:
    from app.graph.workflow import build_research_graph as _build_research_graph

    return _build_research_graph(*args, **kwargs)


def describe_graph() -> Sequence[str]:
    from app.graph.workflow import describe_graph as _describe_graph

    return _describe_graph()


def describe_graph_structure() -> str:
    from app.graph.workflow import describe_graph_structure as _describe_graph_structure

    return _describe_graph_structure()

__all__ = [
    "ResearchState",
    "build_checkpoint_config",
    "build_checkpointer",
    "build_graph_invoke_config",
    "build_resume_config",
    "build_initial_state",
    "build_research_graph",
    "describe_graph",
    "describe_graph_structure",
    "get_latest_checkpoint",
]
