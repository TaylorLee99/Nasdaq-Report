"""Graph visualization helpers."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph


def render_mermaid(graph: CompiledStateGraph) -> str:
    """Return a Mermaid graph description for the compiled workflow."""

    return graph.get_graph().draw_mermaid()
