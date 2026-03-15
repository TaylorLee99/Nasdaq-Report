"""Manual review hook interfaces and no-op implementations."""

from __future__ import annotations

from typing import Protocol

from app.domain import ManualReviewRequest
from app.graph.state import ResearchState


class ManualReviewHook(Protocol):
    """Evaluate node outputs and emit future human-review requests."""

    def evaluate(
        self,
        *,
        node_name: str,
        thread_id: str,
        state: ResearchState,
        update: dict[str, object],
    ) -> list[ManualReviewRequest]:
        """Return manual review requests for this node execution."""


class NoOpManualReviewHook:
    """Default hook that emits no manual review requests."""

    def evaluate(
        self,
        *,
        node_name: str,
        thread_id: str,
        state: ResearchState,
        update: dict[str, object],
    ) -> list[ManualReviewRequest]:
        del node_name, thread_id, state, update
        return []
