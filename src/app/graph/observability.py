"""Node-level logging and state snapshot helpers for LangGraph execution."""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, cast

from langchain_core.runnables import RunnableConfig

from app.config import Settings
from app.domain import ManualReviewRequest
from app.graph.review_hooks import ManualReviewHook, NoOpManualReviewHook
from app.graph.state import ResearchState

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class StateSnapshotLogger:
    """Persist lightweight JSON snapshots for local debugging."""

    def __init__(self, trace_dir: str, *, enabled: bool) -> None:
        self._trace_dir = Path(trace_dir)
        self._enabled = enabled

    def log_snapshot(
        self,
        *,
        thread_id: str,
        node_name: str,
        state: ResearchState,
        update: dict[str, object],
        manual_review_requests: list[ManualReviewRequest],
    ) -> None:
        """Write one JSON snapshot for a node execution."""

        if not self._enabled:
            return
        target_dir = self._trace_dir / thread_id
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        payload = {
            "thread_id": thread_id,
            "node_name": node_name,
            "timestamp": timestamp,
            "state_summary": summarize_state(state),
            "update_summary": summarize_update(update),
            "manual_review_requests": [
                request.model_dump(mode="json") for request in manual_review_requests
            ],
        }
        (target_dir / f"{timestamp}_{node_name}.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )


def instrument_node(
    *,
    node_name: str,
    node_fn: Callable[[ResearchState], dict[str, object]],
    settings: Settings,
    review_hook: ManualReviewHook | None = None,
) -> Callable[..., dict[str, object]]:
    """Wrap a graph node with logging, snapshots, and manual-review evaluation."""

    effective_hook = review_hook or NoOpManualReviewHook()
    snapshot_logger = StateSnapshotLogger(
        settings.graph.trace_dir,
        enabled=settings.graph.enable_state_snapshot_logging,
    )

    def wrapped(
        state: ResearchState,
        config: Optional[RunnableConfig] = None,  # noqa: UP045
    ) -> dict[str, object]:
        thread_id = _resolve_thread_id(state=state, config=config)
        if settings.graph.enable_node_logging:
            logger.info(
                "graph.node.start thread_id=%s node=%s state=%s",
                thread_id,
                node_name,
                summarize_state(state),
            )
        update = node_fn(state)
        manual_review_requests = effective_hook.evaluate(
            node_name=node_name,
            thread_id=thread_id,
            state=state,
            update=update,
        )
        if manual_review_requests:
            existing = cast(list[ManualReviewRequest], update.get("manual_review_requests", []))
            update["manual_review_requests"] = [*existing, *manual_review_requests]
        snapshot_logger.log_snapshot(
            thread_id=thread_id,
            node_name=node_name,
            state=state,
            update=update,
            manual_review_requests=manual_review_requests,
        )
        if settings.graph.enable_node_logging:
            logger.info(
                "graph.node.end thread_id=%s node=%s update=%s manual_reviews=%s",
                thread_id,
                node_name,
                summarize_update(update),
                len(manual_review_requests),
            )
        return update

    return wrapped


def summarize_state(state: ResearchState) -> dict[str, Any]:
    """Return a compact state summary for logs and trace snapshots."""

    request = state.get("request")
    coverage_status = state.get("coverage_status")
    return {
        "request_id": request.request_id if request is not None else None,
        "ticker": request.company.ticker if request is not None else None,
        "completed_steps": list(state.get("completed_steps", [])),
        "route_plan": list(state.get("route_plan", [])),
        "shared_memory_count": len(state.get("shared_memory", [])),
        "agent_packet_count": len(state.get("agent_packets", [])),
        "verification_result_count": len(state.get("verification_results", [])),
        "manual_review_request_count": len(state.get("manual_review_requests", [])),
        "conflict_count": len(state.get("conflicts", [])),
        "coverage_label": coverage_status.label.value if coverage_status is not None else None,
    }


def summarize_update(update: dict[str, object]) -> dict[str, Any]:
    """Return a compact update summary for logs and trace snapshots."""

    summary: dict[str, Any] = {"keys": sorted(update.keys())}
    for key in (
        "completed_steps",
        "route_plan",
        "verification_results",
        "manual_review_requests",
        "conflicts",
        "agent_packets",
    ):
        value = update.get(key)
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
    return summary


def _resolve_thread_id(
    *,
    state: ResearchState,
    config: RunnableConfig | None,
) -> str:
    configurable = {}
    if config is not None:
        configurable = dict(config.get("configurable", {}))
    thread_id = configurable.get("thread_id")
    if thread_id:
        return str(thread_id)
    request = state.get("request")
    return request.request_id if request is not None else "unknown-thread"
