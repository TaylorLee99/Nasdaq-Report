"""Checkpointer wiring for durable LangGraph execution."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings, get_settings

SUPPORTED_CHECKPOINTERS = ("memory",)


@lru_cache(maxsize=8)
def _get_cached_memory_saver(namespace: str) -> InMemorySaver:
    """Return a process-wide shared in-memory saver for one namespace."""

    del namespace
    return InMemorySaver()


def build_checkpointer(settings: Settings | None = None) -> BaseCheckpointSaver:
    """Return the configured checkpointer backend.

    The default backend is in-memory, but the wiring is isolated here so a
    durable backend can be dropped in without changing the graph builder.
    """

    current_settings = settings or get_settings()
    backend = current_settings.graph.checkpointer_backend.strip().lower()
    if backend == "memory":
        return _get_cached_memory_saver(current_settings.graph.checkpoint_namespace)
    msg = (
        f"Unsupported checkpointer backend: {current_settings.graph.checkpointer_backend}. "
        f"Supported backends: {', '.join(SUPPORTED_CHECKPOINTERS)}"
    )
    raise ValueError(msg)


def build_checkpoint_config(
    *,
    thread_id: str,
    settings: Settings | None = None,
    checkpoint_id: str | None = None,
) -> RunnableConfig:
    """Build the config payload required by LangGraph checkpointers."""

    current_settings = settings or get_settings()
    configurable = {
        "thread_id": thread_id,
        "checkpoint_ns": current_settings.graph.checkpoint_namespace,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return RunnableConfig(configurable=configurable)


def get_latest_checkpoint(
    *,
    thread_id: str,
    settings: Settings | None = None,
    saver: BaseCheckpointSaver | None = None,
) -> CheckpointTuple | None:
    """Return the most recent checkpoint tuple for a thread id."""

    effective_saver = saver or build_checkpointer(settings)
    for checkpoint in effective_saver.list(None):
        checkpoint_thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if checkpoint_thread_id == thread_id:
            return checkpoint
    return None


def build_resume_config(
    *,
    thread_id: str,
    settings: Settings | None = None,
    saver: BaseCheckpointSaver | None = None,
) -> RunnableConfig:
    """Build a resume config using the latest checkpoint when available."""

    latest = get_latest_checkpoint(thread_id=thread_id, settings=settings, saver=saver)
    if latest is None:
        return build_checkpoint_config(thread_id=thread_id, settings=settings)
    configurable = latest.config.get("configurable", {})
    checkpoint_id = configurable.get("checkpoint_id")
    checkpoint_namespace = configurable.get("checkpoint_ns")
    return RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_namespace
            or (settings or get_settings()).graph.checkpoint_namespace,
            "checkpoint_id": checkpoint_id,
        }
    )
