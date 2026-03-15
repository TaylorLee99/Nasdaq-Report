"""Factory helpers for live or stub specialized-agent reasoning models."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.llm.gemini import (
    GeminiGenerateContentClient,
    GeminiSpecializedAgentModel,
    live_llm_calls_allowed,
)


def build_specialized_agent_model(
    settings: Settings | None = None,
) -> GeminiSpecializedAgentModel | None:
    """Return a live specialized-agent model when enabled, otherwise None for stub fallback."""

    current_settings = settings or get_settings()
    if not current_settings.llm.enabled or not live_llm_calls_allowed():
        return None
    if current_settings.llm.provider.lower() != "gemini":
        return None
    if not current_settings.llm.api_key:
        return None
    client = GeminiGenerateContentClient(
        api_key=current_settings.llm.api_key,
        model=current_settings.llm.model,
        base_url=current_settings.llm.base_url,
        timeout_seconds=current_settings.llm.timeout_seconds,
    )
    return GeminiSpecializedAgentModel(
        client=client,
        temperature=current_settings.llm.temperature,
    )
