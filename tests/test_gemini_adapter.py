from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import pytest

from app.domain import (
    AnalysisRequest,
    AnalysisTaskType,
    ChunkRecord,
    Company,
    ConfidenceLabel,
    FilingType,
    RoutedTask,
    TaskRoutingStatus,
)
from app.graph.subgraphs import EIGHT_K_PROFILE, TEN_K_PROFILE, TEN_Q_PROFILE
from app.llm.factory import build_specialized_agent_model
from app.llm.gemini import GeminiGenerateContentClient, GeminiSpecializedAgentModel


class FakeJsonPostClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def post_json(
        self,
        url: str,
        *,
        payload: Mapping[str, object],
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, object]:
        self.requests.append(
            {
                "url": url,
                "payload": dict(payload),
                "headers": dict(headers or {}),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


def build_request() -> tuple[AnalysisRequest, list[RoutedTask], list[ChunkRecord]]:
    request = AnalysisRequest(
        request_id="req-gemini",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    tasks = [
        RoutedTask(
            task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
            agent_name=TEN_K_PROFILE.agent_name,
            filing_type=FilingType.FORM_10K,
            status=TaskRoutingStatus.ROUTED,
            reason="test",
            document_available=True,
            parse_confidence=ConfidenceLabel.HIGH,
        )
    ]
    chunks = [
        ChunkRecord(
            chunk_id="chunk-1",
            document_id="doc-1",
            section_id="sec-1",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 1. Business",
            text="NVIDIA expanded data center demand and highlighted platform breadth.",
            token_count=11,
            parse_confidence=ConfidenceLabel.HIGH,
        )
    ]
    return request, tasks, chunks


def build_ten_q_request() -> tuple[AnalysisRequest, list[RoutedTask], list[ChunkRecord]]:
    request = AnalysisRequest(
        request_id="req-gemini-10q",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What changed in the latest quarter?",
        max_reretrievals=2,
    )
    tasks = [
        RoutedTask(
            task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
            agent_name=TEN_Q_PROFILE.agent_name,
            filing_type=FilingType.FORM_10Q,
            status=TaskRoutingStatus.ROUTED,
            reason="test",
            document_available=True,
            parse_confidence=ConfidenceLabel.HIGH,
        )
    ]
    chunks = [
        ChunkRecord(
            chunk_id="chunk-10q-1",
            document_id="doc-10q-1",
            section_id="sec-10q-1",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name=(
                "Management's Discussion and Analysis of Financial Condition and Results "
                "of Operations"
            ),
            text=(
                "Revenue increased in the latest quarter while gross margin tightened and "
                "liquidity remained strong with higher cash from operations."
            ),
            token_count=19,
            parse_confidence=ConfidenceLabel.HIGH,
        )
    ]
    return request, tasks, chunks


def test_gemini_specialized_agent_model_parses_structured_json_response() -> None:
    request, tasks, chunks = build_request()
    fake_http = FakeJsonPostClient(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "provisional_summary": (
                                            "Data center strength remains central to the "
                                            "business narrative."
                                        ),
                                        "unresolved_items": [],
                                        "confidence_label": "high",
                                        "reretrieval_requested": False,
                                        "requested_query": None,
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
    )
    client = GeminiGenerateContentClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        http_client=fake_http,
    )
    model = GeminiSpecializedAgentModel(client)

    draft = model.reason(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=tasks,
        chunks=chunks,
        iteration_count=1,
        max_iterations=2,
    )

    assert draft.provisional_summary.startswith("Data center strength")
    assert draft.confidence_label == ConfidenceLabel.HIGH
    assert draft.reretrieval_requested is False
    headers = fake_http.requests[0]["headers"]
    assert isinstance(headers, dict)
    assert headers["x-goog-api-key"] == "test-key"
    assert "generateContent" in str(fake_http.requests[0]["url"])


def test_gemini_specialized_agent_model_falls_back_on_invalid_json() -> None:
    request, tasks, chunks = build_request()
    fake_http = FakeJsonPostClient(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "not-json"}
                        ]
                    }
                }
            ]
        }
    )
    client = GeminiGenerateContentClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        http_client=fake_http,
    )
    model = GeminiSpecializedAgentModel(client)

    draft = model.reason(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=tasks,
        chunks=chunks,
        iteration_count=1,
        max_iterations=2,
    )

    assert draft.confidence_label == ConfidenceLabel.LOW
    assert draft.unresolved_items == ["llm_reasoning_failed"]
    assert draft.reretrieval_requested is False


def test_gemini_specialized_agent_model_coerces_string_unresolved_items() -> None:
    request, tasks, chunks = build_request()
    fake_http = FakeJsonPostClient(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "provisional_summary": "Demand remains healthy.",
                                        "unresolved_items": "Need fresher liquidity detail.",
                                        "confidence_label": "medium",
                                        "reretrieval_requested": False,
                                        "requested_query": None,
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
    )
    client = GeminiGenerateContentClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        http_client=fake_http,
    )
    model = GeminiSpecializedAgentModel(client)

    draft = model.reason(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=tasks,
        chunks=chunks,
        iteration_count=1,
        max_iterations=2,
    )

    assert draft.unresolved_items == ["Need fresher liquidity detail."]
    assert draft.confidence_label == ConfidenceLabel.MEDIUM


def test_build_specialized_agent_model_stays_offline_under_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_LLM__ENABLED", "true")
    monkeypatch.setenv("APP_LLM__PROVIDER", "gemini")
    monkeypatch.setenv("APP_LLM__MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("APP_LLM__API_KEY", "test-key")

    from app.config import get_settings

    get_settings.cache_clear()
    try:
        assert build_specialized_agent_model(get_settings()) is None
    finally:
        get_settings.cache_clear()


def test_gemini_specialized_agent_model_guards_narrow_10q_output() -> None:
    request, tasks, chunks = build_ten_q_request()
    fake_http = FakeJsonPostClient(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "provisional_summary": (
                                            "Executive compensation targets were updated."
                                        ),
                                        "unresolved_items": [],
                                        "confidence_label": "high",
                                        "reretrieval_requested": False,
                                        "requested_query": None,
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
    )
    client = GeminiGenerateContentClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        http_client=fake_http,
    )
    model = GeminiSpecializedAgentModel(client)

    draft = model.reason(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=tasks,
        chunks=chunks,
        iteration_count=1,
        max_iterations=2,
    )

    assert draft.confidence_label == ConfidenceLabel.LOW
    assert draft.reretrieval_requested is True
    assert "llm_summary_too_narrow" in draft.unresolved_items
    assert draft.requested_query is not None
    assert "latest quarter" in draft.requested_query
    assert "10-Q" in draft.requested_query


def test_gemini_specialized_agent_model_uses_fallback_summary_at_max_iterations() -> None:
    request, tasks, chunks = build_ten_q_request()
    fake_http = FakeJsonPostClient(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "provisional_summary": "Compensation targets changed.",
                                        "unresolved_items": [],
                                        "confidence_label": "medium",
                                        "reretrieval_requested": False,
                                        "requested_query": None,
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
    )
    client = GeminiGenerateContentClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        http_client=fake_http,
    )
    model = GeminiSpecializedAgentModel(client)

    draft = model.reason(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=tasks,
        chunks=chunks,
        iteration_count=2,
        max_iterations=2,
    )

    assert draft.confidence_label == ConfidenceLabel.LOW
    assert draft.reretrieval_requested is False
    assert "llm_summary_too_narrow" in draft.unresolved_items
    assert "Revenue increased in the latest quarter" in draft.provisional_summary


def test_gemini_specialized_agent_model_rejects_low_quality_8k_fragment_summary() -> None:
    request, _, _ = build_request()
    tasks = [
        RoutedTask(
            task_type=AnalysisTaskType.MATERIAL_EVENTS,
            agent_name="run_8k_agent",
            filing_type=FilingType.FORM_8K,
            status=TaskRoutingStatus.ROUTED,
            reason="test",
            document_available=True,
            parse_confidence=ConfidenceLabel.HIGH,
        )
    ]
    chunks = [
        ChunkRecord(
            chunk_id="chunk-8k-1",
            document_id="doc-8k-1",
            section_id="sec-8k-1",
            company=request.company,
            filing_type=FilingType.FORM_8K,
            accession_number="0001045810-26-000024",
            filing_date=None,
            section_name="Item 5.02 Departure of Directors or Certain Officers",
            item_number="5.02",
            text="On March 2, 2026, NVIDIA adopted a new variable compensation plan.",
            token_count=12,
            parse_confidence=ConfidenceLabel.HIGH,
        )
    ]
    fake_http = FakeJsonPostClient(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "provisional_summary": "By the Compensation Comm.",
                                        "unresolved_items": [],
                                        "confidence_label": "high",
                                        "reretrieval_requested": False,
                                        "requested_query": None,
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
    )
    client = GeminiGenerateContentClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        http_client=fake_http,
    )
    model = GeminiSpecializedAgentModel(client)

    draft = model.reason(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=tasks,
        chunks=chunks,
        iteration_count=2,
        max_iterations=2,
    )

    assert draft.provisional_summary == (
        "On March 2, 2026, NVIDIA adopted a new variable compensation plan."
    )
    assert "llm_summary_low_quality" in draft.unresolved_items


def test_gemini_prompt_includes_ten_q_agent_specific_requirements() -> None:
    request, tasks, chunks = build_ten_q_request()
    fake_http = FakeJsonPostClient(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "provisional_summary": "Liquidity remained strong.",
                                        "unresolved_items": [],
                                        "confidence_label": "high",
                                        "reretrieval_requested": False,
                                        "requested_query": None,
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
    )
    client = GeminiGenerateContentClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        http_client=fake_http,
    )
    model = GeminiSpecializedAgentModel(client)

    model.reason(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=tasks,
        chunks=chunks,
        iteration_count=1,
        max_iterations=2,
    )

    payload = cast(dict[str, Any], fake_http.requests[0]["payload"])
    contents = cast(list[dict[str, Any]], payload["contents"])
    parts = cast(list[dict[str, Any]], contents[0]["parts"])
    prompt = cast(str, parts[0]["text"])
    assert "Focus on the latest quarter's change in operations, liquidity, or controls." in prompt
    assert "Do not summarize table headers, numeric blocks, Item 3/4 headings" in prompt


def test_gemini_prompt_includes_ten_k_agent_specific_requirements() -> None:
    request, tasks, chunks = build_request()
    fake_http = FakeJsonPostClient(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "provisional_summary": "Platform breadth remains central.",
                                        "unresolved_items": [],
                                        "confidence_label": "high",
                                        "reretrieval_requested": False,
                                        "requested_query": None,
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
    )
    client = GeminiGenerateContentClient(
        api_key="test-key",
        model="gemini-2.5-flash",
        http_client=fake_http,
    )
    model = GeminiSpecializedAgentModel(client)

    model.reason(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=tasks,
        chunks=chunks,
        iteration_count=1,
        max_iterations=2,
    )

    payload = cast(dict[str, Any], fake_http.requests[0]["payload"])
    contents = cast(list[dict[str, Any]], payload["contents"])
    parts = cast(list[dict[str, Any]], contents[0]["parts"])
    prompt = cast(str, parts[0]["text"])
    assert "Prefer Item 1 Business and Item 7 MD&A for long-term structure." in prompt
    assert "Avoid isolated regional anecdotes" in prompt
