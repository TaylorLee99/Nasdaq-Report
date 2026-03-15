"""Gemini REST adapter for filings-only specialized-agent reasoning."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.domain import AnalysisRequest, ChunkRecord, ConfidenceLabel, RoutedTask
from app.graph.subgraphs.specialized_agents import (
    AgentProfile,
    AgentReasoningDraft,
    SpecializedAgentModel,
    _build_retrieval_query,
)

logger = logging.getLogger(__name__)


class RetryableLlmError(RuntimeError):
    """Raised when an LLM request may succeed on retry."""


class GeminiReasoningPayload(BaseModel):
    """Structured reasoning contract requested from Gemini."""

    provisional_summary: str = Field(min_length=1)
    unresolved_items: list[str] = Field(default_factory=list)
    confidence_label: ConfidenceLabel
    reretrieval_requested: bool = False
    requested_query: str | None = None

    @field_validator("unresolved_items", mode="before")
    @classmethod
    def normalize_unresolved_items(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return []
            return [normalized]
        return [str(value).strip()]


class JsonPostHttpClient(Protocol):
    """Minimal JSON POST client abstraction for LLM calls."""

    def post_json(
        self,
        url: str,
        *,
        payload: Mapping[str, object],
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, object]:
        """POST one JSON payload and return the decoded response body."""


class UrllibJsonPostClient:
    """Standard-library JSON POST client."""

    def post_json(
        self,
        url: str,
        *,
        payload: Mapping[str, object],
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, object]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **dict(headers or {})},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504}:
                raise RetryableLlmError(str(exc)) from exc
            raise
        except URLError as exc:
            raise RetryableLlmError(str(exc)) from exc
        return json.loads(body)


class GeminiGenerateContentClient:
    """Small Gemini REST client using the official generateContent endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: float = 30.0,
        http_client: JsonPostHttpClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client or UrllibJsonPostClient()

    def generate_json(
        self,
        *,
        system_instruction: str,
        prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """Return the primary text payload for one generateContent call."""

        url = f"{self._base_url}/models/{self._model}:generateContent"
        payload: dict[str, object] = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        response = self._http_client.post_json(
            url,
            payload=payload,
            headers={"x-goog-api-key": self._api_key},
            timeout_seconds=self._timeout_seconds,
        )
        return _extract_response_text(response)


class GeminiSpecializedAgentModel(SpecializedAgentModel):
    """Gemini-backed reasoner for filings-only specialized agents."""

    def __init__(
        self,
        client: GeminiGenerateContentClient,
        *,
        temperature: float = 0.2,
    ) -> None:
        self._client = client
        self._temperature = temperature

    def reason(
        self,
        *,
        profile: AgentProfile,
        request: AnalysisRequest,
        tasks: list[RoutedTask],
        chunks: list[ChunkRecord],
        iteration_count: int,
        max_iterations: int,
    ) -> AgentReasoningDraft:
        prompt = _build_reasoning_prompt(
            profile=profile,
            request=request,
            tasks=tasks,
            chunks=chunks,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )
        try:
            raw_text = self._client.generate_json(
                system_instruction=_SYSTEM_INSTRUCTION,
                prompt=prompt,
                temperature=self._temperature,
            )
            payload = GeminiReasoningPayload.model_validate_json(_strip_code_fences(raw_text))
        except (
            RetryableLlmError,
            HTTPError,
            URLError,
            ValidationError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("Gemini reasoning failed for %s: %s", profile.agent_name, exc)
            return AgentReasoningDraft(
                provisional_summary=(
                    f"{profile.agent_name} could not complete model reasoning; "
                    "falling back to low-confidence stub output."
                ),
                unresolved_items=["llm_reasoning_failed"],
                confidence_label=ConfidenceLabel.LOW,
                reretrieval_requested=False,
                requested_query=None,
            )

        requested_query = payload.requested_query
        if not payload.reretrieval_requested:
            requested_query = None
        draft = AgentReasoningDraft(
            provisional_summary=payload.provisional_summary,
            unresolved_items=payload.unresolved_items,
            confidence_label=payload.confidence_label,
            reretrieval_requested=payload.reretrieval_requested,
            requested_query=requested_query,
        )
        return _apply_post_llm_guard(
            draft=draft,
            profile=profile,
            request=request,
            tasks=tasks,
            chunks=chunks,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )


_SYSTEM_INSTRUCTION = (
    "You are a filings-only equity research reasoning module. "
    "Use only the provided 10-K, 10-Q, or 8-K snippets. "
    "Do not infer from earnings calls, news, blogs, or price targets. "
    "Prefer the most recent filing evidence when multiple snippets overlap. "
    "Return exactly one JSON object with keys: provisional_summary, unresolved_items, "
    "confidence_label, reretrieval_requested, requested_query."
)


def _build_reasoning_prompt(
    *,
    profile: AgentProfile,
    request: AnalysisRequest,
    tasks: list[RoutedTask],
    chunks: list[ChunkRecord],
    iteration_count: int,
    max_iterations: int,
) -> str:
    task_names = [task.task_type.value for task in tasks]
    chunk_lines = "\n\n".join(
        _format_chunk(chunk=chunk, ordinal=index + 1) for index, chunk in enumerate(chunks[:6])
    ) or "No chunks retrieved."
    return (
        f"Company: {request.company.ticker} / {request.company.company_name}\n"
        f"Question: {request.question}\n"
        f"Agent: {profile.agent_name}\n"
        f"Tasks: {', '.join(task_names) or 'none'}\n"
        f"Focus sections: {', '.join(profile.section_names) or 'none'}\n"
        f"Iteration: {iteration_count} of {max_iterations}\n"
        "Requirements:\n"
        "- Summarize only what the provided filing snippets support.\n"
        "- Write 1 to 3 concise sentences, not a bullet list.\n"
        "- Prefer recent snippets and the named focus sections over older or generic snippets.\n"
        "- Do not copy long raw excerpts verbatim.\n"
        "- Keep the summary valuation-free.\n"
        "- If evidence is weak or incomplete, list unresolved items.\n"
        "- Request reretrieval only when the current snippets are clearly insufficient "
        "and another query might help.\n"
        "- confidence_label must be one of: high, medium, low.\n"
        "- requested_query must be null unless reretrieval_requested is true.\n\n"
        f"{_agent_specific_requirements(profile)}\n"
        f"Retrieved snippets:\n{chunk_lines}\n"
    )


def _apply_post_llm_guard(
    *,
    draft: AgentReasoningDraft,
    profile: AgentProfile,
    request: AnalysisRequest,
    tasks: list[RoutedTask],
    chunks: list[ChunkRecord],
    iteration_count: int,
    max_iterations: int,
) -> AgentReasoningDraft:
    if not chunks:
        return draft
    low_quality_summary = _summary_is_too_narrow(
        summary=draft.provisional_summary,
        profile=profile,
        tasks=tasks,
        chunks=chunks,
    ) or _summary_has_poor_quality(
        summary=draft.provisional_summary,
        profile=profile,
    )
    if not low_quality_summary:
        return draft
    unresolved_items = list(
        dict.fromkeys(
            [
                *draft.unresolved_items,
                "llm_summary_too_narrow",
                "llm_summary_low_quality",
            ]
        )
    )
    if iteration_count < max_iterations:
        return AgentReasoningDraft(
            provisional_summary=draft.provisional_summary,
            unresolved_items=unresolved_items,
            confidence_label=ConfidenceLabel.LOW,
            reretrieval_requested=True,
            requested_query=_build_retrieval_query(
                profile=profile,
                request=request,
                tasks=tasks,
            ),
        )
    return AgentReasoningDraft(
        provisional_summary=_fallback_chunk_summary(
            chunks,
            profile=profile,
            request=request,
        ),
        unresolved_items=unresolved_items,
        confidence_label=ConfidenceLabel.LOW,
        reretrieval_requested=False,
        requested_query=None,
    )


def _format_chunk(*, chunk: ChunkRecord, ordinal: int) -> str:
    metadata = [
        f"filing_type={chunk.filing_type.value}",
        f"section_name={chunk.section_name}",
        f"parse_confidence={chunk.parse_confidence.value}",
    ]
    if chunk.filing_date is not None:
        metadata.append(f"filing_date={chunk.filing_date.isoformat()}")
    if chunk.item_number is not None:
        metadata.append(f"item_number={chunk.item_number}")
    if chunk.transcript_segment_type is not None:
        metadata.append(f"transcript_segment_type={chunk.transcript_segment_type.value}")
    excerpt = chunk.text.strip().replace("\n", " ")[:1600]
    return f"[{ordinal}] {'; '.join(metadata)}\n{excerpt}"


def _agent_specific_requirements(profile: AgentProfile) -> str:
    if profile.agent_name == "run_10k_agent":
        return (
            "Agent-specific requirements:\n"
            "- Prefer Item 1 Business and Item 7 MD&A for long-term structure.\n"
            "- Use Risk Factors only when the risk is structural and clearly tied to the "
            "business.\n"
            "- Avoid isolated regional anecdotes, export-license examples, or one-off disclosures "
            "unless they change the long-term business structure.\n"
            "- Favor platform, segment, customer, supply-chain, and disclosed structural risk "
            "descriptions over generic cautionary language.\n"
        )
    if profile.agent_name == "run_10q_agent":
        return (
            "Agent-specific requirements:\n"
            "- Focus on the latest quarter's change in operations, liquidity, or controls.\n"
            "- Prefer MD&A and Liquidity sections over raw financial statement tables.\n"
            "- Do not summarize table headers, numeric blocks, Item 3/4 headings, or note cross-"
            "references by themselves.\n"
            "- If the snippets are mostly tabular or procedural, mark unresolved_items instead of "
            "forcing a narrow summary.\n"
        )
    return "Agent-specific requirements:\n- Stay within the stated filing focus.\n"


def _summary_is_too_narrow(
    *,
    summary: str,
    profile: AgentProfile,
    tasks: list[RoutedTask],
    chunks: list[ChunkRecord],
) -> bool:
    summary_tokens = _token_set(summary)
    if len(summary_tokens) < 3:
        return True
    combined_chunk_tokens = _token_set(" ".join(chunk.text for chunk in chunks[:4]))
    lexical_support = _support_ratio(summary_tokens, combined_chunk_tokens)
    if lexical_support < 0.2:
        return True
    if profile.agent_name == "run_10q_agent":
        expected_tokens = _expected_task_tokens(profile=profile, tasks=tasks)
        if expected_tokens and not summary_tokens.intersection(expected_tokens):
            return True
    return False


def _summary_has_poor_quality(
    *,
    summary: str,
    profile: AgentProfile,
) -> bool:
    normalized = " ".join(summary.split()).strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    if re.match(r"^(by|for|with|through)\b", lowered):
        return True
    if lowered.startswith("disclosed that by the"):
        return True
    if profile.agent_name == "run_10q_agent" and re.match(
        r"^(evaluate|maintain|ensure|finance|provide)\b",
        lowered,
    ):
        return True
    if profile.agent_name == "run_8k_agent" and (
        "compensation comm" in lowered
        or re.match(r"^item\s+\d+\.\d{2}.*disclosed that by the", lowered)
    ):
        return True
    return False


def _expected_task_tokens(
    *,
    profile: AgentProfile,
    tasks: list[RoutedTask],
) -> set[str]:
    task_names = {task.task_type.value for task in tasks}
    expected: set[str] = set()
    if profile.agent_name == "run_10q_agent" or "recent_quarter_change" in task_names:
        expected.update(
            {
                "quarter",
                "recent",
                "revenue",
                "margin",
                "liquidity",
                "cash",
                "operations",
                "results",
                "controls",
                "demand",
            }
        )
    if profile.agent_name == "run_10k_agent":
        expected.update({"business", "risk", "operations", "demand", "market", "platform"})
    if profile.agent_name == "run_8k_agent":
        expected.update({"event", "material", "agreement", "item", "announced"})
    return expected


def _support_ratio(summary_tokens: set[str], source_tokens: set[str]) -> float:
    if not summary_tokens:
        return 0.0
    return len(summary_tokens.intersection(source_tokens)) / len(summary_tokens)


def _token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 4}


def _fallback_chunk_summary(
    chunks: list[ChunkRecord],
    *,
    profile: AgentProfile,
    request: AnalysisRequest,
) -> str:
    if not chunks:
        return "Retrieved filing evidence was present but could not be summarized safely."
    primary_chunk = chunks[0]
    company = request.company.ticker
    section_name = primary_chunk.section_name.lower()
    candidate_sentence = _best_chunk_sentence(chunks=chunks, profile=profile)

    if profile.agent_name == "run_8k_agent":
        item_number = primary_chunk.item_number or "8-K"
        if candidate_sentence is not None and item_number == "5.02":
            return _compact_item_502_fallback_sentence(
                sentence=candidate_sentence,
                company=request.company.company_name or company,
            )
        if primary_chunk.filing_date is not None:
            return (
                f"On {primary_chunk.filing_date:%B %-d, %Y}, {company} disclosed a material "
                f"event in an Item {item_number} 8-K filing."
            )
        return f"{company} disclosed a material event in an Item {item_number} 8-K filing."
    if profile.agent_name == "run_10q_agent":
        if candidate_sentence is not None:
            return _wrap_filing_sentence(
                filing_label="10-Q",
                sentence=candidate_sentence,
            )
        if "liquidity" in section_name:
            return (
                f"The latest 10-Q supports a verified liquidity and capital resources update "
                f"for {company}."
            )
        if "controls" in section_name:
            return f"The latest 10-Q supports a verified controls update for {company}."
        return f"The latest 10-Q supports a verified recent-quarter operating update for {company}."
    if profile.agent_name == "run_10k_agent":
        if candidate_sentence is not None:
            return _wrap_filing_sentence(
                filing_label="10-K",
                sentence=candidate_sentence,
            )
        if "business" in section_name:
            return f"The latest 10-K supports a verified long-term business overview for {company}."
        if "risk" in section_name:
            return f"The latest 10-K supports a verified structural risk disclosure for {company}."
        return (
            f"The latest 10-K supports a verified long-term operating and financial context "
            f"for {company}."
        )
    return "Retrieved filing evidence was present but could not be summarized safely."


def _compact_item_502_fallback_sentence(*, sentence: str, company: str) -> str:
    normalized = " ".join(sentence.split()).strip()
    if not normalized:
        return normalized
    company_name = company.split()[0] if company else "Company"
    match = re.search(
        r"(?i)(On [A-Za-z]+ \d{1,2}, \d{4}), .*?adopted (?:the )?"
        r"(Variable Compensation Plan(?: for Fiscal Year \d{4})?).*",
        normalized,
    )
    if match is not None:
        return f"{match.group(1)}, {company_name} adopted the {match.group(2)}."
    normalized = re.sub(
        r"(?i)^On ([A-Za-z]+ \d{1,2}, \d{4}), .*? adopted ",
        rf"On \1, {company_name} adopted ",
        normalized,
        count=1,
    )
    return normalized


def _best_chunk_sentence(
    *,
    chunks: list[ChunkRecord],
    profile: AgentProfile,
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for chunk in chunks[:3]:
        normalized = " ".join(chunk.text.split()).strip()
        if not normalized:
            continue
        for raw_sentence in re.split(r"(?<=[.!?])\s+", normalized):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            candidates.append((_chunk_sentence_quality(sentence, profile), sentence))
    if not candidates:
        return None
    score, sentence = max(candidates, key=lambda item: item[0])
    if score < 20:
        return None
    if not sentence.endswith("."):
        sentence = f"{sentence}."
    return sentence


def _chunk_sentence_quality(sentence: str, profile: AgentProfile) -> int:
    normalized = " ".join(sentence.split()).strip()
    if not normalized:
        return -100
    lowered = normalized.lower()
    score = 0
    if normalized[0].isupper():
        score += 20
    if 40 <= len(normalized) <= 220:
        score += 10
    if re.match(r"^(On|As of|We|Our|NVIDIA|The company|The latest)\b", normalized):
        score += 10
    if re.match(r"^[a-z]", normalized):
        score -= 50
    if any(
        marker in lowered
        for marker in (
            "compensation comm",
            "the following risk factors should be considered",
            "any investment will be completed on expected terms",
            "the following table sets forth",
            "refer to item 1a",
            "pursuant to the requirements",
        )
    ):
        score -= 60
    if profile.agent_name == "run_10q_agent" and any(
        token in lowered
        for token in (
            "liquidity",
            "capital resources",
            "cash",
            "revenue",
            "margin",
        )
    ):
        score += 10
    if profile.agent_name == "run_10k_agent" and any(
        token in lowered
        for token in (
            "business",
            "platform",
            "market",
            "customer",
            "risk",
            "supply",
        )
    ):
        score += 10
    if re.match(r"^(Evaluate|Maintain|Ensure|Provide)\b", normalized):
        score -= 50
    return score


def _wrap_filing_sentence(*, filing_label: str, sentence: str) -> str:
    normalized = " ".join(sentence.split()).strip().rstrip(".")
    if re.match(r"^(On|As of)\b", normalized):
        return f"The latest {filing_label} indicates that {normalized[0].lower()}{normalized[1:]}."
    return f"{normalized}."


def _extract_response_text(response: Mapping[str, object]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response did not include candidates")
    first_candidate = candidates[0]
    if not isinstance(first_candidate, Mapping):
        raise ValueError("Gemini candidate payload is invalid")
    content = first_candidate.get("content")
    if not isinstance(content, Mapping):
        raise ValueError("Gemini candidate content is missing")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Gemini candidate parts are missing")
    texts = [part.get("text", "") for part in parts if isinstance(part, Mapping)]
    merged = "\n".join(text for text in texts if isinstance(text, str) and text.strip()).strip()
    if not merged:
        raise ValueError("Gemini response text is empty")
    return merged


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def live_llm_calls_allowed() -> bool:
    """Keep the test suite offline even if a developer .env enables Gemini."""

    return os.getenv("PYTEST_CURRENT_TEST") is None
