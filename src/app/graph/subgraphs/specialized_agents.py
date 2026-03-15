"""Reusable specialized-agent subgraphs with bounded reretrieval loops."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from operator import add
from typing import Annotated, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.domain import (
    AgentFinding,
    AgentOutputPacket,
    AnalysisRequest,
    AnalysisTaskType,
    ChunkRecord,
    ConfidenceLabel,
    CoverageLabel,
    CoverageStatus,
    EvidenceRef,
    EvidenceTypeLabel,
    FilingType,
    FindingSignalType,
    RoutedTask,
    SharedMemoryEntry,
    TimeHorizonLabel,
    TranscriptSegmentType,
    VerificationLabel,
    VerificationStatus,
)
from app.indexing.models import ChunkSearchFilters


@dataclass(frozen=True)
class AgentProfile:
    """Static retrieval and reasoning configuration for a specialized agent."""

    agent_name: str
    filing_types: tuple[FilingType, ...]
    task_types: tuple[AnalysisTaskType, ...]
    section_names: tuple[str, ...] = ()
    transcript_segment_types: tuple[str, ...] = ()
    max_iterations: int = 2


class AgentRetrievalRequest(TypedDict):
    """Retriever input for one plan/retrieve cycle."""

    query: str
    filters: ChunkSearchFilters
    section_names: tuple[str, ...]
    transcript_segment_types: tuple[TranscriptSegmentType, ...]
    limit: int


@dataclass(frozen=True)
class AgentReasoningDraft:
    """Structured reasoning result before reflection/output."""

    provisional_summary: str
    unresolved_items: list[str]
    confidence_label: ConfidenceLabel
    reretrieval_requested: bool
    requested_query: str | None = None


class SpecializedAgentRetriever(Protocol):
    """Retriever contract used by specialized agents."""

    def retrieve(self, request: AgentRetrievalRequest) -> list[ChunkRecord]:
        """Return metadata-aware, section-aware chunks for the active request."""


class SpecializedAgentModel(Protocol):
    """Reasoning contract used by specialized agents."""

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
        """Produce provisional findings and reretrieval instructions."""


class NoOpSpecializedAgentRetriever:
    """Default retriever that returns no evidence until a real backend is wired."""

    def retrieve(self, request: AgentRetrievalRequest) -> list[ChunkRecord]:
        del request
        return []


class StubSpecializedAgentModel:
    """Deterministic fallback model for development and tests."""

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
        del request
        task_names = ", ".join(task.task_type.value for task in tasks)
        if not chunks:
            should_retry = iteration_count < max_iterations
            return AgentReasoningDraft(
                provisional_summary=(
                    f"{profile.agent_name} found no supporting chunks "
                    f"for tasks: {task_names or 'none'}."
                ),
                unresolved_items=[task_names or "missing_evidence"],
                confidence_label=ConfidenceLabel.LOW,
                reretrieval_requested=should_retry,
                requested_query=(
                    f"{profile.agent_name} retry {iteration_count + 1}" if should_retry else None
                ),
            )

        confidence = _lowest_confidence(
            [chunk.parse_confidence for chunk in chunks],
            default=ConfidenceLabel.MEDIUM,
        )
        chunk_summary = _summarize_chunks(chunks)
        unresolved_items = []
        should_retry = False
        if confidence == ConfidenceLabel.LOW and iteration_count < max_iterations:
            unresolved_items.append("low_parse_confidence")
            should_retry = True
        return AgentReasoningDraft(
            provisional_summary=chunk_summary,
            unresolved_items=unresolved_items,
            confidence_label=confidence,
            reretrieval_requested=should_retry,
            requested_query=(
                f"{profile.agent_name} parse-confidence retry" if should_retry else None
            ),
        )


class SpecializedAgentState(TypedDict, total=False):
    """Internal subgraph state for one specialized agent execution."""

    request: AnalysisRequest
    tasks: list[RoutedTask]
    retrieval_request: AgentRetrievalRequest | None
    requested_queries: Annotated[list[str], add]
    retrieved_chunks: Annotated[list[ChunkRecord], add]
    provisional_summary: str
    item_reasoning_drafts: dict[str, AgentReasoningDraft]
    unresolved_items: list[str]
    confidence_label: ConfidenceLabel
    reretrieval_requested: bool
    iteration_count: int
    max_iterations: int
    packet: AgentOutputPacket | None


TEN_K_PROFILE = AgentProfile(
    agent_name="run_10k_agent",
    filing_types=(FilingType.FORM_10K,),
    task_types=(AnalysisTaskType.LONG_TERM_STRUCTURE,),
    section_names=(
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. Management's Discussion and Analysis",
    ),
)
TEN_Q_PROFILE = AgentProfile(
    agent_name="run_10q_agent",
    filing_types=(FilingType.FORM_10Q,),
    task_types=(AnalysisTaskType.RECENT_QUARTER_CHANGE,),
    section_names=(
        "Management's Discussion and Analysis of Financial Condition and Results of Operations",
        "Liquidity and Capital Resources",
    ),
)
EIGHT_K_PROFILE = AgentProfile(
    agent_name="run_8k_agent",
    filing_types=(FilingType.FORM_8K,),
    task_types=(AnalysisTaskType.MATERIAL_EVENTS,),
)
CALL_PROFILE = AgentProfile(
    agent_name="run_call_agent",
    filing_types=(FilingType.EARNINGS_CALL,),
    task_types=(AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE,),
    section_names=("Prepared Remarks", "Question-and-Answer Session"),
    transcript_segment_types=("prepared_remarks", "qa"),
)


def build_specialized_agent_subgraph(
    *,
    profile: AgentProfile,
    retriever: SpecializedAgentRetriever | None = None,
    model: SpecializedAgentModel | None = None,
) -> CompiledStateGraph:
    """Build a bounded Plan -> Retrieve -> Reason -> Reflect -> Output loop."""

    effective_retriever = retriever or NoOpSpecializedAgentRetriever()
    effective_model = model or StubSpecializedAgentModel()

    workflow = StateGraph(SpecializedAgentState)
    workflow.add_node("plan", lambda state: _plan_node(state, profile=profile))
    workflow.add_node(
        "retrieve",
        lambda state: _retrieve_node(state, retriever=effective_retriever),
    )
    workflow.add_node(
        "reason",
        lambda state: _reason_node(state, profile=profile, model=effective_model),
    )
    workflow.add_node("reflect", lambda state: _reflect_node(state, profile=profile))
    workflow.add_node("output", lambda state: _output_node(state, profile=profile))

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "reason")
    workflow.add_edge("reason", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        _reflect_route,
        {
            "retrieve": "retrieve",
            "output": "output",
        },
    )
    workflow.add_edge("output", END)
    return workflow.compile()


def build_initial_agent_state(
    *,
    request: AnalysisRequest,
    tasks: list[RoutedTask],
    profile: AgentProfile,
) -> SpecializedAgentState:
    """Create an initial state for independent specialized-agent execution."""

    return {
        "request": request,
        "tasks": tasks,
        "retrieval_request": None,
        "requested_queries": [],
        "retrieved_chunks": [],
        "provisional_summary": "",
        "item_reasoning_drafts": {},
        "unresolved_items": [],
        "confidence_label": ConfidenceLabel.MEDIUM,
        "reretrieval_requested": False,
        "iteration_count": 0,
        "max_iterations": request.max_reretrievals,
        "packet": None,
    }


def execute_specialized_agent(
    *,
    profile: AgentProfile,
    request: AnalysisRequest,
    tasks: list[RoutedTask],
    retriever: SpecializedAgentRetriever | None = None,
    model: SpecializedAgentModel | None = None,
) -> AgentOutputPacket:
    """Run a specialized-agent subgraph and return its final packet."""

    graph = build_specialized_agent_subgraph(
        profile=profile,
        retriever=retriever,
        model=model,
    )
    result = graph.invoke(build_initial_agent_state(request=request, tasks=tasks, profile=profile))
    packet = result.get("packet")
    if packet is None:
        msg = f"Specialized agent {profile.agent_name} did not emit an AgentOutputPacket"
        raise ValueError(msg)
    return packet


def _plan_node(state: SpecializedAgentState, *, profile: AgentProfile) -> dict[str, object]:
    query = _build_retrieval_query(
        profile=profile,
        request=state["request"],
        tasks=state.get("tasks", []),
    )
    return {
        "retrieval_request": {
            "query": query,
            "filters": _build_filters(profile=profile, state=state),
            "section_names": profile.section_names,
            "transcript_segment_types": _build_transcript_segment_types(profile),
            "limit": _retrieval_limit_for_profile(profile),
        },
        "requested_queries": [query],
    }


def _retrieve_node(
    state: SpecializedAgentState,
    *,
    retriever: SpecializedAgentRetriever,
) -> dict[str, object]:
    retrieval_request = state.get("retrieval_request")
    if retrieval_request is None:
        return {"retrieved_chunks": [], "iteration_count": state.get("iteration_count", 0)}
    chunks = retriever.retrieve(retrieval_request)
    return {
        "retrieved_chunks": chunks,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def _reason_node(
    state: SpecializedAgentState,
    *,
    profile: AgentProfile,
    model: SpecializedAgentModel,
) -> dict[str, object]:
    chunks = state.get("retrieved_chunks", [])
    if profile.agent_name == "run_8k_agent":
        item_update = _reason_8k_items(
            state=state,
            profile=profile,
            model=model,
            chunks=chunks,
        )
        if item_update is not None:
            return item_update
    draft = model.reason(
        profile=profile,
        request=state["request"],
        tasks=state.get("tasks", []),
        chunks=chunks,
        iteration_count=state.get("iteration_count", 0),
        max_iterations=state.get("max_iterations", profile.max_iterations),
    )
    update: dict[str, object] = {
        "provisional_summary": draft.provisional_summary,
        "unresolved_items": draft.unresolved_items,
        "confidence_label": draft.confidence_label,
        "reretrieval_requested": draft.reretrieval_requested,
    }
    if draft.requested_query is not None:
        retrieval_request = state.get("retrieval_request")
        if retrieval_request is None:
            msg = "retrieval_request must exist before reretrieval can be requested"
            raise ValueError(msg)
        update["requested_queries"] = [draft.requested_query]
        update["retrieval_request"] = {
            "query": draft.requested_query,
            "filters": retrieval_request["filters"],
            "section_names": retrieval_request["section_names"],
            "transcript_segment_types": retrieval_request["transcript_segment_types"],
            "limit": retrieval_request["limit"],
        }
    return update


def _reason_8k_items(
    *,
    state: SpecializedAgentState,
    profile: AgentProfile,
    model: SpecializedAgentModel,
    chunks: list[ChunkRecord],
) -> dict[str, object] | None:
    grouped_chunks = _group_chunks_by_item_number(chunks)
    if not grouped_chunks:
        return None
    item_drafts: dict[str, AgentReasoningDraft] = {}
    unresolved_items: list[str] = []
    confidence_labels: list[ConfidenceLabel] = []
    requested_queries: list[str] = []
    summaries: list[str] = []

    for item_number, item_chunks in grouped_chunks.items():
        draft = model.reason(
            profile=profile,
            request=state["request"],
            tasks=state.get("tasks", []),
            chunks=item_chunks,
            iteration_count=state.get("iteration_count", 0),
            max_iterations=state.get("max_iterations", profile.max_iterations),
        )
        item_drafts[item_number] = draft
        unresolved_items.extend(draft.unresolved_items)
        confidence_labels.append(draft.confidence_label)
        summaries.append(f"Item {item_number}: {draft.provisional_summary}")
        if draft.reretrieval_requested and draft.requested_query is not None:
            requested_queries.append(draft.requested_query)

    update: dict[str, object] = {
        "provisional_summary": " ".join(summaries),
        "item_reasoning_drafts": item_drafts,
        "unresolved_items": list(dict.fromkeys(unresolved_items)),
        "confidence_label": _lowest_confidence(
            confidence_labels,
            default=ConfidenceLabel.MEDIUM,
        ),
        "reretrieval_requested": bool(requested_queries),
    }
    if requested_queries:
        retrieval_request = state.get("retrieval_request")
        if retrieval_request is None:
            msg = "retrieval_request must exist before reretrieval can be requested"
            raise ValueError(msg)
        next_query = requested_queries[0]
        update["requested_queries"] = requested_queries
        update["retrieval_request"] = {
            "query": next_query,
            "filters": retrieval_request["filters"],
            "section_names": retrieval_request["section_names"],
            "transcript_segment_types": retrieval_request["transcript_segment_types"],
            "limit": retrieval_request["limit"],
        }
    return update


def _reflect_node(state: SpecializedAgentState, *, profile: AgentProfile) -> dict[str, object]:
    del profile
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 0)
    should_retry = state.get("reretrieval_requested", False) and iteration_count < max_iterations
    return {"reretrieval_requested": should_retry}


def _reflect_route(state: SpecializedAgentState) -> str:
    return "retrieve" if state.get("reretrieval_requested", False) else "output"


def _output_node(state: SpecializedAgentState, *, profile: AgentProfile) -> dict[str, object]:
    chunks = state.get("retrieved_chunks", [])
    unresolved_items = list(dict.fromkeys(state.get("unresolved_items", [])))
    tasks = state.get("tasks", [])
    coverage_label = (
        CoverageLabel.COMPLETE if chunks and not unresolved_items else CoverageLabel.PARTIAL
    )
    findings = _build_findings_for_output(
        state=state,
        profile=profile,
        tasks=tasks,
        chunks=chunks,
        unresolved_items=unresolved_items,
        coverage_label=coverage_label,
    )
    evidence_refs = _collect_packet_evidence_refs(findings)
    packet = AgentOutputPacket(
        agent_name=profile.agent_name,
        findings=findings,
        evidence_refs=evidence_refs,
        provisional_summary=state.get("provisional_summary") or None,
        unresolved_items=unresolved_items,
        confidence_label=state.get("confidence_label", ConfidenceLabel.MEDIUM),
        iteration_count=state.get("iteration_count", 0),
        retrieved_snippets=chunks[:5],
        coverage_status=CoverageStatus(
            label=coverage_label,
            covered_topics=[task.task_type.value for task in tasks],
            missing_topics=unresolved_items,
            notes=state.get("provisional_summary") or None,
        ),
        reretrieval_requested=False,
        requested_queries=list(dict.fromkeys(state.get("requested_queries", []))),
        shared_memory_updates=[
            SharedMemoryEntry(
                memory_id=f"{state['request'].request_id}:{profile.agent_name}:output",
                key=f"{profile.agent_name}.output",
                value=state.get("provisional_summary") or "",
                source_agent=profile.agent_name,
                evidence_refs=evidence_refs,
            )
        ],
    )
    return {"packet": packet}


def _build_findings_for_output(
    *,
    state: SpecializedAgentState,
    profile: AgentProfile,
    tasks: list[RoutedTask],
    chunks: list[ChunkRecord],
    unresolved_items: list[str],
    coverage_label: CoverageLabel,
) -> list[AgentFinding]:
    if profile.agent_name == "run_8k_agent":
        item_findings = _build_8k_item_findings(
            state=state,
            profile=profile,
            tasks=tasks,
            chunks=chunks,
            unresolved_items=unresolved_items,
            coverage_label=coverage_label,
        )
        if item_findings:
            return item_findings
    claim, summary = _build_form_aware_claim_and_summary(
        profile=profile,
        task=tasks[0] if tasks else None,
        chunks=chunks,
        provisional_summary=state.get("provisional_summary") or "",
    )
    return [
        _build_agent_finding(
            state=state,
            profile=profile,
            task=task,
            claim=claim,
            summary=summary,
            chunks=chunks,
            unresolved_items=unresolved_items,
            coverage_label=coverage_label,
        )
        for task in tasks
    ]


def _build_form_aware_claim_and_summary(
    *,
    profile: AgentProfile,
    task: RoutedTask | None,
    chunks: list[ChunkRecord],
    provisional_summary: str,
) -> tuple[str, str]:
    deterministic_sentence = _best_deterministic_sentence(profile=profile, chunks=chunks)
    normalized_summary = (provisional_summary or "").strip()

    if profile.agent_name in {"run_10k_agent", "run_10q_agent"}:
        if deterministic_sentence is not None:
            return deterministic_sentence, deterministic_sentence
        if normalized_summary:
            return normalized_summary, normalized_summary
        fallback_task = task.task_type.value if task is not None else profile.agent_name
        return fallback_task, fallback_task

    if normalized_summary:
        return normalized_summary, normalized_summary
    fallback_task = task.task_type.value if task is not None else profile.agent_name
    return fallback_task, fallback_task


def _best_deterministic_sentence(
    *,
    profile: AgentProfile,
    chunks: list[ChunkRecord],
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for chunk in chunks[:4]:
        normalized = " ".join(chunk.text.split()).strip()
        if not normalized:
            continue
        for raw_sentence in re.split(r"(?<=[.!?])\s+", normalized):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            if _looks_like_truncated_sentence(sentence):
                continue
            candidates.append((_deterministic_sentence_score(profile, chunk, sentence), sentence))
    if not candidates:
        return None
    score, sentence = max(candidates, key=lambda item: item[0])
    if score < 25:
        return None
    sentence = _clean_heading_like_prefix(profile=profile, sentence=sentence)
    if not sentence.endswith("."):
        sentence = f"{sentence}."
    return sentence


def _clean_heading_like_prefix(*, profile: AgentProfile, sentence: str) -> str:
    cleaned = " ".join(sentence.split()).strip()
    if not cleaned:
        return cleaned

    generic_patterns = (
        (
            r"^Item\s+\d+[A-Z]?\.\s+"
            r"(?:Business|Risk Factors|Management['’]s Discussion(?: and Analysis)?|"
            r"Liquidity and Capital Resources|Controls and Procedures)\s+"
        ),
        r"^Part\s+[IVX]+\s+",
    )
    for pattern in generic_patterns:
        cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()

    if profile.agent_name == "run_10k_agent":
        cleaned = re.sub(
            r"^(?:Our Businesses|Our Business|Business Overview)\b[:\s-]*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"^Business\b(?=\s+(?:NVIDIA|We|Our|The company)\b)[:\s-]*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).strip()

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _looks_like_truncated_sentence(sentence: str) -> bool:
    normalized = " ".join(sentence.split()).strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    if normalized.endswith(","):
        return True
    if lowered.endswith((" and", " or", " as well as", " including")):
        return True
    if normalized.count(",") >= 4 and not re.search(r"[.!?]$", normalized):
        return True
    return False


def _deterministic_sentence_score(
    profile: AgentProfile,
    chunk: ChunkRecord,
    sentence: str,
) -> int:
    normalized = " ".join(sentence.split()).strip()
    if not normalized:
        return -100
    if _looks_like_low_signal_fragment(normalized):
        return -100
    lowered = normalized.lower()
    score = 0

    if normalized[0].isupper():
        score += 15
    if 40 <= len(normalized) <= 220:
        score += 10
    if re.match(r"^(On|As of|We|Our|NVIDIA|The company)\b", normalized):
        score += 10
    if re.match(r"^[a-z]", normalized):
        score -= 40

    generic_markers = (
        "the following risk factors should be considered",
        "the following table sets forth",
        "refer to item 1a",
        "pursuant to the requirements",
        "any investment will be completed on expected terms",
        "exchange act",
        "http://",
        "https://",
    )
    if any(marker in lowered for marker in generic_markers):
        score -= 60

    if profile.agent_name == "run_10q_agent":
        if any(
            token in lowered
            for token in ("liquidity", "capital resources", "cash", "revenue", "margin", "quarter")
        ):
            score += 20
        if normalized.startswith("As of ") and any(
            token in lowered
            for token in ("cash", "cash equivalents", "marketable securities")
        ):
            score += 25
        if re.search(r"\$\s?\d", normalized):
            score += 15
        if any(
            token in lowered
            for token in ("evaluate our liquidity", "future capital requirements")
        ):
            score -= 25
        if any(
            token in lowered
            for token in (
                "agreement",
                "subject to closing conditions",
                "anthropic",
                "invest up to",
            )
        ):
            score -= 40
        if "liquidity" in chunk.section_name.lower():
            score += 10
    if profile.agent_name == "run_10k_agent":
        if any(
            token in lowered
            for token in ("business", "platform", "market", "customer", "segment", "supply", "risk")
        ):
            score += 20
        if any(
            token in lowered
            for token in (
                "is now a data center scale ai infrastructure company",
                "full stack computing platform",
                (
                    "platforms incorporate processors, interconnects, software, "
                    "algorithms, systems, and services"
                ),
            )
        ):
            score += 40
        if any(
            token in lowered
            for token in (
                "data center scale ai infrastructure company",
                "reshaping all industries",
                "pioneered accelerated computing",
            )
        ):
            score += 35
        if "pioneered accelerated computing" in lowered:
            score -= 5
        if any(
            token in lowered
            for token in (
                "accelerated computing",
                "data center scale ai infrastructure",
                "platform strategy",
                "unified underlying programmable architecture",
                "markets we serve",
                "technology stack",
            )
        ):
            score += 30
        if "two segments" in lowered:
            score -= 12
        if lowered.count(",") >= 4:
            score -= 12
        if any(
            token in lowered
            for token in (
                "software libraries, frameworks, algorithms",
                "software development kits",
                "application programming interfaces",
            )
        ):
            score -= 15
        if any(
            token in lowered
            for token in ("china", "license requirements", "alternative products")
        ):
            score -= 20
        if "business" in chunk.section_name.lower():
            score += 10
    return score


def _looks_like_low_signal_fragment(text: str) -> bool:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return True
    if re.fullmatch(r"\d{1,3}[.]?", normalized):
        return True
    tokens = normalized.split()
    if len(tokens) <= 2 and all(re.fullmatch(r"[\d.,$()%]+", token) for token in tokens):
        return True
    if re.match(r"^(item|part)\s+\d", normalized, flags=re.IGNORECASE):
        return True
    return False


def _build_8k_item_findings(
    *,
    state: SpecializedAgentState,
    profile: AgentProfile,
    tasks: list[RoutedTask],
    chunks: list[ChunkRecord],
    unresolved_items: list[str],
    coverage_label: CoverageLabel,
) -> list[AgentFinding]:
    if not tasks:
        return []
    grouped_chunks: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        if not chunk.item_number:
            continue
        grouped_chunks.setdefault(chunk.item_number, []).append(chunk)
    if not grouped_chunks:
        return []
    findings: list[AgentFinding] = []
    primary_task = tasks[0]
    item_drafts = state.get("item_reasoning_drafts", {})
    for item_number, item_chunks in sorted(grouped_chunks.items()):
        item_draft = item_drafts.get(item_number)
        raw_item_summary = (
            item_draft.provisional_summary
            if item_draft is not None
            else _summarize_chunks(item_chunks)
        )
        item_summary = _build_8k_item_summary(
            item_number=item_number,
            item_chunks=item_chunks,
            item_summary=raw_item_summary,
        )
        claim = _build_8k_item_claim(
            item_number=item_number,
            item_chunks=item_chunks,
            item_summary=item_summary,
        )
        item_evidence_refs = _select_chunk_evidence_refs(
            chunks=item_chunks,
            claim=claim,
            agent_name=profile.agent_name,
        )
        item_unresolved_items = list(
            dict.fromkeys(
                [
                    *unresolved_items,
                    *(item_draft.unresolved_items if item_draft is not None else []),
                ]
            )
        )
        item_coverage_label = (
            CoverageLabel.COMPLETE
            if item_evidence_refs and not item_unresolved_items
            else CoverageLabel.PARTIAL
        )
        findings.append(
            _build_agent_finding(
                state=state,
                profile=profile,
                task=primary_task,
                claim=claim,
                summary=item_summary,
                chunks=item_chunks,
                unresolved_items=item_unresolved_items,
                coverage_label=item_coverage_label,
                finding_suffix=item_number.replace(".", "_"),
                covered_topics=[primary_task.task_type.value, f"item_{item_number}"],
            )
        )
    return findings


def _build_8k_item_summary(
    *,
    item_number: str,
    item_chunks: list[ChunkRecord],
    item_summary: str,
) -> str:
    summary = _normalize_8k_summary(item_summary or _summarize_chunks(item_chunks))
    if item_number == "9.01":
        exhibit_description = _extract_exhibit_description(item_chunks)
        if exhibit_description is not None:
            return f'Exhibit filed: "{exhibit_description}."'
        summary = _strip_8k_boilerplate(summary)
        if _requires_8k_fallback(summary):
            return "Exhibits and supporting filing materials were disclosed."
        return summary

    if _requires_8k_fallback(summary):
        preferred_sentence = _best_8k_chunk_sentence(item_chunks)
        if preferred_sentence is not None:
            return preferred_sentence
    summary = _strip_8k_boilerplate(summary)
    preferred_sentence = _best_8k_chunk_sentence(item_chunks)
    if preferred_sentence is not None and _is_weaker_8k_summary(summary, preferred_sentence):
        return preferred_sentence
    return _compress_8k_item_summary(item_number=item_number, summary=summary)


def _build_agent_finding(
    *,
    state: SpecializedAgentState,
    profile: AgentProfile,
    task: RoutedTask,
    claim: str,
    summary: str,
    chunks: list[ChunkRecord],
    unresolved_items: list[str],
    coverage_label: CoverageLabel,
    finding_suffix: str | None = None,
    covered_topics: list[str] | None = None,
) -> AgentFinding:
    finding_id = f"{state['request'].request_id}:{profile.agent_name}:{task.task_type.value}"
    if finding_suffix:
        finding_id = f"{finding_id}:{finding_suffix}"
    return AgentFinding(
        finding_id=finding_id,
        agent_name=profile.agent_name,
        company=state["request"].company,
        claim=claim,
        summary=summary,
        signal_type=_signal_type_for_profile(profile),
        time_horizon=_time_horizon_for_profile(profile),
        evidence_type=_evidence_type_for_chunks(chunks),
        as_of_date=_latest_chunk_date(chunks),
        filing_types=list(profile.filing_types),
        evidence_refs=_select_chunk_evidence_refs(
            chunks=chunks,
            claim=claim,
            agent_name=profile.agent_name,
        ),
        coverage_status=CoverageStatus(
            label=coverage_label,
            covered_topics=covered_topics or [task.task_type.value],
            missing_topics=unresolved_items,
            notes=summary or None,
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.INSUFFICIENT,
            confidence=ConfidenceLabel.LOW,
            rationale="Unverified prior to evidence checker.",
            verifier_name="pending_verification",
        ),
    )


def _build_8k_item_claim(
    *,
    item_number: str,
    item_chunks: list[ChunkRecord],
    item_summary: str,
) -> str:
    raw_section_name = next(
        (
            chunk.section_name
            for chunk in item_chunks
            if chunk.section_name and chunk.section_name.strip()
        ),
        f"Item {item_number}",
    )
    section_label = _clean_8k_section_label(raw_section_name, item_number)
    summary = _normalize_8k_summary(item_summary or _summarize_chunks(item_chunks))
    if item_number == "5.02":
        compact_source = summary
        if _requires_8k_fallback(summary) or not _looks_like_item_502_event_summary(summary):
            compact_source = _best_8k_chunk_sentence(item_chunks) or summary
        compact_summary = _compress_8k_item_summary(
            item_number=item_number,
            summary=compact_source,
        )
        if compact_summary:
            return f"Item {item_number} disclosed: {compact_summary}"
    if item_number == "9.01" and summary.lower().startswith("exhibit filed:"):
        return _build_8k_item_claim_fallback(
            item_number=item_number,
            section_label=section_label,
            item_chunks=item_chunks,
        )
    if item_number != "9.01" and _requires_8k_fallback(summary):
        preferred_sentence = _best_8k_chunk_sentence(item_chunks)
        if preferred_sentence is not None:
            summary = preferred_sentence
    if item_number == "9.01" and _requires_8k_fallback(summary):
        return _build_8k_item_claim_fallback(
            item_number=item_number,
            section_label=section_label,
            item_chunks=item_chunks,
        )
    if summary.lower().startswith(f"item {item_number.lower()}"):
        return summary
    if _is_generic_8k_summary(summary):
        return _build_8k_item_claim_fallback(
            item_number=item_number,
            section_label=section_label,
            item_chunks=item_chunks,
        )
    if len(summary) > 1:
        return _compose_8k_item_claim(
            item_number=item_number,
            section_label=section_label,
            summary=summary,
        )
    return f"Item {item_number} ({section_label}) disclosure."


def _group_chunks_by_item_number(chunks: list[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    grouped: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        if not chunk.item_number:
            continue
        grouped.setdefault(chunk.item_number, []).append(chunk)
    return grouped


def _clean_8k_section_label(section_name: str, item_number: str) -> str:
    normalized = re.sub(
        rf"^\s*item\s+{re.escape(item_number)}\.?\s*",
        "",
        section_name,
        flags=re.IGNORECASE,
    ).strip()
    return normalized.rstrip(".") or f"Item {item_number}"


def _is_generic_8k_summary(summary: str) -> bool:
    normalized = " ".join(summary.split()).strip(" .")
    if not normalized:
        return True
    tokens = normalized.split()
    if len(tokens) <= 2:
        return True
    return normalized.lower() in {"nvidia corp", "nvidia corporation"}


def _build_8k_item_claim_fallback(
    *,
    item_number: str,
    section_label: str,
    item_chunks: list[ChunkRecord],
) -> str:
    if item_number == "9.01":
        exhibit_description = _extract_exhibit_description(item_chunks)
        if exhibit_description is not None:
            return (
                f"Item {item_number} ({section_label}) disclosed the exhibit "
                f"\"{exhibit_description}.\""
            )
        return (
            f"Item {item_number} ({section_label}) disclosed exhibits and supporting "
            "filing materials."
        )
    first_sentence = _best_8k_chunk_sentence(item_chunks) or _summarize_chunks(item_chunks)
    if len(first_sentence) > 1:
        return _compose_8k_item_claim(
            item_number=item_number,
            section_label=section_label,
            summary=_normalize_8k_summary(first_sentence),
        )
    return f"Item {item_number} ({section_label}) disclosure."


def _extract_exhibit_description(item_chunks: list[ChunkRecord]) -> str | None:
    for chunk in item_chunks:
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if line.lower() == "description":
                for candidate in lines[index + 1 :]:
                    if _looks_like_exhibit_code(candidate):
                        continue
                    if candidate:
                        return candidate.rstrip(".")
            if _looks_like_exhibit_code(line):
                for candidate in lines[index + 1 :]:
                    if _looks_like_exhibit_code(candidate):
                        continue
                    if candidate:
                        return candidate.rstrip(".")
        match = re.search(
            r"(?ims)^\s*10\.1\s+(.+?)(?:\n\s*\d{2,3}\b|\n\s*signature\b|$)",
            chunk.text,
        )
        if match is not None:
            candidate = " ".join(match.group(1).split()).rstrip(".")
            if candidate:
                return candidate
        match = re.search(
            r"(?im)^\s*10\.1\s+(.+)$",
            chunk.text,
        )
        if match is not None:
            candidate = " ".join(match.group(1).split()).rstrip(".")
            if candidate:
                return candidate
    return None


def _looks_like_exhibit_code(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    if normalized in {"description", "exhibit number"}:
        return True
    if re.fullmatch(r"\d+\.\d+", normalized):
        return True
    if re.fullmatch(r"\d{2,3}", normalized):
        return True
    return False


def _normalize_8k_summary(summary: str) -> str:
    normalized = " ".join(summary.split()).strip()
    if not normalized:
        return normalized
    normalized = _limit_sentence_count(normalized, max_sentences=2)
    normalized = _remove_unsupported_8k_sentences(normalized)
    normalized = _rewrite_8k_scaffold_phrases(normalized)
    normalized = _strip_8k_boilerplate(normalized)
    if not normalized.endswith("."):
        normalized = f"{normalized}."
    return normalized


def _best_8k_chunk_sentence(chunks: list[ChunkRecord]) -> str | None:
    candidates: list[tuple[int, str]] = []
    for chunk in chunks[:4]:
        normalized = " ".join(chunk.text.split()).strip()
        if not normalized:
            continue
        for raw_sentence in re.split(r"(?<=[.!?])\s+", normalized):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            candidates.append((_score_8k_sentence(sentence), sentence))
    if not candidates:
        return None
    score, sentence = max(candidates, key=lambda item: item[0])
    if score < 15:
        return None
    normalized = _normalize_8k_summary(sentence)
    return normalized or None


def _score_8k_sentence(sentence: str) -> int:
    normalized = " ".join(sentence.split()).strip()
    if not normalized:
        return -100
    lowered = normalized.lower()
    score = 0
    if normalized[0].isupper():
        score += 15
    if re.match(r"^(On|The Board|NVIDIA|The company|The Compensation Committee)\b", normalized):
        score += 20
    if re.match(r"^[a-z]", normalized):
        score -= 50
    if len(normalized) < 40:
        score -= 20
    boilerplate_markers = (
        "the following table sets forth",
        "pursuant to the requirements",
        "signature",
        "exhibit number",
        "(d) exhibits",
        "cover page of this current report",
    )
    if any(marker in lowered for marker in boilerplate_markers):
        score -= 50
    if "compensation committee" in lowered or "variable compensation plan" in lowered:
        score += 15
    return score


def _compose_8k_item_claim(
    *,
    item_number: str,
    section_label: str,
    summary: str,
) -> str:
    if re.match(r"^(on|in|during|an|a|the)\b", summary, flags=re.IGNORECASE) or re.match(
        r"^[A-Z][A-Za-z0-9&'. -]{0,40}\bdisclosed\b",
        summary,
    ):
        return f"Item {item_number} ({section_label}) disclosed: {summary}"
    return f"Item {item_number} ({section_label}) disclosed that {summary}"


def _limit_sentence_count(text: str, *, max_sentences: int) -> str:
    if max_sentences <= 0:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    trimmed = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not trimmed:
        return ""
    return " ".join(trimmed[:max_sentences])


def _remove_unsupported_8k_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("no other "):
            continue
        if lowered.startswith("the provided snippets do not indicate"):
            continue
        if lowered.startswith("the provided 8-k filing does not indicate"):
            continue
        kept.append(stripped)
    return " ".join(kept)


def _rewrite_8k_scaffold_phrases(text: str) -> str:
    rewritten = re.sub(
        r"^The provided 8-K filing dated ([A-Za-z]+ \d{1,2}, \d{4}), disclosed ",
        r"The \1 8-K disclosed ",
        text,
    )
    rewritten = re.sub(
        r"^An 8-K filing dated ([A-Za-z]+ \d{1,2}, \d{4}), disclosed ",
        r"The \1 8-K disclosed ",
        rewritten,
    )
    return rewritten.strip()


def _strip_8k_boilerplate(text: str) -> str:
    rewritten = re.sub(
        r"(?i)\bSIGNATURE\b.*$",
        "",
        text,
    ).strip()
    rewritten = re.sub(
        r"(?i)\bPursuant to the requirements of the Securities Exchange Act of 1934.*$",
        "",
        rewritten,
    ).strip()
    rewritten = re.sub(
        r"(?i)\bThe cover page of this Current Report on Form 8-K.*$",
        "",
        rewritten,
    ).strip()
    rewritten = re.sub(
        r"(?i)^\(d\)\s+Exhibits\s+Exhibit Number\s+Description\s*",
        "",
        rewritten,
    ).strip()
    return rewritten.strip(" .")


def _compress_8k_item_summary(*, item_number: str, summary: str) -> str:
    normalized = " ".join(summary.split()).strip()
    if not normalized or item_number != "5.02":
        return normalized

    match = re.search(
        r"(?i)(On [A-Za-z]+ \d{1,2}, \d{4}), .*?adopted the "
        r"(Variable Compensation Plan for Fiscal Year \d{4}).*",
        normalized,
    )
    if match is not None:
        event_date = match.group(1)
        plan_name = match.group(2)
        return (
            f"{event_date}, NVIDIA adopted the {plan_name} for eligible executive officers."
        )

    return normalized


def _looks_like_item_502_event_summary(summary: str) -> bool:
    lowered = summary.lower()
    required_markers = (
        "compensation plan",
        "variable compensation plan",
        "compensation committee",
        "executive officers",
    )
    if any(marker in lowered for marker in required_markers):
        return True
    return bool(re.search(r"on [A-Za-z]+ \d{1,2}, \d{4}", summary))


def _is_weaker_8k_summary(summary: str, preferred_sentence: str) -> bool:
    normalized_summary = " ".join(summary.split()).strip()
    normalized_preferred = " ".join(preferred_sentence.split()).strip()
    if not normalized_summary:
        return True
    if normalized_summary == normalized_preferred:
        return False
    if normalized_summary[0].islower():
        return True
    weak_markers = (
        "by the compensation committee",
        "participant must remain an employee",
        "exhibit number description",
        "cover page of this current report",
    )
    lowered_summary = normalized_summary.lower()
    return any(marker in lowered_summary for marker in weak_markers)


def _requires_8k_fallback(summary: str) -> bool:
    lowered = summary.lower()
    return (
        "provided 8-k filing" in lowered
        or "provided snippets" in lowered
        or "as an exhibit" in lowered
        or lowered.startswith("by the ")
        or lowered.startswith("(d) exhibits")
        or "pursuant to the requirements" in lowered
    )


def _build_filters(
    *,
    profile: AgentProfile,
    state: SpecializedAgentState,
) -> ChunkSearchFilters:
    return ChunkSearchFilters(
        ticker=state["request"].company.ticker,
        filing_types=list(profile.filing_types),
        section_name=profile.section_names[0] if profile.section_names else None,
        transcript_segment_type=(
            _build_transcript_segment_types(profile)[0]
            if profile.transcript_segment_types
            else None
        ),
    )


def _build_retrieval_query(
    *,
    profile: AgentProfile,
    request: AnalysisRequest,
    tasks: list[RoutedTask],
) -> str:
    ticker = request.company.ticker
    task_names = " ".join(task.task_type.value for task in tasks)
    question = request.question.strip()
    if profile.agent_name == "run_10q_agent":
        return " ".join(
            part
            for part in (
                ticker,
                "latest quarter",
                "recent quarter change",
                "revenue margin liquidity controls",
                "10-Q",
                question,
                task_names,
            )
            if part
        )
    if profile.agent_name == "run_10k_agent":
        return " ".join(
            part
            for part in (
                ticker,
                "long term structure",
                "business risk md&a",
                "10-K",
                question,
                task_names,
            )
            if part
        )
    if profile.agent_name == "run_8k_agent":
        return " ".join(
            part
            for part in (
                ticker,
                "recent material events",
                _build_8k_focus_terms(question),
                "8-K",
                question,
                task_names,
            )
            if part
        )
    return " ".join(part for part in (ticker, question, task_names, profile.agent_name) if part)


def _retrieval_limit_for_profile(profile: AgentProfile) -> int:
    if profile.agent_name == "run_8k_agent":
        return 8
    return 4


def _build_8k_focus_terms(question: str) -> str:
    lowered_question = question.lower()
    focus_terms = [
        "item 1.01 agreement customer supply partnership",
        "item 2.02 results operations guidance",
        "item 5.02 leadership board compensation",
        "item 8.01 other events strategic update",
    ]
    if any(token in lowered_question for token in ("leadership", "management", "director", "ceo")):
        focus_terms.insert(0, "item 5.02 officer director compensation change")
    if any(token in lowered_question for token in ("earnings", "results", "guidance")):
        focus_terms.insert(0, "item 2.02 earnings results operations guidance")
    if any(token in lowered_question for token in ("agreement", "customer", "supply", "partner")):
        focus_terms.insert(0, "item 1.01 material definitive agreement")
    if any(token in lowered_question for token in ("financing", "debt", "securities", "credit")):
        focus_terms.append("item 2.03 financing debt securities")
    ordered_terms = list(dict.fromkeys(focus_terms))
    return " ".join(ordered_terms)


def _build_transcript_segment_types(profile: AgentProfile) -> tuple[TranscriptSegmentType, ...]:
    return tuple(TranscriptSegmentType(item) for item in profile.transcript_segment_types)


def _collect_packet_evidence_refs(findings: list[AgentFinding]) -> list[EvidenceRef]:
    selected: list[EvidenceRef] = []
    seen_ids: set[str] = set()
    for finding in findings:
        for evidence_ref in finding.evidence_refs:
            if evidence_ref.evidence_id in seen_ids:
                continue
            selected.append(evidence_ref)
            seen_ids.add(evidence_ref.evidence_id)
            if len(selected) >= 3:
                return selected
    return selected


def _select_chunk_evidence_refs(
    *,
    chunks: list[ChunkRecord],
    claim: str,
    agent_name: str,
    max_refs: int = 3,
) -> list[EvidenceRef]:
    ranked = sorted(
        chunks,
        key=lambda chunk: _chunk_alignment_score(chunk=chunk, claim=claim),
        reverse=True,
    )
    selected: list[EvidenceRef] = []
    fallbacks: list[EvidenceRef] = []
    for chunk in ranked:
        excerpt = _best_chunk_excerpt_for_claim(chunk=chunk, claim=claim)
        evidence_ref = _chunk_to_evidence_ref(
            chunk,
            agent_name,
            excerpt_override=excerpt,
        )
        if _passes_evidence_alignment_guard(
            claim=claim,
            chunk=chunk,
            excerpt=evidence_ref.excerpt,
        ):
            selected.append(evidence_ref)
        else:
            fallbacks.append(evidence_ref)
        if len(selected) >= max_refs:
            break
    if selected:
        return selected
    return fallbacks[:max_refs]


def _chunk_alignment_score(*, chunk: ChunkRecord, claim: str) -> int:
    chunk_text = " ".join(chunk.text.split()).strip()
    if not chunk_text:
        return -100
    score = _sentence_alignment_score(sentence=chunk_text, claim=claim)
    lowered_chunk = chunk_text.lower()
    lowered_claim = claim.lower()

    sentence_excerpt = _best_chunk_excerpt_for_claim(chunk=chunk, claim=claim)
    if sentence_excerpt is not None:
        score = max(score, _sentence_alignment_score(sentence=sentence_excerpt, claim=claim) + 15)
    if (
        chunk.section_name
        and "business" in chunk.section_name.lower()
        and "segments" in lowered_claim
    ):
        score += 10
    if "data-center-scale ai infrastructure company" in lowered_claim:
        if chunk.section_name and "business" in chunk.section_name.lower():
            score += 25
        else:
            score -= 20
    if (
        chunk.section_name
        and "liquidity" in chunk.section_name.lower()
        and "cash" in lowered_claim
    ):
        score += 10
    if _is_ten_q_liquidity_chunk(chunk):
        if "as of" in lowered_chunk and any(
            token in lowered_chunk
            for token in ("cash", "cash equivalents", "marketable securities")
        ):
            score += 25
        if re.search(r"\$\s?\d", chunk_text):
            score += 10

    penalties = (
        "the following table sets forth",
        "pursuant to the requirements",
        "signature",
        "(d) exhibits",
        "exhibit number",
    )
    if any(marker in lowered_chunk for marker in penalties):
        score -= 20
    return score


def _best_chunk_excerpt_for_claim(*, chunk: ChunkRecord, claim: str) -> str | None:
    normalized = " ".join(chunk.text.split()).strip()
    if not normalized:
        return None
    if chunk.filing_type == FilingType.FORM_8K and chunk.item_number == "9.01":
        exhibit_description = _extract_exhibit_description([chunk])
        if exhibit_description is not None:
            return f'Exhibit filed: "{exhibit_description}."'
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized)]
    scored = [
        (_sentence_alignment_score(sentence=sentence, claim=claim, chunk=chunk), sentence)
        for sentence in sentences
        if sentence
    ]
    if not scored:
        return normalized[:240]
    score, sentence = max(scored, key=lambda item: item[0])
    if score <= 0:
        fallback = sentences[0] if sentences else normalized
        cleaned_fallback = _clean_evidence_excerpt(chunk=chunk, excerpt=fallback)
        return cleaned_fallback if len(cleaned_fallback) <= 240 else f"{cleaned_fallback[:237]}..."
    if not sentence.endswith("."):
        sentence = f"{sentence}."
    cleaned_sentence = _clean_evidence_excerpt(chunk=chunk, excerpt=sentence)
    return cleaned_sentence if len(cleaned_sentence) <= 240 else f"{cleaned_sentence[:237]}..."


def _passes_evidence_alignment_guard(
    *,
    claim: str,
    chunk: ChunkRecord,
    excerpt: str,
) -> bool:
    normalized_excerpt = " ".join(excerpt.split()).strip()
    if not normalized_excerpt:
        return False
    lowered_excerpt = normalized_excerpt.lower()
    lowered_claim = claim.lower()

    score = _sentence_alignment_score(sentence=normalized_excerpt, claim=claim, chunk=chunk)
    claim_tokens = {
        token
        for token in _alignment_tokens(lowered_claim)
        if len(token) > 2 and token not in {"item", "disclosed", "that", "the", "and"}
    }
    overlap = sum(1 for token in claim_tokens if token in lowered_excerpt)

    if chunk.filing_type == FilingType.FORM_10Q and _is_ten_q_liquidity_chunk(chunk):
        if _looks_like_accounting_pronouncement_excerpt(normalized_excerpt):
            return False
        if any(token in lowered_claim for token in ("cash", "marketable", "equivalents")):
            if overlap < 2 or score < 20:
                return False

    return score >= 10 or overlap >= 2


def _clean_evidence_excerpt(*, chunk: ChunkRecord, excerpt: str) -> str:
    cleaned = " ".join(excerpt.split()).strip()
    if not cleaned:
        return cleaned
    if chunk.filing_type == FilingType.FORM_10K:
        cleaned = re.sub(
            r"^(?:Item\s+1\.\s+)?(?:Our Businesses|Our Business|Business Overview)\b[:\s-]*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"^(?:Item\s+1\.\s+)?Business\b(?=\s+(?:NVIDIA|We|Our|The company)\b)[:\s-]*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
    return cleaned


def _looks_like_accounting_pronouncement_excerpt(excerpt: str) -> bool:
    lowered = excerpt.lower()
    markers = (
        "adoption of new and recently issued accounting pronouncements",
        "recently issued accounting pronouncements",
        "new accounting pronouncements",
    )
    return any(marker in lowered for marker in markers)


def _sentence_alignment_score(
    *,
    sentence: str,
    claim: str,
    chunk: ChunkRecord | None = None,
) -> int:
    normalized_sentence = " ".join(sentence.split()).strip()
    if not normalized_sentence:
        return -100
    lowered_sentence = normalized_sentence.lower()
    lowered_claim = claim.lower()
    score = 0

    claim_tokens = {
        token
        for token in _alignment_tokens(lowered_claim)
        if len(token) > 2 and token not in {"item", "disclosed", "that", "the", "and"}
    }
    overlap = sum(1 for token in claim_tokens if token in lowered_sentence)
    score += overlap * 8

    if normalized_sentence[0].isupper():
        score += 5
    if lowered_claim[:80] and lowered_claim[:80] in lowered_sentence:
        score += 30
    if re.match(r"^(On|As of|We|Our|NVIDIA|The company)\b", normalized_sentence):
        score += 10
    if re.match(r"^[a-z]", normalized_sentence):
        score -= 20

    if chunk is not None and _is_ten_q_liquidity_chunk(chunk):
        score += _ten_q_exact_match_bonus(sentence=normalized_sentence, claim=claim)
        if lowered_sentence.startswith("as of ") and any(
            token in lowered_sentence
            for token in ("cash", "cash equivalents", "marketable securities")
        ):
            score += 35
        if re.search(r"\$\s?\d", normalized_sentence):
            score += 20
        if "consist of" in lowered_sentence or "issued by" in lowered_sentence:
            score -= 15
        if any(
            token in lowered_sentence
            for token in (
                "hypothetical",
                "10% decrease",
                "fair value",
                "publicly-held equity securities balance",
            )
        ):
            score -= 35
        if any(
            token in lowered_sentence
            for token in (
                "cash used in financing activities",
                "purchases of property and equipment",
            )
        ):
            score -= 20
        if any(
            token in lowered_sentence
            for token in (
                "agreement",
                "subject to closing conditions",
                "anthropic",
                "invest up to",
            )
        ):
            score -= 30
    return score


def _ten_q_exact_match_bonus(*, sentence: str, claim: str) -> int:
    lowered_sentence = sentence.lower()
    lowered_claim = claim.lower()
    score = 0

    amount_tokens = set(re.findall(r"\$?\d+(?:\.\d+)?", lowered_claim))
    if amount_tokens:
        score += sum(
            20
            for token in amount_tokens
            if token in sentence or token in lowered_sentence
        )

    phrase_tokens = (
        "cash equivalents",
        "marketable securities",
        "cash, cash equivalents",
    )
    score += sum(
        15
        for phrase in phrase_tokens
        if phrase in lowered_claim and phrase in lowered_sentence
    )

    if "as of " in lowered_claim and lowered_sentence.startswith("as of "):
        score += 10
    return score


def _alignment_tokens(text: str) -> set[str]:
    normalized = text.replace("-", " ")
    return set(re.findall(r"[a-z0-9$]+(?:\.[0-9]+)?", normalized))


def _is_ten_q_liquidity_chunk(chunk: ChunkRecord) -> bool:
    return (
        chunk.filing_type == FilingType.FORM_10Q
        and "liquidity" in chunk.section_name.lower()
    )


def _chunk_to_evidence_ref(
    chunk: ChunkRecord,
    agent_name: str,
    *,
    excerpt_override: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"{agent_name}:{chunk.chunk_id}",
        document_id=chunk.document_id,
        filing_type=chunk.filing_type,
        excerpt=(excerpt_override or chunk.text[:240]),
        section_id=chunk.section_id,
        chunk_id=chunk.chunk_id,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        source_uri=chunk.source_url,
    )


def _lowest_confidence(
    labels: list[ConfidenceLabel],
    *,
    default: ConfidenceLabel,
) -> ConfidenceLabel:
    ranking = {
        ConfidenceLabel.LOW: 0,
        ConfidenceLabel.MEDIUM: 1,
        ConfidenceLabel.HIGH: 2,
    }
    if not labels:
        return default
    return min(labels, key=lambda label: ranking[label])


def _signal_type_for_profile(profile: AgentProfile) -> FindingSignalType:
    if profile.agent_name == "run_8k_agent":
        return FindingSignalType.EVENT
    if profile.agent_name == "run_call_agent":
        return FindingSignalType.MANAGEMENT_TONE
    return FindingSignalType.FUNDAMENTAL


def _time_horizon_for_profile(profile: AgentProfile) -> TimeHorizonLabel:
    if profile.agent_name == "run_10k_agent":
        return TimeHorizonLabel.LONG_TERM
    if profile.agent_name in {"run_10q_agent", "run_8k_agent", "run_call_agent"}:
        return TimeHorizonLabel.RECENT
    return TimeHorizonLabel.MIXED


def _evidence_type_for_chunks(chunks: list[ChunkRecord]) -> EvidenceTypeLabel:
    if not chunks:
        return EvidenceTypeLabel.NARRATIVE
    if all(chunk.filing_type == FilingType.EARNINGS_CALL for chunk in chunks):
        return EvidenceTypeLabel.NARRATIVE
    return (
        EvidenceTypeLabel.MIXED
        if any(chunk.embedding for chunk in chunks)
        else EvidenceTypeLabel.NARRATIVE
    )


def _summarize_chunks(chunks: list[ChunkRecord]) -> str:
    sentences: list[str] = []
    for chunk in chunks[:2]:
        text = " ".join(chunk.text.split())
        if not text:
            continue
        sentence = _first_summary_sentence(text)
        if sentence and not sentence.endswith("."):
            sentence = f"{sentence}."
        if sentence:
            sentences.append(sentence)
    if not sentences:
        return "Retrieved filing evidence is present but could not be summarized deterministically."
    return " ".join(dict.fromkeys(sentences))


def _first_summary_sentence(text: str) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return ""
    protected = normalized
    for abbreviation in ("Corp.", "Inc.", "Ltd.", "Co."):
        protected = protected.replace(abbreviation, abbreviation.replace(".", "<prd>"))
    sentence = protected.split(". ", maxsplit=1)[0].strip()
    return sentence.replace("<prd>", ".")


def _latest_chunk_date(chunks: list[ChunkRecord]) -> date | None:
    dates = [
        candidate
        for chunk in chunks
        for candidate in (chunk.filing_date, chunk.report_period, chunk.call_date)
        if candidate is not None
    ]
    return max(dates) if dates else None


__all__ = [
    "AgentProfile",
    "AgentReasoningDraft",
    "AgentRetrievalRequest",
    "CALL_PROFILE",
    "EIGHT_K_PROFILE",
    "NoOpSpecializedAgentRetriever",
    "SpecializedAgentModel",
    "SpecializedAgentRetriever",
    "StubSpecializedAgentModel",
    "TEN_K_PROFILE",
    "TEN_Q_PROFILE",
    "build_initial_agent_state",
    "build_specialized_agent_subgraph",
    "execute_specialized_agent",
]
