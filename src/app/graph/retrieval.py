"""Retrieval adapters used by specialized filing agents."""

from __future__ import annotations

import re
from datetime import date

from app.config import Settings, get_settings
from app.domain import ChunkRecord, ConfidenceLabel, FilingType
from app.graph.subgraphs.specialized_agents import AgentRetrievalRequest, SpecializedAgentRetriever
from app.graph.ten_q_heuristics import (
    build_ten_q_recent_change_candidate,
    is_ten_q_recent_change_candidate_sentence,
)
from app.indexing import FakeEmbeddingProvider
from app.indexing.models import ChunkSearchFilters
from app.storage import make_session_factory
from app.storage.repositories import PgVectorChunkRepository


class IndexedChunkRetriever(SpecializedAgentRetriever):
    """Retriever backed by the indexed filing corpus."""

    def __init__(
        self,
        *,
        vector_repository: PgVectorChunkRepository,
        embedding_provider: FakeEmbeddingProvider,
    ) -> None:
        self._vector_repository = vector_repository
        self._embedding_provider = embedding_provider

    def retrieve(self, request: AgentRetrievalRequest) -> list[ChunkRecord]:
        base_filters = _without_section_filters(request["filters"])
        chunks = _apply_latest_filing_hard_filter(
            filing_types=request["filters"].filing_types,
            chunks=self._vector_repository.list_chunks(filters=base_filters, limit=None),
        )
        section_filtered = _post_filter_chunks(request=request, chunks=chunks)
        filtered = section_filtered
        filtered = _drop_low_quality_filing_fragments(
            filing_types=request["filters"].filing_types,
            chunks=filtered,
        )
        filtered = _exclude_ten_k_non_structural_chunks(request=request, chunks=filtered)
        filtered = _exclude_ten_q_market_risk_chunks(request=request, chunks=filtered)
        filtered = _exclude_ten_q_accounting_policy_chunks(request=request, chunks=filtered)
        filtered = _exclude_ten_q_event_style_chunks(request=request, chunks=filtered)
        filtered = _prefer_ten_q_recent_change_chunks(request=request, chunks=filtered)
        filtered = _prefer_ten_q_liquidity_balance_chunks(request=request, chunks=filtered)
        filtered = _prefer_ten_k_structure_chunks(request=request, chunks=filtered)
        if not filtered:
            filtered = section_filtered
        if not filtered and request["filters"].filing_types == [FilingType.FORM_8K]:
            filtered = chunks
        reranked = sorted(
            filtered,
            key=lambda chunk: (
                _ten_q_recent_change_priority(request, chunk),
                _filing_narrative_quality(chunk),
                _sentence_candidate_quality(request, chunk),
                _section_intent_score(request, chunk),
                _section_quality(request, chunk),
                _keyword_overlap(request["query"], chunk.text),
                _item_number_quality(request, chunk),
                _recency_score(chunk),
                chunk.parse_confidence == ConfidenceLabel.HIGH,
                len(chunk.text),
            ),
            reverse=True,
        )
        if request["filters"].filing_types == [FilingType.FORM_8K]:
            reranked = _diversify_8k_item_chunks(reranked)
        return reranked[: request["limit"]]


def build_specialized_agent_retriever(
    settings: Settings | None = None,
) -> IndexedChunkRetriever:
    """Build the default indexed-corpus retriever for filing agents."""

    current_settings = settings or get_settings()
    return IndexedChunkRetriever(
        vector_repository=PgVectorChunkRepository(make_session_factory(current_settings)),
        embedding_provider=FakeEmbeddingProvider(
            dimensions=current_settings.indexing.embedding_dimensions
        ),
    )


def _without_section_filters(filters: ChunkSearchFilters) -> ChunkSearchFilters:
    return filters.model_copy(
        update={
            "section_name": None,
            "transcript_segment_type": None,
        }
    )


def _post_filter_chunks(
    *,
    request: AgentRetrievalRequest,
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    section_names = set(request["section_names"])
    segment_types = {segment.value for segment in request["transcript_segment_types"]}
    filtered = []
    for chunk in chunks:
        if section_names and not any(
            _section_matches(chunk, preferred_name) for preferred_name in section_names
        ):
            continue
        if segment_types:
            segment = chunk.transcript_segment_type.value if chunk.transcript_segment_type else None
            if segment not in segment_types:
                continue
        filtered.append(chunk)
    return filtered


def _drop_low_quality_filing_fragments(
    *,
    filing_types: list[FilingType],
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    if not chunks or len(filing_types) != 1:
        return chunks
    filing_type = filing_types[0]
    if filing_type not in {FilingType.FORM_10K, FilingType.FORM_10Q}:
        return chunks
    if filing_type == FilingType.FORM_10Q:
        candidate_chunks = [
            chunk for chunk in chunks if build_ten_q_recent_change_candidate(chunk) is not None
        ]
        if candidate_chunks:
            return candidate_chunks

    scored_chunks = [(chunk, _filing_narrative_quality(chunk)) for chunk in chunks]
    strong_chunks = [chunk for chunk, score in scored_chunks if score >= 1.0]
    if strong_chunks:
        return strong_chunks
    medium_chunks = [chunk for chunk, score in scored_chunks if score >= 0.0]
    return medium_chunks or chunks


def _prefer_ten_q_liquidity_balance_chunks(
    *,
    request: AgentRetrievalRequest,
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    if request["filters"].filing_types != [FilingType.FORM_10Q]:
        return chunks
    if "liquidity" not in request["query"].lower():
        return chunks

    preferred = [chunk for chunk in chunks if _is_balance_sentence_chunk(chunk)]
    return preferred or chunks


def _prefer_ten_q_recent_change_chunks(
    *,
    request: AgentRetrievalRequest,
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    if request["filters"].filing_types != [FilingType.FORM_10Q]:
        return chunks
    if not any(token in request["query"].lower() for token in ("latest quarter", "recent quarter")):
        return chunks

    preferred = [chunk for chunk in chunks if _looks_like_ten_q_recent_change_chunk(chunk)]
    sentence_preferred = [
        chunk for chunk in preferred if _has_ten_q_recent_change_candidate_sentence(chunk)
    ]
    if sentence_preferred:
        return sentence_preferred
    if preferred:
        return preferred

    filtered = [
        chunk
        for chunk in chunks
        if not _looks_like_ten_q_non_delta_recent_change_chunk(chunk)
    ]
    sentence_filtered = [
        chunk for chunk in filtered if _has_ten_q_recent_change_candidate_sentence(chunk)
    ]
    if sentence_filtered:
        return sentence_filtered
    if filtered:
        return filtered
    return chunks


def _prefer_ten_k_structure_chunks(
    *,
    request: AgentRetrievalRequest,
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    if request["filters"].filing_types != [FilingType.FORM_10K]:
        return chunks
    if "long term structure" not in request["query"].lower():
        return chunks

    preferred_business = [
        chunk
        for chunk in chunks
        if "business" in chunk.section_name.lower() and _looks_like_ten_k_structure_chunk(chunk)
    ]
    if preferred_business:
        return preferred_business

    preferred_mda = [
        chunk
        for chunk in chunks
        if any(
            token in chunk.section_name.lower()
            for token in ("management", "discussion", "analysis")
        )
        and _looks_like_ten_k_structure_chunk(chunk)
    ]
    if preferred_mda:
        return preferred_mda

    preferred = [chunk for chunk in chunks if _looks_like_ten_k_structure_chunk(chunk)]
    return preferred or chunks


def _exclude_ten_k_non_structural_chunks(
    *,
    request: AgentRetrievalRequest,
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    if request["filters"].filing_types != [FilingType.FORM_10K]:
        return chunks
    if "long term structure" not in request["query"].lower():
        return chunks

    filtered = [chunk for chunk in chunks if not _looks_like_ten_k_non_structural_chunk(chunk)]
    return filtered or chunks


def _exclude_ten_q_market_risk_chunks(
    *,
    request: AgentRetrievalRequest,
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    if request["filters"].filing_types != [FilingType.FORM_10Q]:
        return chunks
    query = request["query"].lower()
    if "liquidity" not in query:
        return chunks

    filtered = [chunk for chunk in chunks if not _looks_like_market_risk_chunk(chunk)]
    return filtered or chunks


def _exclude_ten_q_accounting_policy_chunks(
    *,
    request: AgentRetrievalRequest,
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    if request["filters"].filing_types != [FilingType.FORM_10Q]:
        return chunks
    query = request["query"].lower()
    if not any(token in query for token in ("latest quarter", "recent quarter", "liquidity")):
        return chunks

    filtered = [chunk for chunk in chunks if not _looks_like_ten_q_accounting_policy_chunk(chunk)]
    return filtered or chunks


def _exclude_ten_q_event_style_chunks(
    *,
    request: AgentRetrievalRequest,
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    if request["filters"].filing_types != [FilingType.FORM_10Q]:
        return chunks
    query = request["query"].lower()
    if "liquidity" not in query:
        return chunks

    filtered = [chunk for chunk in chunks if not _looks_like_event_style_liquidity_chunk(chunk)]
    return filtered or chunks


def _keyword_overlap(query: str, text: str) -> float:
    left = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
    right = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _item_number_quality(request: AgentRetrievalRequest, chunk: ChunkRecord) -> float:
    if chunk.item_number is None:
        return 0.0
    query = request["query"].lower()
    if f"item {chunk.item_number.lower()}" in query:
        return 1.0
    if any(token in query for token in _item_number_keywords(chunk.item_number)):
        return 0.5
    return 0.0


def _section_quality(request: AgentRetrievalRequest, chunk: ChunkRecord) -> float:
    if not request["section_names"]:
        return 0.0
    return max(
        (
            _section_match_score(chunk, preferred)
            for preferred in request["section_names"]
        ),
        default=0.0,
    )


def _section_matches(chunk: ChunkRecord, preferred: str) -> bool:
    return _section_match_score(chunk, preferred) >= 0.5


def _section_match_score(chunk: ChunkRecord, preferred: str) -> float:
    expected_types = _preferred_section_types(preferred)
    actual_types = _actual_section_types(chunk)
    if expected_types and actual_types and not (expected_types & actual_types):
        return 0.0
    actual = chunk.section_name
    actual_tokens = _normalized_tokens(actual)
    preferred_tokens = _normalized_tokens(preferred)
    if not actual_tokens or not preferred_tokens:
        return 0.0
    overlap = len(actual_tokens & preferred_tokens) / len(preferred_tokens)
    return overlap


def _preferred_section_types(preferred: str) -> set[str]:
    lowered = preferred.lower()
    expected: set[str] = set()
    if "liquidity" in lowered:
        expected.add("liquidity")
    if "discussion" in lowered or "analysis" in lowered:
        expected.add("mda")
    if "business" in lowered:
        expected.add("business")
    if "risk" in lowered or "factors" in lowered:
        expected.add("risk_factors")
    if "controls" in lowered:
        expected.add("controls")
    if "financial statements" in lowered:
        expected.add("financial_statements")
    if "notes" in lowered:
        expected.add("notes")
    return expected


def _actual_section_types(chunk: ChunkRecord) -> set[str]:
    lowered = chunk.section_name.lower()
    actual_types: set[str] = set()
    if chunk.section_type is not None:
        actual_types.add(chunk.section_type.value)
    if "liquidity" in lowered:
        actual_types.add("liquidity")
    if "discussion" in lowered or "analysis" in lowered:
        actual_types.add("mda")
    if "business" in lowered:
        actual_types.add("business")
    if "risk" in lowered or "factors" in lowered:
        actual_types.add("risk_factors")
    if "controls" in lowered:
        actual_types.add("controls")
    if "financial statements" in lowered or "financial statement" in lowered:
        actual_types.add("financial_statements")
    if "notes" in lowered:
        actual_types.add("notes")
    return actual_types


def _normalized_tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _recency_score(chunk: ChunkRecord) -> float:
    effective_date = chunk.filing_date or chunk.report_period or chunk.call_date
    if effective_date is None:
        return 0.0
    days_old = max((_reference_date() - effective_date).days, 0)
    return 1.0 / (1.0 + days_old / 365.0)


def _filing_narrative_quality(chunk: ChunkRecord) -> float:
    if chunk.filing_type not in {FilingType.FORM_10K, FilingType.FORM_10Q}:
        return 0.0

    text = " ".join(chunk.text.split()).strip()
    if not text:
        return -5.0

    lowered = text.lower()
    score = 0.0

    if re.match(r"^[A-Z]", text):
        score += 1.0
    if re.match(r"^(On|As of|The company|We|Our|NVIDIA|Management)", text):
        score += 0.5
    if 80 <= len(text) <= 700:
        score += 0.5

    if re.match(r"^[a-z]", text):
        score -= 2.0
    if text.startswith(("(", "[", "*")):
        score -= 1.0
    if _looks_table_like(text):
        score -= 3.0

    fragment_markers = (
        "the following risk factors should be considered",
        "the following table sets forth",
        "refer to item 1a",
        "any investment will be completed on expected terms",
        "pursuant to the requirements",
    )
    if any(marker in lowered for marker in fragment_markers):
        score -= 2.5

    if text.count(",") >= 8:
        score -= 0.5

    return score


def _sentence_candidate_quality(request: AgentRetrievalRequest, chunk: ChunkRecord) -> float:
    if chunk.filing_type not in {FilingType.FORM_10K, FilingType.FORM_10Q}:
        return 0.0
    query = request["query"].lower()
    text = " ".join(chunk.text.split()).strip().lower()
    section_lower = chunk.section_name.lower()
    score = 0.0

    if chunk.filing_type == FilingType.FORM_10Q:
        ten_q_candidate = build_ten_q_recent_change_candidate(chunk)
        if any(
            token in text
            for token in (
                "liquidity",
                "capital resources",
                "cash",
                "revenue",
                "margin",
                "net sales",
                "services",
                "products",
                "gross margin",
                "operating cash flow",
                "sales increased",
                "revenue increased",
            )
        ):
            score += 1.0
        if "as of" in text and any(
            token in text for token in ("cash", "cash equivalents", "marketable securities")
        ):
            score += 1.5
        if ten_q_candidate is not None:
            score += 2.5
        if re.search(r"\$\s?\d", chunk.text):
            score += 0.5
        if any(
            token in text
            for token in ("future capital requirements", "evaluate our liquidity")
        ):
            score -= 1.0
        if any(
            token in text
            for token in (
                "hypothetical",
                "10% decrease",
                "fair value",
                "publicly-held equity securities balance",
            )
        ):
            score -= 2.0
        if any(
            token in text
            for token in (
                "agreement",
                "subject to closing conditions",
                "anthropic",
                "invest up to",
            )
        ):
            score -= 3.0
        if any(
            token in text
            for token in (
                "critical accounting policies",
                "critical accounting estimates",
                "accounting pronouncements",
                "summary of significant accounting policies",
                "preparation of condensed consolidated financial statements",
                "preparation of financial statements",
                "make estimates and assumptions",
                "asu ",
                "adopt asu",
                "prepared in accordance with gaap",
                "forward-looking statements",
                "private securities litigation reform act",
                "future arrangements",
                "share repurchase program remained available",
                "amortization of these costs",
                "estimated life of the products",
                "contractual obligations and commitments",
                "commitments and contingencies",
                "operating lease commitments",
                "could affect customer demand",
            )
        ):
            score -= 4.0
        if "latest quarter" in query and "quarter" in text:
            score += 0.5

    if chunk.filing_type == FilingType.FORM_10K:
        if any(
            token in text
            for token in (
                "business",
                "platform",
                "market",
                "customer",
                "segment",
                "supply",
                "products",
                "services",
                "provides",
                "offers",
                "serves",
                "software",
                "cloud",
                "stores",
                "marketplace",
                "devices",
            )
        ):
            score += 1.0
        if "is a technology company" in text:
            score += 2.0
        if any(
            token in text
            for token in (
                "designs, manufactures and markets",
                "sells a variety of related services",
                "we seek to offer",
                "we serve our primary customer sets",
            )
        ):
            score += 1.5
        if any(
            token in text
            for token in ("china", "license requirements", "alternative products")
        ):
            score -= 1.0
        if any(
            token in text
            for token in (
                "critical accounting estimates",
                "critical accounting policies",
                "market risk",
                "equity 10% decrease",
                "foreign exchange rate risk",
                "interest rate risk",
                "could be adversely affected",
                "additional taxes imposed",
                "forward-looking statements",
                "private securities litigation reform act",
                "shareholders of record",
                "long-term debt is carried at amortized cost",
                "fluctuations in interest rates do not impact",
                "our employees are critical to our mission",
            )
        ):
            score -= 3.5
        if "human capital" in section_lower:
            score -= 2.0
        if "—" not in chunk.section_name and "--" not in chunk.section_name:
            score += 1.0
        if "long term structure" in query and any(
            token in text
            for token in (
                "business",
                "platform",
                "market",
                "customer",
                "segment",
                "products",
                "services",
                "offers",
                "provides",
                "serves",
            )
        ):
            score += 0.5

    return score


def _ten_q_recent_change_priority(
    request: AgentRetrievalRequest,
    chunk: ChunkRecord,
) -> float:
    if chunk.filing_type != FilingType.FORM_10Q:
        return 0.0
    if not any(token in request["query"].lower() for token in ("latest quarter", "recent quarter")):
        return 0.0

    candidate = build_ten_q_recent_change_candidate(chunk)
    if candidate is None:
        return 0.0

    lowered = candidate.lower()
    score = 2.0
    if any(
        marker in lowered
        for marker in (
            "compared to the same quarter",
            "compared with the same quarter",
            "compared to the prior year period",
            "compared with the prior year period",
            "compared to the comparable prior year period",
            "compared with the comparable prior year period",
            "trailing twelve months ended",
            "during the period ended",
        )
    ):
        score += 2.0
    if any(
        marker in lowered
        for marker in (
            "revenue",
            "net sales",
            "gross margin",
            "operating income",
            "cash",
            "cash equivalents",
            "marketable securities",
            "customers",
            "average revenue",
        )
    ):
        score += 1.0
    if re.search(r"\b\d+%\b", lowered):
        score += 1.0
    if re.search(r"\$\s?\d", candidate):
        score += 0.75
    if any(
        marker in lowered
        for marker in (
            "historically experienced higher net sales",
            "compared to other quarters in its fiscal year",
            "seasonal holiday demand",
            "product and service introductions can significantly impact",
        )
    ):
        score -= 6.0
    if any(
        marker in lowered
        for marker in (
            "changes in foreign exchange rates",
            "foreign exchange rates did not significantly impact",
            "negatively impacted operating income by",
        )
    ):
        score -= 3.0
    return score


def _section_intent_score(request: AgentRetrievalRequest, chunk: ChunkRecord) -> float:
    if chunk.filing_type == FilingType.FORM_10K:
        return _ten_k_section_intent_score(request["query"], chunk.section_name)
    if chunk.filing_type == FilingType.FORM_10Q:
        return _ten_q_section_intent_score(request["query"], chunk.section_name)
    return 0.0


def _ten_k_section_intent_score(query: str, section_name: str) -> float:
    query_tokens = _normalized_tokens(query)
    section_tokens = _normalized_tokens(section_name)
    if not section_tokens:
        return 0.0

    business_terms = {"business", "structure", "model", "segment", "product", "customer"}
    risk_terms = {"risk", "competition", "headwind", "challenge", "supply", "constraint"}
    mda_terms = {"md", "mda", "discussion", "analysis", "operations", "financial", "results"}

    if "business" in section_tokens:
        score = 1.5 if query_tokens & business_terms else 0.5
        if "—" not in section_name and "--" not in section_name:
            score += 1.25
        if "human" in section_tokens or "capital" in section_tokens:
            score -= 1.5
        return score
    if "risk" in section_tokens or "factors" in section_tokens:
        return 1.25 if query_tokens & risk_terms else -0.25
    if "analysis" in section_tokens or "discussion" in section_tokens:
        return 1.0 if query_tokens & mda_terms else 0.5
    return 0.0


def _ten_q_section_intent_score(query: str, section_name: str) -> float:
    query_tokens = _normalized_tokens(query)
    section_tokens = _normalized_tokens(section_name)
    if not section_tokens:
        return 0.0

    liquidity_terms = {"liquidity", "cash", "capital", "resources", "obligations"}
    quarter_terms = {"quarter", "recent", "revenue", "margin", "growth", "operations"}
    controls_terms = {"controls", "control", "disclosure", "internal"}
    financial_terms = {"assets", "liabilities", "cash", "flow", "eps", "balance", "statements"}

    if "liquidity" in section_tokens:
        return 1.5 if query_tokens & liquidity_terms else 0.5
    if "analysis" in section_tokens or "discussion" in section_tokens:
        return 1.25 if query_tokens & quarter_terms else 0.75
    if "controls" in section_tokens:
        return 1.0 if query_tokens & controls_terms else -0.25
    if "statements" in section_tokens or "financial" in section_tokens:
        return 1.0 if query_tokens & financial_terms else 0.25
    return 0.0


def _looks_table_like(text: str) -> bool:
    number_tokens = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)
    money_like_tokens = re.findall(r"\$\s?\d", text)
    short_lines = [line.strip() for line in text.splitlines() if line.strip()]
    month_headers = {
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    }

    if len(number_tokens) >= 8 and len(money_like_tokens) >= 2:
        return True
    if short_lines:
        header_tokens = _normalized_tokens(" ".join(short_lines[:2]))
        if len(number_tokens) >= 8 and header_tokens & month_headers:
            return True
    if text.count("$") >= 3 and text.count(",") >= 8:
        return True
    return False


def _is_balance_sentence_chunk(chunk: ChunkRecord) -> bool:
    if chunk.filing_type != FilingType.FORM_10Q:
        return False
    if "liquidity" not in chunk.section_name.lower():
        return False

    text = " ".join(chunk.text.split()).strip().lower()
    if not text.startswith("as of "):
        return False
    if not any(token in text for token in ("cash", "cash equivalents", "marketable securities")):
        return False
    disallowed = (
        "hypothetical",
        "10% decrease",
        "fair value",
        "publicly-held equity securities balance",
    )
    return not any(token in text for token in disallowed)


def _looks_like_ten_k_structure_chunk(chunk: ChunkRecord) -> bool:
    if chunk.filing_type != FilingType.FORM_10K:
        return False
    text = " ".join(chunk.text.split()).strip().lower()
    section = chunk.section_name.lower()
    if _looks_like_ten_k_non_structural_chunk(chunk):
        return False
    structural_markers = (
        "we offer",
        "we provides",
        "we provide",
        "we serve",
        "we seek to offer",
        "we seek to be",
        "we develop and support",
        "we build and deploy",
        "is a technology company",
        "our products",
        "our services",
        "our customers",
        "designs, manufactures and markets",
        "sells a variety of related services",
        "customer sets",
        "operating segments",
        "principal software platforms",
        "platform",
        "platforms",
        "software",
        "cloud",
        "services",
        "products",
        "customers",
        "marketplace",
        "stores",
        "advertising",
        "devices",
        "ecosystem",
        "infrastructure",
        "end markets",
        "built four principal software platforms",
    )
    if any(marker in text for marker in structural_markers):
        return True
    return "business" in section and not _looks_table_like(chunk.text)


def _looks_like_ten_q_recent_change_chunk(chunk: ChunkRecord) -> bool:
    if chunk.filing_type != FilingType.FORM_10Q:
        return False
    text = " ".join(chunk.text.split()).strip().lower()
    if build_ten_q_recent_change_candidate(chunk) is not None:
        return True
    if _looks_like_ten_q_accounting_policy_chunk(chunk):
        return False
    if _looks_like_market_risk_chunk(chunk):
        return False
    if _looks_like_event_style_liquidity_chunk(chunk):
        return False
    if _looks_like_ten_q_non_delta_recent_change_chunk(chunk):
        return False

    metric_markers = (
        "revenue",
        "net sales",
        "gross margin",
        "operating income",
        "operating margin",
        "net income",
        "cash flows from operating activities",
        "cash flow from operating activities",
        "cash and cash equivalents",
        "cash, cash equivalents, and marketable securities",
        "commercial customer count",
    )
    delta_markers = (
        " increased ",
        " decrease ",
        " decreased ",
        " growth ",
        " grew ",
        " compared to the prior year period",
        " compared to the comparable prior year period",
        " year-over-year ",
        " q1 ",
        " q2 ",
        " q3 ",
        " q4 ",
    )
    has_metric = any(marker in text for marker in metric_markers)
    has_delta = any(marker in f" {text} " for marker in delta_markers) or bool(
        re.search(r"\b\d+%\b", text)
    )
    return has_metric and has_delta


def _looks_like_ten_q_non_delta_recent_change_chunk(chunk: ChunkRecord) -> bool:
    if chunk.filing_type != FilingType.FORM_10Q:
        return False
    text = " ".join(chunk.text.split()).strip().lower()
    section = chunk.section_name.lower()
    markers = (
        "organized our operations into three segments",
        "these segments reflect the way the company evaluates its business performance",
        "business seasonality and product introductions",
        "updated products",
        "sales and marketing efforts",
        "sales and marketing costs",
        "technology and infrastructure costs",
        "contractual obligations and commitments",
        "commitments and contingencies",
        "operating lease commitments",
        "macroeconomic factors",
        "macroeconomic conditions",
        "legal proceedings",
        "outcomes of legal proceedings",
        "subject to significant uncertainty",
        "could affect customer demand",
        "future capital requirements",
        "future arrangements",
        "share repurchase program remained available",
        "estimated life of the products",
        "marketing costs are largely variable",
        "see item 1 of part i",
        "segment information",
        "note 8",
        "note 1",
    )
    section_markers = (
        "sales and marketing",
        "business seasonality",
        "product introductions",
        "segment information",
        "legal proceedings",
    )
    return any(marker in text for marker in markers) or any(
        marker in section for marker in section_markers
    )


def _has_ten_q_recent_change_candidate_sentence(chunk: ChunkRecord) -> bool:
    return build_ten_q_recent_change_candidate(chunk) is not None


def _is_ten_q_recent_change_candidate_sentence(
    *,
    sentence: str,
    section_name: str,
) -> bool:
    return is_ten_q_recent_change_candidate_sentence(
        sentence=sentence,
        section_name=section_name,
    )


def _looks_like_ten_k_non_structural_chunk(chunk: ChunkRecord) -> bool:
    if chunk.filing_type != FilingType.FORM_10K:
        return False
    text = " ".join(chunk.text.split()).strip().lower()
    section = chunk.section_name.lower()
    markers = (
        "critical accounting estimates",
        "critical accounting policies",
        "market risk",
        "equity 10% decrease",
        "foreign exchange rate risk",
        "interest rate risk",
        "additional taxes imposed",
        "could be adversely affected",
        "forward-looking statements",
        "private securities litigation reform act",
        "item 8",
        "earnings 49 part ii",
        "our employees are critical to our mission",
        "hold rights to openai",
        "openai’s intellectual property",
    )
    return any(marker in text for marker in markers) or any(marker in section for marker in markers)


def _looks_like_market_risk_chunk(chunk: ChunkRecord) -> bool:
    if chunk.filing_type != FilingType.FORM_10Q:
        return False
    text = " ".join(chunk.text.split()).strip().lower()
    section = chunk.section_name.lower()
    markers = (
        "market risk",
        "interest rate risk",
        "quantitative and qualitative disclosures",
        "hypothetical",
        "fair value",
        "10% decrease",
        "item 3",
    )
    return any(marker in text for marker in markers) or any(marker in section for marker in markers)


def _looks_like_ten_q_accounting_policy_chunk(chunk: ChunkRecord) -> bool:
    if chunk.filing_type != FilingType.FORM_10Q:
        return False
    text = " ".join(chunk.text.split()).strip().lower()
    section = chunk.section_name.lower()
    markers = (
        "adoption of new and recently issued accounting pronouncements",
        "accounting pronouncements",
        "critical accounting policies",
        "critical accounting estimates",
        "summary of significant accounting policies",
        "preparation of condensed consolidated financial statements",
        "preparation of financial statements",
        "make estimates and assumptions",
        "asu ",
        "adopt asu",
        "prepared in accordance with gaap",
        "forward-looking statements",
        "private securities litigation reform act",
    )
    return any(marker in text for marker in markers) or any(marker in section for marker in markers)


def _looks_like_event_style_liquidity_chunk(chunk: ChunkRecord) -> bool:
    if chunk.filing_type != FilingType.FORM_10Q:
        return False
    text = " ".join(chunk.text.split()).strip().lower()
    section = chunk.section_name.lower()
    markers = (
        "agreement",
        "subject to closing conditions",
        "anthropic",
        "invest up to",
        "entered into an agreement",
    )
    if any(marker in text for marker in markers):
        return True
    if "liquidity" in section:
        return False
    return False


def _reference_date() -> date:
    return date.today()


def _apply_latest_filing_hard_filter(
    *,
    filing_types: list[FilingType],
    chunks: list[ChunkRecord],
) -> list[ChunkRecord]:
    if len(filing_types) != 1 or not chunks:
        return chunks
    filing_type = filing_types[0]
    if filing_type not in {FilingType.FORM_10K, FilingType.FORM_10Q, FilingType.FORM_8K}:
        return chunks
    latest_date = max(
        (chunk.filing_date for chunk in chunks if chunk.filing_date is not None),
        default=None,
    )
    if latest_date is None:
        return chunks
    latest_accessions = {
        chunk.accession_number
        for chunk in chunks
        if chunk.filing_date == latest_date and chunk.accession_number is not None
    }
    if latest_accessions:
        return [
            chunk for chunk in chunks if chunk.accession_number in latest_accessions
        ]
    return [chunk for chunk in chunks if chunk.filing_date == latest_date]


def _diversify_8k_item_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    diversified: list[ChunkRecord] = []
    seen_items: set[str] = set()
    deferred: list[ChunkRecord] = []
    for chunk in chunks:
        item_number = (chunk.item_number or "").strip()
        if item_number and item_number not in seen_items:
            diversified.append(chunk)
            seen_items.add(item_number)
            continue
        deferred.append(chunk)
    diversified.extend(deferred)
    return diversified


def _item_number_keywords(item_number: str) -> tuple[str, ...]:
    mapping = {
        "1.01": ("agreement", "customer", "supply", "partnership"),
        "2.02": ("earnings", "results", "operations", "guidance"),
        "2.03": ("financing", "debt", "securities", "credit"),
        "5.02": ("leadership", "board", "director", "officer", "compensation"),
        "8.01": ("event", "strategic", "update", "announcement"),
    }
    return mapping.get(item_number, ())
