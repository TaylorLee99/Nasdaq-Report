"""Retrieval adapters used by specialized filing agents."""

from __future__ import annotations

import re
from datetime import date

from app.config import Settings, get_settings
from app.domain import ChunkRecord, ConfidenceLabel, FilingType
from app.graph.subgraphs.specialized_agents import AgentRetrievalRequest, SpecializedAgentRetriever
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
        filtered = _post_filter_chunks(request=request, chunks=chunks)
        filtered = _drop_low_quality_filing_fragments(
            filing_types=request["filters"].filing_types,
            chunks=filtered,
        )
        filtered = _exclude_ten_q_market_risk_chunks(request=request, chunks=filtered)
        filtered = _exclude_ten_q_event_style_chunks(request=request, chunks=filtered)
        filtered = _prefer_ten_q_liquidity_balance_chunks(request=request, chunks=filtered)
        if not filtered:
            filtered = chunks
        reranked = sorted(
            filtered,
            key=lambda chunk: (
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
            _section_matches(chunk.section_name, preferred_name) for preferred_name in section_names
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
            _section_match_score(chunk.section_name, preferred)
            for preferred in request["section_names"]
        ),
        default=0.0,
    )


def _section_matches(actual: str, preferred: str) -> bool:
    return _section_match_score(actual, preferred) >= 0.5


def _section_match_score(actual: str, preferred: str) -> float:
    actual_tokens = _normalized_tokens(actual)
    preferred_tokens = _normalized_tokens(preferred)
    if not actual_tokens or not preferred_tokens:
        return 0.0
    overlap = len(actual_tokens & preferred_tokens) / len(preferred_tokens)
    return overlap


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
    score = 0.0

    if chunk.filing_type == FilingType.FORM_10Q:
        if any(
            token in text
            for token in ("liquidity", "capital resources", "cash", "revenue", "margin")
        ):
            score += 1.0
        if "as of" in text and any(
            token in text for token in ("cash", "cash equivalents", "marketable securities")
        ):
            score += 1.5
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
        if "latest quarter" in query and "quarter" in text:
            score += 0.5

    if chunk.filing_type == FilingType.FORM_10K:
        if any(
            token in text
            for token in ("business", "platform", "market", "customer", "segment", "supply")
        ):
            score += 1.0
        if any(
            token in text
            for token in ("china", "license requirements", "alternative products")
        ):
            score -= 1.0
        if "long term structure" in query and any(
            token in text for token in ("business", "platform", "market", "customer", "segment")
        ):
            score += 0.5

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
        return 1.5 if query_tokens & business_terms else 0.5
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
