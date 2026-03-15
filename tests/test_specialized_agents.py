from __future__ import annotations

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
    TranscriptSegmentType,
)
from app.graph.subgraphs import (
    CALL_PROFILE,
    EIGHT_K_PROFILE,
    TEN_K_PROFILE,
    TEN_Q_PROFILE,
    AgentProfile,
    AgentReasoningDraft,
    AgentRetrievalRequest,
    StubSpecializedAgentModel,
    build_initial_agent_state,
    build_specialized_agent_subgraph,
    execute_specialized_agent,
)
from app.graph.subgraphs.specialized_agents import (
    _build_8k_item_claim,
    _select_chunk_evidence_refs,
)


class RecordingRetriever:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self._chunks = chunks
        self.requests: list[AgentRetrievalRequest] = []

    def retrieve(self, request: AgentRetrievalRequest) -> list[ChunkRecord]:
        self.requests.append(request)
        return list(self._chunks)


class EmptyRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, request: AgentRetrievalRequest) -> list[ChunkRecord]:
        del request
        self.calls += 1
        return []


class AlwaysRetryModel:
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
        del request, tasks, chunks
        return AgentReasoningDraft(
            provisional_summary=f"{profile.agent_name} retry loop iteration {iteration_count}",
            unresolved_items=["needs_more_evidence"],
            confidence_label=ConfidenceLabel.LOW,
            reretrieval_requested=iteration_count < max_iterations,
            requested_query=f"{profile.agent_name} retry {iteration_count + 1}",
        )


class RecordingItemModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, list[str]]] = []

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
        del profile, request, tasks, iteration_count, max_iterations
        item_number = chunks[0].item_number if chunks else None
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.calls.append((item_number, chunk_ids))
        return AgentReasoningDraft(
            provisional_summary=f"Modeled summary for item {item_number}",
            unresolved_items=[],
            confidence_label=ConfidenceLabel.HIGH,
            reretrieval_requested=False,
        )


@pytest.mark.parametrize(
    ("profile", "task_type", "filing_type"),
    [
        (TEN_K_PROFILE, AnalysisTaskType.LONG_TERM_STRUCTURE, FilingType.FORM_10K),
        (TEN_Q_PROFILE, AnalysisTaskType.RECENT_QUARTER_CHANGE, FilingType.FORM_10Q),
        (EIGHT_K_PROFILE, AnalysisTaskType.MATERIAL_EVENTS, FilingType.FORM_8K),
        (CALL_PROFILE, AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE, FilingType.EARNINGS_CALL),
    ],
)
def test_specialized_agent_subgraphs_execute_with_profile_specific_filters(
    profile: AgentProfile,
    task_type: AnalysisTaskType,
    filing_type: FilingType,
) -> None:
    request = AnalysisRequest(
        request_id=f"req-{profile.agent_name}",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Run specialized agent.",
        include_transcript=filing_type == FilingType.EARNINGS_CALL,
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=task_type,
        agent_name=profile.agent_name,
        filing_type=filing_type,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id=f"{profile.agent_name}:chunk-1",
        document_id="doc-1",
        section_id="section-1",
        company=request.company,
        filing_type=filing_type,
        section_name=profile.section_names[0] if profile.section_names else "Item 1.01",
        text=(
            "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, and "
            "marketable securities."
            if filing_type == FilingType.FORM_10Q
            else "Evidence chunk for specialized agent execution."
        ),
        token_count=6,
        parse_confidence=ConfidenceLabel.HIGH,
        transcript_segment_type=(
            TranscriptSegmentType.QA if filing_type == FilingType.EARNINGS_CALL else None
        ),
    )
    retriever = RecordingRetriever([chunk])

    packet = execute_specialized_agent(
        profile=profile,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=StubSpecializedAgentModel(),
    )

    assert packet.agent_name == profile.agent_name
    assert packet.iteration_count == 1
    assert packet.evidence_refs
    assert packet.confidence_label == ConfidenceLabel.HIGH
    assert retriever.requests[0]["filters"].filing_types == list(profile.filing_types)
    assert retriever.requests[0]["section_names"] == profile.section_names
    assert retriever.requests[0]["transcript_segment_types"] == tuple(
        TranscriptSegmentType(value) for value in profile.transcript_segment_types
    )


def test_specialized_agent_reretrieval_loop_is_bounded() -> None:
    request = AnalysisRequest(
        request_id="req-loop",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Force bounded retries.",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
        agent_name=TEN_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_10K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for retry test",
        document_available=True,
        parse_confidence=ConfidenceLabel.LOW,
    )
    retriever = EmptyRetriever()
    graph = build_specialized_agent_subgraph(
        profile=TEN_K_PROFILE,
        retriever=retriever,
        model=AlwaysRetryModel(),
    )

    result = graph.invoke(
        build_initial_agent_state(request=request, tasks=[task], profile=TEN_K_PROFILE)
    )
    packet = result["packet"]

    assert retriever.calls == 2
    assert packet is not None
    assert packet.iteration_count == 2
    assert packet.confidence_label == ConfidenceLabel.LOW
    assert packet.unresolved_items == ["needs_more_evidence"]
    assert len(packet.requested_queries) >= 2


def test_ten_q_planner_uses_latest_quarter_query_template() -> None:
    request = AnalysisRequest(
        request_id="req-10q-query",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for query test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenq-query:chunk-1",
        document_id="doc-10q-query",
        section_id="section-10q-query",
        company=request.company,
        filing_type=FilingType.FORM_10Q,
        section_name=TEN_Q_PROFILE.section_names[0],
        text="Quarterly revenue and liquidity evidence.",
        token_count=5,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever([chunk])

    execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=StubSpecializedAgentModel(),
    )

    query = retriever.requests[0]["query"]
    assert "latest quarter" in query
    assert "revenue margin liquidity operating income customer growth controls" in query
    assert "10-Q" in query


def test_ten_q_agent_prefers_deterministic_excerpt_claim_over_fragmentary_summary() -> None:
    request = AnalysisRequest(
        request_id="req-10q-deterministic",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for deterministic claim test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenq-deterministic:chunk-1",
        document_id="doc-10q-deterministic",
        section_id="section-10q-deterministic",
        company=request.company,
        filing_type=FilingType.FORM_10Q,
        section_name="Liquidity and Capital Resources",
        text=(
            "Evaluate our liquidity and capital resources to ensure we can finance future "
            "capital requirements. As of October 26, 2025, we had $60.6 billion in cash, "
            "cash equivalents, and marketable securities."
        ),
        token_count=26,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever([chunk])

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim.startswith("As of October 26, 2025")


def test_ten_q_agent_ignores_numeric_residue_fragments() -> None:
    request = AnalysisRequest(
        request_id="req-10q-numeric-fragment",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for numeric residue test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenq-numeric-residue:chunk-1",
            document_id="doc-10q-numeric-residue",
            section_id="section-10q-numeric-residue-1",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name=(
                "Management's Discussion and Analysis of Financial Condition and Results "
                "of Operations"
            ),
            text="25",
            token_count=1,
            parse_confidence=ConfidenceLabel.MEDIUM,
        ),
        ChunkRecord(
            chunk_id="tenq-numeric-residue:chunk-2",
            document_id="doc-10q-numeric-residue",
            section_id="section-10q-numeric-residue-2",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name="Liquidity and Capital Resources",
            text=(
                "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, "
                "and marketable securities."
            ),
            token_count=17,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim.startswith("As of October 26, 2025")
    assert packet.findings[0].claim != "25."


def test_ten_q_agent_omits_recent_quarter_finding_when_only_non_delta_chunks_exist() -> None:
    request = AnalysisRequest(
        request_id="req-10q-no-delta-finding",
        company=Company(cik="0000320193", ticker="AAPL", company_name="Apple Inc."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for non-delta exclusion test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenq-no-delta:chunk-products",
            document_id="doc-10q-no-delta",
            section_id="section-10q-no-delta-products",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name=(
                "Management's Discussion and Analysis — Business Seasonality and Product "
                "Introductions"
            ),
            text=(
                "During the first quarter of 2026, the Company announced the following "
                "updated products: 14-inch MacBook Pro and iPad Pro."
            ),
            token_count=18,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
        ChunkRecord(
            chunk_id="tenq-no-delta:chunk-legal",
            document_id="doc-10q-no-delta",
            section_id="section-10q-no-delta-legal",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name="Management's Discussion and Analysis — Legal Proceedings",
            text=(
                "The outcomes of legal proceedings and claims are subject to significant "
                "uncertainty."
            ),
            token_count=12,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings == []
    assert "missing_recent_quarter_delta" in packet.unresolved_items


def test_ten_q_agent_builds_claim_from_overview_table_row() -> None:
    request = AnalysisRequest(
        request_id="req-10q-table-row",
        company=Company(cik="0000789019", ticker="MSFT", company_name="Microsoft Corporation"),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for overview table-row test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenq-table-row:chunk-1",
        document_id="doc-10q-table-row",
        section_id="section-10q-table-row",
        company=request.company,
        filing_type=FilingType.FORM_10Q,
        section_name="ITEM 2. MANAGEMENT’S DISCUSSION AND ANALYSIS OF — OVERVIEW",
        text="Productivity and Business Processes Revenue $ 34,116 $ 29,437 16%",
        token_count=12,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever([chunk]),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim == (
        "Productivity and Business Processes revenue increased 16% compared with the "
        "prior-year period."
    )


def test_ten_q_agent_accepts_strong_delta_sentence_from_macro_subsection() -> None:
    request = AnalysisRequest(
        request_id="req-10q-macro-delta",
        company=Company(cik="0000320193", ticker="AAPL", company_name="Apple Inc."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for macro subsection delta test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenq-macro-delta:chunk-1",
        document_id="doc-10q-macro-delta",
        section_id="section-10q-macro-delta",
        company=request.company,
        filing_type=FilingType.FORM_10Q,
        section_name=(
            "Item 2. Management’s Discussion and Analysis of Financial Condition and "
            "Results of Operations — Macroeconomic Conditions"
        ),
        text=(
            "Europe Europe net sales increased during the first quarter of 2026 compared "
            "to the same quarter in 2025 primarily due to higher net sales of iPhone "
            "and Services."
        ),
        token_count=24,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever([chunk]),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim.startswith(
        "Europe net sales increased during the first quarter of 2026"
    )


def test_ten_q_agent_rejects_business_seasonality_sentence_as_recent_quarter_change() -> None:
    request = AnalysisRequest(
        request_id="req-10q-seasonality-reject",
        company=Company(cik="0000320193", ticker="AAPL", company_name="Apple Inc."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for seasonality rejection test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenq-seasonality:chunk-1",
        document_id="doc-10q-seasonality",
        section_id="section-10q-seasonality",
        company=request.company,
        filing_type=FilingType.FORM_10Q,
        section_name=(
            "Item 2. Management’s Discussion and Analysis of Financial Condition and "
            "Results of Operations — Business Seasonality and Product Introductions"
        ),
        text=(
            "The Company has historically experienced higher net sales in its first quarter "
            "compared to other quarters in its fiscal year due in part to seasonal holiday demand."
        ),
        token_count=24,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever([chunk]),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings == []
    assert "missing_recent_quarter_delta" in packet.unresolved_items


def test_8k_debt_offering_claim_prefers_opening_sale_sentence_for_evidence() -> None:
    company = Company(cik="0001018724", ticker="AMZN", company_name="Amazon.com Inc.")
    chunk_opening = ChunkRecord(
        chunk_id="8k-debt-offering:chunk-1",
        document_id="doc-8k-debt-offering",
        section_id="section-8k-debt-offering",
        company=company,
        filing_type=FilingType.FORM_8K,
        section_name="Item 8.01 Other Events",
        text=(
            "On March 13, 2026, Amazon.com, Inc. closed the sale of $1,750,000,000 "
            "aggregate principal amount of its floating rate notes due 2028 and "
            "multiple other series of notes."
        ),
        token_count=30,
        parse_confidence=ConfidenceLabel.HIGH,
        item_number="8.01",
    )
    chunk_underwriting = ChunkRecord(
        chunk_id="8k-debt-offering:chunk-2",
        document_id="doc-8k-debt-offering",
        section_id="section-8k-debt-offering",
        company=company,
        filing_type=FilingType.FORM_8K,
        section_name="Item 8.01 Other Events",
        text=(
            "000,000 aggregate principal amount of its 5.950% notes due 2066 and "
            "$3,000,000,000 aggregate principal amount of its 6.050% notes due 2076 "
            "pursuant to an Underwriting Agreement dated March 10, 2026."
        ),
        token_count=35,
        parse_confidence=ConfidenceLabel.HIGH,
        item_number="8.01",
    )

    refs = _select_chunk_evidence_refs(
        chunks=[chunk_underwriting, chunk_opening],
        claim=(
            "On March 13, 2026, Amazon.com, Inc. completed the sale of multiple "
            "series of notes, including both fixed and floating rate notes with "
            "various maturities up to 2076."
        ),
        agent_name=EIGHT_K_PROFILE.agent_name,
    )

    assert refs[0].chunk_id == chunk_opening.chunk_id
    assert "closed the sale" in refs[0].excerpt.lower()


def test_ten_q_agent_accepts_customer_growth_sentence_from_overview() -> None:
    request = AnalysisRequest(
        request_id="req-10q-customer-growth",
        company=Company(cik="0001321655", ticker="PLTR", company_name="Palantir Technologies Inc."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for customer growth test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenq-customer-growth:chunk-1",
        document_id="doc-10q-customer-growth",
        section_id="section-10q-customer-growth",
        company=request.company,
        filing_type=FilingType.FORM_10Q,
        section_name=(
            "ITEM 2. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION "
            "AND RESULTS OF OPERATIONS — Overview"
        ),
        text=(
            "Our average revenue for the top twenty customers during the trailing twelve "
            "months ended September 30, 2025 was $83.0 million, which grew 38% from an "
            "average of $60.1 million in revenue from the top twenty customers during "
            "the trailing twelve months ended September 30, 2024."
        ),
        token_count=42,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever([chunk]),
        model=StubSpecializedAgentModel(),
    )

    assert "top twenty customers" in packet.findings[0].claim.lower()
    assert "38%" in packet.findings[0].claim


def test_ten_k_agent_prefers_business_structure_sentence_over_regional_anecdote() -> None:
    request = AnalysisRequest(
        request_id="req-10k-deterministic",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Describe the long-term structure.",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
        agent_name=TEN_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_10K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for deterministic claim test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenk-deterministic:chunk-1",
        document_id="doc-10k-deterministic",
        section_id="section-10k-deterministic",
        company=request.company,
        filing_type=FilingType.FORM_10K,
        section_name="Item 1. Business",
        text=(
            "We have engaged with customers in China to provide alternative products not "
            "subject to the new license requirements. NVIDIA serves gaming, data center, "
            "professional visualization, and automotive end markets through accelerated "
            "computing platforms."
        ),
        token_count=31,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever([chunk])

    packet = execute_specialized_agent(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=StubSpecializedAgentModel(),
    )

    assert "accelerated computing platforms" in packet.findings[0].claim


def test_ten_k_agent_prefers_business_description_over_market_risk_fragment() -> None:
    request = AnalysisRequest(
        request_id="req-10k-market-risk-generalization",
        company=Company(cik="0000789019", ticker="MSFT", company_name="Microsoft Corp."),
        question="Describe the long-term structure.",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
        agent_name=TEN_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_10K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for generalized 10-K structure test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenk-generalized:chunk-market-risk",
            document_id="doc-10k-generalized",
            section_id="section-10k-market-risk",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 7A. Quantitative and Qualitative Disclosures about Market Risk",
            text=(
                "Equity 10% decrease in equity market prices would reduce earnings and other "
                "income in the current fiscal year."
            ),
            token_count=17,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
        ChunkRecord(
            chunk_id="tenk-generalized:chunk-business",
            document_id="doc-10k-generalized",
            section_id="section-10k-business",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 1. Business",
            text=(
                "Microsoft provides cloud-based and productivity software services to "
                "commercial and consumer customers across a broad portfolio."
            ),
            token_count=16,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert "cloud-based and productivity software services" in packet.findings[0].claim


def test_ten_k_agent_strips_heading_like_business_prefixes_from_claims() -> None:
    request = AnalysisRequest(
        request_id="req-10k-heading-prefix",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Describe the business structure.",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
        agent_name=TEN_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_10K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for heading cleanup test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenk-heading-prefix:chunk-1",
        document_id="doc-10k-heading-prefix",
        section_id="section-10k-heading-prefix",
        company=request.company,
        filing_type=FilingType.FORM_10K,
        section_name="Item 1. Business",
        text=(
            "Our Businesses We report our business results in two segments and serve gaming "
            "and data center end markets."
        ),
        token_count=17,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    packet = execute_specialized_agent(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever([chunk]),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim.startswith("We report our business results")
    assert not packet.findings[0].claim.startswith("Our Businesses")


def test_ten_k_agent_strips_item_heading_prefixes_from_claims() -> None:
    request = AnalysisRequest(
        request_id="req-10k-item-prefix",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Describe the business structure.",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
        agent_name=TEN_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_10K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for item prefix cleanup test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenk-item-prefix:chunk-1",
        document_id="doc-10k-item-prefix",
        section_id="section-10k-item-prefix",
        company=request.company,
        filing_type=FilingType.FORM_10K,
        section_name="Item 1. Business",
        text="Item 1. Business NVIDIA serves gaming and data center customers worldwide.",
        token_count=11,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    packet = execute_specialized_agent(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever([chunk]),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim.startswith("NVIDIA serves gaming and data center")
    assert not packet.findings[0].claim.startswith("Item 1. Business")


def test_ten_k_agent_ranks_claim_aligned_evidence_refs_first() -> None:
    request = AnalysisRequest(
        request_id="req-10k-evidence-alignment",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Describe the business structure.",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
        agent_name=TEN_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_10K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for evidence alignment test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenk-evidence-alignment:chunk-regional",
            document_id="doc-10k-evidence-alignment",
            section_id="section-10k-regional",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 1. Business",
            text=(
                "We have engaged with customers in China to provide alternative products not "
                "subject to the new license requirements."
            ),
            token_count=18,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
        ChunkRecord(
            chunk_id="tenk-evidence-alignment:chunk-structure",
            document_id="doc-10k-evidence-alignment",
            section_id="section-10k-structure",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 1. Business",
            text="Our Businesses We report our business results in two segments.",
            token_count=10,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim == "We report our business results in two segments."
    assert (
        packet.findings[0].evidence_refs[0].chunk_id
        == "tenk-evidence-alignment:chunk-structure"
    )
    assert (
        packet.findings[0].evidence_refs[0].excerpt
        == "We report our business results in two segments."
    )


def test_ten_k_agent_prefers_richer_business_platform_sentence_over_segment_stub() -> None:
    request = AnalysisRequest(
        request_id="req-10k-richer-business",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Describe the long-term business structure.",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
        agent_name=TEN_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_10K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for richer 10-K business test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenk-rich-business",
            document_id="doc-10k-rich-business",
            section_id="section-10k-rich-business",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 1. Business",
            text=(
                "NVIDIA is a data center scale AI infrastructure company with a broad "
                "technology stack spanning GPUs, software, and networking."
            ),
            token_count=21,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
        ChunkRecord(
            chunk_id="tenk-segment-stub",
            document_id="doc-10k-rich-business",
            section_id="section-10k-segment-stub",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 1. Business",
            text="We report our business results in two segments.",
            token_count=8,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert (
        packet.findings[0].claim
        == (
            "NVIDIA is a data center scale AI infrastructure company with a broad "
            "technology stack spanning GPUs, software, and networking."
        )
    )


def test_ten_k_agent_skips_truncated_technology_stack_fragment() -> None:
    request = AnalysisRequest(
        request_id="req-10k-truncated-tech-stack",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Describe the long-term business structure.",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
        agent_name=TEN_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_10K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for truncated 10-K business sentence test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenk-truncated-tech-stack",
            document_id="doc-10k-truncated-tech-stack",
            section_id="section-10k-truncated-tech-stack",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 1. Business",
            text=(
                "Our technology stack includes the foundational NVIDIA CUDA development "
                "platform that runs on all NVIDIA GPUs, as well as hundreds of "
                "domain-specific software libraries, frameworks, algorithms,"
            ),
            token_count=24,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
        ChunkRecord(
            chunk_id="tenk-complete-business",
            document_id="doc-10k-truncated-tech-stack",
            section_id="section-10k-complete-business",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 1. Business",
            text=(
                "NVIDIA is a data center scale AI infrastructure company reshaping "
                "all industries."
            ),
            token_count=12,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim == (
        "NVIDIA is a data center scale AI infrastructure company reshaping all industries."
    )


def test_ten_k_agent_prefers_data_center_infrastructure_sentence_over_pioneer_sentence() -> None:
    request = AnalysisRequest(
        request_id="req-10k-infrastructure-sentence",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="Describe the long-term business structure.",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.LONG_TERM_STRUCTURE,
        agent_name=TEN_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_10K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for 10-K business sentence preference test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenk-business-multi-sentence",
            document_id="doc-10k-business-multi-sentence",
            section_id="section-10k-business-multi-sentence",
            company=request.company,
            filing_type=FilingType.FORM_10K,
            section_name="Item 1. Business",
            text=(
                "NVIDIA pioneered accelerated computing to help solve the most challenging "
                "computational problems. NVIDIA is now a data center scale AI infrastructure "
                "company reshaping all industries."
            ),
            token_count=25,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim == (
        "NVIDIA is now a data center scale AI infrastructure company reshaping all industries."
    )


def test_ten_q_agent_uses_sentence_aligned_evidence_excerpt() -> None:
    request = AnalysisRequest(
        request_id="req-10q-evidence-excerpt",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for sentence excerpt test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="tenq-evidence-excerpt:chunk-1",
        document_id="doc-10q-evidence-excerpt",
        section_id="section-10q-evidence-excerpt",
        company=request.company,
        filing_type=FilingType.FORM_10Q,
        section_name="Liquidity and Capital Resources",
        text=(
            "Cash used in financing activities increased in the first nine months of fiscal "
            "year 2026. As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, "
            "and marketable securities."
        ),
        token_count=29,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever([chunk]),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].claim.startswith("As of October 26, 2025")
    assert packet.findings[0].evidence_refs[0].excerpt.startswith("As of October 26, 2025")


def test_ten_q_agent_prefers_recent_quarter_sentence_over_accounting_policy_sentence() -> None:
    request = AnalysisRequest(
        request_id="req-10q-accounting-generalization",
        company=Company(cik="0001321655", ticker="PLTR", company_name="Palantir Technologies Inc."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for generalized 10-Q recent-quarter test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenq-generalized:chunk-accounting",
            document_id="doc-10q-generalized",
            section_id="section-10q-accounting",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name="Critical Accounting Policies and Estimates",
            text=(
                "Critical Accounting Policies and Estimates Our condensed consolidated "
                "financial statements and the accompanying notes thereto are prepared in "
                "accordance with GAAP."
            ),
            token_count=22,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
        ChunkRecord(
            chunk_id="tenq-generalized:chunk-quarterly-change",
            document_id="doc-10q-generalized",
            section_id="section-10q-quarterly-change",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name=(
                "Management's Discussion and Analysis of Financial Condition "
                "and Results of Operations"
            ),
            text=(
                "Revenue increased in the quarter as commercial customer count and software "
                "deployment activity expanded."
            ),
            token_count=16,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert "Revenue increased in the quarter" in packet.findings[0].claim


def test_ten_k_evidence_refs_prefer_business_chunk_for_data_center_claim() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    business_chunk = ChunkRecord(
        chunk_id="tenk-business-evidence",
        document_id="doc-10k-business-evidence",
        section_id="section-10k-business-evidence",
        company=company,
        filing_type=FilingType.FORM_10K,
        section_name="Item 1. Business",
        text=(
            "NVIDIA is now a data center scale AI infrastructure company reshaping "
            "all industries."
        ),
        token_count=12,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    risk_chunk = ChunkRecord(
        chunk_id="tenk-risk-evidence",
        document_id="doc-10k-risk-evidence",
        section_id="section-10k-risk-evidence",
        company=company,
        filing_type=FilingType.FORM_10K,
        section_name="Foreign Exchange Rate Risk",
        text=(
            "Foreign Exchange Rate Risk We consider our direct exposure to foreign "
            "exchange rate fluctuations to be minimal."
        ),
        token_count=15,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    evidence_refs = _select_chunk_evidence_refs(
        chunks=[risk_chunk, business_chunk],
        claim="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        agent_name=TEN_K_PROFILE.agent_name,
    )

    assert evidence_refs[0].chunk_id == "tenk-business-evidence"
    assert "data center scale ai infrastructure company" in evidence_refs[0].excerpt.lower()


def test_ten_q_agent_prefers_balance_sentence_over_security_mix_excerpt() -> None:
    request = AnalysisRequest(
        request_id="req-10q-balance-sentence",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for liquidity sentence preference test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenq-balance-sentence:chunk-mix",
            document_id="doc-10q-balance-sentence",
            section_id="section-10q-balance-mix",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name="Liquidity and Capital Resources",
            text=(
                "Our marketable securities consist of publicly-held equity securities, debt "
                "securities issued by the USG and its agencies, and foreign government "
                "entities."
            ),
            token_count=24,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
        ChunkRecord(
            chunk_id="tenq-balance-sentence:chunk-balance",
            document_id="doc-10q-balance-sentence",
            section_id="section-10q-balance-balance",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name="Liquidity and Capital Resources",
            text=(
                "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, "
                "and marketable securities."
            ),
            token_count=17,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert packet.findings[0].evidence_refs[0].chunk_id == "tenq-balance-sentence:chunk-balance"
    assert packet.findings[0].evidence_refs[0].excerpt.startswith("As of October 26, 2025")


def test_ten_q_agent_prefers_exact_amount_match_for_evidence_excerpt() -> None:
    request = AnalysisRequest(
        request_id="req-10q-exact-amount-match",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for exact amount evidence preference test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenq-exact-amount:chunk-generic-balance",
            document_id="doc-10q-exact-amount",
            section_id="section-10q-exact-amount-generic",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name="Liquidity and Capital Resources",
            text=(
                "As of October 26, 2025, liquidity remained strong and marketable securities "
                "were available to support operating needs."
            ),
            token_count=18,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
        ChunkRecord(
            chunk_id="tenq-exact-amount:chunk-exact-balance",
            document_id="doc-10q-exact-amount",
            section_id="section-10q-exact-amount-exact",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name="Liquidity and Capital Resources",
            text=(
                "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, "
                "and marketable securities."
            ),
            token_count=17,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert (
        packet.findings[0].evidence_refs[0].chunk_id
        == "tenq-exact-amount:chunk-exact-balance"
    )
    assert "$60.6 billion" in packet.findings[0].evidence_refs[0].excerpt


def test_ten_q_agent_skips_low_alignment_accounting_pronouncement_evidence() -> None:
    request = AnalysisRequest(
        request_id="req-10q-accounting-evidence-guard",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What changed recently?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.RECENT_QUARTER_CHANGE,
        agent_name=TEN_Q_PROFILE.agent_name,
        filing_type=FilingType.FORM_10Q,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for accounting pronouncement evidence guard test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunks = [
        ChunkRecord(
            chunk_id="tenq-accounting-guard:chunk-accounting",
            document_id="doc-10q-accounting-guard",
            section_id="section-10q-accounting-guard-accounting",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name="Liquidity and Capital Resources",
            text=(
                "Adoption of New and Recently Issued Accounting Pronouncements There has "
                "been no adoption of any new and recently issued accounting pronouncements."
            ),
            token_count=20,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
        ChunkRecord(
            chunk_id="tenq-accounting-guard:chunk-balance",
            document_id="doc-10q-accounting-guard",
            section_id="section-10q-accounting-guard-balance",
            company=request.company,
            filing_type=FilingType.FORM_10Q,
            section_name="Liquidity and Capital Resources",
            text=(
                "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, "
                "and marketable securities."
            ),
            token_count=17,
            parse_confidence=ConfidenceLabel.HIGH,
        ),
    ]

    packet = execute_specialized_agent(
        profile=TEN_Q_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever(chunks),
        model=StubSpecializedAgentModel(),
    )

    assert (
        packet.findings[0].evidence_refs[0].chunk_id
        == "tenq-accounting-guard:chunk-balance"
    )
    assert "Accounting Pronouncements" not in packet.findings[0].evidence_refs[0].excerpt


def test_eight_k_agent_prefers_sentence_start_over_mid_sentence_fragment() -> None:
    request = AnalysisRequest(
        request_id="req-8k-sentence-start",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What material event was disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for 8-K sentence start test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="eightk-sentence-start:chunk-1",
        document_id="doc-8k-sentence-start",
        section_id="section-8k-sentence-start",
        company=request.company,
        filing_type=FilingType.FORM_8K,
        section_name="Item 5.02 Departure of Directors or Certain Officers",
        item_number="5.02",
        text=(
            "by the Compensation Committee, a participant must remain an employee through "
            "the payment date under the 2027 Plan to be eligible to earn an award. "
            "On March 2, 2026, the Compensation Committee adopted the Variable Compensation "
            "Plan for Fiscal Year 2027."
        ),
        token_count=36,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    packet = execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=RecordingRetriever([chunk]),
        model=StubSpecializedAgentModel(),
    )

    assert "disclosed: On March 2, 2026" in packet.findings[0].claim
    assert "disclosed that by the Compensation Committee" not in packet.findings[0].claim


def test_eight_k_planner_uses_item_type_query_template() -> None:
    request = AnalysisRequest(
        request_id="req-8k-query",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What recent leadership and material events were disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for query test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    chunk = ChunkRecord(
        chunk_id="eightk-query:chunk-1",
        document_id="doc-8k-query",
        section_id="section-8k-query",
        company=request.company,
        filing_type=FilingType.FORM_8K,
        section_name="Item 5.02 Departure of Directors or Certain Officers",
        item_number="5.02",
        text="Leadership transition evidence.",
        token_count=4,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever([chunk])

    execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=StubSpecializedAgentModel(),
    )

    query = retriever.requests[0]["query"]
    assert "item 5.02" in query
    assert "leadership board compensation" in query
    assert "item 2.02" in query
    assert "8-K" in query


def test_eight_k_agent_emits_separate_findings_per_item_number() -> None:
    request = AnalysisRequest(
        request_id="req-8k-items",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What recent material events were disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for item split test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever(
        [
            ChunkRecord(
                chunk_id="8k-item-1-01",
                document_id="doc-8k-items",
                section_id="section-1-01",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 1.01 Entry into a Material Definitive Agreement",
                item_number="1.01",
                text="The company entered into a material definitive agreement with a partner.",
                token_count=11,
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            ChunkRecord(
                chunk_id="8k-item-5-02",
                document_id="doc-8k-items",
                section_id="section-5-02",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text="The board approved a leadership compensation update.",
                token_count=9,
                parse_confidence=ConfidenceLabel.HIGH,
            ),
        ]
    )

    packet = execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=StubSpecializedAgentModel(),
    )

    assert len(packet.findings) == 2
    assert any("Item 1.01" in finding.claim for finding in packet.findings)
    assert any("Item 5.02" in finding.claim for finding in packet.findings)


def test_eight_k_agent_reasons_each_item_group_separately() -> None:
    request = AnalysisRequest(
        request_id="req-8k-item-reasoning",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What recent material events were disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for item reasoning test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever(
        [
            ChunkRecord(
                chunk_id="8k-item-1-01-a",
                document_id="doc-8k-reason",
                section_id="section-1-01-a",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 1.01 Entry into a Material Definitive Agreement",
                item_number="1.01",
                text="The company entered into a material definitive agreement with a partner.",
                token_count=11,
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            ChunkRecord(
                chunk_id="8k-item-1-01-b",
                document_id="doc-8k-reason",
                section_id="section-1-01-b",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 1.01 Entry into a Material Definitive Agreement",
                item_number="1.01",
                text="The agreement includes supply commitments.",
                token_count=6,
                parse_confidence=ConfidenceLabel.HIGH,
            ),
            ChunkRecord(
                chunk_id="8k-item-5-02",
                document_id="doc-8k-reason",
                section_id="section-5-02",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text="The board approved a leadership compensation update.",
                token_count=9,
                parse_confidence=ConfidenceLabel.HIGH,
            ),
        ]
    )
    model = RecordingItemModel()

    packet = execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=model,
    )

    assert model.calls == [
        ("1.01", ["8k-item-1-01-a", "8k-item-1-01-b"]),
        ("5.02", ["8k-item-5-02"]),
    ]
    assert packet.findings[0].summary == "Modeled summary for item 1.01"
    assert packet.findings[1].summary == "Modeled summary for item 5.02"


def test_eight_k_item_claim_fallback_uses_section_label_for_generic_summary() -> None:
    request = AnalysisRequest(
        request_id="req-8k-901",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What exhibits were disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for item 9.01 fallback test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever(
        [
            ChunkRecord(
                chunk_id="8k-item-9-01",
                document_id="doc-8k-901",
                section_id="section-9-01",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 9.01. Financial Statements and Exhibits.",
                item_number="9.01",
                text="Description\nVariable Compensation Plan - Fiscal Year 2027\n",
                token_count=7,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ]
    )
    model = RecordingItemModel()
    model.reason = lambda **kwargs: AgentReasoningDraft(  # type: ignore[method-assign]
        provisional_summary="NVIDIA Corp",
        unresolved_items=[],
        confidence_label=ConfidenceLabel.HIGH,
        reretrieval_requested=False,
    )

    packet = execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=model,
    )

    assert "Financial Statements and Exhibits" in packet.findings[0].claim
    assert '"Variable Compensation Plan - Fiscal Year 2027."' in packet.findings[0].claim
    assert (
        packet.findings[0].claim
        == (
            'Item 9.01 (Financial Statements and Exhibits) disclosed the exhibit '
            '"Variable Compensation Plan - Fiscal Year 2027."'
        )
    )


def test_eight_k_item_claim_normalizes_casing_and_limits_sentences() -> None:
    request = AnalysisRequest(
        request_id="req-8k-casing",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What recent material events were disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for casing test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever(
        [
            ChunkRecord(
                chunk_id="8k-item-5-02-casing",
                document_id="doc-8k-casing",
                section_id="section-5-02-casing",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text="Leadership compensation update evidence.",
                token_count=4,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ]
    )
    model = RecordingItemModel()
    model.reason = lambda **kwargs: AgentReasoningDraft(  # type: ignore[method-assign]
        provisional_summary=(
            "On March 2, 2026, NVIDIA disclosed a Variable Compensation Plan for Fiscal Year 2027. "
            "The filing details target award opportunities. "
            "A third sentence should be dropped."
        ),
        unresolved_items=[],
        confidence_label=ConfidenceLabel.HIGH,
        reretrieval_requested=False,
    )

    packet = execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=model,
    )

    claim = packet.findings[0].claim
    assert "nVIDIA" not in claim
    assert "A third sentence should be dropped" not in claim
    assert claim.count(".") <= 3
    assert "disclosed: On March 2, 2026, NVIDIA disclosed a Variable Compensation Plan" in claim


def test_eight_k_item_summary_prefers_sentence_start_over_mid_sentence_fragment() -> None:
    request = AnalysisRequest(
        request_id="req-8k-summary-sentence-start",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What recent material events were disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for 8-K summary sentence start test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever(
        [
            ChunkRecord(
                chunk_id="8k-item-5-02-summary-start",
                document_id="doc-8k-summary-start",
                section_id="section-5-02-summary-start",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text=(
                    "by the Compensation Committee, a participant must remain an employee "
                    "through the payment date under the 2027 Plan to be eligible to earn "
                    "an award. On March 2, 2026, the Compensation Committee adopted the "
                    "Variable Compensation Plan for Fiscal Year 2027."
                ),
                token_count=40,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ]
    )
    model = RecordingItemModel()
    model.reason = lambda **kwargs: AgentReasoningDraft(  # type: ignore[method-assign]
        provisional_summary=(
            "by the Compensation Committee, a participant must remain an employee "
            "through the payment date under the 2027 Plan to be eligible to earn an award."
        ),
        unresolved_items=[],
        confidence_label=ConfidenceLabel.HIGH,
        reretrieval_requested=False,
    )

    packet = execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=model,
    )

    assert packet.findings[0].summary.startswith("On March 2, 2026")
    assert not packet.findings[0].summary.startswith("by the Compensation Committee")


def test_eight_k_item_502_summary_is_compressed_for_readability() -> None:
    request = AnalysisRequest(
        request_id="req-8k-502-compressed-summary",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What recent material events were disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for 8-K summary compression test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever(
        [
            ChunkRecord(
                chunk_id="8k-item-5-02-compressed-summary",
                document_id="doc-8k-compressed-summary",
                section_id="section-5-02-compressed-summary",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text=(
                    "On March 2, 2026, the Compensation Committee of the Board of Directors, "
                    "or the Board, of NVIDIA Corporation, or the Company, adopted the Variable "
                    "Compensation Plan for Fiscal Year 2027, or the 2027 Plan, which provides "
                    "eligible executive officers the opportunity to earn a variable cash payment."
                ),
                token_count=46,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ]
    )
    model = RecordingItemModel()
    model.reason = lambda **kwargs: AgentReasoningDraft(  # type: ignore[method-assign]
        provisional_summary=(
            "On March 2, 2026, the Compensation Committee of the Board of Directors, or "
            "the Board, of NVIDIA Corporation, or the Company, adopted the Variable "
            "Compensation Plan for Fiscal Year 2027, or the 2027 Plan, which provides "
            "eligible executive officers the opportunity to earn a variable cash payment."
        ),
        unresolved_items=[],
        confidence_label=ConfidenceLabel.HIGH,
        reretrieval_requested=False,
    )

    packet = execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=model,
    )

    assert packet.findings[0].summary == (
        "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan for Fiscal Year "
        "2027 for eligible executive officers."
    )


def test_eight_k_item_502_claim_uses_compact_event_sentence() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    claim = _build_8k_item_claim(
        item_number="5.02",
        item_chunks=[
            ChunkRecord(
                chunk_id="8k-item-5-02-compact-claim",
                document_id="doc-8k-compact-claim",
                section_id="section-5-02-compact-claim",
                company=company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text=(
                    "On March 2, 2026, the Compensation Committee of NVIDIA adopted the "
                    "Variable Compensation Plan for Fiscal Year 2027."
                ),
                token_count=18,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ],
        item_summary=(
            "On March 2, 2026, the Compensation Committee of NVIDIA adopted the Variable "
            "Compensation Plan for Fiscal Year 2027."
        ),
    )

    assert claim == (
        "Item 5.02 disclosed: On March 2, 2026, NVIDIA adopted the Variable Compensation "
        "Plan for Fiscal Year 2027 for eligible executive officers."
    )


def test_eight_k_item_502_claim_ignores_garbled_summary_and_uses_chunk_sentence() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    claim = _build_8k_item_claim(
        item_number="5.02",
        item_chunks=[
            ChunkRecord(
                chunk_id="8k-item-5-02-garbled-claim",
                document_id="doc-8k-garbled-claim",
                section_id="section-5-02-garbled-claim",
                company=company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text=(
                    "On March 2, 2026, the Compensation Committee of NVIDIA adopted the "
                    "Variable Compensation Plan for Fiscal Year 2027."
                ),
                token_count=18,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ],
        item_summary="Kress Executive Vice President and Chief Financial Officer $1,500,000",
    )

    assert claim == (
        "Item 5.02 disclosed: On March 2, 2026, NVIDIA adopted the Variable Compensation "
        "Plan for Fiscal Year 2027 for eligible executive officers."
    )


def test_eight_k_item_claim_removes_unsupported_negative_follow_up() -> None:
    request = AnalysisRequest(
        request_id="req-8k-negative",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What recent material events were disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for unsupported negative test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever(
        [
            ChunkRecord(
                chunk_id="8k-item-9-01-negative",
                document_id="doc-8k-negative",
                section_id="section-9-01-negative",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 9.01. Financial Statements and Exhibits.",
                item_number="9.01",
                text="Description\nVariable Compensation Plan - Fiscal Year 2027\n",
                token_count=7,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ]
    )
    model = RecordingItemModel()
    model.reason = lambda **kwargs: AgentReasoningDraft(  # type: ignore[method-assign]
        provisional_summary=(
            "The provided 8-K filing dated March 6, 2026, disclosed the Variable "
            "Compensation Plan for Fiscal Year 2027 as an exhibit. "
            "No other material events or leadership changes were mentioned in the "
            "provided snippets."
        ),
        unresolved_items=[],
        confidence_label=ConfidenceLabel.HIGH,
        reretrieval_requested=False,
    )

    packet = execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=model,
    )

    claim = packet.findings[0].claim
    assert "No other material events" not in claim
    assert 'disclosed the exhibit "Variable Compensation Plan - Fiscal Year 2027."' in claim


def test_eight_k_item_901_summary_strips_boilerplate_and_uses_exhibit_description() -> None:
    request = AnalysisRequest(
        request_id="req-8k-901-summary",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        question="What exhibits were disclosed?",
        max_reretrievals=2,
    )
    task = RoutedTask(
        task_type=AnalysisTaskType.MATERIAL_EVENTS,
        agent_name=EIGHT_K_PROFILE.agent_name,
        filing_type=FilingType.FORM_8K,
        status=TaskRoutingStatus.ROUTED,
        reason="routed for item 9.01 summary cleanup test",
        document_available=True,
        parse_confidence=ConfidenceLabel.HIGH,
    )
    retriever = RecordingRetriever(
        [
            ChunkRecord(
                chunk_id="8k-item-9-01-summary",
                document_id="doc-8k-901-summary",
                section_id="section-9-01-summary",
                company=request.company,
                filing_type=FilingType.FORM_8K,
                section_name="Item 9.01. Financial Statements and Exhibits.",
                item_number="9.01",
                text=(
                    "(d) Exhibits\nExhibit Number\nDescription\n10.1\n"
                    "Variable Compensation Plan - Fiscal Year 2027\n104\n"
                    "The cover page of this Current Report on Form 8-K, formatted in inline "
                    "XBRL\nSIGNATURE\nPursuant to the requirements of the Securities Exchange "
                    "Act of 1934, the registrant has duly caused this report to be signed.\n"
                ),
                token_count=36,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ]
    )
    model = RecordingItemModel()
    model.reason = lambda **kwargs: AgentReasoningDraft(  # type: ignore[method-assign]
        provisional_summary=(
            "(d) Exhibits Exhibit Number Description 10.1 Variable Compensation Plan - "
            "Fiscal Year 2027 104 The cover page of this Current Report on Form 8-K "
            "SIGNATURE Pursuant to the requirements of the Securities Exchange Act of 1934."
        ),
        unresolved_items=[],
        confidence_label=ConfidenceLabel.HIGH,
        reretrieval_requested=False,
    )

    packet = execute_specialized_agent(
        profile=EIGHT_K_PROFILE,
        request=request,
        tasks=[task],
        retriever=retriever,
        model=model,
    )

    assert (
        packet.findings[0].summary
        == 'Exhibit filed: "Variable Compensation Plan - Fiscal Year 2027."'
    )
    assert "SIGNATURE" not in packet.findings[0].summary
    assert (
        packet.findings[0].evidence_refs[0].excerpt
        == 'Exhibit filed: "Variable Compensation Plan - Fiscal Year 2027."'
    )
