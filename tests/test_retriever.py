from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.config import get_settings
from app.domain import (
    ChunkRecord,
    Company,
    ConfidenceLabel,
    FilingType,
    UniverseSnapshot,
    UniverseSnapshotConstituent,
)
from app.graph.retrieval import IndexedChunkRetriever
from app.graph.subgraphs import AgentRetrievalRequest
from app.indexing import ChunkSearchFilters, FakeEmbeddingProvider
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.repositories import PgVectorChunkRepository, UniverseRepository


def configure_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "retriever.db"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    get_settings.cache_clear()


def seed_company_master(universe_repository: UniverseRepository) -> Company:
    constituent = UniverseSnapshotConstituent(
        snapshot_date=date(2024, 12, 31),
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA Corp.",
        exchange="NASDAQ",
        is_domestic_filer=True,
    )
    universe_repository.replace_snapshot(
        UniverseSnapshot(
            snapshot_date=constituent.snapshot_date,
            constituents=[constituent],
        )
    )
    return constituent.to_company()


def test_vector_repository_searches_with_metadata_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    chunk_texts = [
        "Revenue growth accelerated in the quarter due to data center demand.",
        "Question-and-answer session discussed margin pressure and supply constraints.",
    ]
    embeddings = embedding_provider.embed_texts(chunk_texts)
    chunks = [
        ChunkRecord(
            chunk_id="chunk-10q",
            document_id="doc-10q",
            section_id="section-10q",
            company=company,
            filing_type=FilingType.FORM_10Q,
            accession_number="0001045810-24-000101",
            filing_date=date(2024, 11, 20),
            report_period=date(2024, 10, 27),
            section_name=(
                "Management's Discussion and Analysis of Financial Condition "
                "and Results of Operations"
            ),
            text=chunk_texts[0],
            token_count=10,
            parse_confidence=ConfidenceLabel.HIGH,
            embedding=embeddings[0],
            embedding_key=embedding_provider.model_name,
        ),
        ChunkRecord(
            chunk_id="chunk-call",
            document_id="doc-call",
            section_id="section-call",
            company=company,
            filing_type=FilingType.EARNINGS_CALL,
            transcript_id="NVDA-2024-11-20-Q3FY2025",
            call_date=date(2024, 11, 20),
            fiscal_quarter="Q3FY2025",
            section_name="Question-and-Answer Session",
            text=chunk_texts[1],
            token_count=9,
            parse_confidence=ConfidenceLabel.MEDIUM,
            source_reliability=ConfidenceLabel.MEDIUM,
            embedding=embeddings[1],
            embedding_key=embedding_provider.model_name,
        ),
    ]

    repository.replace_document_chunks(document_id="doc-10q", chunks=[chunks[0]])
    repository.replace_document_chunks(document_id="doc-call", chunks=[chunks[1]])

    results = repository.search(
        query_embedding=embedding_provider.embed_texts(["revenue demand quarter"])[0],
        filters=ChunkSearchFilters(
            ticker="NVDA",
            filing_types=[FilingType.FORM_10Q],
        ),
        limit=2,
    )
    listed = repository.list_chunks(
        filters=ChunkSearchFilters(
            ticker="NVDA",
            filing_types=[FilingType.EARNINGS_CALL],
        )
    )

    assert results
    assert results[0].chunk.chunk_id == "chunk-10q"
    assert len(listed) == 1
    assert listed[0].chunk_id == "chunk-call"


def test_indexed_chunk_retriever_applies_section_aware_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        "Revenue growth accelerated in the quarter due to data center demand.",
        "Liquidity and capital resources remained strong with significant cash.",
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-1",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-mda",
                document_id="doc-1",
                section_id="section-mda",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-24-000101",
                filing_date=date(2024, 11, 20),
                report_period=date(2024, 10, 27),
                section_name=(
                    "Management's Discussion and Analysis of Financial Condition "
                    "and Results of Operations"
                ),
                text=texts[0],
                token_count=10,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-liquidity",
                document_id="doc-1",
                section_id="section-liquidity",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-24-000101",
                filing_date=date(2024, 11, 20),
                report_period=date(2024, 10, 27),
                section_name="Liquidity and Capital Resources",
                text=texts[1],
                token_count=9,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    request: AgentRetrievalRequest = {
        "query": "liquidity cash resources",
        "filters": ChunkSearchFilters(
            ticker="NVDA",
            filing_types=[FilingType.FORM_10Q],
        ),
        "section_names": ("Liquidity and Capital Resources",),
        "transcript_segment_types": (),
        "limit": 3,
    }
    results = retriever.retrieve(request)

    assert results
    assert [chunk.chunk_id for chunk in results] == ["chunk-liquidity"]


def test_indexed_chunk_retriever_prefers_more_recent_filings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        "Liquidity and capital resources remained strong with significant cash.",
        "Liquidity and capital resources remained strong with significant cash.",
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-old",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-old",
                document_id="doc-old",
                section_id="section-old",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-23-000001",
                filing_date=date(2023, 8, 20),
                report_period=date(2023, 7, 30),
                section_name="Liquidity and Capital Resources",
                text=texts[0],
                token_count=9,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            )
        ],
    )
    repository.replace_document_chunks(
        document_id="doc-new",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-new",
                document_id="doc-new",
                section_id="section-new",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-24-000001",
                filing_date=date(2024, 11, 20),
                report_period=date(2024, 10, 27),
                section_name="Liquidity and Capital Resources",
                text=texts[1],
                token_count=9,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            )
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "liquidity cash resources",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10Q],
            ),
            "section_names": ("Liquidity and Capital Resources",),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-new"]


def test_indexed_chunk_retriever_filters_fragmentary_10q_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        (
            "any investment will be completed on expected terms, if at all. Refer to Item 1A. "
            "Risk Factors for additional information regarding our investments."
        ),
        (
            "As of October 26, 2025, liquidity and capital resources remained strong, with no "
            "material changes outside the obligations described in Note 11."
        ),
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-fragment-vs-narrative",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-fragment",
                document_id="doc-fragment-vs-narrative",
                section_id="section-fragment",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Liquidity and Capital Resources",
                text=texts[0],
                token_count=20,
                parse_confidence=ConfidenceLabel.MEDIUM,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-narrative",
                document_id="doc-fragment-vs-narrative",
                section_id="section-narrative",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Liquidity and Capital Resources",
                text=texts[1],
                token_count=23,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "latest quarter liquidity capital resources",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10Q],
            ),
            "section_names": ("Liquidity and Capital Resources",),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-narrative"]


def test_indexed_chunk_retriever_filters_generic_10k_risk_heading_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        (
            "The following risk factors should be considered in addition to the other information "
            "in this Annual Report on Form 10-K."
        ),
        (
            "Our ability to meet evolving industry needs and manage long manufacturing lead times "
            "could pressure results if demand and supply timing diverge."
        ),
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-10k-risk",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-risk-heading",
                document_id="doc-10k-risk",
                section_id="section-risk-heading",
                company=company,
                filing_type=FilingType.FORM_10K,
                accession_number="0001045810-26-000021",
                filing_date=date(2026, 2, 25),
                report_period=date(2026, 1, 25),
                section_name="Risk Factors",
                text=texts[0],
                token_count=18,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-risk-narrative",
                document_id="doc-10k-risk",
                section_id="section-risk-narrative",
                company=company,
                filing_type=FilingType.FORM_10K,
                accession_number="0001045810-26-000021",
                filing_date=date(2026, 2, 25),
                report_period=date(2026, 1, 25),
                section_name="Risk Factors",
                text=texts[1],
                token_count=21,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "long-term business risk demand supply constraints",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10K],
            ),
            "section_names": ("Risk Factors",),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-risk-narrative"]


def test_indexed_chunk_retriever_penalizes_table_like_10q_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        (
            "Oct 26, 2025 Jan 26, 2025 (In millions) Cash and cash equivalents $ 11,486 "
            "$ 8,589 Marketable securities 49,122 34,621 Cash, cash equivalents and "
            "marketable securities $ 60,608 $ 43,210."
        ),
        (
            "As of October 26, 2025, cash and marketable securities remained available to "
            "support operating needs and planned capital expenditures."
        ),
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-10q-table",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-table",
                document_id="doc-10q-table",
                section_id="section-table",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Liquidity and Capital Resources",
                text=texts[0],
                token_count=30,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-narrative-liquidity",
                document_id="doc-10q-table",
                section_id="section-narrative-liquidity",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Liquidity and Capital Resources",
                text=texts[1],
                token_count=20,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "latest quarter liquidity cash capital resources",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10Q],
            ),
            "section_names": ("Liquidity and Capital Resources",),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-narrative-liquidity"]


def test_indexed_chunk_retriever_prefers_ten_q_balance_chunks_over_hypothetical_sentences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        (
            "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, and "
            "marketable securities."
        ),
        (
            "A hypothetical 10% decrease in our publicly-held equity securities would "
            "decrease the fair value of the publicly-held equity securities balance."
        ),
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-10q-liquidity-balance",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-balance-sentence",
                document_id="doc-10q-liquidity-balance",
                section_id="section-balance-sentence",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Liquidity and Capital Resources",
                text=texts[0],
                token_count=17,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-hypothetical-balance",
                document_id="doc-10q-liquidity-balance",
                section_id="section-hypothetical-balance",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Liquidity and Capital Resources",
                text=texts[1],
                token_count=20,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "latest quarter liquidity cash capital resources",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10Q],
            ),
            "section_names": ("Liquidity and Capital Resources",),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-balance-sentence"]


def test_indexed_chunk_retriever_excludes_ten_q_market_risk_chunks_for_liquidity_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        (
            "As of October 26, 2025, liquidity and capital resources remained strong, with "
            "cash and marketable securities available to support operating needs."
        ),
        (
            "Quantitative and Qualitative Disclosures about Market Risk. A hypothetical "
            "10% decrease in fair value would reduce the balance."
        ),
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-10q-market-risk-filter",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-liquidity-balance",
                document_id="doc-10q-market-risk-filter",
                section_id="section-liquidity-balance",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Liquidity and Capital Resources",
                text=texts[0],
                token_count=22,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-market-risk",
                document_id="doc-10q-market-risk-filter",
                section_id="section-market-risk",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Item 3. Quantitative and Qualitative Disclosures about Market Risk",
                text=texts[1],
                token_count=18,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "latest quarter liquidity cash capital resources",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10Q],
            ),
            "section_names": ("Liquidity and Capital Resources",),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-liquidity-balance"]


def test_indexed_chunk_retriever_excludes_ten_q_event_style_chunks_for_liquidity_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        (
            "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, and "
            "marketable securities."
        ),
        (
            "In November 2025, we entered into an agreement, subject to certain closing "
            "conditions, to invest up to $10 billion in Anthropic."
        ),
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-10q-event-style-filter",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-liquidity-balance",
                document_id="doc-10q-event-style-filter",
                section_id="section-liquidity-balance",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Liquidity and Capital Resources",
                text=texts[0],
                token_count=17,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-event-style",
                document_id="doc-10q-event-style-filter",
                section_id="section-event-style",
                company=company,
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filing_date=date(2025, 11, 19),
                report_period=date(2025, 10, 26),
                section_name="Liquidity and Capital Resources",
                text=texts[1],
                token_count=22,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "latest quarter liquidity cash capital resources",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10Q],
            ),
            "section_names": ("Liquidity and Capital Resources",),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-liquidity-balance"]


def test_indexed_chunk_retriever_prefers_business_section_for_10k_structure_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        (
            "NVIDIA serves gaming, data center, professional visualization, and automotive "
            "end markets through accelerated computing platforms."
        ),
        (
            "Competition, supply constraints, and product transition risks could pressure "
            "results if demand timing changes."
        ),
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-10k-sections",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-business",
                document_id="doc-10k-sections",
                section_id="section-business",
                company=company,
                filing_type=FilingType.FORM_10K,
                accession_number="0001045810-26-000021",
                filing_date=date(2026, 2, 25),
                report_period=date(2026, 1, 25),
                section_name="Item 1. Business",
                text=texts[0],
                token_count=17,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-risk",
                document_id="doc-10k-sections",
                section_id="section-risk",
                company=company,
                filing_type=FilingType.FORM_10K,
                accession_number="0001045810-26-000021",
                filing_date=date(2026, 2, 25),
                report_period=date(2026, 1, 25),
                section_name="Item 1A. Risk Factors",
                text=texts[1],
                token_count=15,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "NVDA long term structure business model products segments 10-K",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10K],
            ),
            "section_names": (
                "Item 1. Business",
                "Item 1A. Risk Factors",
                "Item 7. Management's Discussion and Analysis",
            ),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-business", "chunk-risk"]


def test_indexed_chunk_retriever_penalizes_regional_anecdotes_for_10k_structure_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        (
            "We have engaged with customers in China to provide alternative products not "
            "subject to the new license requirements."
        ),
        (
            "NVIDIA serves gaming, data center, professional visualization, and automotive "
            "end markets through accelerated computing platforms."
        ),
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-10k-business-rank",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-regional",
                document_id="doc-10k-business-rank",
                section_id="section-regional",
                company=company,
                filing_type=FilingType.FORM_10K,
                accession_number="0001045810-26-000021",
                filing_date=date(2026, 2, 25),
                report_period=date(2026, 1, 25),
                section_name="Item 1. Business",
                text=texts[0],
                token_count=18,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-structure",
                document_id="doc-10k-business-rank",
                section_id="section-structure",
                company=company,
                filing_type=FilingType.FORM_10K,
                accession_number="0001045810-26-000021",
                filing_date=date(2026, 2, 25),
                report_period=date(2026, 1, 25),
                section_name="Item 1. Business",
                text=texts[1],
                token_count=16,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "NVDA long term structure business model products segments 10-K",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10K],
            ),
            "section_names": ("Item 1. Business",),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-structure", "chunk-regional"]


def test_indexed_chunk_retriever_prefers_latest_10k_filing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        "NVIDIA is a data center scale AI infrastructure company with a broad technology stack.",
        "NVIDIA reports business results in two segments.",
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-10k-latest-filter",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-10k-old",
                document_id="doc-10k-latest-filter",
                section_id="section-10k-old",
                company=company,
                filing_type=FilingType.FORM_10K,
                accession_number="0001045810-25-000023",
                filing_date=date(2025, 2, 26),
                report_period=date(2025, 1, 26),
                section_name="Item 1. Business",
                text=texts[1],
                token_count=8,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-10k-latest",
                document_id="doc-10k-latest-filter",
                section_id="section-10k-latest",
                company=company,
                filing_type=FilingType.FORM_10K,
                accession_number="0001045810-26-000021",
                filing_date=date(2026, 2, 25),
                report_period=date(2026, 1, 25),
                section_name="Item 1. Business",
                text=texts[0],
                token_count=13,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "NVDA long term structure business model platforms 10-K",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_10K],
            ),
            "section_names": ("Item 1. Business",),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-10k-latest"]


def test_indexed_chunk_retriever_ranks_8k_item_specific_chunks_higher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        "The company disclosed a new material definitive agreement with a strategic partner.",
        "The board approved compensation changes and disclosed an officer transition.",
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-8k-items",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-1-01",
                document_id="doc-8k-items",
                section_id="section-1-01",
                company=company,
                filing_type=FilingType.FORM_8K,
                accession_number="0001045810-24-000201",
                filing_date=date(2024, 12, 1),
                section_name="Item 1.01 Entry into a Material Definitive Agreement",
                item_number="1.01",
                text=texts[0],
                token_count=11,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-5-02",
                document_id="doc-8k-items",
                section_id="section-5-02",
                company=company,
                filing_type=FilingType.FORM_8K,
                accession_number="0001045810-24-000201",
                filing_date=date(2024, 12, 1),
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text=texts[1],
                token_count=11,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": "NVDA recent material events item 5.02 leadership board compensation 8-K",
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_8K],
            ),
            "section_names": (),
            "transcript_segment_types": (),
            "limit": 2,
        }
    )

    assert [chunk.chunk_id for chunk in results] == ["chunk-5-02", "chunk-1-01"]


def test_indexed_chunk_retriever_diversifies_8k_items_before_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    company = seed_company_master(universe_repository)
    repository = PgVectorChunkRepository(session_factory)
    embedding_provider = FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions)
    texts = [
        "The board approved a leadership compensation update.",
        "The board approved another leadership compensation update.",
        "The company entered into a material definitive agreement with a partner.",
    ]
    embeddings = embedding_provider.embed_texts(texts)
    repository.replace_document_chunks(
        document_id="doc-8k-diverse",
        chunks=[
            ChunkRecord(
                chunk_id="chunk-5-02-a",
                document_id="doc-8k-diverse",
                section_id="section-5-02-a",
                company=company,
                filing_type=FilingType.FORM_8K,
                accession_number="0001045810-24-000202",
                filing_date=date(2024, 12, 2),
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text=texts[0],
                token_count=8,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[0],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-5-02-b",
                document_id="doc-8k-diverse",
                section_id="section-5-02-b",
                company=company,
                filing_type=FilingType.FORM_8K,
                accession_number="0001045810-24-000202",
                filing_date=date(2024, 12, 2),
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                item_number="5.02",
                text=texts[1],
                token_count=9,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[1],
                embedding_key=embedding_provider.model_name,
            ),
            ChunkRecord(
                chunk_id="chunk-1-01",
                document_id="doc-8k-diverse",
                section_id="section-1-01",
                company=company,
                filing_type=FilingType.FORM_8K,
                accession_number="0001045810-24-000202",
                filing_date=date(2024, 12, 2),
                section_name="Item 1.01 Entry into a Material Definitive Agreement",
                item_number="1.01",
                text=texts[2],
                token_count=11,
                parse_confidence=ConfidenceLabel.HIGH,
                embedding=embeddings[2],
                embedding_key=embedding_provider.model_name,
            ),
        ],
    )
    retriever = IndexedChunkRetriever(
        vector_repository=repository,
        embedding_provider=embedding_provider,
    )

    results = retriever.retrieve(
        {
            "query": (
                "NVDA recent material events item 5.02 leadership board compensation "
                "item 1.01 agreement 8-K"
            ),
            "filters": ChunkSearchFilters(
                ticker="NVDA",
                filing_types=[FilingType.FORM_8K],
            ),
            "section_names": (),
            "transcript_segment_types": (),
            "limit": 3,
        }
    )

    assert results[0].item_number == "5.02"
    assert results[1].item_number == "1.01"
