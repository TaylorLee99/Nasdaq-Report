from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.config import get_settings
from app.domain import (
    ConfidenceLabel,
    CoverageLabel,
    CoverageStatus,
    DocumentSectionType,
    DownloadStatusLabel,
    FilingType,
    ParsedSection,
    SecRawDocumentRecord,
    SecSubmissionMetadata,
    TranscriptMetadata,
    TranscriptSegmentType,
    TranscriptSourceType,
    UniverseSnapshot,
    UniverseSnapshotConstituent,
)
from app.indexing import (
    ChunkingConfig,
    ChunkSearchFilters,
    FakeEmbeddingProvider,
    RetrievalIndexingService,
    SectionChunker,
)
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.raw_store import LocalRawDocumentStore
from app.storage.repositories import (
    PgVectorChunkRepository,
    SecMetadataRepository,
    SecRawDocumentRepository,
    TranscriptRepository,
    UniverseRepository,
)

FILING_FIXTURE = Path("tests/fixtures/parser_10q.txt")
TRANSCRIPT_FIXTURE = Path("tests/fixtures/parser_transcript.txt")


def configure_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "indexing.db"
    raw_store_path = tmp_path / "raw-store"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    monkeypatch.setenv("APP_STORAGE__RAW_STORE_DIR", str(raw_store_path))
    monkeypatch.setenv("APP_INDEXING__CHUNK_SIZE_CHARS", "120")
    monkeypatch.setenv("APP_INDEXING__CHUNK_OVERLAP_CHARS", "20")
    get_settings.cache_clear()


def build_repositories() -> tuple[
    UniverseRepository,
    SecMetadataRepository,
    SecRawDocumentRepository,
    TranscriptRepository,
    PgVectorChunkRepository,
    LocalRawDocumentStore,
]:
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    return (
        UniverseRepository(session_factory),
        SecMetadataRepository(session_factory),
        SecRawDocumentRepository(session_factory),
        TranscriptRepository(session_factory),
        PgVectorChunkRepository(session_factory),
        LocalRawDocumentStore(settings.storage.raw_store_dir),
    )


def seed_company_master(universe_repository: UniverseRepository) -> None:
    universe_repository.replace_snapshot(
        UniverseSnapshot(
            snapshot_date=date(2024, 12, 31),
            constituents=[
                UniverseSnapshotConstituent(
                    snapshot_date=date(2024, 12, 31),
                    ticker="NVDA",
                    cik="0001045810",
                    company_name="NVIDIA Corp.",
                    exchange="NASDAQ",
                    is_domestic_filer=True,
                )
            ],
        )
    )


def build_service(
    universe_repository: UniverseRepository,
    sec_metadata_repository: SecMetadataRepository,
    sec_raw_document_repository: SecRawDocumentRepository,
    transcript_repository: TranscriptRepository,
    vector_repository: PgVectorChunkRepository,
    raw_store: LocalRawDocumentStore,
) -> RetrievalIndexingService:
    return RetrievalIndexingService(
        universe_repository=universe_repository,
        sec_metadata_repository=sec_metadata_repository,
        sec_raw_document_repository=sec_raw_document_repository,
        transcript_repository=transcript_repository,
        vector_repository=vector_repository,
        raw_document_store=raw_store,
        embedding_provider=FakeEmbeddingProvider(
            dimensions=get_settings().indexing.embedding_dimensions
        ),
        chunking_config=ChunkingConfig(
            chunk_size_chars=get_settings().indexing.chunk_size_chars,
            chunk_overlap_chars=get_settings().indexing.chunk_overlap_chars,
        ),
    )


def seed_filing(
    sec_metadata_repository: SecMetadataRepository,
    sec_raw_document_repository: SecRawDocumentRepository,
    raw_store: LocalRawDocumentStore,
) -> None:
    filing = SecSubmissionMetadata(
        cik="0001045810",
        ticker="NVDA",
        accession_number="0001045810-24-000123",
        filing_date=date(2024, 11, 20),
        form_type=FilingType.FORM_10Q,
        primary_document="nvda-10q.txt",
        report_period=date(2024, 10, 27),
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000123/nvda-10q.txt",
    )
    sec_metadata_repository.upsert_filings(filings=[filing], cik=filing.cik)
    stored_path = raw_store.save(
        "sec/NVDA/10-Q/2024-11-20/0001045810-24-000123/nvda-10q.txt",
        FILING_FIXTURE.read_bytes(),
    )
    sec_raw_document_repository.upsert_download_record(
        SecRawDocumentRecord(
            accession_number=filing.accession_number,
            cik=filing.cik,
            ticker=filing.ticker,
            form_type=filing.form_type,
            source_url=filing.source_url,
            storage_path=str(stored_path),
            checksum_sha256="fixture",
            content_type="text/plain; charset=utf-8",
            size_bytes=len(FILING_FIXTURE.read_bytes()),
            status=DownloadStatusLabel.DOWNLOADED,
            downloaded_at=datetime.now(UTC),
            retry_count=1,
        )
    )


def seed_transcript(
    transcript_repository: TranscriptRepository,
    raw_store: LocalRawDocumentStore,
) -> None:
    stored_path = raw_store.save(
        "transcripts/NVDA/2024-11-20/Q3FY2025/NVDA-2024-11-20-Q3FY2025.txt",
        TRANSCRIPT_FIXTURE.read_bytes(),
    )
    transcript_repository.upsert_transcript(
        TranscriptMetadata(
            transcript_id="NVDA-2024-11-20-Q3FY2025",
            company=UniverseSnapshotConstituent(
                snapshot_date=date(2024, 12, 31),
                ticker="NVDA",
                cik="0001045810",
                company_name="NVIDIA Corp.",
                exchange="NASDAQ",
                is_domestic_filer=True,
            ).to_company(),
            call_date=date(2024, 11, 20),
            fiscal_quarter="Q3FY2025",
            source_url="https://example.com/nvda-q3fy2025-transcript",
            speaker_separated=True,
            source_reliability=ConfidenceLabel.MEDIUM,
            source_type=TranscriptSourceType.FILE,
            availability_coverage=CoverageStatus(
                label=CoverageLabel.COMPLETE,
                covered_topics=["transcript"],
            ),
            storage_path=str(stored_path),
            checksum_sha256="fixture",
            content_type="text/plain; charset=utf-8",
            size_bytes=len(TRANSCRIPT_FIXTURE.read_bytes()),
        )
    )


def test_indexing_service_indexes_filings_and_supports_metadata_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    (
        universe_repository,
        sec_metadata_repository,
        sec_raw_document_repository,
        transcript_repository,
        vector_repository,
        raw_store,
    ) = build_repositories()
    seed_company_master(universe_repository)
    seed_filing(sec_metadata_repository, sec_raw_document_repository, raw_store)
    runner = CliRunner()
    cli_result = runner.invoke(
        app,
        ["index", "filing", "--ticker", "NVDA", "--forms", "10-Q"],
    )
    service = build_service(
        universe_repository,
        sec_metadata_repository,
        sec_raw_document_repository,
        transcript_repository,
        vector_repository,
        raw_store,
    )
    listed = vector_repository.list_chunks(
        filters=ChunkSearchFilters(
            ticker="NVDA",
            filing_types=[FilingType.FORM_10Q],
            section_name="Liquidity and Capital Resources",
        )
    )
    search_results = service.search(
        query="cash flow and liquidity",
        filters=ChunkSearchFilters(
            ticker="NVDA",
            filing_types=[FilingType.FORM_10Q],
            section_name="Liquidity and Capital Resources",
        ),
        limit=3,
    )

    assert cli_result.exit_code == 0
    assert '"indexed_documents": 1' in cli_result.stdout
    assert listed
    assert all(chunk.section_name == "Liquidity and Capital Resources" for chunk in listed)
    assert search_results
    assert search_results[0].chunk.report_period == date(2024, 10, 27)
    assert search_results[0].chunk.source_reliability == ConfidenceLabel.HIGH
    get_settings.cache_clear()


def test_indexing_service_supports_legacy_root_relative_storage_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    (
        universe_repository,
        sec_metadata_repository,
        sec_raw_document_repository,
        transcript_repository,
        vector_repository,
        raw_store,
    ) = build_repositories()
    seed_company_master(universe_repository)

    filing = SecSubmissionMetadata(
        cik="0001045810",
        ticker="NVDA",
        accession_number="0001045810-24-000124",
        filing_date=date(2024, 11, 21),
        form_type=FilingType.FORM_10Q,
        primary_document="nvda-10q-legacy.txt",
        report_period=date(2024, 10, 27),
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000124/nvda-10q.txt",
    )
    sec_metadata_repository.upsert_filings(filings=[filing], cik=filing.cik)
    relative_path = "sec/NVDA/10-Q/2024-11-21/0001045810-24-000124/nvda-10q-legacy.txt"
    stored_path = raw_store.save(relative_path, FILING_FIXTURE.read_bytes())
    root_relative_path = os.path.relpath(stored_path, Path.cwd())
    sec_raw_document_repository.upsert_download_record(
        SecRawDocumentRecord(
            accession_number=filing.accession_number,
            cik=filing.cik,
            ticker=filing.ticker,
            form_type=filing.form_type,
            source_url=filing.source_url,
            storage_path=root_relative_path,
            checksum_sha256="fixture",
            content_type="text/plain; charset=utf-8",
            size_bytes=len(FILING_FIXTURE.read_bytes()),
            status=DownloadStatusLabel.DOWNLOADED,
            downloaded_at=datetime.now(UTC),
            retry_count=1,
        )
    )

    service = build_service(
        universe_repository,
        sec_metadata_repository,
        sec_raw_document_repository,
        transcript_repository,
        vector_repository,
        raw_store,
    )
    result = service.index_filings(ticker="NVDA", forms=[FilingType.FORM_10Q])

    assert result.indexed_documents == 1
    assert result.indexed_chunks >= 1
    get_settings.cache_clear()


def test_index_transcript_cli_indexes_chunks_and_searches_qa_segment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    (
        universe_repository,
        sec_metadata_repository,
        sec_raw_document_repository,
        transcript_repository,
        vector_repository,
        raw_store,
    ) = build_repositories()
    seed_company_master(universe_repository)
    seed_transcript(transcript_repository, raw_store)
    runner = CliRunner()

    result = runner.invoke(app, ["index", "transcript", "--ticker", "NVDA"])

    assert result.exit_code == 0
    assert '"indexed_documents": 1' in result.stdout

    service = build_service(
        universe_repository,
        sec_metadata_repository,
        sec_raw_document_repository,
        transcript_repository,
        vector_repository,
        raw_store,
    )
    search_results = service.search(
        query="supply outlook",
        filters=ChunkSearchFilters(
            ticker="NVDA",
            filing_types=[FilingType.EARNINGS_CALL],
            section_name="Question-and-Answer Session",
            transcript_segment_type=TranscriptSegmentType.QA,
        ),
        limit=3,
    )

    assert search_results
    assert search_results[0].chunk.transcript_id == "NVDA-2024-11-20-Q3FY2025"
    assert search_results[0].chunk.transcript_segment_type == TranscriptSegmentType.QA
    assert search_results[0].chunk.source_reliability == ConfidenceLabel.MEDIUM
    get_settings.cache_clear()


def test_section_chunker_skips_table_like_10q_blocks() -> None:
    company = UniverseSnapshotConstituent(
        snapshot_date=date(2024, 12, 31),
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA Corp.",
        exchange="NASDAQ",
        is_domestic_filer=True,
    ).to_company()
    section = ParsedSection(
        section_id="doc-1:section:1",
        document_id="doc-1",
        company=company,
        filing_type=FilingType.FORM_10Q,
        heading="Liquidity and Capital Resources",
        text=(
            "Oct 26, 2025 Jan 26, 2025 (In millions) Cash and cash equivalents $ 11,486 "
            "$ 8,589 Marketable securities 49,122 34,621 Cash, cash equivalents and "
            "marketable securities $ 60,608 $ 43,210."
        ),
        ordinal=1,
        section_type=DocumentSectionType.LIQUIDITY,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    chunker = SectionChunker(ChunkingConfig(chunk_size_chars=400, chunk_overlap_chars=40))
    spans = chunker.chunk_section(section)

    assert spans == []


def test_section_chunker_skips_item_heading_fragments_for_10q() -> None:
    company = UniverseSnapshotConstituent(
        snapshot_date=date(2024, 12, 31),
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA Corp.",
        exchange="NASDAQ",
        is_domestic_filer=True,
    ).to_company()
    section = ParsedSection(
        section_id="doc-2:section:1",
        document_id="doc-2",
        company=company,
        filing_type=FilingType.FORM_10Q,
        heading=(
            "Management's Discussion and Analysis of Financial Condition and Results "
            "of Operations"
        ),
        text="25 Item 3. Quantitative and Qualitative Disclosures About Market Risk.",
        ordinal=1,
        section_type=DocumentSectionType.MDA,
        parse_confidence=ConfidenceLabel.MEDIUM,
    )

    chunker = SectionChunker(ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=20))
    spans = chunker.chunk_section(section)

    assert spans == []


def test_section_chunker_keeps_overlap_on_word_boundaries() -> None:
    company = UniverseSnapshotConstituent(
        snapshot_date=date(2024, 12, 31),
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA Corp.",
        exchange="NASDAQ",
        is_domestic_filer=True,
    ).to_company()
    section = ParsedSection(
        section_id="doc-3:section:1",
        document_id="doc-3",
        company=company,
        filing_type=FilingType.FORM_10Q,
        heading="Liquidity and Capital Resources",
        text=(
            "We continuously evaluate our liquidity and capital resources to ensure the "
            "company can finance future requirements and maintain flexibility."
        ),
        ordinal=1,
        section_type=DocumentSectionType.LIQUIDITY,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    chunker = SectionChunker(ChunkingConfig(chunk_size_chars=90, chunk_overlap_chars=20))
    spans = chunker.chunk_section(section)

    assert len(spans) >= 2
    assert not spans[1].text.startswith("ously")
    assert spans[1].text.startswith("the company can finance")


def test_section_chunker_skips_numeric_only_residue_for_10q() -> None:
    company = UniverseSnapshotConstituent(
        snapshot_date=date(2024, 12, 31),
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA Corp.",
        exchange="NASDAQ",
        is_domestic_filer=True,
    ).to_company()
    section = ParsedSection(
        section_id="doc-4:section:1",
        document_id="doc-4",
        company=company,
        filing_type=FilingType.FORM_10Q,
        heading=(
            "Management's Discussion and Analysis of Financial Condition and Results "
            "of Operations"
        ),
        text="25",
        ordinal=1,
        section_type=DocumentSectionType.MDA,
        parse_confidence=ConfidenceLabel.MEDIUM,
    )

    chunker = SectionChunker(ChunkingConfig(chunk_size_chars=50, chunk_overlap_chars=10))
    spans = chunker.chunk_section(section)

    assert spans == []


def test_section_chunker_skips_accounting_pronouncement_fragment_for_10q() -> None:
    company = UniverseSnapshotConstituent(
        snapshot_date=date(2024, 12, 31),
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA Corp.",
        exchange="NASDAQ",
        is_domestic_filer=True,
    ).to_company()
    section = ParsedSection(
        section_id="doc-5:section:1",
        document_id="doc-5",
        company=company,
        filing_type=FilingType.FORM_10Q,
        heading="Liquidity and Capital Resources",
        text=(
            "Adoption of New and Recently Issued Accounting Pronouncements\n"
            "There has been no adoption of any new accounting pronouncements."
        ),
        ordinal=1,
        section_type=DocumentSectionType.LIQUIDITY,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    chunker = SectionChunker(ChunkingConfig(chunk_size_chars=120, chunk_overlap_chars=20))
    spans = chunker.chunk_section(section)

    assert spans == []


def test_section_chunker_skips_significant_accounting_policy_fragment_for_10q() -> None:
    company = UniverseSnapshotConstituent(
        snapshot_date=date(2024, 12, 31),
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        exchange="NASDAQ",
        is_domestic_filer=True,
    ).to_company()
    section = ParsedSection(
        section_id="doc-6:section:1",
        document_id="doc-6",
        company=company,
        filing_type=FilingType.FORM_10Q,
        heading="Liquidity and Capital Resources",
        text=(
            "Note 1, Summary of Significant Accounting Policies, describes the significant "
            "accounting policies and methods used in the preparation of the Company’s "
            "condensed consolidated financial statements."
        ),
        ordinal=1,
        section_type=DocumentSectionType.LIQUIDITY,
        parse_confidence=ConfidenceLabel.HIGH,
    )

    chunker = SectionChunker(ChunkingConfig(chunk_size_chars=220, chunk_overlap_chars=20))
    spans = chunker.chunk_section(section)

    assert spans == []
