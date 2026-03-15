from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from app.application.service import AnalysisApplicationService
from app.config import get_settings
from app.data_sources import TranscriptIngestionService
from app.domain import (
    AnalysisRunRequest,
    ConfidenceLabel,
    DownloadStatusLabel,
    FilingType,
    SecRawDocumentRecord,
    SecSubmissionMetadata,
    UniverseSnapshot,
    UniverseSnapshotConstituent,
)
from app.indexing import ChunkingConfig, FakeEmbeddingProvider, RetrievalIndexingService
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.raw_store import LocalRawDocumentStore
from app.storage.repositories import (
    AnalysisJobRepository,
    PgVectorChunkRepository,
    SecMetadataRepository,
    SecRawDocumentRepository,
    TranscriptRepository,
    UniverseRepository,
)


def configure_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "e2e-smoke.db"
    raw_store_path = tmp_path / "raw-store"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    monkeypatch.setenv("APP_STORAGE__RAW_STORE_DIR", str(raw_store_path))
    monkeypatch.setenv("APP_INDEXING__CHUNK_SIZE_CHARS", "120")
    monkeypatch.setenv("APP_INDEXING__CHUNK_OVERLAP_CHARS", "20")
    get_settings.cache_clear()


def test_sample_data_end_to_end_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_10q_path: Path,
    sample_transcript_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    sec_metadata_repository = SecMetadataRepository(session_factory)
    sec_raw_document_repository = SecRawDocumentRepository(session_factory)
    transcript_repository = TranscriptRepository(session_factory)
    analysis_job_repository = AnalysisJobRepository(session_factory)
    vector_repository = PgVectorChunkRepository(session_factory)
    raw_store = LocalRawDocumentStore(settings.storage.raw_store_dir)

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
    filing = SecSubmissionMetadata(
        cik="0001045810",
        ticker="NVDA",
        accession_number="0001045810-24-000123",
        filing_date=date(2024, 11, 20),
        form_type=FilingType.FORM_10Q,
        primary_document="sample_10q.txt",
        report_period=date(2024, 10, 27),
        source_url="https://www.sec.gov/Archives/example/sample_10q.txt",
    )
    sec_metadata_repository.upsert_filings(filings=[filing], cik=filing.cik)
    stored_path = raw_store.save(
        "sec/NVDA/10-Q/2024-11-20/0001045810-24-000123/sample_10q.txt",
        sample_10q_path.read_bytes(),
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
            size_bytes=len(sample_10q_path.read_bytes()),
            status=DownloadStatusLabel.DOWNLOADED,
            downloaded_at=datetime.now(UTC),
            retry_count=1,
        )
    )

    transcript_service = TranscriptIngestionService(
        transcript_repository=transcript_repository,
        universe_repository=universe_repository,
        raw_document_store=raw_store,
    )
    transcript_service.import_transcript(
        ticker="NVDA",
        file_path=sample_transcript_path,
        call_date=date(2024, 11, 20),
        fiscal_quarter="Q3FY2025",
        source_reliability=ConfidenceLabel.MEDIUM,
        speaker_separated=True,
    )

    indexing_service = RetrievalIndexingService(
        universe_repository=universe_repository,
        sec_metadata_repository=sec_metadata_repository,
        sec_raw_document_repository=sec_raw_document_repository,
        transcript_repository=transcript_repository,
        vector_repository=vector_repository,
        raw_document_store=raw_store,
        embedding_provider=FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions),
        chunking_config=ChunkingConfig(
            chunk_size_chars=settings.indexing.chunk_size_chars,
            chunk_overlap_chars=settings.indexing.chunk_overlap_chars,
        ),
    )
    filing_index_result = indexing_service.index_filings(ticker="NVDA", forms=[FilingType.FORM_10Q])
    transcript_index_result = indexing_service.index_transcripts(ticker="NVDA")

    analysis_service = AnalysisApplicationService(
        universe_repository=universe_repository,
        sec_metadata_repository=sec_metadata_repository,
        transcript_repository=transcript_repository,
        analysis_job_repository=analysis_job_repository,
    )
    job = analysis_service.run_analysis(
        AnalysisRunRequest(
            ticker="NVDA",
            question="Build a disclosure-grounded thesis from sample fixtures.",
            include_transcript=True,
            as_of_date=date(2024, 12, 31),
        )
    )
    fetched = analysis_service.get_analysis(job.analysis_id)

    assert filing_index_result.indexed_documents == 1
    assert transcript_index_result.indexed_documents == 1
    assert job.status.value == "completed"
    assert job.final_report is not None
    assert job.final_report_markdown is not None
    assert fetched is not None
    assert fetched.analysis_id == job.analysis_id
