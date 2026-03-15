"""SQLAlchemy ORM models for persistent storage."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.db import Base
from app.storage.vector_types import PgVectorCompatible


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class CompanyMasterModel(Base):
    """Canonical company master record."""

    __tablename__ = "company_master"

    cik: Mapped[str] = mapped_column(String(32), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    is_domestic_filer: Mapped[bool] = mapped_column(Boolean, nullable=False)
    latest_snapshot_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class UniverseSnapshotConstituentModel(Base):
    """Stored universe membership for a specific snapshot date."""

    __tablename__ = "universe_snapshot_constituents"
    __table_args__ = (
        UniqueConstraint("snapshot_date", "cik", name="uq_universe_snapshot_constituents"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    cik: Mapped[str] = mapped_column(ForeignKey("company_master.cik"), nullable=False)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    is_domestic_filer: Mapped[bool] = mapped_column(Boolean, nullable=False)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SecSubmissionMetadataModel(Base):
    """Normalized SEC filing metadata row."""

    __tablename__ = "sec_submission_metadata"
    __table_args__ = (UniqueConstraint("accession_number", name="uq_sec_submission_metadata"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(ForeignKey("company_master.cik"), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    accession_number: Mapped[str] = mapped_column(String(32), nullable=False)
    filing_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    form_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    primary_document: Mapped[str] = mapped_column(String(255), nullable=False)
    report_period: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SecSubmissionsRawModel(Base):
    """Optional raw SEC submissions payload storage."""

    __tablename__ = "sec_submissions_raw"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(ForeignKey("company_master.cik"), nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class SecRawDocumentModel(Base):
    """Raw SEC filing download state linked to metadata rows."""

    __tablename__ = "sec_raw_documents"
    __table_args__ = (UniqueConstraint("accession_number", name="uq_sec_raw_documents"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    accession_number: Mapped[str] = mapped_column(
        ForeignKey("sec_submission_metadata.accession_number"),
        nullable=False,
    )
    cik: Mapped[str] = mapped_column(ForeignKey("company_master.cik"), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    form_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_count: Mapped[int] = mapped_column(nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class TranscriptMetadataModel(Base):
    """Transcript metadata and provenance for the optional channel."""

    __tablename__ = "transcript_metadata"
    __table_args__ = (UniqueConstraint("transcript_id", name="uq_transcript_metadata"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    transcript_id: Mapped[str] = mapped_column(String(128), nullable=False)
    cik: Mapped[str] = mapped_column(ForeignKey("company_master.cik"), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    call_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    fiscal_quarter: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fiscal_year: Mapped[int | None] = mapped_column(nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    speaker_separated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_reliability: Mapped[str] = mapped_column(String(16), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    coverage_label: Mapped[str] = mapped_column(String(16), nullable=False)
    covered_topics: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    missing_topics: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    coverage_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(nullable=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TranscriptAvailabilityGapModel(Base):
    """Recorded missing transcript coverage for reduced-mode operation."""

    __tablename__ = "transcript_availability_gaps"
    __table_args__ = (UniqueConstraint("gap_id", name="uq_transcript_availability_gaps"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    gap_id: Mapped[str] = mapped_column(String(128), nullable=False)
    cik: Mapped[str] = mapped_column(ForeignKey("company_master.cik"), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    coverage_label: Mapped[str] = mapped_column(String(16), nullable=False)
    covered_topics: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    missing_topics: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    coverage_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class XbrlFactModel(Base):
    """Canonical XBRL fact storage."""

    __tablename__ = "xbrl_facts"
    __table_args__ = (UniqueConstraint("fact_id", name="uq_xbrl_facts"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fact_id: Mapped[str] = mapped_column(String(128), nullable=False)
    accession_number: Mapped[str] = mapped_column(
        ForeignKey("sec_submission_metadata.accession_number"),
        nullable=False,
        index=True,
    )
    cik: Mapped[str] = mapped_column(ForeignKey("company_master.cik"), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    filing_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    canonical_fact: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_concept: Mapped[str] = mapped_column(String(255), nullable=False)
    numeric_value: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    frame: Mapped[str | None] = mapped_column(String(64), nullable=True)
    context_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decimals: Mapped[int | None] = mapped_column(nullable=True)
    filing_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    fiscal_period: Mapped[str | None] = mapped_column(String(32), nullable=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class IndexedChunkModel(Base):
    """Retrieval-ready chunks stored in the vector index."""

    __tablename__ = "indexed_chunks"
    __table_args__ = (UniqueConstraint("chunk_id", name="uq_indexed_chunks"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(128), nullable=False)
    document_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    section_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    cik: Mapped[str] = mapped_column(ForeignKey("company_master.cik"), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    filing_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    accession_number: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    transcript_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    filing_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    report_period: Mapped[date | None] = mapped_column(Date, nullable=True)
    call_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    fiscal_quarter: Mapped[str | None] = mapped_column(String(32), nullable=True)
    section_name: Mapped[str] = mapped_column(String(255), nullable=False)
    section_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunk_index: Mapped[int] = mapped_column(nullable=False, default=1)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(nullable=False)
    char_start: Mapped[int | None] = mapped_column(nullable=True)
    char_end: Mapped[int | None] = mapped_column(nullable=True)
    item_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    speaker: Mapped[str | None] = mapped_column(String(255), nullable=True)
    transcript_segment_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parse_confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    used_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_reliability: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    embedding: Mapped[list[float]] = mapped_column(
        PgVectorCompatible(12),
        nullable=False,
    )
    embedding_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AnalysisJobModel(Base):
    """Persisted synchronous analysis job and final report artifacts."""

    __tablename__ = "analysis_jobs"

    analysis_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    cik: Mapped[str | None] = mapped_column(
        ForeignKey("company_master.cik"),
        nullable=True,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    document_signals_json: Mapped[list[dict[str, object]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    request_payload_json: Mapped[dict[str, object]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    final_report_json: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    final_report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )
