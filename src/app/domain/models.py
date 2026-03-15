"""Canonical entity and report models."""

from datetime import UTC, date, datetime
from decimal import Decimal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import (
    ConfidenceLabel,
    ConflictType,
    CoverageLabel,
    DocumentSectionType,
    EvidenceTypeLabel,
    FilingType,
    FindingSignalType,
    TimeHorizonLabel,
    TranscriptSegmentType,
    TranscriptSourceType,
    VerificationLabel,
    XbrlCanonicalFact,
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class CanonicalModel(BaseModel):
    """Base model for all canonical schemas."""

    model_config = ConfigDict(extra="forbid")


class Company(CanonicalModel):
    """Canonical company identifier."""

    cik: str
    ticker: str
    company_name: str = Field(validation_alias=AliasChoices("company_name", "legal_name"))
    exchange: str = "NASDAQ"
    is_domestic_filer: bool = True
    latest_snapshot_date: date | None = None
    universe_name: str = "NASDAQ_100"
    snapshot_label: str | None = None

    @property
    def legal_name(self) -> str:
        """Backward-compatible alias for company_name."""

        return self.company_name


class FilingMetadata(CanonicalModel):
    """Metadata for 10-K, 10-Q, and 8-K filings."""

    filing_id: str
    company: Company
    filing_type: FilingType
    accession_number: str
    filed_at: date
    period_end_date: date | None = None
    fiscal_period: str | None = None
    is_amendment: bool = False
    source_uri: str | None = None

    @model_validator(mode="after")
    def validate_filing_type(self) -> "FilingMetadata":
        """Reject transcript metadata in filing records."""

        if self.filing_type == FilingType.EARNINGS_CALL:
            msg = "FilingMetadata cannot use EARNINGS_CALL"
            raise ValueError(msg)
        return self


class TranscriptMetadata(CanonicalModel):
    """Metadata for the optional earnings call channel."""

    transcript_id: str
    company: Company
    call_date: date
    fiscal_quarter: str | None = None
    fiscal_year: int | None = None
    source_url: str | None = None
    speaker_separated: bool = False
    source_reliability: ConfidenceLabel = ConfidenceLabel.MEDIUM
    source_type: TranscriptSourceType = TranscriptSourceType.FILE
    availability_coverage: "CoverageStatus" = Field(
        default_factory=lambda: CoverageStatus.not_started()
    )
    storage_path: str | None = None
    checksum_sha256: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    imported_at: datetime = Field(default_factory=utc_now)


class RawDocument(CanonicalModel):
    """Normalized raw text artifact consumed by parsing and indexing."""

    document_id: str
    company: Company
    document_type: FilingType
    content: str
    filing_metadata: FilingMetadata | None = None
    transcript_metadata: TranscriptMetadata | None = None
    source_uri: str | None = None
    language: str = "en"
    ingested_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_metadata_pairing(self) -> "RawDocument":
        """Require the correct metadata type for each document."""

        if self.document_type == FilingType.EARNINGS_CALL:
            if self.transcript_metadata is None:
                msg = "Transcript documents require transcript_metadata"
                raise ValueError(msg)
        elif self.filing_metadata is None:
            msg = "Filing documents require filing_metadata"
            raise ValueError(msg)
        return self


class XbrlFact(CanonicalModel):
    """Typed XBRL fact attached to a filing."""

    fact_id: str
    accession_number: str
    company: Company
    filing_type: FilingType
    canonical_fact: XbrlCanonicalFact
    original_concept: str
    numeric_value: Decimal
    unit: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    frame: str | None = None
    context_ref: str | None = None
    decimals: int | None = None
    filing_date: date | None = None
    fiscal_period: str | None = None


class ParsedSection(CanonicalModel):
    """Structured section extracted from a raw document."""

    section_id: str
    document_id: str
    company: Company
    filing_type: FilingType
    heading: str
    text: str
    ordinal: int = Field(ge=1)
    parent_section_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    section_type: DocumentSectionType | None = None
    parse_confidence: ConfidenceLabel = ConfidenceLabel.MEDIUM
    used_fallback: bool = False
    fallback_reason: str | None = None
    item_number: str | None = None
    transcript_segment_type: TranscriptSegmentType | None = None
    speaker: str | None = None


class ChunkRecord(CanonicalModel):
    """Searchable chunk produced from a parsed section."""

    chunk_id: str
    document_id: str
    section_id: str
    company: Company
    filing_type: FilingType
    accession_number: str | None = None
    transcript_id: str | None = None
    filing_date: date | None = None
    report_period: date | None = None
    call_date: date | None = None
    fiscal_quarter: str | None = None
    section_name: str
    section_type: DocumentSectionType | None = None
    chunk_index: int = Field(default=1, ge=1)
    text: str
    token_count: int = Field(ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    item_number: str | None = None
    speaker: str | None = None
    transcript_segment_type: TranscriptSegmentType | None = None
    parse_confidence: ConfidenceLabel = ConfidenceLabel.MEDIUM
    used_fallback: bool = False
    source_reliability: ConfidenceLabel | None = None
    source_url: str | None = None
    embedding: list[float] = Field(default_factory=list)
    embedding_key: str | None = None


class EvidenceRef(CanonicalModel):
    """Reference back to specific supporting disclosure spans."""

    evidence_id: str
    document_id: str
    filing_type: FilingType
    excerpt: str
    section_id: str | None = None
    chunk_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    source_uri: str | None = None


class CoverageStatus(CanonicalModel):
    """Coverage snapshot for findings and reports."""

    label: CoverageLabel
    covered_topics: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    notes: str | None = None

    @classmethod
    def not_started(cls) -> "CoverageStatus":
        """Return a reusable default coverage object."""

        return cls(label=CoverageLabel.NOT_STARTED)


class VerificationStatus(CanonicalModel):
    """Verification outcome for a claim or conflict."""

    label: VerificationLabel
    confidence: ConfidenceLabel
    rationale: str | None = None
    verifier_name: str | None = None
    checked_at: datetime = Field(default_factory=utc_now)


class AgentFinding(CanonicalModel):
    """A claim plus its evidence, coverage, and verification status."""

    finding_id: str
    agent_name: str
    company: Company
    claim: str
    summary: str
    signal_type: FindingSignalType = FindingSignalType.FUNDAMENTAL
    time_horizon: TimeHorizonLabel = TimeHorizonLabel.MIXED
    evidence_type: EvidenceTypeLabel = EvidenceTypeLabel.NARRATIVE
    as_of_date: date | None = None
    filing_types: list[FilingType] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    coverage_status: CoverageStatus = Field(default_factory=CoverageStatus.not_started)
    verification_status: VerificationStatus
    created_at: datetime = Field(default_factory=utc_now)


class SharedMemoryEntry(CanonicalModel):
    """Persistent memory item exchanged between graph agents."""

    memory_id: str
    key: str
    value: str
    source_agent: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class ConflictCandidate(CanonicalModel):
    """Potential contradiction between agent findings."""

    conflict_id: str
    conflict_type: ConflictType
    claim: str
    finding_ids: list[str] = Field(default_factory=list)
    reason: str
    shared_topic: str | None = None
    recency_gap_days: int | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    verification_status: VerificationStatus = Field(
        default_factory=lambda: VerificationStatus(
            label=VerificationLabel.CONFLICTING,
            confidence=ConfidenceLabel.MEDIUM,
        )
    )


class ReportClaim(CanonicalModel):
    """Claim entry rendered into the final structured thesis report."""

    finding_id: str
    agent_name: str
    title: str
    claim: str
    summary: str | None = None
    importance: ConfidenceLabel | None = None
    time_horizon: TimeHorizonLabel | None = None
    trigger_to_monitor: str | None = None
    confidence: ConfidenceLabel
    verification_label: VerificationLabel
    filing_types: list[FilingType] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class AnalysisScope(CanonicalModel):
    """Explicit analysis scope embedded in the final report payload."""

    question: str
    as_of_date: date | None = None
    allowed_sources: list[FilingType] = Field(default_factory=list)
    transcript_included: bool = False
    max_reretrievals: int = 0


class DataCoverageSummary(CanonicalModel):
    """Structured data coverage summary for the final report."""

    coverage_label: CoverageLabel
    covered_channels: list[FilingType] = Field(default_factory=list)
    missing_channels: list[FilingType] = Field(default_factory=list)
    covered_topics: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    notes: str | None = None


class ExecutiveSummarySection(CanonicalModel):
    """Top-level summary constrained to evidence-backed statements."""

    summary: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    verification_label: VerificationLabel
    confidence: ConfidenceLabel


class FinalInvestmentThesisSection(CanonicalModel):
    """Final valuation-free thesis section."""

    thesis: str
    stance: str
    uncertainty_statement: str | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    verification_label: VerificationLabel
    confidence: ConfidenceLabel


class EvidenceGroundedReportSection(CanonicalModel):
    """Structured report body grouped by document channel."""

    business_overview: list[ReportClaim] = Field(default_factory=list)
    recent_quarter_change: list[ReportClaim] = Field(default_factory=list)
    material_events: list[ReportClaim] = Field(default_factory=list)
    cross_document_tensions: list[ReportClaim] = Field(default_factory=list)


class VerificationSummary(CanonicalModel):
    """Verification totals for downstream consumers."""

    total_claims: int = 0
    supported_claim_count: int = 0
    partially_supported_claim_count: int = 0
    insufficient_claim_count: int = 0
    conflicting_claim_count: int = 0
    unavailable_claim_count: int = 0
    high_confidence_claim_count: int = 0
    medium_confidence_claim_count: int = 0
    low_confidence_claim_count: int = 0


class CoverageGap(CanonicalModel):
    """Structured coverage gap emitted in the report appendix."""

    topic: str
    reason: str | None = None


class SourceDocumentRef(CanonicalModel):
    """Document provenance captured in the appendix."""

    document_id: str
    filing_type: FilingType
    source_uri: str | None = None


class AppendixSection(CanonicalModel):
    """Appendix with excluded claims and provenance details."""

    conflicts: list[ConflictCandidate] = Field(default_factory=list)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    excluded_claims: list[ReportClaim] = Field(default_factory=list)
    source_documents: list[SourceDocumentRef] = Field(default_factory=list)


class FinalReport(CanonicalModel):
    """Valuation-free report grounded in 10-K, 10-Q, and 8-K findings."""

    report_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    company: Company
    analysis_scope: AnalysisScope
    coverage_summary: DataCoverageSummary
    executive_summary: ExecutiveSummarySection
    key_risks: list[ReportClaim] = Field(default_factory=list)
    key_opportunities: list[ReportClaim] = Field(default_factory=list)
    watchpoints: list[ReportClaim] = Field(default_factory=list)
    final_investment_thesis: FinalInvestmentThesisSection
    evidence_grounded_report: EvidenceGroundedReportSection
    verification_summary: VerificationSummary
    appendix: AppendixSection

    @property
    def coverage_status(self) -> CoverageStatus:
        """Backward-compatible view derived from coverage_summary."""

        return CoverageStatus(
            label=self.coverage_summary.coverage_label,
            covered_topics=self.coverage_summary.covered_topics,
            missing_topics=self.coverage_summary.missing_topics,
            notes=self.coverage_summary.notes,
        )


class HealthStatus(CanonicalModel):
    """Health probe payload."""

    status: str = "ok"
    app_name: str
    environment: str
