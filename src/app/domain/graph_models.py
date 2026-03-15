"""LangGraph-facing state and packet models."""

from datetime import UTC, date, datetime

from pydantic import Field, model_validator

from app.domain.enums import (
    AnalysisTaskType,
    ConfidenceLabel,
    CoverageLabel,
    FilingType,
    ManualReviewStatus,
    TaskRoutingStatus,
)
from app.domain.models import (
    AgentFinding,
    CanonicalModel,
    ChunkRecord,
    Company,
    ConflictCandidate,
    CoverageStatus,
    EvidenceRef,
    FinalReport,
    ParsedSection,
    RawDocument,
    SharedMemoryEntry,
    XbrlFact,
)
from app.domain.verifier_models import ClaimVerificationResult


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def default_requested_filing_types() -> list[FilingType]:
    """Return the default filing forms for a research request."""

    return [
        FilingType.FORM_10K,
        FilingType.FORM_10Q,
        FilingType.FORM_8K,
    ]


def default_requested_tasks() -> list[AnalysisTaskType]:
    """Return the default planner task set for a research request."""

    return [
        AnalysisTaskType.LONG_TERM_STRUCTURE,
        AnalysisTaskType.RECENT_QUARTER_CHANGE,
        AnalysisTaskType.MATERIAL_EVENTS,
        AnalysisTaskType.MANAGEMENT_TONE_GUIDANCE,
        AnalysisTaskType.FINAL_THESIS,
    ]


class DocumentSignal(CanonicalModel):
    """Document availability and quality hints consumed by planner/router nodes."""

    filing_type: FilingType
    available: bool = True
    document_date: date | None = None
    report_period: date | None = None
    parse_confidence: ConfidenceLabel = ConfidenceLabel.MEDIUM
    notes: str | None = None


class PlannedTask(CanonicalModel):
    """Planner output before channel-specific routing."""

    task_type: AnalysisTaskType
    description: str
    preferred_filing_type: FilingType | None = None
    preferred_agent: str


class RoutedTask(CanonicalModel):
    """Task routed to a specialized agent or marked as a coverage gap."""

    task_type: AnalysisTaskType
    agent_name: str
    filing_type: FilingType | None = None
    status: TaskRoutingStatus
    reason: str
    document_available: bool
    recency_days: int | None = None
    parse_confidence: ConfidenceLabel | None = None
    missing_coverage_topics: list[str] = Field(default_factory=list)


class ExecutionPlan(CanonicalModel):
    """Deterministic execution plan emitted by the planner/router stage."""

    plan_id: str
    request_id: str
    tasks: list[PlannedTask] = Field(default_factory=list)
    routed_tasks: list[RoutedTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class AnalysisRequest(CanonicalModel):
    """Top-level request passed into the graph."""

    request_id: str
    company: Company
    question: str
    as_of_date: date | None = None
    requested_filing_types: list[FilingType] = Field(default_factory=default_requested_filing_types)
    requested_tasks: list[AnalysisTaskType] = Field(default_factory=default_requested_tasks)
    document_signals: list[DocumentSignal] = Field(default_factory=list)
    include_transcript: bool = False
    max_reretrievals: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def normalize_transcript_channel(self) -> "AnalysisRequest":
        """Keep transcript inclusion consistent with the filing list."""

        filing_types: list[FilingType] = [
            filing_type
            for filing_type in self.requested_filing_types
            if filing_type != FilingType.EARNINGS_CALL
        ]
        if self.include_transcript:
            filing_types.append(FilingType.EARNINGS_CALL)
        self.requested_filing_types = filing_types
        self.requested_tasks = list(dict.fromkeys(self.requested_tasks))
        return self


class AgentOutputPacket(CanonicalModel):
    """Structured output contract for every graph agent."""

    agent_name: str
    findings: list[AgentFinding] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    retrieved_snippets: list[ChunkRecord] = Field(default_factory=list)
    provisional_summary: str | None = None
    unresolved_items: list[str] = Field(default_factory=list)
    confidence_label: ConfidenceLabel = ConfidenceLabel.MEDIUM
    iteration_count: int = Field(default=0, ge=0)
    shared_memory_updates: list[SharedMemoryEntry] = Field(default_factory=list)
    conflicts: list[ConflictCandidate] = Field(default_factory=list)
    coverage_status: CoverageStatus = Field(default_factory=CoverageStatus.not_started)
    reretrieval_requested: bool = False
    requested_queries: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ManualReviewRequest(CanonicalModel):
    """Future-facing human-review request emitted by graph instrumentation."""

    review_id: str
    thread_id: str
    node_name: str
    reason: str
    blocking: bool = False
    status: ManualReviewStatus = ManualReviewStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)


class GraphState(CanonicalModel):
    """Canonical shared state for the LangGraph workflow."""

    request: AnalysisRequest
    raw_documents: list[RawDocument] = Field(default_factory=list)
    parsed_sections: list[ParsedSection] = Field(default_factory=list)
    chunk_records: list[ChunkRecord] = Field(default_factory=list)
    xbrl_facts: list[XbrlFact] = Field(default_factory=list)
    shared_memory: list[SharedMemoryEntry] = Field(default_factory=list)
    agent_packets: list[AgentOutputPacket] = Field(default_factory=list)
    verification_results: list[ClaimVerificationResult] = Field(default_factory=list)
    manual_review_requests: list[ManualReviewRequest] = Field(default_factory=list)
    execution_plan: ExecutionPlan | None = None
    routed_tasks: list[RoutedTask] = Field(default_factory=list)
    conflicts: list[ConflictCandidate] = Field(default_factory=list)
    final_report: FinalReport | None = None
    coverage_status: CoverageStatus = Field(default_factory=CoverageStatus.not_started)
    completed_steps: list[str] = Field(default_factory=list)
    reretrieval_count: int = Field(default=0, ge=0)
    max_reretrievals: int | None = Field(default=None, ge=0, le=5)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def apply_request_defaults(self) -> "GraphState":
        """Inherit graph limits from the request and enforce bounded retries."""

        if self.max_reretrievals is None:
            self.max_reretrievals = self.request.max_reretrievals
        if self.reretrieval_count > self.max_reretrievals:
            msg = "reretrieval_count cannot exceed max_reretrievals"
            raise ValueError(msg)
        if self.coverage_status.label == CoverageLabel.NOT_STARTED and self.raw_documents:
            self.coverage_status = CoverageStatus(label=CoverageLabel.PARTIAL)
        return self
