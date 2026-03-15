"""Canonical enums shared across ETL, DB, and graph layers."""

from enum import StrEnum


class FilingType(StrEnum):
    """Supported disclosure channels."""

    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    EARNINGS_CALL = "EARNINGS_CALL"


class VerificationLabel(StrEnum):
    """Verifier labels for evidence-grounded claims."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"
    UNAVAILABLE = "unavailable"


class ConfidenceLabel(StrEnum):
    """Coarse confidence levels for findings and verification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CoverageLabel(StrEnum):
    """Coverage state for a request, finding set, or report."""

    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    COMPLETE = "complete"


class RuntimeEnvironment(StrEnum):
    """Runtime environment names."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class DownloadStatusLabel(StrEnum):
    """Download lifecycle state for raw filings."""

    PENDING = "pending"
    DOWNLOADED = "downloaded"
    FAILED = "failed"
    SKIPPED = "skipped"


class TranscriptSourceType(StrEnum):
    """Transcript ingestion source types."""

    FILE = "file"
    GENERIC_HTTP = "generic_http"


class DocumentSectionType(StrEnum):
    """Semantic section types emitted by form-aware parsers."""

    BUSINESS = "business"
    RISK_FACTORS = "risk_factors"
    MDA = "mda"
    FINANCIAL_STATEMENTS = "financial_statements"
    NOTES = "notes"
    LIQUIDITY = "liquidity"
    CONTROLS = "controls"
    ITEM = "item"
    PREPARED_REMARKS = "prepared_remarks"
    QA = "qa"
    SPEAKER_TURN = "speaker_turn"
    FALLBACK = "fallback"


class TranscriptSegmentType(StrEnum):
    """Transcript segment types emitted by transcript parsing."""

    PREPARED_REMARKS = "prepared_remarks"
    QA = "qa"
    SPEAKER_TURN = "speaker_turn"


class XbrlCanonicalFact(StrEnum):
    """Canonical XBRL facts supported by the MVP."""

    REVENUE = "revenue"
    OPERATING_INCOME = "operating_income"
    NET_INCOME = "net_income"
    DILUTED_EPS = "diluted_eps"
    CASH_AND_CASH_EQUIVALENTS = "cash_and_cash_equivalents"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    TOTAL_ASSETS = "total_assets"
    TOTAL_LIABILITIES = "total_liabilities"


class AnalysisTaskType(StrEnum):
    """Planner task categories for disclosure-grounded analysis."""

    LONG_TERM_STRUCTURE = "long_term_structure"
    RECENT_QUARTER_CHANGE = "recent_quarter_change"
    MATERIAL_EVENTS = "material_events"
    MANAGEMENT_TONE_GUIDANCE = "management_tone_guidance"
    FINAL_THESIS = "final_thesis"


class TaskRoutingStatus(StrEnum):
    """Deterministic routing outcomes for planned tasks."""

    ROUTED = "routed"
    MISSING_COVERAGE = "missing_coverage"
    SKIPPED = "skipped"


class FindingSignalType(StrEnum):
    """High-level signal families emitted by specialized agents."""

    FUNDAMENTAL = "fundamental"
    EVENT = "event"
    MANAGEMENT_TONE = "management_tone"
    GUIDANCE = "guidance"
    RISK = "risk"


class TimeHorizonLabel(StrEnum):
    """Time horizon associated with a finding."""

    LONG_TERM = "long_term"
    RECENT = "recent"
    POINT_IN_TIME = "point_in_time"
    MIXED = "mixed"


class EvidenceTypeLabel(StrEnum):
    """Primary evidence shape supporting a finding."""

    NUMERIC = "numeric"
    NARRATIVE = "narrative"
    MIXED = "mixed"


class ConflictType(StrEnum):
    """Explainable conflict categories emitted by the checker."""

    FACTUAL_CONFLICT = "factual_conflict"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    NARRATIVE_TENSION = "narrative_tension"


class AnalysisJobStatus(StrEnum):
    """Synchronous analysis job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ManualReviewStatus(StrEnum):
    """Lifecycle state for future human-review checkpoints."""

    PENDING = "pending"
    BYPASSED = "bypassed"
    APPROVED = "approved"
    REJECTED = "rejected"
