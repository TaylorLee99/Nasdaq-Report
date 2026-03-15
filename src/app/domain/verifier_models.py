"""Schemas for confidence-calibrated claim verification."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field

from app.domain.enums import ConfidenceLabel, VerificationLabel
from app.domain.models import CanonicalModel, ChunkRecord, ConflictCandidate, EvidenceRef, XbrlFact


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class ClaimVerificationRequest(CanonicalModel):
    """Input payload for claim-level evidence verification."""

    claim_id: str
    agent_name: str
    claim: str
    cited_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    retrieved_snippets: list[ChunkRecord] = Field(default_factory=list)
    numeric_facts: list[XbrlFact] = Field(default_factory=list)
    conflict_candidates: list[ConflictCandidate] = Field(default_factory=list)
    missing_coverage_topics: list[str] = Field(default_factory=list)


class ClaimVerificationResult(CanonicalModel):
    """Structured output from the confidence-calibrated verifier."""

    claim_id: str
    label: VerificationLabel
    confidence: ConfidenceLabel
    rationale: str
    alignment_score: float = Field(ge=0.0, le=1.0)
    sufficiency_score: float = Field(ge=0.0, le=1.0)
    reretrieval_recommended: bool = False
    requested_queries: list[str] = Field(default_factory=list)
    missing_coverage_topics: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    conflicting_conflict_ids: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utc_now)
