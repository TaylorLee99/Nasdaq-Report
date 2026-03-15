"""Explainable heuristic conflict classification across agent outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from app.domain import (
    AgentFinding,
    AgentOutputPacket,
    ConfidenceLabel,
    ConflictCandidate,
    ConflictType,
    EvidenceTypeLabel,
    FindingSignalType,
    TimeHorizonLabel,
    VerificationLabel,
    VerificationStatus,
)

POSITIVE_TERMS = {
    "improving",
    "improvement",
    "strong",
    "stable",
    "resilient",
    "optimistic",
    "confidence",
    "accelerating",
    "growth",
    "positive",
    "benefit",
    "upside",
}
NEGATIVE_TERMS = {
    "pressure",
    "decline",
    "deteriorating",
    "weak",
    "soft",
    "risk",
    "headwind",
    "negative",
    "downside",
    "compression",
    "adverse",
    "worsening",
}
TOPIC_KEYWORDS = {
    "margin": {"margin", "gross margin", "operating margin", "compression"},
    "guidance": {"guidance", "outlook", "forecast"},
    "liquidity": {"liquidity", "cash", "cash flow", "working capital"},
    "demand": {"demand", "orders", "bookings", "pipeline"},
    "risk": {"risk", "exposure", "uncertainty"},
    "event": {"event", "investigation", "impairment", "charge", "restructuring"},
}


@dataclass(frozen=True)
class ClassifiedConflict:
    """Internal result before conversion into the domain candidate."""

    conflict_type: ConflictType
    reason: str
    shared_topic: str | None
    recency_gap_days: int | None
    confidence: ConfidenceLabel


class HeuristicConflictChecker:
    """Rule-based checker that separates contradiction, time mismatch, and tension."""

    def generate_candidates(
        self,
        packets: list[AgentOutputPacket],
    ) -> list[ConflictCandidate]:
        findings = [finding for packet in packets for finding in packet.findings]
        conflicts: list[ConflictCandidate] = []
        for index, left in enumerate(findings):
            for right in findings[index + 1 :]:
                classified = self._classify_pair(left, right)
                if classified is None:
                    continue
                conflicts.append(
                    ConflictCandidate(
                        conflict_id=f"conflict:{left.finding_id}:{right.finding_id}",
                        conflict_type=classified.conflict_type,
                        claim=f"{left.claim} <> {right.claim}",
                        finding_ids=[left.finding_id, right.finding_id],
                        reason=classified.reason,
                        shared_topic=classified.shared_topic,
                        recency_gap_days=classified.recency_gap_days,
                        evidence_refs=[*left.evidence_refs, *right.evidence_refs],
                        verification_status=VerificationStatus(
                            label=VerificationLabel.CONFLICTING,
                            confidence=classified.confidence,
                            rationale=classified.reason,
                            verifier_name="heuristic_conflict_checker",
                        ),
                    )
                )
        return conflicts

    def _classify_pair(
        self,
        left: AgentFinding,
        right: AgentFinding,
    ) -> ClassifiedConflict | None:
        shared_topic = self._shared_topic(left, right)
        if shared_topic is None:
            return None

        left_polarity = self._polarity(left.claim, left.summary)
        right_polarity = self._polarity(right.claim, right.summary)
        if left_polarity == 0 or right_polarity == 0 or left_polarity == right_polarity:
            return None

        recency_gap_days = self._recency_gap_days(left.as_of_date, right.as_of_date)
        if self._is_narrative_tension(left, right):
            return ClassifiedConflict(
                conflict_type=ConflictType.NARRATIVE_TENSION,
                reason=(
                    f"Opposing signals on '{shared_topic}' appear as "
                    "numeric-vs-narrative or tone-vs-fact tension, "
                    "not a direct contradiction."
                ),
                shared_topic=shared_topic,
                recency_gap_days=recency_gap_days,
                confidence=ConfidenceLabel.MEDIUM,
            )

        if self._is_temporal_mismatch(left, right, recency_gap_days):
            return ClassifiedConflict(
                conflict_type=ConflictType.TEMPORAL_MISMATCH,
                reason=(
                    f"Opposing signals on '{shared_topic}' likely reflect "
                    "different disclosure dates or horizons rather than "
                    "the same factual state."
                ),
                shared_topic=shared_topic,
                recency_gap_days=recency_gap_days,
                confidence=ConfidenceLabel.MEDIUM,
            )

        return ClassifiedConflict(
            conflict_type=ConflictType.FACTUAL_CONFLICT,
            reason=(
                f"Opposing claims on '{shared_topic}' share a similar "
                "horizon and evidence basis, making a factual conflict plausible."
            ),
            shared_topic=shared_topic,
            recency_gap_days=recency_gap_days,
            confidence=ConfidenceLabel.HIGH,
        )

    @staticmethod
    def _shared_topic(left: AgentFinding, right: AgentFinding) -> str | None:
        left_text = f"{left.claim} {left.summary}".lower()
        right_text = f"{right.claim} {right.summary}".lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(keyword in left_text for keyword in keywords) and any(
                keyword in right_text for keyword in keywords
            ):
                return topic
        left_topics = set(left.coverage_status.covered_topics)
        right_topics = set(right.coverage_status.covered_topics)
        overlap = sorted(left_topics & right_topics)
        return overlap[0] if overlap else None

    @staticmethod
    def _polarity(*texts: str) -> int:
        tokens = re.findall(r"[a-zA-Z_]+", " ".join(texts).lower())
        positive = sum(token in POSITIVE_TERMS for token in tokens)
        negative = sum(token in NEGATIVE_TERMS for token in tokens)
        if positive == negative:
            return 0
        return 1 if positive > negative else -1

    @staticmethod
    def _recency_gap_days(left_date: date | None, right_date: date | None) -> int | None:
        if left_date is None or right_date is None:
            return None
        return abs((left_date - right_date).days)

    @staticmethod
    def _is_narrative_tension(left: AgentFinding, right: AgentFinding) -> bool:
        evidence_pair = {left.evidence_type, right.evidence_type}
        signal_pair = {left.signal_type, right.signal_type}
        return (
            evidence_pair == {EvidenceTypeLabel.NUMERIC, EvidenceTypeLabel.NARRATIVE}
            or FindingSignalType.MANAGEMENT_TONE in signal_pair
            or FindingSignalType.GUIDANCE in signal_pair
            and EvidenceTypeLabel.NUMERIC in evidence_pair
        )

    @staticmethod
    def _is_temporal_mismatch(
        left: AgentFinding,
        right: AgentFinding,
        recency_gap_days: int | None,
    ) -> bool:
        horizon_pair = {left.time_horizon, right.time_horizon}
        if recency_gap_days is not None and recency_gap_days >= 120:
            return True
        return horizon_pair in (
            {TimeHorizonLabel.LONG_TERM, TimeHorizonLabel.RECENT},
            {TimeHorizonLabel.LONG_TERM, TimeHorizonLabel.POINT_IN_TIME},
        )


def collect_conflicts(packets: list[AgentOutputPacket]) -> list[ConflictCandidate]:
    """Convenience wrapper for the default heuristic checker."""

    return HeuristicConflictChecker().generate_candidates(packets)
