"""Confidence-calibrated claim verification heuristics and interfaces."""

from __future__ import annotations

import re
from typing import Protocol

from app.domain import (
    AgentFinding,
    AgentOutputPacket,
    ClaimVerificationRequest,
    ClaimVerificationResult,
    ConfidenceLabel,
    ConflictCandidate,
    ConflictType,
    FilingType,
    VerificationLabel,
    VerificationStatus,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
STRONG_CLAIM_TERMS = {
    "always",
    "clearly",
    "definitely",
    "no",
    "never",
    "proves",
    "will",
}
EIGHT_K_MATERIAL_ITEM_NUMBERS = {"1.01", "2.02", "2.03", "5.02", "8.01"}
EVENT_KEYWORDS = {
    "adopted",
    "agreement",
    "announced",
    "appointed",
    "approved",
    "board",
    "committee",
    "compensation",
    "customer",
    "definitive",
    "director",
    "disclosed",
    "earnings",
    "entered",
    "event",
    "guidance",
    "leadership",
    "material",
    "officer",
    "operations",
    "results",
    "strategic",
    "transition",
}
GENERIC_FALLBACK_PATTERNS = (
    "the latest 10-k supports a verified",
    "the latest 10-q supports a verified",
    "supports a verified long-term structure finding",
    "supports a verified recent-quarter change finding",
    "supports a verified liquidity and capital resources update",
    "supports a verified controls update",
)


class ModelAssistedVerifier(Protocol):
    """Optional interface for model-assisted verification implementations."""

    def verify(self, request: ClaimVerificationRequest) -> ClaimVerificationResult:
        """Return a verification result for one claim."""


class HeuristicEvidenceVerifier:
    """Deterministic verifier for claim-evidence alignment and sufficiency."""

    def verify_claim(self, request: ClaimVerificationRequest) -> ClaimVerificationResult:
        claim_tokens = _tokens(request.claim)
        alignment_score = self._alignment_score(
            claim_tokens=claim_tokens,
            request=request,
        )
        sufficiency_score = self._sufficiency_score(request)
        strong_claim = self._is_strong_claim(claim_tokens)
        conflict_ids = [conflict.conflict_id for conflict in request.conflict_candidates]
        eight_k_event_request = _is_eight_k_event_request(request)
        supported_threshold = 0.67
        partial_threshold = 0.34

        if eight_k_event_request:
            alignment_score = min(alignment_score + _eight_k_alignment_bonus(request), 1.0)
            sufficiency_score = min(sufficiency_score + _eight_k_sufficiency_bonus(request), 1.0)
            supported_threshold = 0.55
            partial_threshold = 0.28
            if _is_item_502_compensation_event(request):
                alignment_score = min(alignment_score + 0.12, 1.0)
                sufficiency_score = min(sufficiency_score + 0.08, 1.0)
                supported_threshold = 0.5
                partial_threshold = 0.24

        evidence_available = bool(request.cited_evidence_refs or request.retrieved_snippets)

        if request.missing_coverage_topics and not evidence_available:
            label = VerificationLabel.UNAVAILABLE
            confidence = ConfidenceLabel.LOW
            rationale = (
                "Required coverage is unavailable for this claim: "
                f"{', '.join(request.missing_coverage_topics)}."
            )
            reretrieval_recommended = False
        elif self._has_factual_conflict(request.conflict_candidates):
            label = VerificationLabel.CONFLICTING
            confidence = ConfidenceLabel.MEDIUM
            rationale = "At least one factual conflict candidate overlaps with this claim."
            reretrieval_recommended = False
        elif alignment_score < 0.2 and sufficiency_score < 0.34:
            label = VerificationLabel.INSUFFICIENT
            confidence = ConfidenceLabel.LOW
            rationale = (
                "Evidence is weakly aligned to the claim and insufficient in quantity or "
                "diversity."
            )
            reretrieval_recommended = True
        elif alignment_score >= 0.5 and sufficiency_score >= supported_threshold:
            label = VerificationLabel.SUPPORTED
            confidence = _supported_confidence_label(
                request=request,
                alignment_score=alignment_score,
                sufficiency_score=sufficiency_score,
                strong_claim=strong_claim,
            )
            rationale = (
                "Evidence aligns well with the claim and is sufficient across the "
                "retrieved support set."
            )
            if eight_k_event_request:
                rationale = (
                    "Item-anchored 8-K evidence aligns with the event claim and is sufficient "
                    "for a current-report disclosure."
                )
            reretrieval_recommended = False
        elif alignment_score >= 0.35 and sufficiency_score >= partial_threshold:
            label = VerificationLabel.PARTIALLY_SUPPORTED
            confidence = ConfidenceLabel.MEDIUM if not strong_claim else ConfidenceLabel.LOW
            rationale = (
                "Evidence partially supports the claim but leaves unresolved gaps or "
                "limited support breadth."
            )
            if eight_k_event_request:
                rationale = (
                    "The 8-K item disclosure supports the event directionally, but the current "
                    "support set is still narrow."
                )
            reretrieval_recommended = True
        else:
            label = VerificationLabel.INSUFFICIENT
            confidence = ConfidenceLabel.LOW
            rationale = "Available evidence does not adequately support the claim."
            reretrieval_recommended = True

        if strong_claim and label in {
            VerificationLabel.PARTIALLY_SUPPORTED,
            VerificationLabel.INSUFFICIENT,
        }:
            confidence = ConfidenceLabel.LOW

        requested_queries: list[str] = []
        if reretrieval_recommended:
            requested_queries.append(_build_reretrieval_query(request.claim, claim_tokens))

        return ClaimVerificationResult(
            claim_id=request.claim_id,
            label=label,
            confidence=confidence,
            rationale=rationale,
            alignment_score=alignment_score,
            sufficiency_score=sufficiency_score,
            reretrieval_recommended=reretrieval_recommended,
            requested_queries=requested_queries,
            missing_coverage_topics=request.missing_coverage_topics,
            supporting_evidence_ids=[
                *[ref.evidence_id for ref in request.cited_evidence_refs],
                *[chunk.chunk_id for chunk in request.retrieved_snippets],
            ],
            conflicting_conflict_ids=conflict_ids,
        )

    def verify_packets(
        self,
        packets: list[AgentOutputPacket],
        conflicts: list[ConflictCandidate],
    ) -> list[ClaimVerificationResult]:
        results: list[ClaimVerificationResult] = []
        for packet in packets:
            for finding in packet.findings:
                request = ClaimVerificationRequest(
                    claim_id=finding.finding_id,
                    agent_name=finding.agent_name,
                    claim=finding.claim,
                    cited_evidence_refs=finding.evidence_refs,
                    retrieved_snippets=packet.retrieved_snippets,
                    numeric_facts=[],
                    conflict_candidates=_relevant_conflicts(finding, conflicts),
                    missing_coverage_topics=finding.coverage_status.missing_topics,
                )
                results.append(self.verify_claim(request))
        return results

    def _alignment_score(
        self,
        *,
        claim_tokens: set[str],
        request: ClaimVerificationRequest,
    ) -> float:
        scores = _alignment_terms(request, claim_tokens)
        if not scores:
            return 0.0
        adjusted = max(
            max(scores)
            + _excerpt_grounding_bonus(request)
            - _coverage_penalty(request)
            - _generic_claim_penalty(request),
            0.0,
        )
        return min(adjusted, 1.0)

    @staticmethod
    def _sufficiency_score(request: ClaimVerificationRequest) -> float:
        score = 0.0
        if request.cited_evidence_refs:
            score += min(len(request.cited_evidence_refs), 2) * 0.25
        if request.retrieved_snippets:
            score += min(len(request.retrieved_snippets), 2) * 0.2
        if request.numeric_facts:
            score += 0.2
        if request.conflict_candidates:
            score -= 0.15
        if request.missing_coverage_topics:
            score -= 0.2
        score += _excerpt_grounding_bonus(request) * 0.25
        score -= _generic_claim_penalty(request) * 0.25
        return max(min(score, 1.0), 0.0)

    @staticmethod
    def _is_strong_claim(claim_tokens: set[str]) -> bool:
        return any(token in STRONG_CLAIM_TERMS for token in claim_tokens)

    @staticmethod
    def _has_factual_conflict(conflicts: list[ConflictCandidate]) -> bool:
        return any(
            conflict.conflict_type == ConflictType.FACTUAL_CONFLICT for conflict in conflicts
        )


def apply_verification_results(
    packets: list[AgentOutputPacket],
    results: list[ClaimVerificationResult],
) -> list[AgentOutputPacket]:
    """Update packet findings with verification results for downstream consumers."""

    result_map = {result.claim_id: result for result in results}
    updated_packets: list[AgentOutputPacket] = []
    for packet in packets:
        updated_findings: list[AgentFinding] = []
        for finding in packet.findings:
            result = result_map.get(finding.finding_id)
            if result is None:
                updated_findings.append(finding)
                continue
            updated_findings.append(
                finding.model_copy(
                    update={
                        "verification_status": VerificationStatus(
                            label=result.label,
                            confidence=result.confidence,
                            rationale=result.rationale,
                            verifier_name="heuristic_evidence_verifier",
                        )
                    }
                )
            )
        updated_packets.append(packet.model_copy(update={"findings": updated_findings}))
    return updated_packets


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if token not in STOPWORDS}


def _build_reretrieval_query(claim: str, claim_tokens: set[str]) -> str:
    salient = " ".join(sorted(claim_tokens)[:6])
    return f"claim support evidence {salient or claim}"


def _relevant_conflicts(
    finding: AgentFinding,
    conflicts: list[ConflictCandidate],
) -> list[ConflictCandidate]:
    return [conflict for conflict in conflicts if finding.finding_id in conflict.finding_ids]


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    return len(overlap) / max(len(left), 1)


def _snippet_alignment(claim_tokens: set[str], text: str) -> float:
    return _overlap_ratio(claim_tokens, _tokens(text))


def _best_alignment_from_conflicts(
    claim_tokens: set[str],
    conflicts: list[ConflictCandidate],
) -> float:
    if not conflicts:
        return 0.0
    return max(
        (_snippet_alignment(claim_tokens, conflict.claim) for conflict in conflicts),
        default=0.0,
    )


def _numeric_alignment_bonus(request: ClaimVerificationRequest) -> float:
    if not request.numeric_facts:
        return 0.0
    claim = request.claim.lower()
    financial_terms = {
        "revenue",
        "income",
        "margin",
        "cash",
        "assets",
        "liabilities",
        "eps",
    }
    if any(term in claim for term in financial_terms):
        return 0.2
    return 0.05


def _coverage_penalty(request: ClaimVerificationRequest) -> float:
    return 0.2 if request.missing_coverage_topics else 0.0


def _alignment_terms(
    request: ClaimVerificationRequest,
    claim_tokens: set[str],
) -> list[float]:
    scores = [_snippet_alignment(claim_tokens, ref.excerpt) for ref in request.cited_evidence_refs]
    scores.extend(
        _snippet_alignment(claim_tokens, chunk.text) for chunk in request.retrieved_snippets
    )
    if request.conflict_candidates:
        scores.append(_best_alignment_from_conflicts(claim_tokens, request.conflict_candidates))
    if request.numeric_facts:
        scores.append(_numeric_alignment_bonus(request))
    return scores


def _excerpt_grounding_bonus(request: ClaimVerificationRequest) -> float:
    if not request.cited_evidence_refs:
        return 0.0
    claim_tokens = _tokens(request.claim)
    if not claim_tokens:
        return 0.0
    best_excerpt_alignment = max(
        (_snippet_alignment(claim_tokens, ref.excerpt) for ref in request.cited_evidence_refs),
        default=0.0,
    )
    return 0.12 if best_excerpt_alignment >= 0.45 else 0.0


def _generic_claim_penalty(request: ClaimVerificationRequest) -> float:
    lowered = request.claim.lower()
    if any(pattern in lowered for pattern in GENERIC_FALLBACK_PATTERNS):
        return 0.12
    return 0.0


def _is_eight_k_event_request(request: ClaimVerificationRequest) -> bool:
    if request.agent_name == "run_8k_agent":
        return True
    return any(
        ref.filing_type == FilingType.FORM_8K for ref in request.cited_evidence_refs
    ) or any(chunk.filing_type == FilingType.FORM_8K for chunk in request.retrieved_snippets)


def _eight_k_alignment_bonus(request: ClaimVerificationRequest) -> float:
    item_numbers = _detected_item_numbers(request)
    if not item_numbers.intersection(EIGHT_K_MATERIAL_ITEM_NUMBERS):
        return 0.0
    claim_tokens = _tokens(request.claim)
    if not claim_tokens:
        return 0.0
    event_overlap = len(claim_tokens.intersection(EVENT_KEYWORDS)) / len(claim_tokens)
    return 0.15 if event_overlap >= 0.1 else 0.08


def _eight_k_sufficiency_bonus(request: ClaimVerificationRequest) -> float:
    item_numbers = _detected_item_numbers(request)
    if not item_numbers.intersection(EIGHT_K_MATERIAL_ITEM_NUMBERS):
        return 0.0
    evidence_count = len(request.cited_evidence_refs) + len(request.retrieved_snippets)
    if evidence_count == 0:
        return 0.0
    return 0.18 if evidence_count >= 2 else 0.1


def _detected_item_numbers(request: ClaimVerificationRequest) -> set[str]:
    detected = {
        chunk.item_number.strip()
        for chunk in request.retrieved_snippets
        if chunk.item_number and chunk.item_number.strip()
    }
    detected.update(
        match.group(1)
        for ref in request.cited_evidence_refs
        for match in [re.search(r"\bitem\s+(\d+\.\d{2})\b", ref.excerpt.lower())]
        if match is not None
    )
    return detected


def _is_item_502_compensation_event(request: ClaimVerificationRequest) -> bool:
    item_numbers = _detected_item_numbers(request)
    if "5.02" not in item_numbers:
        return False
    lowered = " ".join(
        [
            request.claim,
            *[ref.excerpt for ref in request.cited_evidence_refs],
            *[chunk.text for chunk in request.retrieved_snippets],
        ]
    ).lower()
    markers = (
        "compensation committee",
        "variable compensation plan",
        "executive officers",
        "compensation plan",
    )
    return any(marker in lowered for marker in markers)


def _supported_confidence_label(
    *,
    request: ClaimVerificationRequest,
    alignment_score: float,
    sufficiency_score: float,
    strong_claim: bool,
) -> ConfidenceLabel:
    if strong_claim:
        return ConfidenceLabel.MEDIUM
    if _is_eight_k_event_request(request):
        return ConfidenceLabel.MEDIUM
    if _is_item_901_exhibit_request(request):
        return ConfidenceLabel.MEDIUM
    if _is_simple_segment_claim(request):
        return ConfidenceLabel.MEDIUM
    if _is_numeric_financial_claim(request) and alignment_score >= 0.6 and sufficiency_score >= 0.7:
        return ConfidenceLabel.HIGH
    return ConfidenceLabel.MEDIUM


def _is_item_901_exhibit_request(request: ClaimVerificationRequest) -> bool:
    item_numbers = _detected_item_numbers(request)
    if "9.01" not in item_numbers:
        return False
    lowered = " ".join(
        [
            request.claim,
            *[ref.excerpt for ref in request.cited_evidence_refs],
            *[chunk.text for chunk in request.retrieved_snippets],
        ]
    ).lower()
    return "exhibit filed" in lowered or "financial statements and exhibits" in lowered


def _is_simple_segment_claim(request: ClaimVerificationRequest) -> bool:
    lowered = request.claim.lower().strip()
    markers = (
        "reports business results in two segments",
        "report our business results in two segments",
    )
    return any(marker in lowered for marker in markers)


def _is_generic_supported_claim(request: ClaimVerificationRequest) -> bool:
    lowered = request.claim.lower()
    return any(pattern in lowered for pattern in GENERIC_FALLBACK_PATTERNS)


def _is_numeric_financial_claim(request: ClaimVerificationRequest) -> bool:
    if request.numeric_facts:
        return True
    lowered = request.claim.lower()
    markers = ("revenue", "income", "cash", "margin", "eps", "$")
    return any(marker in lowered for marker in markers)
