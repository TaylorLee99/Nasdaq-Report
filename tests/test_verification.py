from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.domain import (
    AgentFinding,
    AgentOutputPacket,
    ChunkRecord,
    ClaimVerificationRequest,
    Company,
    ConfidenceLabel,
    ConflictCandidate,
    ConflictType,
    CoverageLabel,
    CoverageStatus,
    EvidenceRef,
    FilingType,
    VerificationLabel,
    VerificationStatus,
    XbrlCanonicalFact,
    XbrlFact,
)
from app.graph.verification import HeuristicEvidenceVerifier, apply_verification_results

COMPANY = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")


def make_evidence(evidence_id: str, excerpt: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        document_id="doc-1",
        filing_type=FilingType.FORM_10Q,
        excerpt=excerpt,
    )


def make_8k_evidence(evidence_id: str, excerpt: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        document_id="doc-8k",
        filing_type=FilingType.FORM_8K,
        excerpt=excerpt,
    )


def make_numeric_fact() -> XbrlFact:
    return XbrlFact(
        fact_id="fact-1",
        accession_number="0001045810-24-000001",
        company=COMPANY,
        filing_type=FilingType.FORM_10Q,
        canonical_fact=XbrlCanonicalFact.REVENUE,
        original_concept="RevenueFromContractWithCustomerExcludingAssessedTax",
        numeric_value=Decimal("100.0"),
        filing_date=date(2024, 11, 20),
    )


def make_conflict(conflict_type: ConflictType) -> ConflictCandidate:
    return ConflictCandidate(
        conflict_id="conflict-1",
        conflict_type=conflict_type,
        claim="Revenue declined sharply in the quarter.",
        finding_ids=["finding-1"],
        reason="Opposing signal detected.",
    )


def make_finding() -> AgentFinding:
    return AgentFinding(
        finding_id="finding-1",
        agent_name="run_10q_agent",
        company=COMPANY,
        claim="Revenue growth remained strong in the quarter.",
        summary="Quarterly revenue claim.",
        filing_types=[FilingType.FORM_10Q],
        verification_status=VerificationStatus(
            label=VerificationLabel.INSUFFICIENT,
            confidence=ConfidenceLabel.LOW,
        ),
        coverage_status=CoverageStatus(label=CoverageLabel.COMPLETE),
    )


def test_verifier_marks_supported_when_alignment_and_sufficiency_are_strong() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-1",
        agent_name="run_10q_agent",
        claim="Revenue growth remained strong in the quarter.",
        cited_evidence_refs=[
            make_evidence("ev-1", "Revenue growth remained strong in the quarter."),
            make_evidence(
                "ev-2",
                "The quarter showed strong revenue growth from data center demand.",
            ),
        ],
        numeric_facts=[make_numeric_fact()],
    )

    result = verifier.verify_claim(request)

    assert result.label == VerificationLabel.SUPPORTED
    assert result.confidence == ConfidenceLabel.HIGH
    assert result.reretrieval_recommended is False
    assert set(result.supporting_evidence_ids) == {"ev-1", "ev-2"}


def test_verifier_keeps_non_numeric_supported_claims_at_medium_confidence() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-1b",
        agent_name="run_10k_agent",
        claim="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        cited_evidence_refs=[
            make_evidence(
                "ev-1b",
                (
                    "NVIDIA is now a data center scale AI infrastructure company "
                    "reshaping all industries."
                ),
            ),
            make_evidence(
                "ev-2b",
                "NVIDIA reshapes industries with data center scale AI infrastructure.",
            ),
        ],
        retrieved_snippets=[
            ChunkRecord(
                chunk_id="chunk-1b",
                document_id="doc-1b",
                section_id="sec-1b",
                company=COMPANY,
                filing_type=FilingType.FORM_10K,
                section_name="Business",
                text=(
                    "NVIDIA is now a data center scale AI infrastructure company "
                    "reshaping all industries."
                ),
                token_count=14,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ],
    )

    result = verifier.verify_claim(request)

    assert result.label == VerificationLabel.SUPPORTED
    assert result.confidence == ConfidenceLabel.MEDIUM


def test_verifier_marks_partially_supported_when_alignment_is_moderate() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-2",
        agent_name="run_10q_agent",
        claim="Revenue growth remained strong in the quarter.",
        cited_evidence_refs=[make_evidence("ev-1", "Revenue growth improved this quarter.")],
        retrieved_snippets=[],
        numeric_facts=[make_numeric_fact()],
    )

    result = verifier.verify_claim(request)

    assert result.label == VerificationLabel.PARTIALLY_SUPPORTED
    assert result.confidence == ConfidenceLabel.MEDIUM
    assert result.reretrieval_recommended is True
    assert result.requested_queries


def test_verifier_marks_insufficient_for_unsupported_strong_claim() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-3",
        agent_name="run_call_agent",
        claim="Margin will definitely recover immediately.",
    )

    result = verifier.verify_claim(request)

    assert result.label == VerificationLabel.INSUFFICIENT
    assert result.confidence == ConfidenceLabel.LOW
    assert result.reretrieval_recommended is True


def test_verifier_marks_conflicting_for_factual_conflict() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-4",
        agent_name="run_10q_agent",
        claim="Revenue growth remained strong in the quarter.",
        cited_evidence_refs=[
            make_evidence("ev-1", "Revenue growth remained strong in the quarter.")
        ],
        conflict_candidates=[make_conflict(ConflictType.FACTUAL_CONFLICT)],
    )

    result = verifier.verify_claim(request)

    assert result.label == VerificationLabel.CONFLICTING
    assert result.conflicting_conflict_ids == ["conflict-1"]
    assert result.reretrieval_recommended is False


def test_verifier_marks_unavailable_when_coverage_is_missing() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-5",
        agent_name="run_call_agent",
        claim="Management guided to stronger margins.",
        missing_coverage_topics=["transcript"],
    )

    result = verifier.verify_claim(request)

    assert result.label == VerificationLabel.UNAVAILABLE
    assert result.confidence == ConfidenceLabel.LOW
    assert result.missing_coverage_topics == ["transcript"]
    assert result.reretrieval_recommended is False


def test_verifier_uses_retrieved_snippets_before_marking_claim_unavailable() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-6",
        agent_name="run_10q_agent",
        claim="Liquidity remained strong in the quarter.",
        cited_evidence_refs=[
            make_evidence("ev-1", "Liquidity remained strong in the quarter.")
        ],
        retrieved_snippets=[
            ChunkRecord(
                chunk_id="chunk-1",
                document_id="doc-1",
                section_id="sec-1",
                company=COMPANY,
                filing_type=FilingType.FORM_10Q,
                section_name="Liquidity and Capital Resources",
                text="Liquidity remained strong in the quarter with significant cash.",
                token_count=9,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ],
        missing_coverage_topics=["some_unresolved_detail"],
    )

    result = verifier.verify_claim(request)

    assert result.label in {
        VerificationLabel.SUPPORTED,
        VerificationLabel.PARTIALLY_SUPPORTED,
        VerificationLabel.INSUFFICIENT,
    }
    assert result.label != VerificationLabel.UNAVAILABLE


def test_verifier_gives_8k_item_disclosures_event_specific_credit() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-7",
        agent_name="run_8k_agent",
        claim=(
            "The 8-K disclosed a compensation committee action and leadership-related "
            "update."
        ),
        cited_evidence_refs=[
            make_8k_evidence(
                "ev-8k-1",
                (
                    "Item 5.02 On March 2, 2026, NVIDIA's Compensation Committee adopted "
                    "the Fiscal Year 2027 Variable Compensation Plan."
                ),
            )
        ],
        retrieved_snippets=[
            ChunkRecord(
                chunk_id="chunk-8k-1",
                document_id="doc-8k-1",
                section_id="sec-8k-1",
                company=COMPANY,
                filing_type=FilingType.FORM_8K,
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                text=(
                    "On March 2, 2026, NVIDIA's Compensation Committee adopted the "
                    "Fiscal Year 2027 Variable Compensation Plan."
                ),
                token_count=18,
                parse_confidence=ConfidenceLabel.HIGH,
                item_number="5.02",
            )
        ],
    )

    result = verifier.verify_claim(request)

    assert result.label in {
        VerificationLabel.SUPPORTED,
        VerificationLabel.PARTIALLY_SUPPORTED,
    }
    assert result.confidence in {ConfidenceLabel.HIGH, ConfidenceLabel.MEDIUM}
    assert "8-K" in result.rationale or "Item-anchored" in result.rationale


def test_verifier_prefers_compact_item_502_event_claims() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-7b",
        agent_name="run_8k_agent",
        claim=(
            "Item 5.02 disclosed: On March 2, 2026, NVIDIA adopted the Variable "
            "Compensation Plan for Fiscal Year 2027 for eligible executive officers."
        ),
        cited_evidence_refs=[
            make_8k_evidence(
                "ev-8k-502-compact",
                (
                    "Item 5.02 On March 2, 2026, NVIDIA's Compensation Committee adopted "
                    "the Fiscal Year 2027 Variable Compensation Plan."
                ),
            )
        ],
        retrieved_snippets=[
            ChunkRecord(
                chunk_id="chunk-8k-502-compact",
                document_id="doc-8k-502-compact",
                section_id="sec-8k-502-compact",
                company=COMPANY,
                filing_type=FilingType.FORM_8K,
                section_name="Item 5.02 Departure of Directors or Certain Officers",
                text=(
                    "On March 2, 2026, NVIDIA's Compensation Committee adopted the "
                    "Fiscal Year 2027 Variable Compensation Plan."
                ),
                token_count=16,
                parse_confidence=ConfidenceLabel.HIGH,
                item_number="5.02",
            )
        ],
    )

    result = verifier.verify_claim(request)

    assert result.label == VerificationLabel.SUPPORTED
    assert result.confidence in {ConfidenceLabel.HIGH, ConfidenceLabel.MEDIUM}


def test_verifier_downshifts_confidence_for_item_901_exhibit_claims() -> None:
    verifier = HeuristicEvidenceVerifier()
    request = ClaimVerificationRequest(
        claim_id="claim-901",
        agent_name="run_8k_agent",
        claim='Exhibit filed: "Variable Compensation Plan - Fiscal Year 2027."',
        cited_evidence_refs=[
            make_8k_evidence(
                "ev-8k-901",
                (
                    "Item 9.01 Financial Statements and Exhibits. Exhibit filed: "
                    '"Variable Compensation Plan - Fiscal Year 2027."'
                ),
            )
        ],
        retrieved_snippets=[
            ChunkRecord(
                chunk_id="chunk-8k-901",
                document_id="doc-8k-901",
                section_id="sec-8k-901",
                company=COMPANY,
                filing_type=FilingType.FORM_8K,
                section_name="Item 9.01 Financial Statements and Exhibits",
                text='Exhibit filed: "Variable Compensation Plan - Fiscal Year 2027."',
                token_count=9,
                parse_confidence=ConfidenceLabel.HIGH,
                item_number="9.01",
            )
        ],
    )

    result = verifier.verify_claim(request)

    assert result.label == VerificationLabel.PARTIALLY_SUPPORTED
    assert result.confidence == ConfidenceLabel.MEDIUM


def test_verifier_prefers_excerpt_grounded_claim_over_generic_fallback_claim() -> None:
    verifier = HeuristicEvidenceVerifier()
    generic_request = ClaimVerificationRequest(
        claim_id="claim-generic",
        agent_name="run_10q_agent",
        claim="The latest 10-Q supports a verified recent-quarter change finding for NVIDIA.",
        cited_evidence_refs=[
            make_evidence(
                "ev-generic",
                "Revenue increased in the latest quarter while liquidity remained strong.",
            )
        ],
        retrieved_snippets=[
            ChunkRecord(
                chunk_id="chunk-generic",
                document_id="doc-generic",
                section_id="sec-generic",
                company=COMPANY,
                filing_type=FilingType.FORM_10Q,
                section_name="Liquidity and Capital Resources",
                text="Revenue increased in the latest quarter while liquidity remained strong.",
                token_count=11,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ],
    )
    grounded_request = ClaimVerificationRequest(
        claim_id="claim-grounded",
        agent_name="run_10q_agent",
        claim="Revenue increased in the latest quarter while liquidity remained strong.",
        cited_evidence_refs=generic_request.cited_evidence_refs,
        retrieved_snippets=generic_request.retrieved_snippets,
    )

    generic_result = verifier.verify_claim(generic_request)
    grounded_result = verifier.verify_claim(grounded_request)

    assert grounded_result.alignment_score > generic_result.alignment_score
    assert grounded_result.sufficiency_score >= generic_result.sufficiency_score


def test_apply_verification_results_updates_finding_statuses() -> None:
    finding = make_finding().model_copy(
        update={
            "evidence_refs": [
                make_evidence("ev-1", "Revenue growth remained strong in the quarter.")
            ]
        }
    )
    packet = AgentOutputPacket(
        agent_name="run_10q_agent",
        findings=[finding],
        retrieved_snippets=[
            ChunkRecord(
                chunk_id="chunk-1",
                document_id="doc-1",
                section_id="sec-1",
                company=COMPANY,
                filing_type=FilingType.FORM_10Q,
                section_name="Management's Discussion and Analysis",
                text="Revenue growth remained strong in the quarter.",
                token_count=7,
                parse_confidence=ConfidenceLabel.HIGH,
            )
        ],
    )
    verifier = HeuristicEvidenceVerifier()
    results = verifier.verify_packets([packet], conflicts=[])

    updated_packets = apply_verification_results([packet], results)

    assert (
        updated_packets[0].findings[0].verification_status.label
        == VerificationLabel.PARTIALLY_SUPPORTED
    )
    assert (
        updated_packets[0].findings[0].verification_status.verifier_name
        == "heuristic_evidence_verifier"
    )
