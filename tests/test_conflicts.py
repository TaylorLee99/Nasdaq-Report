from __future__ import annotations

from datetime import date

from app.domain import (
    AgentFinding,
    AgentOutputPacket,
    Company,
    ConfidenceLabel,
    ConflictType,
    CoverageLabel,
    CoverageStatus,
    EvidenceTypeLabel,
    FilingType,
    FindingSignalType,
    TimeHorizonLabel,
    VerificationLabel,
    VerificationStatus,
)
from app.graph.conflicts import collect_conflicts


def build_company() -> Company:
    return Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")


def build_finding(
    *,
    finding_id: str,
    agent_name: str,
    claim: str,
    summary: str,
    signal_type: FindingSignalType,
    time_horizon: TimeHorizonLabel,
    evidence_type: EvidenceTypeLabel,
    as_of_date: date,
    filing_type: FilingType,
    covered_topics: list[str],
) -> AgentFinding:
    return AgentFinding(
        finding_id=finding_id,
        agent_name=agent_name,
        company=build_company(),
        claim=claim,
        summary=summary,
        signal_type=signal_type,
        time_horizon=time_horizon,
        evidence_type=evidence_type,
        as_of_date=as_of_date,
        filing_types=[filing_type],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=covered_topics,
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )


def test_conflict_checker_emits_narrative_tension_for_margin_pressure_vs_call_optimism() -> None:
    packets = [
        AgentOutputPacket(
            agent_name="run_10q_agent",
            findings=[
                build_finding(
                    finding_id="f-10q",
                    agent_name="run_10q_agent",
                    claim="Gross margin faces pressure this quarter.",
                    summary="Quarterly disclosures show margin compression and weaker mix.",
                    signal_type=FindingSignalType.FUNDAMENTAL,
                    time_horizon=TimeHorizonLabel.RECENT,
                    evidence_type=EvidenceTypeLabel.NUMERIC,
                    as_of_date=date(2024, 11, 20),
                    filing_type=FilingType.FORM_10Q,
                    covered_topics=["margin"],
                )
            ],
        ),
        AgentOutputPacket(
            agent_name="run_call_agent",
            findings=[
                build_finding(
                    finding_id="f-call",
                    agent_name="run_call_agent",
                    claim="Management remains optimistic about margin trajectory.",
                    summary="Prepared remarks frame gross margin as stable and improving.",
                    signal_type=FindingSignalType.MANAGEMENT_TONE,
                    time_horizon=TimeHorizonLabel.RECENT,
                    evidence_type=EvidenceTypeLabel.NARRATIVE,
                    as_of_date=date(2024, 11, 20),
                    filing_type=FilingType.EARNINGS_CALL,
                    covered_topics=["margin"],
                )
            ],
        ),
    ]

    conflicts = collect_conflicts(packets)

    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.NARRATIVE_TENSION
    assert conflicts[0].shared_topic == "margin"


def test_conflict_checker_emits_temporal_mismatch_for_old_optimism_vs_recent_event() -> None:
    packets = [
        AgentOutputPacket(
            agent_name="run_10k_agent",
            findings=[
                build_finding(
                    finding_id="f-10k",
                    agent_name="run_10k_agent",
                    claim="Management described demand as strong and resilient.",
                    summary="The annual filing emphasized stable demand conditions.",
                    signal_type=FindingSignalType.FUNDAMENTAL,
                    time_horizon=TimeHorizonLabel.LONG_TERM,
                    evidence_type=EvidenceTypeLabel.NARRATIVE,
                    as_of_date=date(2024, 2, 21),
                    filing_type=FilingType.FORM_10K,
                    covered_topics=["demand"],
                )
            ],
        ),
        AgentOutputPacket(
            agent_name="run_8k_agent",
            findings=[
                build_finding(
                    finding_id="f-8k",
                    agent_name="run_8k_agent",
                    claim="A recent adverse event creates downside risk to demand.",
                    summary="The 8-K disclosed a negative development affecting orders.",
                    signal_type=FindingSignalType.EVENT,
                    time_horizon=TimeHorizonLabel.POINT_IN_TIME,
                    evidence_type=EvidenceTypeLabel.NARRATIVE,
                    as_of_date=date(2024, 12, 10),
                    filing_type=FilingType.FORM_8K,
                    covered_topics=["demand"],
                )
            ],
        ),
    ]

    conflicts = collect_conflicts(packets)

    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.TEMPORAL_MISMATCH
    assert conflicts[0].recency_gap_days is not None
    assert conflicts[0].recency_gap_days >= 120


def test_conflict_checker_emits_factual_conflict_for_same_fact_pattern() -> None:
    packets = [
        AgentOutputPacket(
            agent_name="run_10q_agent",
            findings=[
                build_finding(
                    finding_id="f-liq-pos",
                    agent_name="run_10q_agent",
                    claim="Liquidity remains strong with stable cash flow.",
                    summary="Quarterly liquidity indicators remain resilient.",
                    signal_type=FindingSignalType.FUNDAMENTAL,
                    time_horizon=TimeHorizonLabel.RECENT,
                    evidence_type=EvidenceTypeLabel.NARRATIVE,
                    as_of_date=date(2024, 11, 20),
                    filing_type=FilingType.FORM_10Q,
                    covered_topics=["liquidity"],
                )
            ],
        ),
        AgentOutputPacket(
            agent_name="run_10q_agent_b",
            findings=[
                build_finding(
                    finding_id="f-liq-neg",
                    agent_name="run_10q_agent_b",
                    claim="Liquidity is weakening and cash flow is under pressure.",
                    summary="The same quarter suggests deteriorating liquidity.",
                    signal_type=FindingSignalType.FUNDAMENTAL,
                    time_horizon=TimeHorizonLabel.RECENT,
                    evidence_type=EvidenceTypeLabel.NARRATIVE,
                    as_of_date=date(2024, 11, 25),
                    filing_type=FilingType.FORM_10Q,
                    covered_topics=["liquidity"],
                )
            ],
        ),
    ]

    conflicts = collect_conflicts(packets)

    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.FACTUAL_CONFLICT
    assert conflicts[0].shared_topic == "liquidity"
    assert conflicts[0].verification_status.confidence == ConfidenceLabel.HIGH


def test_conflict_checker_avoids_over_detection_when_topics_do_not_overlap() -> None:
    packets = [
        AgentOutputPacket(
            agent_name="run_10k_agent",
            findings=[
                build_finding(
                    finding_id="f-risk",
                    agent_name="run_10k_agent",
                    claim="Risk exposure is increasing.",
                    summary="Risk factors became more adverse.",
                    signal_type=FindingSignalType.RISK,
                    time_horizon=TimeHorizonLabel.LONG_TERM,
                    evidence_type=EvidenceTypeLabel.NARRATIVE,
                    as_of_date=date(2024, 2, 21),
                    filing_type=FilingType.FORM_10K,
                    covered_topics=["risk"],
                )
            ],
        ),
        AgentOutputPacket(
            agent_name="run_call_agent",
            findings=[
                build_finding(
                    finding_id="f-demand",
                    agent_name="run_call_agent",
                    claim="Demand remains strong and improving.",
                    summary="Management remains optimistic about orders.",
                    signal_type=FindingSignalType.MANAGEMENT_TONE,
                    time_horizon=TimeHorizonLabel.RECENT,
                    evidence_type=EvidenceTypeLabel.NARRATIVE,
                    as_of_date=date(2024, 11, 20),
                    filing_type=FilingType.EARNINGS_CALL,
                    covered_topics=["demand"],
                )
            ],
        ),
    ]

    conflicts = collect_conflicts(packets)

    assert conflicts == []
