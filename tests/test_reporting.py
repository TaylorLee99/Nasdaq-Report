from __future__ import annotations

import json

from app.domain import (
    AgentFinding,
    AgentOutputPacket,
    ClaimVerificationResult,
    Company,
    ConfidenceLabel,
    CoverageLabel,
    CoverageStatus,
    EvidenceRef,
    FilingType,
    FindingSignalType,
    VerificationLabel,
    VerificationStatus,
)
from app.reporting import build_verified_report, render_report_markdown, serialize_report_json


def make_evidence(evidence_id: str, excerpt: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        filing_type=FilingType.FORM_10Q,
        excerpt=excerpt,
        section_id="section-1",
        chunk_id=f"chunk-{evidence_id}",
    )


def make_finding(
    *,
    finding_id: str,
    claim: str,
    signal_type: FindingSignalType,
    verification_label: VerificationLabel,
    confidence: ConfidenceLabel,
) -> AgentFinding:
    return AgentFinding(
        finding_id=finding_id,
        agent_name="run_10q_agent",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        claim=claim,
        summary=claim,
        signal_type=signal_type,
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[make_evidence(finding_id, claim)],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        verification_status=VerificationStatus(
            label=verification_label,
            confidence=confidence,
        ),
    )


def make_result(
    *,
    finding_id: str,
    label: VerificationLabel,
    confidence: ConfidenceLabel,
    missing_topics: list[str] | None = None,
) -> ClaimVerificationResult:
    return ClaimVerificationResult(
        claim_id=finding_id,
        label=label,
        confidence=confidence,
        rationale="verification result",
        alignment_score=0.8 if label == VerificationLabel.SUPPORTED else 0.45,
        sufficiency_score=0.7 if label == VerificationLabel.SUPPORTED else 0.4,
        missing_coverage_topics=missing_topics or [],
    )


def test_build_verified_report_excludes_unsupported_claims_and_renders_sections() -> None:
    supported = make_finding(
        finding_id="finding-supported",
        claim="Data center demand remains strong.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        verification_label=VerificationLabel.INSUFFICIENT,
        confidence=ConfidenceLabel.LOW,
    )
    partial = make_finding(
        finding_id="finding-partial",
        claim="Gross margin pressure remains a watchpoint.",
        signal_type=FindingSignalType.RISK,
        verification_label=VerificationLabel.INSUFFICIENT,
        confidence=ConfidenceLabel.LOW,
    )
    unsupported = make_finding(
        finding_id="finding-unsupported",
        claim="Management will definitely restore margins next quarter.",
        signal_type=FindingSignalType.GUIDANCE,
        verification_label=VerificationLabel.INSUFFICIENT,
        confidence=ConfidenceLabel.LOW,
    )
    packet = AgentOutputPacket(
        agent_name="run_10q_agent",
        findings=[supported, partial, unsupported],
        coverage_status=CoverageStatus(
            label=CoverageLabel.PARTIAL,
            covered_topics=["recent_quarter_change"],
            missing_topics=["transcript"],
        ),
    )
    report = build_verified_report(
        report_id="report-1",
        company=supported.company,
        question="What changed recently?",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-supported",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-partial",
                label=VerificationLabel.PARTIALLY_SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            ),
            make_result(
                finding_id="finding-unsupported",
                label=VerificationLabel.INSUFFICIENT,
                confidence=ConfidenceLabel.LOW,
            ),
        ],
        coverage_status=packet.coverage_status,
        include_transcript=False,
    )
    included_ids = {
        claim.finding_id
        for claim in (
            report.evidence_grounded_report.business_overview
            + report.evidence_grounded_report.recent_quarter_change
            + report.evidence_grounded_report.material_events
        )
    }

    assert included_ids == {"finding-supported", "finding-partial"}
    assert report.key_opportunities[0].finding_id == "finding-supported"
    assert report.key_risks[0].finding_id == "finding-partial"
    assert report.coverage_summary.missing_topics == []
    assert report.watchpoints[0].finding_id == "finding-partial"
    assert report.analysis_scope.allowed_sources == [
        FilingType.FORM_10K,
        FilingType.FORM_10Q,
        FilingType.FORM_8K,
    ]
    assert 1 <= report.executive_summary.summary.count(".") <= 4
    assert len(report.executive_summary.evidence_refs) <= 3
    assert report.verification_summary.insufficient_claim_count == 1
    assert report.final_investment_thesis.thesis
    assert report.appendix.excluded_claims[0].finding_id == "finding-unsupported"


def test_report_falls_back_to_structural_and_liquidity_opportunities_and_watchpoints() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    long_term = AgentFinding(
        finding_id="finding-10k-structure-fallback",
        agent_name="run_10k_agent",
        company=company,
        claim="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        summary="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-10k-structure-fallback",
                document_id="doc-10k-structure-fallback",
                filing_type=FilingType.FORM_10K,
                excerpt=(
                    "NVIDIA is now a data center scale AI infrastructure company "
                    "reshaping all industries."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )
    liquidity = AgentFinding(
        finding_id="finding-10q-liquidity-fallback",
        agent_name="run_10q_agent",
        company=company,
        claim=(
            "As of October 26, 2025, NVIDIA had $60.6 billion in cash, cash "
            "equivalents, and marketable securities."
        ),
        summary=(
            "As of October 26, 2025, NVIDIA had $60.6 billion in cash, cash "
            "equivalents, and marketable securities."
        ),
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-10q-liquidity-fallback",
                document_id="doc-10q-liquidity-fallback",
                filing_type=FilingType.FORM_10Q,
                excerpt=(
                    "As of October 26, 2025, we had $60.6 billion in cash, cash "
                    "equivalents, and marketable securities."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="mixed", findings=[long_term, liquidity])

    report = build_verified_report(
        report_id="report-fallback-opportunities",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-10k-structure-fallback",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            ),
            make_result(
                finding_id="finding-10q-liquidity-fallback",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure", "recent_quarter_change"],
        ),
        include_transcript=False,
    )

    assert {claim.finding_id for claim in report.key_opportunities} == {
        "finding-10k-structure-fallback",
        "finding-10q-liquidity-fallback",
    }
    assert report.watchpoints[0].finding_id == "finding-10k-structure-fallback"
    assert report.watchpoints[0].trigger_to_monitor == "long_term_structure"


def test_report_normalizes_missing_topics_and_trigger_labels() -> None:
    finding = make_finding(
        finding_id="finding-gap",
        claim="Liquidity remained stable in the quarter.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        verification_label=VerificationLabel.INSUFFICIENT,
        confidence=ConfidenceLabel.LOW,
    ).model_copy(
        update={
            "coverage_status": CoverageStatus(
                label=CoverageLabel.PARTIAL,
                covered_topics=["recent_quarter_change"],
                missing_topics=[
                    (
                        "The provided snippets focus on a specific executive compensation "
                        "plan and do not offer a comprehensive overview of the company's "
                        "long-term structure or other recent quarter changes."
                    )
                ],
            ),
            "verification_status": VerificationStatus(
                label=VerificationLabel.PARTIALLY_SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            ),
        }
    )
    packet = AgentOutputPacket(agent_name="run_10q_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-gaps",
        company=finding.company,
        question="What changed recently?",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-gap",
                label=VerificationLabel.PARTIALLY_SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
                missing_topics=[
                    (
                        "The provided snippets focus on a specific executive compensation "
                        "plan and do not offer a comprehensive overview of the company's "
                        "long-term structure or other recent quarter changes."
                    )
                ],
            )
        ],
        coverage_status=finding.coverage_status,
        include_transcript=False,
    )

    assert report.coverage_summary.missing_topics == [
        "long_term_structure",
        "material_events",
    ]
    assert report.watchpoints[0].trigger_to_monitor == "long_term_structure"
    assert report.appendix.coverage_gaps[0].topic == "long_term_structure"


def test_report_renderers_output_markdown_and_json() -> None:
    finding = make_finding(
        finding_id="finding-supported",
        claim="Data center demand remains strong.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        verification_label=VerificationLabel.SUPPORTED,
        confidence=ConfidenceLabel.HIGH,
    )
    packet = AgentOutputPacket(agent_name="run_10q_agent", findings=[finding])
    report = build_verified_report(
        report_id="report-2",
        company=finding.company,
        question="What changed recently?",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-supported",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        include_transcript=True,
    )

    markdown = render_report_markdown(report)
    payload = json.loads(serialize_report_json(report))

    assert "## Executive Summary" in markdown
    assert "## Evidence-Grounded Report" in markdown
    assert "## Verification Summary" in markdown
    assert "## Appendix" in markdown
    assert "Presentation Note:" in markdown
    assert "- Claim: Data center demand remains strong." in markdown
    assert "- Evidence `finding-supported`" in markdown
    assert "Missing channels: 10-K, 8-K" in markdown
    assert "No verified risk finding" in markdown
    assert "Monitoring heuristic" in markdown
    assert (
        payload["evidence_grounded_report"]["recent_quarter_change"][0]["finding_id"]
        == "finding-supported"
    )
    assert payload["verification_summary"]["supported_claim_count"] == 1


def test_final_thesis_prefers_material_event_before_liquidity_and_structure() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    ten_k = AgentFinding(
        finding_id="finding-order-10k",
        agent_name="run_10k_agent",
        company=company,
        claim="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        summary="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[make_evidence("order-10k", "Business structure evidence.")],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )
    ten_q = AgentFinding(
        finding_id="finding-order-10q",
        agent_name="run_10q_agent",
        company=company,
        claim=(
            "As of October 26, 2025, NVIDIA had $60.6 billion in cash, cash "
            "equivalents, and marketable securities."
        ),
        summary=(
            "As of October 26, 2025, NVIDIA had $60.6 billion in cash, cash "
            "equivalents, and marketable securities."
        ),
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[make_evidence("order-10q", "Liquidity evidence.")],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    eight_k = AgentFinding(
        finding_id="finding-order-8k",
        agent_name="run_8k_agent",
        company=company,
        claim=(
            "On March 2, 2026, NVIDIA disclosed an Item 5.02 compensation update and "
            "adopted the Variable Compensation Plan for Fiscal Year 2027 for eligible "
            "executive officers."
        ),
        summary=(
            "On March 2, 2026, NVIDIA disclosed an Item 5.02 compensation update and "
            "adopted the Variable Compensation Plan for Fiscal Year 2027 for eligible "
            "executive officers."
        ),
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="order-8k",
                document_id="doc-order-8k",
                filing_type=FilingType.FORM_8K,
                excerpt="Item 5.02 compensation update evidence.",
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )
    packet = AgentOutputPacket(agent_name="mixed", findings=[ten_k, ten_q, eight_k])
    report = build_verified_report(
        report_id="report-order",
        company=company,
        question="Build a thesis.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-order-10k",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            ),
            make_result(
                finding_id="finding-order-10q",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-order-8k",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure", "recent_quarter_change", "material_events"],
        ),
        include_transcript=False,
    )

    thesis = report.final_investment_thesis.thesis
    assert thesis.index("March 2, 2026") < thesis.index("October 26, 2025")


def test_markdown_renderer_rewrites_supported_watchpoints_into_monitoring_notes() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding = AgentFinding(
        finding_id="finding-watchpoint-render",
        agent_name="run_10k_agent",
        company=company,
        claim="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        summary="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-watchpoint-render",
                document_id="doc-watchpoint-render",
                filing_type=FilingType.FORM_10K,
                excerpt=(
                    "NVIDIA is now a data center scale AI infrastructure company "
                    "reshaping all industries."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_10k_agent", findings=[finding])
    report = build_verified_report(
        report_id="report-watchpoint-render",
        company=company,
        question="Describe the business structure.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-watchpoint-render",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            )
        ],
        coverage_status=finding.coverage_status,
        include_transcript=False,
    )

    markdown = render_report_markdown(report)

    assert "## Watchpoints" in markdown
    assert "Monitoring heuristic" in markdown
    assert "Strategic Positioning Check" in markdown


def test_final_thesis_polishes_company_case_and_duplicate_punctuation() -> None:
    opportunity = make_finding(
        finding_id="finding-opportunity",
        claim=(
            "nvidia's long-term structure includes substantial liquidity as of october 26, 2025.."
        ),
        signal_type=FindingSignalType.FUNDAMENTAL,
        verification_label=VerificationLabel.INSUFFICIENT,
        confidence=ConfidenceLabel.LOW,
    ).model_copy(
        update={
            "summary": (
                "nvidia's long-term structure includes substantial liquidity as of october "
                "26, 2025.."
            ),
            "verification_status": VerificationStatus(
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
        }
    )
    risk = make_finding(
        finding_id="finding-risk",
        claim="capital expenditure remains elevated.",
        signal_type=FindingSignalType.RISK,
        verification_label=VerificationLabel.INSUFFICIENT,
        confidence=ConfidenceLabel.LOW,
    ).model_copy(
        update={
            "summary": "capital expenditure remains elevated..",
            "verification_status": VerificationStatus(
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            ),
        }
    )
    packet = AgentOutputPacket(agent_name="run_10q_agent", findings=[opportunity, risk])

    report = build_verified_report(
        report_id="report-thesis",
        company=opportunity.company,
        question="Build a thesis.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-opportunity",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-risk",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.PARTIAL,
            covered_topics=["recent_quarter_change"],
            missing_topics=["transcript"],
        ),
        include_transcript=False,
    )

    assert "NVIDIA" in report.final_investment_thesis.thesis
    assert "October 26, 2025" in report.final_investment_thesis.thesis
    assert ".." not in report.final_investment_thesis.thesis


def test_final_thesis_uses_multiple_supported_findings_across_filings() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    ten_k = AgentFinding(
        finding_id="finding-10k",
        agent_name="run_10k_agent",
        company=company,
        claim="The latest 10-K supports a verified long-term structure finding for NVIDIA.",
        summary="The latest 10-K supports a verified long-term structure finding for NVIDIA.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[make_evidence("10k", "Long-term structure evidence.")],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    ten_q = AgentFinding(
        finding_id="finding-10q",
        agent_name="run_10q_agent",
        company=company,
        claim="The latest 10-Q supports a verified recent-quarter change finding for NVIDIA.",
        summary="The latest 10-Q supports a verified recent-quarter change finding for NVIDIA.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[make_evidence("10q", "Recent-quarter change evidence.")],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    eight_k = AgentFinding(
        finding_id="finding-8k",
        agent_name="run_8k_agent",
        company=company,
        claim=(
            "On March 6, 2026, NVIDIA disclosed officer and compensation updates in an "
            "Item 5.02 8-K filing."
        ),
        summary=(
            "On March 6, 2026, NVIDIA disclosed officer and compensation updates in an "
            "Item 5.02 8-K filing."
        ),
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[make_evidence("8k", "Material event evidence.")],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="mixed", findings=[eight_k, ten_q, ten_k])

    report = build_verified_report(
        report_id="report-thesis-multi",
        company=company,
        question="Build a filing-only thesis.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-8k",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-10q",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-10k",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events", "recent_quarter_change", "long_term_structure"],
        ),
        include_transcript=False,
    )

    assert "Additional filing support shows" in report.final_investment_thesis.thesis
    assert "Further filings indicate" in report.final_investment_thesis.thesis


def test_final_thesis_adds_that_for_we_clause() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding = AgentFinding(
        finding_id="finding-10k-we-clause",
        agent_name="run_10k_agent",
        company=company,
        claim="We report our business results in two segments.",
        summary="We report our business results in two segments.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[make_evidence("10k-we", "We report our business results in two segments.")],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_10k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-thesis-we",
        company=company,
        question="Build a filing-only thesis.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-10k-we-clause",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        include_transcript=False,
    )

    assert "Verified disclosures show that NVIDIA reports business results in two segments." in (
        report.final_investment_thesis.thesis
    )


def test_final_thesis_rewrites_first_person_filing_clauses_to_analyst_style() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    liquidity_sentence = (
        "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, "
        "and marketable securities."
    )
    segment_sentence = "We report our business results in two segments."
    ten_q = AgentFinding(
        finding_id="finding-10q-analyst-style",
        agent_name="run_10q_agent",
        company=company,
        claim=liquidity_sentence,
        summary=liquidity_sentence,
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[make_evidence("10q-analyst-style", liquidity_sentence)],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    ten_k = AgentFinding(
        finding_id="finding-10k-analyst-style",
        agent_name="run_10k_agent",
        company=company,
        claim=segment_sentence,
        summary=segment_sentence,
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[make_evidence("10k-analyst-style", segment_sentence)],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="mixed", findings=[ten_q, ten_k])

    report = build_verified_report(
        report_id="report-thesis-analyst-style",
        company=company,
        question="Build a filing-only thesis.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-10q-analyst-style",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-10k-analyst-style",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change", "long_term_structure"],
        ),
        include_transcript=False,
    )

    assert "NVIDIA had $60.6 billion in cash" in report.final_investment_thesis.thesis
    assert (
        "NVIDIA reports business results in two segments"
        in report.final_investment_thesis.thesis
    )


def test_summary_and_thesis_prefer_evidence_backed_sentence_over_generic_fallback() -> None:
    finding = AgentFinding(
        finding_id="finding-evidence-backed",
        agent_name="run_10q_agent",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        claim="The latest 10-Q supports a verified recent-quarter change finding for NVIDIA.",
        summary="The latest 10-Q supports a verified recent-quarter change finding for NVIDIA.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-backed",
                document_id="doc-backed",
                filing_type=FilingType.FORM_10Q,
                excerpt=(
                    "Revenue increased in the latest quarter while liquidity remained strong."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_10q_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-evidence-backed",
        company=finding.company,
        question="What changed recently?",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-evidence-backed",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        include_transcript=False,
    )

    assert "Revenue increased in the latest quarter" in report.executive_summary.summary
    assert "Revenue increased in the latest quarter" in report.final_investment_thesis.thesis
    assert "we had" not in report.executive_summary.summary.lower()


def test_executive_summary_falls_back_from_fragmentary_sentences() -> None:
    finding = AgentFinding(
        finding_id="finding-8k-fragment",
        agent_name="run_8k_agent",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        claim=(
            "Item 5.02 disclosed that d by the Compensation Committee, a participant must "
            "remain an employee through the payment date."
        ),
        summary="d by the Compensation Committee, a participant must remain an employee.",
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        as_of_date=None,
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-8k-fragment",
                document_id="doc-8k-fragment",
                filing_type=FilingType.FORM_8K,
                excerpt="Item 5.02 disclosed a compensation committee action.",
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events", "item_5.02"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding])
    report = build_verified_report(
        report_id="report-fragment",
        company=finding.company,
        question="What happened?",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-8k-fragment",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        include_transcript=False,
    )

    assert "d by the Compensation Committee" not in report.executive_summary.summary
    assert "Item 5.02" in report.final_investment_thesis.thesis


def test_report_titles_surface_8k_item_numbers() -> None:
    finding = AgentFinding(
        finding_id="finding-8k",
        agent_name="run_8k_agent",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        claim=(
            "Item 5.02 disclosed that NVIDIA's Compensation Committee adopted the Fiscal "
            "Year 2027 Variable Compensation Plan."
        ),
        summary="Compensation committee action under Item 5.02.",
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-8k",
                document_id="doc-8k",
                filing_type=FilingType.FORM_8K,
                excerpt="Item 5.02 disclosed compensation committee action.",
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events", "item_5.02"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-8k",
        company=finding.company,
        question="What recent material events were disclosed?",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-8k",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        include_transcript=False,
    )

    assert (
        report.evidence_grounded_report.material_events[0].title
        == "Item 5.02 - Officer and Compensation Updates"
    )


def test_material_events_section_keeps_partial_event_findings() -> None:
    supported_event = AgentFinding(
        finding_id="finding-8k-supported",
        agent_name="run_8k_agent",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        claim="Item 5.02 disclosed a compensation committee action.",
        summary="Compensation committee action under Item 5.02.",
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-8k-supported",
                document_id="doc-8k-supported",
                filing_type=FilingType.FORM_8K,
                excerpt="Item 5.02 disclosed a compensation committee action.",
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events", "item_5.02"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    partial_event = AgentFinding(
        finding_id="finding-8k-partial",
        agent_name="run_8k_agent",
        company=supported_event.company,
        claim="Item 1.01 disclosed a new material agreement.",
        summary="Material agreement disclosure under Item 1.01.",
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-8k-partial",
                document_id="doc-8k-partial",
                filing_type=FilingType.FORM_8K,
                excerpt="Item 1.01 disclosed a new material agreement.",
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.PARTIAL,
            covered_topics=["material_events", "item_1.01"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.PARTIALLY_SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )
    packet = AgentOutputPacket(
        agent_name="run_8k_agent",
        findings=[supported_event, partial_event],
    )

    report = build_verified_report(
        report_id="report-8k-events",
        company=supported_event.company,
        question="What recent material events were disclosed?",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-8k-supported",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-8k-partial",
                label=VerificationLabel.PARTIALLY_SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        include_transcript=False,
    )

    assert [claim.finding_id for claim in report.evidence_grounded_report.material_events] == [
        "finding-8k-supported",
        "finding-8k-partial",
    ]


def test_report_titles_prefer_8k_section_labels_over_generic_summary() -> None:
    finding = AgentFinding(
        finding_id="finding-8k-901",
        agent_name="run_8k_agent",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        claim=(
            "Item 9.01 (Financial Statements and Exhibits) disclosed exhibits including "
            "Variable Compensation Plan - Fiscal Year 2027."
        ),
        summary="NVIDIA Corp",
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-8k-901",
                document_id="doc-8k-901",
                filing_type=FilingType.FORM_8K,
                excerpt=(
                    "Item 9.01 (Financial Statements and Exhibits) disclosed exhibits "
                    "including Variable Compensation Plan - Fiscal Year 2027."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events", "item_9.01"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-8k-901",
        company=finding.company,
        question="What exhibits were disclosed?",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-8k-901",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        include_transcript=False,
    )

    assert (
        report.evidence_grounded_report.material_events[0].title
        == "Item 9.01 - Financial Statements and Exhibits"
    )


def test_report_titles_use_template_titles_for_10k_and_10q_findings() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    ten_k_finding = AgentFinding(
        finding_id="finding-10k",
        agent_name="run_10k_agent",
        company=company,
        claim="The filing describes the company's platform mix and end markets.",
        summary="The filing describes the company's platform mix and end markets.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-10k",
                document_id="doc-10k",
                filing_type=FilingType.FORM_10K,
                excerpt="Item 1. Business describes platform mix and end markets.",
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    ten_q_finding = AgentFinding(
        finding_id="finding-10q",
        agent_name="run_10q_agent",
        company=company,
        claim="Liquidity and capital resources remained available during the quarter.",
        summary="Liquidity and capital resources remained available during the quarter.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-10q",
                document_id="doc-10q",
                filing_type=FilingType.FORM_10Q,
                excerpt="Liquidity and capital resources remained available during the quarter.",
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="mixed", findings=[ten_k_finding, ten_q_finding])

    report = build_verified_report(
        report_id="report-template-titles",
        company=company,
        question="Build a filing-only view.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-10k",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-10q",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure", "recent_quarter_change"],
        ),
        include_transcript=False,
    )

    assert report.evidence_grounded_report.business_overview[0].title == "10-K Long-Term Structure"
    assert (
        report.evidence_grounded_report.recent_quarter_change[0].title
        == "10-Q Liquidity and Recent Quarter Change"
    )


def test_report_claims_use_analyst_style_and_prefer_compact_8k_summary() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    ten_q = AgentFinding(
        finding_id="finding-10q-claim",
        agent_name="run_10q_agent",
        company=company,
        claim="As of October 26, 2025, we had $60.6 billion in cash and securities.",
        summary="As of October 26, 2025, we had $60.6 billion in cash and securities.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[
            make_evidence(
                "ev-10q-claim",
                "As of October 26, 2025, we had $60.6 billion in cash and securities.",
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    eight_k = AgentFinding(
        finding_id="finding-8k-compact",
        agent_name="run_8k_agent",
        company=company,
        claim=(
            "Item 5.02 (Departure of Directors or Certain Officers; Election of Directors; "
            "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers) "
            "disclosed: On March 2, 2026, the Compensation Committee of the Board of "
            "Directors, or the Board, of NVIDIA Corporation, or the Company, adopted the "
            "Variable Compensation Plan for Fiscal Year 2027."
        ),
        summary=(
            "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan for Fiscal "
            "Year 2027 for eligible executive officers."
        ),
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-8k-compact",
                document_id="doc-8k-compact",
                filing_type=FilingType.FORM_8K,
                excerpt=(
                    "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan "
                    "for Fiscal Year 2027 for eligible executive officers."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_10q_agent", findings=[ten_q, eight_k])
    report = build_verified_report(
        report_id="report-claim-normalization",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-10q-claim",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-8k-compact",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change", "material_events"],
        ),
        include_transcript=False,
    )

    assert report.evidence_grounded_report.recent_quarter_change[0].claim.startswith(
        "As of October 26, 2025, NVIDIA had"
    )
    assert (
        report.evidence_grounded_report.material_events[0].claim
        == (
            "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan for "
            "Fiscal Year 2027 for eligible executive officers."
        )
    )


def test_executive_summary_and_thesis_deprioritize_item_901_exhibit_only_findings() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    exhibit_only = AgentFinding(
        finding_id="finding-901",
        agent_name="run_8k_agent",
        company=company,
        claim='Item 9.01 (Financial Statements and Exhibits) disclosed the exhibit "Plan".',
        summary='Exhibit filed: "Plan".',
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-901",
                document_id="doc-901",
                filing_type=FilingType.FORM_8K,
                excerpt='Exhibit filed: "Plan".',
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    ten_q = make_finding(
        finding_id="finding-liquidity",
        claim="As of October 26, 2025, we had $60.6 billion in cash and securities.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        verification_label=VerificationLabel.SUPPORTED,
        confidence=ConfidenceLabel.HIGH,
    )
    packet = AgentOutputPacket(agent_name="run_10q_agent", findings=[exhibit_only, ten_q])
    report = build_verified_report(
        report_id="report-priority",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-901",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-liquidity",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
                missing_topics=["generic unresolved evidence gap"],
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        include_transcript=False,
    )

    assert "Exhibit filed" not in report.executive_summary.summary
    assert "Exhibit filed" not in report.final_investment_thesis.thesis
    assert report.coverage_summary.missing_topics == []


def test_executive_summary_handles_company_name_abbreviation_without_false_split() -> None:
    finding = AgentFinding(
        finding_id="finding-8k-corp",
        agent_name="run_8k_agent",
        company=Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp."),
        claim=(
            "NVIDIA Corp. adopted the Variable Compensation Plan for Fiscal Year 2027 "
            "on March 2, 2026."
        ),
        summary=(
            "NVIDIA Corp. adopted the Variable Compensation Plan for Fiscal Year 2027 "
            "on March 2, 2026."
        ),
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-8k-corp",
                document_id="doc-8k-corp",
                filing_type=FilingType.FORM_8K,
                excerpt=(
                    "NVIDIA Corp. adopted the Variable Compensation Plan for Fiscal "
                    "Year 2027 on March 2, 2026."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-corp-split",
        company=finding.company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-8k-corp",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        include_transcript=False,
    )

    assert report.executive_summary.summary.startswith("NVIDIA Corp. adopted")
    assert "NVIDIA Corp. Additional" not in report.final_investment_thesis.thesis


def test_material_events_prioritize_item_502_over_item_901() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding_901 = AgentFinding(
        finding_id="finding-901-order",
        agent_name="run_8k_agent",
        company=company,
        claim='Exhibit filed: "Plan".',
        summary='Exhibit filed: "Plan".',
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-901-order",
                document_id="doc-901-order",
                filing_type=FilingType.FORM_8K,
                excerpt='Exhibit filed: "Plan".',
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    finding_502 = AgentFinding(
        finding_id="finding-502-order",
        agent_name="run_8k_agent",
        company=company,
        claim=(
            "Item 5.02 disclosed: On March 2, 2026, NVIDIA adopted the Variable "
            "Compensation Plan for Fiscal Year 2027 for eligible executive officers."
        ),
        summary=(
            "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan for Fiscal "
            "Year 2027 for eligible executive officers."
        ),
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-502-order",
                document_id="doc-502-order",
                filing_type=FilingType.FORM_8K,
                excerpt=(
                    "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan for "
                    "Fiscal Year 2027 for eligible executive officers."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding_901, finding_502])

    report = build_verified_report(
        report_id="report-material-order",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-901-order",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
            make_result(
                finding_id="finding-502-order",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        include_transcript=False,
    )

    assert report.evidence_grounded_report.material_events[0].finding_id == "finding-502-order"
    assert [claim.finding_id for claim in report.evidence_grounded_report.material_events] == [
        "finding-502-order"
    ]


def test_report_claims_fallback_to_item_502_claim_when_summary_is_garbled() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding_502 = AgentFinding(
        finding_id="finding-502-garbled",
        agent_name="run_8k_agent",
        company=company,
        claim=(
            "Item 5.02 disclosed: On March 2, 2026, NVIDIA adopted the Variable "
            "Compensation Plan for Fiscal Year 2027 for eligible executive officers."
        ),
        summary="Kress Executive Vice President and Chief Financial Officer $1,500,000",
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-502-garbled",
                document_id="doc-502-garbled",
                filing_type=FilingType.FORM_8K,
                excerpt=(
                    "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan for "
                    "Fiscal Year 2027 for eligible executive officers."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding_502])

    report = build_verified_report(
        report_id="report-502-garbled",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-502-garbled",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        include_transcript=False,
    )

    material_claim = report.evidence_grounded_report.material_events[0].claim
    assert "Kress Executive Vice President" not in material_claim
    assert "Variable Compensation Plan for Fiscal Year 2027" in material_claim


def test_item_502_claims_are_rewritten_into_more_natural_report_sentences() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding_502 = AgentFinding(
        finding_id="finding-502-natural",
        agent_name="run_8k_agent",
        company=company,
        claim=(
            "Item 5.02 disclosed: On March 2, 2026, NVIDIA adopted the Variable "
            "Compensation Plan for Fiscal Year 2027 for eligible executive officers."
        ),
        summary=(
            "Item 5.02 disclosed: On March 2, 2026, NVIDIA adopted the Variable "
            "Compensation Plan for Fiscal Year 2027 for eligible executive officers."
        ),
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-502-natural",
                document_id="doc-502-natural",
                filing_type=FilingType.FORM_8K,
                excerpt=(
                    "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan for "
                    "Fiscal Year 2027 for eligible executive officers."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding_502])

    report = build_verified_report(
        report_id="report-502-natural",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-502-natural",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        include_transcript=False,
    )

    expected_prefix = (
        "On March 2, 2026, NVIDIA disclosed an Item 5.02 compensation update and adopted"
    )
    assert report.executive_summary.summary.startswith(expected_prefix)
    assert report.evidence_grounded_report.material_events[0].claim.startswith(expected_prefix)


def test_watchpoints_exclude_supported_high_confidence_events() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding = AgentFinding(
        finding_id="finding-502-watchpoint",
        agent_name="run_8k_agent",
        company=company,
        claim=(
            "Item 5.02 disclosed: On March 2, 2026, NVIDIA adopted the Variable "
            "Compensation Plan for Fiscal Year 2027 for eligible executive officers."
        ),
        summary=(
            "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan for Fiscal "
            "Year 2027 for eligible executive officers."
        ),
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-502-watchpoint",
                document_id="doc-502-watchpoint",
                filing_type=FilingType.FORM_8K,
                excerpt=(
                    "On March 2, 2026, NVIDIA adopted the Variable Compensation Plan for "
                    "Fiscal Year 2027 for eligible executive officers."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.PARTIAL,
            covered_topics=["material_events"],
            missing_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-watchpoint-filter",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-502-watchpoint",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
                missing_topics=["material_events"],
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.PARTIAL,
            covered_topics=["material_events"],
            missing_topics=["material_events"],
        ),
        include_transcript=False,
    )

    assert report.watchpoints == []


def test_watchpoints_include_supported_medium_confidence_findings_when_needed() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding = AgentFinding(
        finding_id="finding-supported-medium-watchpoint",
        agent_name="run_10q_agent",
        company=company,
        claim=(
            "As of October 26, 2025, NVIDIA had $60.6 billion in cash, cash "
            "equivalents, and marketable securities."
        ),
        summary=(
            "As of October 26, 2025, NVIDIA had $60.6 billion in cash, cash "
            "equivalents, and marketable securities."
        ),
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-supported-medium-watchpoint",
                document_id="doc-supported-medium-watchpoint",
                filing_type=FilingType.FORM_10Q,
                excerpt=(
                    "As of October 26, 2025, we had $60.6 billion in cash, cash "
                    "equivalents, and marketable securities."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["recent_quarter_change"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_10q_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-supported-medium-watchpoint-filter",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-supported-medium-watchpoint",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            )
        ],
        coverage_status=finding.coverage_status,
        include_transcript=False,
    )

    assert report.watchpoints[0].finding_id == "finding-supported-medium-watchpoint"
    assert report.watchpoints[0].trigger_to_monitor == "recent_quarter_change"


def test_coverage_summary_does_not_list_missing_topic_when_supported_finding_covers_it() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding = AgentFinding(
        finding_id="finding-material-events-covered",
        agent_name="run_8k_agent",
        company=company,
        claim='Item 9.01 (Financial Statements and Exhibits) disclosed the exhibit "Plan."',
        summary='Exhibit filed: "Plan."',
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-material-events-covered",
                document_id="doc-material-events-covered",
                filing_type=FilingType.FORM_8K,
                excerpt='Exhibit filed: "Plan."',
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events", "item_9.01"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-material-events-covered",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-material-events-covered",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
                missing_topics=["material_events"],
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.PARTIAL,
            covered_topics=["material_events"],
            missing_topics=["material_events"],
        ),
        include_transcript=False,
    )

    assert report.coverage_summary.missing_topics == []


def test_report_rewrites_verbose_10k_technology_stack_sentence() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding = AgentFinding(
        finding_id="finding-10k-tech-stack-rewrite",
        agent_name="run_10k_agent",
        company=company,
        claim=(
            "NVIDIA's technology stack includes the foundational NVIDIA CUDA development "
            "platform that runs on all NVIDIA GPUs, as well as hundreds of domain-specific "
            "software libraries, frameworks, algorithms,"
        ),
        summary=(
            "NVIDIA's technology stack includes the foundational NVIDIA CUDA development "
            "platform that runs on all NVIDIA GPUs, as well as hundreds of domain-specific "
            "software libraries, frameworks, algorithms,"
        ),
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-10k-tech-stack-rewrite",
                document_id="doc-10k-tech-stack-rewrite",
                filing_type=FilingType.FORM_10K,
                excerpt=(
                    "Our technology stack includes the foundational NVIDIA CUDA development "
                    "platform that runs on all NVIDIA GPUs, as well as hundreds of "
                    "domain-specific software libraries, frameworks, algorithms,"
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_10k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-10k-tech-stack-rewrite",
        company=company,
        question="Describe the business structure.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-10k-tech-stack-rewrite",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            )
        ],
        coverage_status=finding.coverage_status,
        include_transcript=False,
    )

    expected = (
        "NVIDIA's technology stack includes CUDA, GPUs, and domain-specific software "
        "libraries and frameworks."
    )
    assert report.executive_summary.summary == expected
    assert report.final_investment_thesis.thesis.startswith(
        "Verified disclosures show that NVIDIA's technology stack includes CUDA, GPUs,"
    )


def test_report_rewrites_10k_data_center_infrastructure_sentence() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding = AgentFinding(
        finding_id="finding-10k-data-center-infrastructure",
        agent_name="run_10k_agent",
        company=company,
        claim=(
            "NVIDIA is now a data center scale AI infrastructure company reshaping "
            "all industries."
        ),
        summary=(
            "NVIDIA is now a data center scale AI infrastructure company reshaping "
            "all industries."
        ),
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-10k-data-center-infrastructure",
                document_id="doc-10k-data-center-infrastructure",
                filing_type=FilingType.FORM_10K,
                excerpt=(
                    "NVIDIA is now a data center scale AI infrastructure company reshaping "
                    "all industries."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.PARTIALLY_SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_10k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-10k-data-center-infrastructure",
        company=company,
        question="Describe the business structure.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-10k-data-center-infrastructure",
                label=VerificationLabel.PARTIALLY_SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            )
        ],
        coverage_status=finding.coverage_status,
        include_transcript=False,
    )

    expected = "NVIDIA is a data-center-scale AI infrastructure company reshaping industries."
    assert report.executive_summary.summary == expected


def test_report_selects_claim_aligned_evidence_ref() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding = AgentFinding(
        finding_id="finding-claim-aligned-evidence",
        agent_name="run_10k_agent",
        company=company,
        claim="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        summary="NVIDIA is a data-center-scale AI infrastructure company reshaping industries.",
        signal_type=FindingSignalType.FUNDAMENTAL,
        filing_types=[FilingType.FORM_10K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-risk-misaligned",
                document_id="doc-risk-misaligned",
                filing_type=FilingType.FORM_10K,
                excerpt=(
                    "Foreign Exchange Rate Risk We consider our direct exposure to foreign "
                    "exchange rate fluctuations to be minimal."
                ),
            ),
            EvidenceRef(
                evidence_id="ev-business-aligned",
                document_id="doc-business-aligned",
                filing_type=FilingType.FORM_10K,
                excerpt=(
                    "NVIDIA is now a data center scale AI infrastructure company reshaping "
                    "all industries."
                ),
            ),
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["long_term_structure"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_10k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-claim-aligned-evidence",
        company=company,
        question="Describe the business structure.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-claim-aligned-evidence",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.MEDIUM,
            )
        ],
        coverage_status=finding.coverage_status,
        include_transcript=False,
    )

    assert report.evidence_grounded_report.business_overview[0].evidence_refs[0].evidence_id == (
        "ev-business-aligned"
    )


def test_report_preserves_company_abbreviation_case_after_period() -> None:
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    finding = AgentFinding(
        finding_id="finding-corp-abbreviation-case",
        agent_name="run_8k_agent",
        company=company,
        claim=(
            "NVIDIA Corp. adopted the Variable Compensation Plan for Fiscal Year 2027 "
            "on March 2, 2026."
        ),
        summary=(
            "NVIDIA Corp. adopted the Variable Compensation Plan for Fiscal Year 2027 "
            "on March 2, 2026."
        ),
        signal_type=FindingSignalType.EVENT,
        filing_types=[FilingType.FORM_8K],
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-corp-abbreviation-case",
                document_id="doc-corp-abbreviation-case",
                filing_type=FilingType.FORM_8K,
                excerpt=(
                    "NVIDIA Corp. adopted the Variable Compensation Plan for Fiscal Year "
                    "2027 on March 2, 2026."
                ),
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["material_events"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    packet = AgentOutputPacket(agent_name="run_8k_agent", findings=[finding])

    report = build_verified_report(
        report_id="report-corp-abbreviation-case",
        company=company,
        question="Summarize the latest filings.",
        packets=[packet],
        verification_results=[
            make_result(
                finding_id="finding-corp-abbreviation-case",
                label=VerificationLabel.SUPPORTED,
                confidence=ConfidenceLabel.HIGH,
            )
        ],
        coverage_status=finding.coverage_status,
        include_transcript=False,
    )

    assert "Corp. adopted" in report.executive_summary.summary
