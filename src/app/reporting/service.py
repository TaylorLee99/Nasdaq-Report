"""Final report assembly and rendering."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from app.domain import (
    AgentFinding,
    AgentOutputPacket,
    AnalysisScope,
    AppendixSection,
    ClaimVerificationResult,
    Company,
    ConfidenceLabel,
    ConflictCandidate,
    CoverageGap,
    CoverageLabel,
    CoverageStatus,
    DataCoverageSummary,
    EvidenceGroundedReportSection,
    ExecutiveSummarySection,
    FilingType,
    FinalInvestmentThesisSection,
    FinalReport,
    ReportClaim,
    SourceDocumentRef,
    VerificationLabel,
    VerificationSummary,
)
from app.synthesis.service import build_structured_thesis_report


def build_report(
    report_id: str,
    company: Company,
    executive_summary: str,
    findings: Sequence[AgentFinding],
) -> FinalReport:
    """Backward-compatible report builder for direct finding inputs."""

    source_documents = []
    for finding in findings:
        for evidence_ref in finding.evidence_refs:
            source_documents.append(
                SourceDocumentRef(
                    document_id=evidence_ref.document_id,
                    filing_type=evidence_ref.filing_type,
                    source_uri=evidence_ref.source_uri,
                )
            )
    return FinalReport(
        report_id=report_id,
        company=company,
        analysis_scope=AnalysisScope(
            question="Direct report build",
            allowed_sources=[FilingType.FORM_10K, FilingType.FORM_10Q, FilingType.FORM_8K],
            transcript_included=False,
            max_reretrievals=0,
        ),
        coverage_summary=DataCoverageSummary(
            coverage_label=(CoverageLabel.COMPLETE if findings else CoverageLabel.NOT_STARTED),
            covered_channels=sorted(
                {filing_type for finding in findings for filing_type in finding.filing_types},
                key=lambda filing_type: filing_type.value,
            ),
            covered_topics=list(
                dict.fromkeys(
                    topic
                    for finding in findings
                    for topic in finding.coverage_status.covered_topics
                )
            ),
        ),
        executive_summary=ExecutiveSummarySection(
            summary=executive_summary,
            evidence_refs=[
                evidence_ref
                for finding in findings[:3]
                for evidence_ref in finding.evidence_refs[:1]
            ],
            verification_label=(
                findings[0].verification_status.label if findings else VerificationLabel.UNAVAILABLE
            ),
            confidence=(
                findings[0].verification_status.confidence if findings else ConfidenceLabel.LOW
            ),
        ),
        key_risks=[],
        key_opportunities=[],
        watchpoints=[],
        final_investment_thesis=FinalInvestmentThesisSection(
            thesis=executive_summary,
            stance="legacy_report_builder",
            verification_label=(
                findings[0].verification_status.label if findings else VerificationLabel.UNAVAILABLE
            ),
            confidence=(
                findings[0].verification_status.confidence if findings else ConfidenceLabel.LOW
            ),
        ),
        evidence_grounded_report=EvidenceGroundedReportSection(),
        verification_summary=VerificationSummary(total_claims=len(findings)),
        appendix=AppendixSection(
            conflicts=[],
            coverage_gaps=[],
            excluded_claims=[],
            source_documents=source_documents,
        ),
    )


def build_verified_report(
    *,
    report_id: str,
    company: Company,
    question: str,
    packets: Sequence[AgentOutputPacket],
    verification_results: Sequence[ClaimVerificationResult],
    coverage_status: CoverageStatus | None = None,
    include_transcript: bool = False,
    as_of_date: date | None = None,
    max_reretrievals: int = 0,
    conflicts: Sequence[ConflictCandidate] | None = None,
) -> FinalReport:
    """Build the final structured thesis report from verified findings only."""

    return build_structured_thesis_report(
        report_id=report_id,
        company=company,
        question=question,
        packets=packets,
        verification_results=verification_results,
        coverage_status=coverage_status,
        include_transcript=include_transcript,
        as_of_date=as_of_date,
        max_reretrievals=max_reretrievals,
        conflicts=conflicts,
    )


def render_report_markdown(report: FinalReport) -> str:
    """Render the final report to markdown."""

    covered_channels = ", ".join(
        item.value for item in report.coverage_summary.covered_channels
    ) or "None"
    presentation_fill_active = _has_presentation_fill(report)
    key_risks_lines = _render_key_risks_section(report)
    key_opportunities_lines = _render_key_opportunities_section(report)
    watchpoints_lines = _render_watchpoints_section(report)
    lines = [
        "# Evidence-Grounded Report",
        "",
        f"- Report ID: `{report.report_id}`",
        f"- Company: `{report.company.ticker}` ({report.company.company_name})",
        f"- Generated At: `{report.generated_at.isoformat()}`",
        (
            "- Allowed Sources: "
            f"{', '.join(source.value for source in report.analysis_scope.allowed_sources)}"
        ),
    ]
    if presentation_fill_active:
        lines.append(
            "- Presentation Note: Some Key Risks, Key Opportunities, or Watchpoints "
            "entries are renderer-level fill heuristics for readability and are not "
            "canonical report objects."
        )
    lines.extend(
        [
            "",
        "## Executive Summary",
        report.executive_summary.summary,
        "",
        "## Data Coverage and Confidence Summary",
        f"Coverage label: {report.coverage_summary.coverage_label.value}",
        f"Covered channels: {covered_channels}",
        ]
    )
    if report.coverage_summary.missing_channels:
        missing_channels = ", ".join(
            item.value for item in report.coverage_summary.missing_channels
        )
        lines.append(f"Missing channels: {missing_channels}")
    lines.extend(
        [
            report.coverage_summary.notes or "No additional coverage notes.",
            "",
            "## Key Risks",
        ]
    )
    lines.extend(key_risks_lines)
    lines.extend(["", "## Key Opportunities"])
    lines.extend(key_opportunities_lines)
    lines.extend(["", "## Watchpoints"])
    lines.extend(watchpoints_lines)
    lines.extend(
        [
            "",
            "## Final Investment Thesis",
            report.final_investment_thesis.thesis,
        ]
    )
    if report.final_investment_thesis.uncertainty_statement:
        lines.append(report.final_investment_thesis.uncertainty_statement)
    lines.extend(["", "## Evidence-Grounded Report", "", "### Business Overview"])
    lines.extend(_render_claim_list(report.evidence_grounded_report.business_overview))
    lines.extend(["", "### Recent Quarter Change"])
    lines.extend(_render_claim_list(report.evidence_grounded_report.recent_quarter_change))
    lines.extend(["", "### Material Events"])
    lines.extend(_render_claim_list(report.evidence_grounded_report.material_events))
    lines.extend(["", "### Cross-Document Tensions"])
    lines.extend(_render_claim_list(report.evidence_grounded_report.cross_document_tensions))
    lines.extend(
        [
            "",
            "## Verification Summary",
            f"- Total claims: {report.verification_summary.total_claims}",
            f"- Supported: {report.verification_summary.supported_claim_count}",
            (
                "- Partially supported: "
                f"{report.verification_summary.partially_supported_claim_count}"
            ),
            f"- Insufficient: {report.verification_summary.insufficient_claim_count}",
            f"- Conflicting: {report.verification_summary.conflicting_claim_count}",
            f"- Unavailable: {report.verification_summary.unavailable_claim_count}",
            f"- High confidence: {report.verification_summary.high_confidence_claim_count}",
            f"- Medium confidence: {report.verification_summary.medium_confidence_claim_count}",
            f"- Low confidence: {report.verification_summary.low_confidence_claim_count}",
            "",
            "## Appendix",
            "",
            "### Coverage Gaps",
        ]
    )
    lines.extend(_render_coverage_gaps(report.appendix.coverage_gaps))
    lines.extend(["", "### Excluded Claims"])
    lines.extend(_render_claim_list(report.appendix.excluded_claims))
    lines.extend(["", "### Source Documents"])
    lines.extend(_render_source_documents(report.appendix.source_documents))
    return "\n".join(lines).strip()


def serialize_report_json(report: FinalReport) -> str:
    """Serialize the final report to JSON."""

    return report.model_dump_json(indent=2)


def _render_claim_list(claims: Sequence[ReportClaim]) -> list[str]:
    if not claims:
        return ["- None"]
    rendered: list[str] = []
    for claim in claims:
        rendered.extend(
            [
                f"- {claim.title}",
                f"  - Claim: {claim.claim}",
                (
                    "  - Verification: "
                    f"{claim.verification_label.value} / confidence={claim.confidence.value}"
                ),
            ]
        )
        if claim.summary and claim.summary != claim.claim:
            rendered.append(f"  - Summary: {claim.summary}")
        if claim.trigger_to_monitor:
            rendered.append(f"  - Trigger To Monitor: {claim.trigger_to_monitor}")
        for evidence_ref in claim.evidence_refs[:3]:
            rendered.append(
                f"  - Evidence `{evidence_ref.evidence_id}`"
            )
            rendered.append(f"    - Excerpt: {evidence_ref.excerpt[:220]}")
    return rendered


def _render_key_risks_section(report: FinalReport) -> list[str]:
    if report.key_risks:
        return _render_claim_list(report.key_risks)
    return _render_note_list(
        [
            {
                "title": "No verified risk finding",
                "note": "Presentation fill",
                "claim": (
                    "The current filing-only pass did not surface a disclosure-grounded "
                    "risk claim that met the inclusion threshold."
                ),
            }
        ]
    )


def _render_key_opportunities_section(report: FinalReport) -> list[str]:
    selected = list(report.key_opportunities)
    seen_ids = {claim.finding_id for claim in selected}
    for candidate in (
        list(report.evidence_grounded_report.recent_quarter_change)
        + list(report.evidence_grounded_report.business_overview)
    ):
        if candidate.finding_id in seen_ids:
            continue
        selected.append(candidate)
        seen_ids.add(candidate.finding_id)
        if len(selected) >= 3:
            break
    return _render_claim_list(selected) if selected else ["- None"]


def _render_watchpoints_section(report: FinalReport) -> list[str]:
    if report.watchpoints and any(
        claim.verification_label != VerificationLabel.SUPPORTED for claim in report.watchpoints
    ):
        return _render_claim_list(report.watchpoints)
    if report.watchpoints:
        return _render_note_list(_heuristic_watchpoint_notes_from_claims(report.watchpoints))
    notes: list[dict[str, str]] = []
    if report.evidence_grounded_report.recent_quarter_change:
        claim = report.evidence_grounded_report.recent_quarter_change[0]
        notes.append(
            {
                "title": "Liquidity Watchpoint",
                "note": "Monitoring heuristic",
                "claim": (
                    "Monitor later 10-Q filings for changes in cash, cash equivalents, "
                    "and marketable securities relative to the current quarter baseline."
                ),
                "source": claim.title,
            }
        )
    if report.evidence_grounded_report.material_events:
        claim = report.evidence_grounded_report.material_events[0]
        notes.append(
            {
                "title": "Material Event Follow-Through",
                "note": "Monitoring heuristic",
                "claim": (
                    "Monitor subsequent 8-K or 10-Q disclosures for amendments, "
                    "implementation details, or downstream effects tied to the latest "
                    "material event disclosure."
                ),
                "source": claim.title,
            }
        )
    if report.evidence_grounded_report.business_overview:
        claim = report.evidence_grounded_report.business_overview[0]
        notes.append(
            {
                "title": "Strategic Positioning Check",
                "note": "Monitoring heuristic",
                "claim": (
                    "Monitor whether later filings continue to support the current "
                    "long-term business positioning described in the latest 10-K."
                ),
                "source": claim.title,
            }
        )
    return _render_note_list(notes) if notes else ["- None"]


def _has_presentation_fill(report: FinalReport) -> bool:
    if not report.key_risks:
        return True
    if not report.key_opportunities:
        return True
    if not report.watchpoints:
        return True
    return all(
        claim.verification_label == VerificationLabel.SUPPORTED
        for claim in report.watchpoints
    )


def _heuristic_watchpoint_notes_from_claims(claims: Sequence[ReportClaim]) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    for claim in claims:
        if claim.agent_name == "run_10q_agent":
            notes.append(
                {
                    "title": "Liquidity Watchpoint",
                    "note": "Monitoring heuristic",
                    "claim": (
                        "Monitor later 10-Q filings for changes in cash, cash equivalents, "
                        "and marketable securities relative to the current quarter baseline."
                    ),
                    "source": claim.title,
                }
            )
            continue
        if claim.agent_name == "run_10k_agent":
            notes.append(
                {
                    "title": "Strategic Positioning Check",
                    "note": "Monitoring heuristic",
                    "claim": (
                        "Monitor whether later filings continue to support the current "
                        "long-term business positioning described in the latest 10-K."
                    ),
                    "source": claim.title,
                }
            )
            continue
        notes.append(
            {
                "title": "Material Event Follow-Through",
                "note": "Monitoring heuristic",
                "claim": (
                    "Monitor subsequent 8-K or 10-Q disclosures for amendments, "
                    "implementation details, or downstream effects tied to the latest "
                    "material event disclosure."
                ),
                "source": claim.title,
            }
        )
    return notes


def _render_note_list(notes: Sequence[dict[str, str]]) -> list[str]:
    if not notes:
        return ["- None"]
    rendered: list[str] = []
    for note in notes:
        rendered.append(f"- {note['title']}")
        if note.get("note"):
            rendered.append(f"  - Note: {note['note']}")
        if note.get("claim"):
            rendered.append(f"  - Claim: {note['claim']}")
        if note.get("source"):
            rendered.append(f"  - Based on: {note['source']}")
    return rendered


def _render_coverage_gaps(coverage_gaps: Sequence[CoverageGap]) -> list[str]:
    if not coverage_gaps:
        return ["- None"]
    rendered: list[str] = []
    for gap in coverage_gaps:
        topic = getattr(gap, "topic", None) or "unknown"
        reason = getattr(gap, "reason", None)
        rendered.append(f"- {topic}")
        if reason:
            rendered.append(f"  - Reason: {reason}")
    return rendered


def _render_source_documents(source_documents: Sequence[SourceDocumentRef]) -> list[str]:
    if not source_documents:
        return ["- None"]
    rendered: list[str] = []
    for document in source_documents:
        rendered.append(f"- `{document.document_id}` ({document.filing_type.value})")
        if document.source_uri:
            rendered.append(f"  - Source: {document.source_uri}")
    return rendered
