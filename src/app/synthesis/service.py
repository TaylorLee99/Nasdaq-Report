"""Verified-finding synthesis for the final disclosure-grounded thesis."""

from __future__ import annotations

import re
from collections import Counter
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
    EvidenceRef,
    ExecutiveSummarySection,
    FilingType,
    FinalInvestmentThesisSection,
    FinalReport,
    FindingSignalType,
    ReportClaim,
    SourceDocumentRef,
    VerificationLabel,
    VerificationStatus,
    VerificationSummary,
)

REPORTABLE_LABELS = {
    VerificationLabel.SUPPORTED,
    VerificationLabel.PARTIALLY_SUPPORTED,
}
TRANSCRIPT_ONLY_MISSING_TOPICS = frozenset({"transcript", "management_tone_guidance"})
NEGATIVE_TERMS = {
    "adverse",
    "challenge",
    "decline",
    "headwind",
    "pressure",
    "risk",
    "slowdown",
    "weakness",
}
POSITIVE_TERMS = {
    "demand",
    "expand",
    "growth",
    "improve",
    "opportunity",
    "resilient",
    "strength",
}
CANONICAL_MISSING_TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "transcript": ("transcript", "earnings call", "call transcript"),
    "management_tone_guidance": (
        "management_tone_guidance",
        "tone",
        "guidance",
        "management commentary",
    ),
    "long_term_structure": (
        "long_term_structure",
        "long-term structure",
        "item 1. business",
        "core business model",
        "product offerings",
        "market segments",
        "business overview",
    ),
    "recent_quarter_change": (
        "recent_quarter_change",
        "recent quarter",
        "quarter change",
        "quarterly change",
        "liquidity",
        "controls",
        "md&a",
        "mda",
    ),
    "material_events": (
        "material_events",
        "material event",
        "business developments",
        "8-k",
        "item 5.02",
        "item 9.01",
        "variable compensation plan",
        "compensation plan",
    ),
    "final_thesis": ("final_thesis", "final thesis", "synthesis", "investment thesis"),
}
CANONICAL_MISSING_TOPICS = frozenset(CANONICAL_MISSING_TOPIC_PATTERNS)
ALLOWED_REPORT_SOURCES = [
    FilingType.FORM_10K,
    FilingType.FORM_10Q,
    FilingType.FORM_8K,
]
MONTH_NAME_MAP = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
}
SENTENCE_ABBREVIATIONS = ("Corp.", "Inc.", "Ltd.", "Co.")


def collect_verified_findings(
    packets: Sequence[AgentOutputPacket],
    verification_results: Sequence[ClaimVerificationResult],
) -> tuple[list[AgentFinding], list[AgentFinding]]:
    """Return included and excluded findings after applying claim verification."""

    result_map = {result.claim_id: result for result in verification_results}
    included: list[AgentFinding] = []
    excluded: list[AgentFinding] = []
    for packet in packets:
        for finding in packet.findings:
            result = result_map.get(finding.finding_id)
            if result is None:
                excluded.append(finding)
                continue
            verified_finding = finding.model_copy(
                update={
                    "verification_status": VerificationStatus(
                        label=result.label,
                        confidence=result.confidence,
                        rationale=result.rationale,
                        verifier_name="heuristic_evidence_verifier",
                        checked_at=result.checked_at,
                    )
                }
            )
            if result.label in REPORTABLE_LABELS and verified_finding.evidence_refs:
                included.append(verified_finding)
            else:
                excluded.append(verified_finding)
    return _sort_findings(included), _sort_findings(excluded)


def build_structured_thesis_report(
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
    """Build the final valuation-free report from verified findings only."""

    findings, excluded_findings = collect_verified_findings(packets, verification_results)
    key_risks = _select_key_risks(findings)
    key_opportunities = _select_key_opportunities(findings)
    watchpoints = _select_watchpoints(findings)
    missing_topics = _collect_missing_topics(
        packets=packets,
        verification_results=verification_results,
        coverage_status=coverage_status,
    )
    suppressed_topics = set(TRANSCRIPT_ONLY_MISSING_TOPICS)
    suppressed_topics.add("evidence_gap")
    if not include_transcript:
        missing_topics = [topic for topic in missing_topics if topic not in suppressed_topics]
    else:
        missing_topics = [topic for topic in missing_topics if topic != "evidence_gap"]
    missing_topics = _reconcile_missing_topics(findings=findings, missing_topics=missing_topics)
    missing_topics = _augment_missing_topics_for_body_coverage(
        findings=findings,
        missing_topics=missing_topics,
    )
    report_conflicts = list(conflicts or [])
    coverage_summary = _build_coverage_summary(
        packets=packets,
        findings=findings,
        excluded_findings=excluded_findings,
        missing_topics=missing_topics,
    )
    source_documents = _collect_source_documents([*findings, *excluded_findings])

    return FinalReport(
        report_id=report_id,
        company=company,
        analysis_scope=AnalysisScope(
            question=question,
            as_of_date=as_of_date,
            allowed_sources=ALLOWED_REPORT_SOURCES,
            transcript_included=include_transcript,
            max_reretrievals=max_reretrievals,
        ),
        coverage_summary=coverage_summary,
        executive_summary=_build_executive_summary(findings),
        key_risks=[_to_report_claim(finding) for finding in key_risks],
        key_opportunities=[_to_report_claim(finding) for finding in key_opportunities],
        watchpoints=[_to_report_claim(finding) for finding in watchpoints],
        final_investment_thesis=_build_final_thesis(
            company=company,
            findings=findings,
            opportunities=key_opportunities,
            risks=key_risks,
            watchpoints=watchpoints,
            missing_topics=missing_topics,
        ),
        evidence_grounded_report=_build_evidence_grounded_report(
            findings=findings,
            conflicts=report_conflicts,
        ),
        verification_summary=_build_verification_summary(verification_results),
        appendix=_build_appendix(
            excluded_findings=excluded_findings,
            missing_topics=missing_topics,
            conflicts=report_conflicts,
            source_documents=source_documents,
        ),
    )


def _sort_findings(findings: Sequence[AgentFinding]) -> list[AgentFinding]:
    confidence_rank = {
        ConfidenceLabel.HIGH: 0,
        ConfidenceLabel.MEDIUM: 1,
        ConfidenceLabel.LOW: 2,
    }
    verification_rank = {
        VerificationLabel.SUPPORTED: 0,
        VerificationLabel.PARTIALLY_SUPPORTED: 1,
        VerificationLabel.INSUFFICIENT: 2,
        VerificationLabel.CONFLICTING: 3,
        VerificationLabel.UNAVAILABLE: 4,
    }
    return sorted(
        findings,
        key=lambda finding: (
            verification_rank[finding.verification_status.label],
            confidence_rank[finding.verification_status.confidence],
            -(finding.as_of_date.toordinal() if finding.as_of_date is not None else 0),
            -len(finding.evidence_refs),
            finding.finding_id,
        ),
    )


def _to_report_claim(finding: AgentFinding) -> ReportClaim:
    trigger_to_monitor = None
    if (
        finding.verification_status.label != VerificationLabel.SUPPORTED
        or finding.verification_status.confidence != ConfidenceLabel.HIGH
    ):
        trigger_to_monitor = _first_canonical_missing_topic(finding.coverage_status.missing_topics)
        if trigger_to_monitor is None:
            trigger_to_monitor = _default_monitor_topic(finding)
    claim = _report_claim_text(finding)
    summary = _report_summary_text(finding)
    return ReportClaim(
        finding_id=finding.finding_id,
        agent_name=finding.agent_name,
        title=_title_from_finding(finding),
        claim=claim,
        summary=summary,
        importance=finding.verification_status.confidence,
        time_horizon=finding.time_horizon,
        trigger_to_monitor=trigger_to_monitor,
        confidence=finding.verification_status.confidence,
        verification_label=finding.verification_status.label,
        filing_types=finding.filing_types,
        evidence_refs=_select_evidence_refs(finding),
    )


def _default_monitor_topic(finding: AgentFinding) -> str | None:
    if finding.agent_name == "run_10k_agent":
        return "long_term_structure"
    if finding.agent_name == "run_10q_agent":
        return "recent_quarter_change"
    if finding.agent_name == "run_8k_agent":
        return "material_events"
    return None


def _title_from_claim(claim: str) -> str:
    return claim.split(".", maxsplit=1)[0][:120].strip() or "Disclosure-backed claim"


def _title_from_finding(finding: AgentFinding) -> str:
    item_number = _extract_item_number(finding)
    base_title = _title_from_claim(finding.claim)
    if item_number is None:
        template_title = _filing_template_title(finding)
        return template_title or base_title
    section_label = _extract_item_section_label(finding, item_number)
    if section_label is not None:
        return f"Item {item_number} - {_shorten_item_section_label(item_number, section_label)}"
    preferred_item_label = _preferred_item_section_label(item_number)
    if preferred_item_label is not None:
        return f"Item {item_number} - {preferred_item_label}"
    if f"item {item_number.lower()}" in base_title.lower():
        return base_title
    summary_title = _title_from_claim(finding.summary)
    if _title_has_signal(summary_title):
        return f"Item {item_number} - {summary_title}"
    return f"Item {item_number} - {base_title}"


def _filing_template_title(finding: AgentFinding) -> str | None:
    text = f"{finding.claim} {finding.summary}".lower()
    if finding.agent_name == "run_10k_agent":
        if "risk" in text or finding.signal_type == FindingSignalType.RISK:
            return "10-K Structural Risks"
        return "10-K Long-Term Structure"
    if finding.agent_name == "run_10q_agent":
        if "control" in text:
            return "10-Q Controls and Procedures"
        if any(term in text for term in ("liquidity", "capital resources", "cash")):
            return "10-Q Liquidity and Recent Quarter Change"
        return "10-Q Recent Quarter Change"
    return None


def _extract_item_number(finding: AgentFinding) -> str | None:
    for pattern_source in (
        finding.claim,
        finding.summary,
        *[evidence_ref.excerpt for evidence_ref in finding.evidence_refs],
    ):
        match = re.search(r"\bitem\s+(\d+\.\d{2})\b", pattern_source, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1)
    return None


def _extract_item_section_label(finding: AgentFinding, item_number: str) -> str | None:
    patterns = (
        finding.claim,
        finding.summary,
        *[evidence_ref.excerpt for evidence_ref in finding.evidence_refs],
    )
    for source in patterns:
        match = re.search(
            rf"item\s+{re.escape(item_number)}\s*\(([^)]+)\)",
            source,
            flags=re.IGNORECASE,
        )
        if match is not None:
            label = " ".join(match.group(1).split()).strip().rstrip(".")
            if label:
                return label
    return None


def _title_has_signal(title: str) -> bool:
    normalized = title.strip()
    if not normalized or normalized == "Disclosure-backed claim":
        return False
    tokens = normalized.split()
    if len(tokens) <= 2:
        return False
    return normalized.lower() not in {"nvidia corp", "nvidia corporation"}


def _shorten_item_section_label(item_number: str, section_label: str) -> str:
    preferred_label = _preferred_item_section_label(item_number)
    if preferred_label is not None:
        return preferred_label
    shortened = section_label.split(";", maxsplit=1)[0].strip()
    return shortened or section_label


def _preferred_item_section_label(item_number: str) -> str | None:
    preferred = {
        "1.01": "Material Definitive Agreements",
        "2.02": "Results of Operations",
        "2.03": "Financing Obligations",
        "5.02": "Officer and Compensation Updates",
        "8.01": "Other Events",
        "9.01": "Financial Statements and Exhibits",
    }
    return preferred.get(item_number)


def _select_key_risks(findings: Sequence[AgentFinding]) -> list[AgentFinding]:
    selected = [
        finding
        for finding in findings
        if finding.signal_type == FindingSignalType.RISK or _contains_term(finding, NEGATIVE_TERMS)
    ]
    return selected[:3]


def _select_key_opportunities(findings: Sequence[AgentFinding]) -> list[AgentFinding]:
    risk_ids = {finding.finding_id for finding in _select_key_risks(findings)}
    selected = [
        finding
        for finding in findings
        if (
            finding.signal_type == FindingSignalType.GUIDANCE
            or _contains_term(finding, POSITIVE_TERMS)
        )
        and finding.signal_type != FindingSignalType.EVENT
        and finding.finding_id not in risk_ids
    ]
    if not selected:
        selected = [
            finding
            for finding in findings
            if finding.finding_id not in risk_ids
            and finding.signal_type != FindingSignalType.EVENT
            and finding.verification_status.label == VerificationLabel.SUPPORTED
            and _looks_like_structural_or_liquidity_strength(finding)
        ]
    return selected[:3]


def _select_watchpoints(findings: Sequence[AgentFinding]) -> list[AgentFinding]:
    selected = [
        finding
        for finding in findings
        if finding.verification_status.label
        in {
            VerificationLabel.PARTIALLY_SUPPORTED,
            VerificationLabel.INSUFFICIENT,
            VerificationLabel.CONFLICTING,
            VerificationLabel.UNAVAILABLE,
        }
        and _extract_item_number(finding) != "9.01"
    ]
    if not selected:
        selected = [
            finding
            for finding in findings
            if finding.verification_status.label == VerificationLabel.SUPPORTED
            and finding.verification_status.confidence == ConfidenceLabel.MEDIUM
            and finding.signal_type != FindingSignalType.EVENT
            and _extract_item_number(finding) != "9.01"
        ]
    return selected[:3]


def _contains_term(finding: AgentFinding, terms: set[str]) -> bool:
    text = f"{finding.claim} {finding.summary}".lower()
    return any(term in text for term in terms)


def _looks_like_structural_or_liquidity_strength(finding: AgentFinding) -> bool:
    text = f"{finding.claim} {finding.summary}".lower()
    positive_markers = (
        "cash, cash equivalents",
        "marketable securities",
        "liquidity",
        "capital resources",
        "data-center-scale ai infrastructure company",
        "accelerated computing",
        "technology stack",
        "software libraries and frameworks",
        "platform",
    )
    return any(marker in text for marker in positive_markers)


def _collect_missing_topics(
    *,
    packets: Sequence[AgentOutputPacket],
    verification_results: Sequence[ClaimVerificationResult],
    coverage_status: CoverageStatus | None,
) -> list[str]:
    topics: list[str] = []
    if coverage_status is not None:
        topics.extend(coverage_status.missing_topics)
    for packet in packets:
        topics.extend(packet.coverage_status.missing_topics)
    for result in verification_results:
        topics.extend(result.missing_coverage_topics)
    normalized_topics = [
        canonical_topic
        for topic in topics
        if topic
        for canonical_topic in _canonicalize_missing_topic(topic)
    ]
    return list(dict.fromkeys(topic for topic in normalized_topics if topic))


def _build_coverage_summary(
    *,
    packets: Sequence[AgentOutputPacket],
    findings: Sequence[AgentFinding],
    excluded_findings: Sequence[AgentFinding],
    missing_topics: Sequence[str],
) -> DataCoverageSummary:
    covered_topics = list(
        dict.fromkeys(
            topic
            for finding in findings
            for topic in finding.coverage_status.covered_topics
            if topic
        )
    )
    covered_channels = _derive_covered_channels(packets, findings)
    missing_channels = [
        filing_type for filing_type in ALLOWED_REPORT_SOURCES if filing_type not in covered_channels
    ]
    notes_parts = [
        f"Included claims: {len(findings)}",
        f"Excluded claims: {len(excluded_findings)}",
    ]
    if missing_topics:
        notes_parts.append(f"Missing topics: {', '.join(missing_topics)}")
    return DataCoverageSummary(
        coverage_label=(
            CoverageLabel.COMPLETE
            if findings and not missing_topics and not missing_channels
            else CoverageLabel.PARTIAL
        ),
        covered_channels=covered_channels,
        missing_channels=missing_channels,
        covered_topics=covered_topics,
        missing_topics=list(missing_topics),
        notes=". ".join(notes_parts) + ".",
    )


def _reconcile_missing_topics(
    *,
    findings: Sequence[AgentFinding],
    missing_topics: Sequence[str],
) -> list[str]:
    covered_topics = {
        topic
        for finding in findings
        for topic in finding.coverage_status.covered_topics
        if topic
    }
    return [topic for topic in missing_topics if topic not in covered_topics]


def _augment_missing_topics_for_body_coverage(
    *,
    findings: Sequence[AgentFinding],
    missing_topics: Sequence[str],
) -> list[str]:
    effective_topics = list(dict.fromkeys(topic for topic in missing_topics if topic))
    covered_channels = _derive_covered_channels([], findings)
    channel_topics = {
        FilingType.FORM_10K: "long_term_structure",
        FilingType.FORM_10Q: "recent_quarter_change",
        FilingType.FORM_8K: "material_events",
    }
    for filing_type, topic in channel_topics.items():
        if filing_type not in covered_channels and topic not in effective_topics:
            effective_topics.append(topic)
    return effective_topics


def _derive_covered_channels(
    packets: Sequence[AgentOutputPacket],
    findings: Sequence[AgentFinding],
) -> list[FilingType]:
    covered: list[FilingType] = []
    if any(FilingType.FORM_10K in finding.filing_types for finding in findings):
        covered.append(FilingType.FORM_10K)
    if any(FilingType.FORM_10Q in finding.filing_types for finding in findings):
        covered.append(FilingType.FORM_10Q)
    if _material_event_findings(findings):
        covered.append(FilingType.FORM_8K)
    return covered


def _select_executive_findings(findings: Sequence[AgentFinding]) -> list[AgentFinding]:
    candidate_findings = list(findings)
    preferred_findings = [
        finding for finding in candidate_findings if not _is_exhibit_only_finding(finding)
    ]
    if preferred_findings:
        candidate_findings = preferred_findings

    selected: list[AgentFinding] = []
    seen_channels: set[FilingType] = set()
    seen_signatures: set[str] = set()

    for finding in candidate_findings:
        filing_type = finding.filing_types[0] if finding.filing_types else None
        signature = _finding_signature(finding)
        if filing_type is not None and filing_type in seen_channels:
            continue
        if signature in seen_signatures:
            continue
        selected.append(finding)
        if filing_type is not None:
            seen_channels.add(filing_type)
        seen_signatures.add(signature)
        if len(selected) == 3:
            return selected

    for finding in candidate_findings:
        signature = _finding_signature(finding)
        if signature in seen_signatures:
            continue
        selected.append(finding)
        seen_signatures.add(signature)
        if len(selected) == 3:
            break
    return selected


def _build_executive_summary(findings: Sequence[AgentFinding]) -> ExecutiveSummarySection:
    if not findings:
        return ExecutiveSummarySection(
            summary="No supported disclosure-grounded thesis is available yet.",
            verification_label=VerificationLabel.UNAVAILABLE,
            confidence=ConfidenceLabel.LOW,
        )
    top_findings = _select_executive_findings(findings)
    sentences = [_summary_sentence_from_finding(finding) for finding in top_findings]
    sentences = [sentence for sentence in sentences if sentence]
    return ExecutiveSummarySection(
        summary=_polish_report_paragraph(" ".join(sentences[:3])),
        evidence_refs=_collect_evidence_refs(top_findings),
        verification_label=_lowest_verification_label(top_findings),
        confidence=_lowest_confidence(top_findings),
    )


def _build_final_thesis(
    *,
    company: Company,
    findings: Sequence[AgentFinding],
    opportunities: Sequence[AgentFinding],
    risks: Sequence[AgentFinding],
    watchpoints: Sequence[AgentFinding],
    missing_topics: Sequence[str],
) -> FinalInvestmentThesisSection:
    if not findings:
        return FinalInvestmentThesisSection(
            thesis=(
                f"{company.ticker} does not yet have enough verified filing support "
                "for a final thesis."
            ),
            stance="evidence_limited",
            uncertainty_statement="Coverage and verification remain insufficient.",
            verification_label=VerificationLabel.UNAVAILABLE,
            confidence=ConfidenceLabel.LOW,
        )

    parts: list[str] = []
    thesis_findings = _select_thesis_support_findings(
        findings=findings,
        opportunities=opportunities,
        risks=risks,
    )
    if thesis_findings:
        parts.append(
            _build_thesis_sentence(
                "Verified disclosures show",
                _finding_clause(thesis_findings[0], company=company),
            )
        )
    if len(thesis_findings) >= 2:
        parts.append(
            _build_thesis_sentence(
                "Additional filing support shows",
                _finding_clause(thesis_findings[1], company=company),
            )
        )
    if len(thesis_findings) >= 3:
        parts.append(
            _build_thesis_sentence(
                "Further filings indicate",
                _finding_clause(thesis_findings[2], company=company),
            )
        )
    if not parts:
        parts.append("Verified disclosures remain mixed and do not support a stronger synthesis")
    uncertainty_statement = None
    if watchpoints or missing_topics:
        uncertainty_statement = (
            "The conclusion remains uncertainty-aware because some signals are only partially "
            "supported or coverage is incomplete."
        )
    return FinalInvestmentThesisSection(
        thesis=_polish_report_paragraph(" ".join(parts), company=company),
        stance="valuation_free_disclosure_thesis",
        uncertainty_statement=uncertainty_statement,
        evidence_refs=_collect_evidence_refs([*opportunities[:1], *risks[:1], *watchpoints[:1]]),
        verification_label=_lowest_verification_label(findings[:3]),
        confidence=_lowest_confidence(findings[:3]),
    )


def _select_thesis_support_findings(
    *,
    findings: Sequence[AgentFinding],
    opportunities: Sequence[AgentFinding],
    risks: Sequence[AgentFinding],
) -> list[AgentFinding]:
    ordered_candidates = sorted(
        [
            *opportunities,
            *findings,
            *risks,
        ],
        key=_thesis_priority,
    )
    preferred_candidates = [
        finding for finding in ordered_candidates if not _is_exhibit_only_finding(finding)
    ]
    if preferred_candidates:
        ordered_candidates = preferred_candidates
    selected: list[AgentFinding] = []
    seen_ids: set[str] = set()
    seen_filing_types: set[FilingType] = set()

    for finding in ordered_candidates:
        if finding.finding_id in seen_ids:
            continue
        primary_filing_type = finding.filing_types[0] if finding.filing_types else None
        if primary_filing_type is not None and primary_filing_type in seen_filing_types:
            continue
        selected.append(finding)
        seen_ids.add(finding.finding_id)
        if primary_filing_type is not None:
            seen_filing_types.add(primary_filing_type)
        if len(selected) == 3:
            return selected

    for finding in ordered_candidates:
        if finding.finding_id in seen_ids:
            continue
        selected.append(finding)
        seen_ids.add(finding.finding_id)
        if len(selected) == 3:
            break
    return selected


def _thesis_priority(finding: AgentFinding) -> tuple[int, int, int, int]:
    primary_filing_type = finding.filing_types[0] if finding.filing_types else None
    filing_priority_map = {
        FilingType.FORM_8K: 0,
        FilingType.FORM_10Q: 1,
        FilingType.FORM_10K: 2,
    }
    filing_priority = (
        filing_priority_map.get(primary_filing_type, 3)
        if primary_filing_type is not None
        else 3
    )
    confidence_priority = {
        ConfidenceLabel.HIGH: 0,
        ConfidenceLabel.MEDIUM: 1,
        ConfidenceLabel.LOW: 2,
    }[finding.verification_status.confidence]
    return (
        filing_priority,
        0 if finding.signal_type == FindingSignalType.EVENT else 1,
        confidence_priority,
        -(finding.as_of_date.toordinal() if finding.as_of_date is not None else 0),
    )


def _build_evidence_grounded_report(
    *,
    findings: Sequence[AgentFinding],
    conflicts: Sequence[ConflictCandidate],
) -> EvidenceGroundedReportSection:
    business_overview = [
        _to_report_claim(finding)
        for finding in findings
        if FilingType.FORM_10K in finding.filing_types
    ]
    recent_quarter_change = [
        _to_report_claim(finding)
        for finding in findings
        if FilingType.FORM_10Q in finding.filing_types
    ]
    material_events = [_to_report_claim(finding) for finding in _material_event_findings(findings)]
    cross_document_tensions = [
        ReportClaim(
            finding_id=conflict.conflict_id,
            agent_name="conflict_checker",
            title=conflict.shared_topic or conflict.conflict_type.value,
            claim=conflict.claim,
            summary=conflict.reason,
            confidence=conflict.verification_status.confidence,
            verification_label=conflict.verification_status.label,
            filing_types=[],
            evidence_refs=conflict.evidence_refs,
        )
        for conflict in conflicts
    ]
    return EvidenceGroundedReportSection(
        business_overview=business_overview,
        recent_quarter_change=recent_quarter_change,
        material_events=material_events,
        cross_document_tensions=cross_document_tensions,
    )


def _material_event_findings(findings: Sequence[AgentFinding]) -> list[AgentFinding]:
    selected = [
        finding
        for finding in findings
        if FilingType.FORM_8K in finding.filing_types
        or finding.signal_type == FindingSignalType.EVENT
    ]
    non_exhibit_events = [finding for finding in selected if not _is_exhibit_only_finding(finding)]
    if non_exhibit_events:
        selected = non_exhibit_events
    selected = [finding for finding in selected if not _is_weak_exhibit_only_finding(finding)]
    return sorted(
        selected,
        key=lambda finding: (
            finding.signal_type == FindingSignalType.EVENT,
            finding.verification_status.label == VerificationLabel.SUPPORTED,
            finding.verification_status.label == VerificationLabel.PARTIALLY_SUPPORTED,
            _material_event_item_priority(finding),
            len(finding.evidence_refs),
            finding.as_of_date.toordinal() if finding.as_of_date is not None else 0,
        ),
        reverse=True,
    )


def _build_verification_summary(
    verification_results: Sequence[ClaimVerificationResult],
) -> VerificationSummary:
    label_counts = Counter(result.label for result in verification_results)
    confidence_counts = Counter(result.confidence for result in verification_results)
    return VerificationSummary(
        total_claims=len(verification_results),
        supported_claim_count=label_counts[VerificationLabel.SUPPORTED],
        partially_supported_claim_count=label_counts[VerificationLabel.PARTIALLY_SUPPORTED],
        insufficient_claim_count=label_counts[VerificationLabel.INSUFFICIENT],
        conflicting_claim_count=label_counts[VerificationLabel.CONFLICTING],
        unavailable_claim_count=label_counts[VerificationLabel.UNAVAILABLE],
        high_confidence_claim_count=confidence_counts[ConfidenceLabel.HIGH],
        medium_confidence_claim_count=confidence_counts[ConfidenceLabel.MEDIUM],
        low_confidence_claim_count=confidence_counts[ConfidenceLabel.LOW],
    )


def _build_appendix(
    *,
    excluded_findings: Sequence[AgentFinding],
    missing_topics: Sequence[str],
    conflicts: Sequence[ConflictCandidate],
    source_documents: Sequence[SourceDocumentRef],
) -> AppendixSection:
    return AppendixSection(
        conflicts=list(conflicts),
        coverage_gaps=[CoverageGap(topic=topic) for topic in missing_topics],
        excluded_claims=[_to_report_claim(finding) for finding in excluded_findings],
        source_documents=list(source_documents),
    )


def _collect_source_documents(findings: Sequence[AgentFinding]) -> list[SourceDocumentRef]:
    documents: dict[str, SourceDocumentRef] = {}
    for finding in findings:
        for evidence_ref in finding.evidence_refs:
            documents.setdefault(
                evidence_ref.document_id,
                SourceDocumentRef(
                    document_id=evidence_ref.document_id,
                    filing_type=evidence_ref.filing_type,
                    source_uri=evidence_ref.source_uri,
                ),
            )
    return list(documents.values())


def _collect_evidence_refs(findings: Sequence[AgentFinding]) -> list[EvidenceRef]:
    seen: set[str] = set()
    refs: list[EvidenceRef] = []
    for finding in findings:
        for evidence_ref in _select_evidence_refs(finding):
            if evidence_ref.evidence_id in seen:
                continue
            seen.add(evidence_ref.evidence_id)
            refs.append(evidence_ref)
    return refs[:3]


def _lowest_verification_label(findings: Sequence[AgentFinding]) -> VerificationLabel:
    if not findings:
        return VerificationLabel.UNAVAILABLE
    ranking = {
        VerificationLabel.SUPPORTED: 0,
        VerificationLabel.PARTIALLY_SUPPORTED: 1,
        VerificationLabel.INSUFFICIENT: 2,
        VerificationLabel.CONFLICTING: 3,
        VerificationLabel.UNAVAILABLE: 4,
    }
    return max(
        (finding.verification_status.label for finding in findings),
        key=lambda label: ranking[label],
    )


def _lowest_confidence(findings: Sequence[AgentFinding]) -> ConfidenceLabel:
    if not findings:
        return ConfidenceLabel.LOW
    ranking = {
        ConfidenceLabel.HIGH: 2,
        ConfidenceLabel.MEDIUM: 1,
        ConfidenceLabel.LOW: 0,
    }
    return min(
        (finding.verification_status.confidence for finding in findings),
        key=lambda label: ranking[label],
    )


def _summary_sentence_from_finding(finding: AgentFinding) -> str:
    sentence = _best_summary_sentence(finding)
    sentence = _sanitize_summary_sentence(sentence)
    sentence = _rewrite_8k_report_sentence(sentence)
    sentence = _to_analyst_style_clause(
        sentence.rstrip("."),
        finding=finding,
        company=finding.company,
    )
    report_claim = _report_claim_text(finding).rstrip(".")
    if report_claim and (
        sentence.strip().lower().startswith("item ")
        or
        _is_generic_filing_support_sentence(sentence)
        or _looks_fragmentary_sentence(sentence)
        or _summary_quality_score(sentence) + 10 < _summary_quality_score(report_claim)
    ):
        sentence = report_claim
        if finding.agent_name == "run_8k_agent":
            sentence = _first_sentence(sentence)
    if len(sentence) > 200:
        sentence = _truncate_report_sentence(sentence, max_length=200)
    return _polish_report_sentence(sentence, company=finding.company)


def _report_claim_text(finding: AgentFinding) -> str:
    item_number = _extract_item_number(finding)
    if (
        item_number == "5.02"
        and finding.summary
        and _looks_like_item_502_claim_text(finding.summary)
    ):
        candidate = finding.summary
    elif item_number == "5.02":
        candidate = finding.claim or finding.summary
    elif (
        item_number is not None
        and item_number != "9.01"
        and finding.summary
        and finding.claim
        and (
            finding.summary.strip().lower().startswith("item ")
            or "disclosed that" in finding.summary.lower()
            or _looks_fragmentary_sentence(finding.summary)
        )
    ):
        candidate = finding.claim
    elif item_number is not None and finding.summary:
        candidate = finding.summary
    else:
        candidate = finding.claim or finding.summary
    normalized = _sanitize_summary_sentence(candidate)
    normalized = _rewrite_8k_report_sentence(normalized)
    normalized = _to_analyst_style_clause(
        normalized.rstrip("."),
        finding=finding,
        company=finding.company,
    )
    return _polish_report_sentence(normalized, company=finding.company)


def _report_summary_text(finding: AgentFinding) -> str:
    if not finding.summary:
        return ""
    item_number = _extract_item_number(finding)
    if item_number == "5.02" and not _looks_like_item_502_claim_text(finding.summary):
        normalized = _sanitize_summary_sentence(finding.claim)
        normalized = _rewrite_8k_report_sentence(normalized)
        normalized = _to_analyst_style_clause(
            normalized.rstrip("."),
            finding=finding,
            company=finding.company,
        )
        return _polish_report_sentence(normalized, company=finding.company)
    if (
        item_number is not None
        and item_number != "9.01"
        and finding.claim
        and (
            finding.summary.strip().lower().startswith("item ")
            or "disclosed that" in finding.summary.lower()
            or _looks_fragmentary_sentence(finding.summary)
        )
    ):
        normalized = _sanitize_summary_sentence(finding.claim)
        normalized = _rewrite_8k_report_sentence(normalized)
        normalized = _to_analyst_style_clause(
            normalized.rstrip("."),
            finding=finding,
            company=finding.company,
        )
        return _polish_report_sentence(normalized, company=finding.company)
    normalized = _sanitize_summary_sentence(finding.summary)
    normalized = _rewrite_8k_report_sentence(normalized)
    normalized = _to_analyst_style_clause(
        normalized.rstrip("."),
        finding=finding,
        company=finding.company,
    )
    return _polish_report_sentence(normalized, company=finding.company)


def _finding_signature(finding: AgentFinding) -> str:
    text = _summary_sentence_from_finding(finding).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [token for token in text.split() if len(token) > 2]
    return " ".join(tokens[:12])


def _best_summary_sentence(finding: AgentFinding) -> str:
    candidates = [
        _first_sentence(text)
        for text in (finding.summary, finding.claim)
        if text
    ]
    if _is_generic_filing_support_sentence(" ".join(candidates)):
        evidence_sentence = _evidence_backed_sentence_from_finding(finding)
        if evidence_sentence is not None:
            candidates.append(evidence_sentence)
    scored_candidates = sorted(
        candidates,
        key=lambda sentence: _summary_quality_score(sentence),
        reverse=True,
    )
    if scored_candidates and _summary_quality_score(scored_candidates[0]) >= 20:
        return scored_candidates[0]
    return _fallback_summary_sentence(finding)


def _summary_quality_score(sentence: str) -> int:
    normalized = " ".join(sentence.split()).strip()
    if not normalized:
        return -100
    lowered = normalized.lower()
    score = 0
    if normalized[0].isupper():
        score += 20
    if re.match(r"^[A-Za-z][A-Za-z'’-]{2,}", normalized):
        score += 10
    if 40 <= len(normalized) <= 180:
        score += 10
    if re.match(r"^(On|As of|The|NVIDIA|Item)\b", normalized):
        score += 15
    if re.match(r"^[a-z]\b", normalized):
        score -= 50
    fragment_markers = (
        "(d) exhibits",
        "disclosed that d by",
        "disclosed that by the",
        "compensation comm",
        "the following risk factors should be considered",
        "any investment will be completed on expected terms",
        "pursuant to the requirements",
        "the following table sets forth",
        "refer to item 1a",
        "evaluate our liquidity and capital resources",
        "engaged with customers in china",
        "license requirements",
    )
    if any(marker in lowered for marker in fragment_markers):
        score -= 50
    if _is_generic_filing_support_sentence(normalized):
        score -= 20
    if re.search(r"\bdisclosed(?: that|:)\s+[a-z]\b", normalized):
        score -= 55
    if re.match(r"^(Evaluate|Maintain|Ensure|Provide)\b", normalized):
        score -= 50
    if normalized.count(",") > 5:
        score -= 10
    if _looks_fragmentary_sentence(normalized):
        score -= 60
    return score


def _fallback_summary_sentence(finding: AgentFinding) -> str:
    company_name = _company_display_name(finding.company)
    item_number = _extract_item_number(finding)
    if item_number is not None:
        item_label = _preferred_item_section_label(item_number) or "material event"
        date_prefix = (
            f"On {finding.as_of_date:%B %-d, %Y}, " if finding.as_of_date is not None else ""
        )
        return (
            f"{date_prefix}{company_name} disclosed {item_label.lower()} in an "
            f"Item {item_number} 8-K filing."
        )
    if FilingType.FORM_10K in finding.filing_types:
        return (
            "The latest 10-K supports a verified long-term structure finding for "
            f"{company_name}."
        )
    if FilingType.FORM_10Q in finding.filing_types:
        return (
            "The latest 10-Q supports a verified recent-quarter change finding for "
            f"{company_name}."
        )
    if FilingType.FORM_8K in finding.filing_types:
        return f"The latest 8-K supports a verified material-events finding for {company_name}."
    return f"Verified filing disclosures support this finding for {company_name}."


def _evidence_backed_sentence_from_finding(finding: AgentFinding) -> str | None:
    candidates = [
        _first_sentence(evidence_ref.excerpt)
        for evidence_ref in finding.evidence_refs
        if evidence_ref.excerpt
    ]
    scored_candidates = sorted(
        candidates,
        key=lambda sentence: (
            _evidence_sentence_relevance_score(finding, sentence),
            _summary_quality_score(sentence),
        ),
        reverse=True,
    )
    if not scored_candidates:
        return None
    best_candidate = scored_candidates[0]
    if _summary_quality_score(best_candidate) < 25:
        return None
    if _evidence_sentence_relevance_score(finding, best_candidate) < 2:
        return None
    return best_candidate


def _is_generic_filing_support_sentence(text: str) -> bool:
    lowered = " ".join(text.split()).lower()
    generic_markers = (
        "the latest 10-k supports a verified",
        "the latest 10-q supports a verified",
        "supports a verified long-term structure finding",
        "supports a verified recent-quarter change finding",
        "supports a verified liquidity and capital resources update",
        "supports a verified controls update",
    )
    return any(marker in lowered for marker in generic_markers)


def _evidence_sentence_relevance_score(finding: AgentFinding, sentence: str) -> int:
    lowered = sentence.lower()
    score = 0
    if finding.agent_name == "run_10q_agent":
        if any(
            token in lowered
            for token in ("liquidity", "capital resources", "cash", "revenue", "margin", "quarter")
        ):
            score += 2
    if finding.agent_name == "run_10k_agent":
        if any(
            token in lowered
            for token in ("business", "platform", "market", "customer", "segment", "supply", "risk")
        ):
            score += 2
    if finding.agent_name == "run_8k_agent":
        if any(
            token in lowered
            for token in ("item", "compensation", "agreement", "officer", "director", "event")
        ):
            score += 2
    if any(token in lowered for token in ("exchange act", "http://", "https://", "website")):
        score -= 2
    return score


def _company_display_name(company: Company) -> str:
    if company.company_name:
        token = re.split(r"[\s.]+", company.company_name.strip())[0]
        if token:
            return token
    return company.ticker


def _first_sentence(text: str) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return ""
    protected = normalized
    for abbreviation in SENTENCE_ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", "<prd>"))
    parts = protected.split(". ")
    fallback_sentence = ""
    for part in parts:
        sentence = part.strip().replace("<prd>", ".")
        if not sentence:
            continue
        if sentence and not sentence.endswith("."):
            sentence += "."
        if not fallback_sentence:
            fallback_sentence = sentence
        if not _looks_fragmentary_sentence(sentence):
            return sentence
    return fallback_sentence


def _sanitize_summary_sentence(text: str) -> str:
    normalized = (
        text.replace("U.S.", "US")
        .replace("U.K.", "UK")
        .replace("e.g.", "for example")
        .replace("i.e.", "that is")
    )
    normalized = re.sub(
        r"^Highlights from .*?included:\s*[•\-]\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"^([A-Z][A-Za-z&.'-]+)\s+\1\b",
        r"\1",
        normalized,
    )
    normalized = re.sub(r"\b([A-Z][a-z]+)\s+\1\b", r"\1", normalized)
    return normalized


def _rewrite_8k_report_sentence(text: str) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return normalized
    match = re.match(
        r"^Item 5\.02 disclosed:\s*(On [A-Za-z]+ \d{1,2}, \d{4}),\s+(.+?)\s+(adopted .+)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if match is not None:
        event_date = match.group(1)
        company_phrase = match.group(2).strip().rstrip(",")
        action = match.group(3).strip()
        return (
            f"{event_date}, {company_phrase} disclosed an Item 5.02 compensation update and "
            f"{action}"
        )
    return normalized


def _finding_clause(finding: AgentFinding, *, company: Company | None = None) -> str:
    summary_sentence = _report_claim_text(finding)
    if _is_generic_filing_support_sentence(summary_sentence) or _summary_quality_score(
        summary_sentence
    ) < 20:
        evidence_sentence = _evidence_backed_sentence_from_finding(finding)
        if evidence_sentence is not None:
            summary_sentence = evidence_sentence
        else:
            summary_sentence = _fallback_summary_sentence(finding)
    clause = summary_sentence.rstrip(".")
    clause = re.sub(
        r"^Item\s+\d+\.\d{2}\s*\([^)]*\)\s*disclosed(?::| that)\s*",
        "",
        clause,
        flags=re.IGNORECASE,
    )
    clause = _polish_report_fragment(clause.strip(), company=company)
    clause = _to_analyst_style_clause(clause, finding=finding, company=company)
    if not clause:
        return "relevant evidence remains mixed"
    return clause


def _build_thesis_sentence(prefix: str, clause: str) -> str:
    normalized_clause = clause.strip().rstrip(".")
    if re.match(r"^[A-Z][a-z]+ing\b", normalized_clause):
        normalized_clause = normalized_clause[0].lower() + normalized_clause[1:]
        return f"{prefix} that {normalized_clause}."
    if re.match(r"^(On|As of|The|We|Our)\b", normalized_clause):
        normalized_clause = normalized_clause[0].lower() + normalized_clause[1:]
        return f"{prefix} that {normalized_clause}."
    if re.match(
        r"^[A-Z][A-Za-z0-9&.'’\-\s]{0,90}\b"
        r"(?:announced|adopted|completed|designs|disclosed|grew|had|has|holds|"
        r"increased|includes|is|issued|makes|manufactures|markets|offers|posted|"
        r"provides|reports|serve|serves)\b",
        normalized_clause,
    ):
        return f"{prefix} that {normalized_clause}."
    return f"{prefix} {normalized_clause}."


def _to_analyst_style_clause(
    clause: str,
    *,
    finding: AgentFinding,
    company: Company | None,
) -> str:
    normalized = clause.strip()
    if not normalized or company is None:
        return normalized
    if finding.agent_name not in {"run_10k_agent", "run_10q_agent"}:
        return normalized

    company_name = _company_display_name(company)
    rewritten = re.sub(
        r"^The Company\b",
        company_name,
        normalized,
        count=1,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"^Making our suite of cloud-based services platform-agnostic,\s*(.+)$",
        rf"{company_name} makes its suite of cloud-based services platform-agnostic, \1",
        rewritten,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"^(As of [A-Za-z]+ \d{1,2}, \d{4}),\s+we had\b",
        rf"\1, {company_name} had",
        rewritten,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"^We report our business results in\b",
        f"{company_name} reports business results in",
        rewritten,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"^We\b",
        company_name,
        rewritten,
        count=1,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"^Our\b",
        f"{company_name}'s",
        rewritten,
        count=1,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(r"\bour\b", "its", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(
        rf"^{re.escape(company_name)}\s+serve\b",
        f"{company_name} serves",
        rewritten,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        rf"^{re.escape(company_name)}\s+have\b",
        f"{company_name} has",
        rewritten,
        flags=re.IGNORECASE,
    )
    return rewritten


def _canonicalize_missing_topic(topic: str) -> list[str]:
    normalized = topic.strip()
    if not normalized:
        return []
    lowered = normalized.lower()
    snake = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if snake in CANONICAL_MISSING_TOPICS:
        return [snake]

    labels = [
        label
        for label, patterns in CANONICAL_MISSING_TOPIC_PATTERNS.items()
        if any(pattern in lowered for pattern in patterns)
    ]
    if labels:
        return list(dict.fromkeys(labels))
    return ["evidence_gap"]


def _first_canonical_missing_topic(topics: Sequence[str]) -> str | None:
    for topic in topics:
        canonical_topics = _canonicalize_missing_topic(topic)
        if canonical_topics:
            if canonical_topics[0] == "evidence_gap":
                continue
            return canonical_topics[0]
    return None


def _is_exhibit_only_finding(finding: AgentFinding) -> bool:
    item_number = _extract_item_number(finding)
    if item_number == "9.01":
        return True
    lowered = f"{finding.claim} {finding.summary}".lower()
    return "exhibit filed:" in lowered


def _is_weak_exhibit_only_finding(finding: AgentFinding) -> bool:
    item_number = _extract_item_number(finding)
    combined_text = " ".join(part for part in (finding.claim, finding.summary) if part)
    lowered = combined_text.lower()
    if item_number != "9.01" and "exhibit filed:" not in lowered:
        return False
    if "exhibits and supporting filing materials were disclosed" in lowered:
        return True
    match = re.search(r'Exhibit filed:\s*"([^"]+)"', combined_text, flags=re.IGNORECASE)
    if match is None:
        return False
    description = match.group(1).strip().strip(".").lower()
    if description in {"underwriting", "agreement", "indenture", "plan"}:
        return True
    if re.search(r",\s*(as|amended|restated)$", description):
        return True
    return False


def _material_event_item_priority(finding: AgentFinding) -> int:
    item_number = _extract_item_number(finding)
    priority = {
        "5.02": 5,
        "2.02": 4,
        "1.01": 3,
        "8.01": 2,
        "9.01": 1,
    }
    return priority.get(item_number or "", 0)


def _looks_like_item_502_claim_text(text: str) -> bool:
    lowered = " ".join(text.split()).lower()
    markers = (
        "variable compensation plan",
        "compensation committee",
        "executive officers",
        "fiscal year 2027",
    )
    return any(marker in lowered for marker in markers)


def _polish_report_paragraph(text: str, *, company: Company | None = None) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return normalized
    normalized = _rewrite_10k_business_report_sentence(normalized, company=company)
    normalized = re.sub(r"\.\.+", ".", normalized)
    normalized = re.sub(r",\s*\.", ".", normalized)
    normalized = re.sub(r"\s+\.", ".", normalized)
    normalized = re.sub(r"\s+,", ",", normalized)
    normalized = re.sub(r":\s+([a-z])", lambda m: f": {m.group(1).upper()}", normalized)
    normalized = _normalize_company_and_date_text(normalized, company=company)
    if not normalized.endswith(".") and not re.search(r'\."\s*$', normalized):
        normalized = f"{normalized}."
    return normalized


def _polish_report_sentence(text: str, *, company: Company | None = None) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return normalized
    normalized = _rewrite_10k_business_report_sentence(normalized, company=company)
    normalized = re.sub(
        r"via a press release attached as Exhibit 99\.1(?: to an 8-K filing)?",
        "via an 8-K press release",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r",\s*\.", ".", normalized)
    normalized = re.sub(
        r"^the provided snippets\b",
        "The snippets",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"^the provided 8-k filing dated ([A-Za-z]+ \d{1,2}, \d{4})",
        r"The \1 8-K",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b([A-Z][a-z]+)\s+\1\b",
        r"\1",
        normalized,
    )
    normalized = _normalize_company_and_date_text(normalized, company=company)
    if not normalized.endswith(".") and not re.search(r'\."\s*$', normalized):
        normalized = f"{normalized}."
    return normalized


def _rewrite_10k_business_report_sentence(text: str, *, company: Company | None = None) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return normalized
    lowered = normalized.lower()
    company_name = _company_display_name(company) if company is not None else "The company"
    if "is now a data center scale ai infrastructure company reshaping all industries" in lowered:
        return (
            f"{company_name} is a data-center-scale AI infrastructure company reshaping "
            "industries"
        )
    if "technology stack includes the foundational nvidia cuda development platform" in lowered:
        return (
            f"{company_name}'s technology stack includes CUDA, GPUs, and "
            "domain-specific software libraries and frameworks"
        )
    if "makes its suite of cloud-based services platform-agnostic" in lowered:
        return (
            f"{company_name} makes its cloud-based services available across a wide range "
            "of devices and ecosystems, including competitor platforms"
        )
    if "making our suite of cloud-based services platform-agnostic" in lowered:
        return (
            f"{company_name} makes its cloud-based services available across a wide range "
            "of devices and ecosystems, including competitor platforms"
        )
    return normalized


def _polish_report_fragment(text: str, *, company: Company | None = None) -> str:
    normalized = " ".join(text.split()).strip().rstrip(".")
    if not normalized:
        return normalized
    return _normalize_company_and_date_text(normalized, company=company).rstrip(".")


def _normalize_company_and_date_text(text: str, *, company: Company | None = None) -> str:
    normalized = text
    for lowercase_month, title_month in MONTH_NAME_MAP.items():
        normalized = re.sub(
            rf"\b{lowercase_month}\b",
            title_month,
            normalized,
            flags=re.IGNORECASE,
        )
    if company is not None:
        normalized = _normalize_company_mentions(normalized, company)
    for abbreviation in SENTENCE_ABBREVIATIONS:
        normalized = normalized.replace(abbreviation, abbreviation.replace(".", "<prd>"))
    normalized = re.sub(
        r"([.!?]\s+)([a-z])",
        lambda match: f"{match.group(1)}{match.group(2).upper()}",
        normalized,
    )
    normalized = re.sub(
        r"^([a-z])",
        lambda match: match.group(1).upper(),
        normalized,
    )
    normalized = normalized.replace("<prd>", ".")
    return normalized


def _normalize_company_mentions(text: str, company: Company) -> str:
    normalized = text
    if company.company_name:
        normalized = re.sub(
            rf"{re.escape(company.company_name)}\s*\([^)]*\)",
            company.company_name,
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            re.escape(company.company_name),
            company.company_name,
            normalized,
            flags=re.IGNORECASE,
        )
        company_tokens = [token for token in re.split(r"[\s.]+", company.company_name) if token]
        if company_tokens:
            primary_name = company_tokens[0]
            normalized = re.sub(
                rf"\b{re.escape(primary_name)}\b",
                primary_name,
                normalized,
                flags=re.IGNORECASE,
            )
    normalized = re.sub(
        rf"\b{re.escape(company.ticker)}\b",
        company.ticker,
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized


def _looks_fragmentary_sentence(text: str) -> bool:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    if any(
        lowered.endswith(fragment)
        for fragment in (
            "quarter en.",
            "revenue from.",
            "customers duri.",
            "press release is.",
            "elected directors, ra.",
        )
    ):
        return True
    if re.search(r"\b(is|are|was|were|from|en|ra|duri)\.$", lowered):
        return True
    if re.search(r"\b(increased|decreased|declined|grew|offset|reflects)\.$", lowered):
        return True
    if re.search(r"\b(and|or|but)\.$", lowered):
        return True
    if " quarter en." in lowered or " revenue from." in lowered:
        return True
    return False


def _truncate_report_sentence(text: str, *, max_length: int) -> str:
    normalized = " ".join(text.split()).strip()
    if len(normalized) <= max_length:
        return normalized
    truncated = normalized[: max_length - 3].rstrip()
    truncated = re.sub(r"\s+\S*$", "", truncated).rstrip(",;:- ")
    trailing_stopword = re.compile(
        r"\b("
        r"a|an|the|and|or|but|to|of|for|in|on|at|by|"
        r"increased|decreased|declined|grew|reflects|reflecting|"
        r"partially|primarily|including|inclusive|offset|offsetting"
        r")$"
    )
    while truncated and trailing_stopword.search(truncated.lower()):
        updated = re.sub(r"\s+\S+$", "", truncated).rstrip(",;:- ")
        if not updated or updated == truncated:
            break
        truncated = updated
    if not truncated:
        truncated = normalized[: max_length - 3].rstrip()
    return f"{truncated}..."


def _select_evidence_refs(finding: AgentFinding, *, max_refs: int = 2) -> list[EvidenceRef]:
    report_claim = _report_claim_text(finding)
    ranked = sorted(
        finding.evidence_refs,
        key=lambda ref: (
            _evidence_ref_alignment_score(report_claim, ref),
            _evidence_ref_quality(ref),
            ref.source_uri is not None,
            ref.char_start is not None,
            len(ref.excerpt),
        ),
        reverse=True,
    )
    selected: list[EvidenceRef] = []
    seen_documents: set[str] = set()
    for ref in ranked:
        if _evidence_ref_quality(ref) <= 0:
            continue
        if ref.document_id in seen_documents and len(selected) >= 1:
            continue
        selected.append(_clean_report_evidence_ref(ref, company=finding.company))
        seen_documents.add(ref.document_id)
        if len(selected) >= max_refs:
            break
    if selected:
        return selected
    return [_clean_report_evidence_ref(ref, company=finding.company) for ref in ranked[:1]]


def _clean_report_evidence_ref(
    ref: EvidenceRef,
    *,
    company: Company | None = None,
) -> EvidenceRef:
    cleaned_excerpt = _clean_report_evidence_excerpt(ref.excerpt, company=company)
    if cleaned_excerpt == ref.excerpt:
        return ref
    return ref.model_copy(update={"excerpt": cleaned_excerpt})


def _clean_report_evidence_excerpt(
    text: str,
    *,
    company: Company | None = None,
) -> str:
    normalized = _sanitize_summary_sentence(" ".join(text.split()).strip())
    if not normalized:
        return normalized
    normalized = _normalize_company_and_date_text(normalized, company=company)
    best_sentence = _first_sentence(normalized)
    if best_sentence and not _looks_fragmentary_sentence(best_sentence):
        normalized = best_sentence
    normalized = re.sub(r"\b([A-Z][a-z]+)\s+\1\b", r"\1", normalized)
    normalized = re.sub(r"\b([A-Z]{2,})\s+\1\b", r"\1", normalized)
    normalized = re.sub(r"\s+,", ",", normalized)
    normalized = re.sub(r"\s+\.", ".", normalized)
    normalized = re.sub(r"\s*\([^)]{81,}\)", "", normalized)
    return _truncate_report_sentence(normalized, max_length=200)


def _evidence_ref_quality(ref: EvidenceRef) -> int:
    excerpt = " ".join(ref.excerpt.split())
    if len(excerpt) < 40:
        return 0
    if len(excerpt) > 420:
        return 1
    if excerpt.count("  ") > 0:
        return 1
    return 2


def _evidence_ref_alignment_score(claim: str, ref: EvidenceRef) -> int:
    excerpt = " ".join(ref.excerpt.split()).lower()
    normalized_claim = " ".join(claim.split()).lower()
    if not excerpt or not normalized_claim:
        return 0
    claim_tokens = {
        token
        for token in re.findall(r"[a-z0-9$]+(?:\.[0-9]+)?", normalized_claim.replace("-", " "))
        if len(token) > 2 and token not in {"item", "disclosed", "that", "the", "and"}
    }
    score = sum(10 for token in claim_tokens if token in excerpt)
    if normalized_claim[:80] and normalized_claim[:80] in excerpt:
        score += 25
    if (
        "data-center-scale ai infrastructure company" in normalized_claim
        and "data center scale ai infrastructure company" in excerpt
    ):
        score += 25
    if (
        "cash equivalents" in normalized_claim
        and "cash equivalents" in excerpt
        and "marketable securities" in normalized_claim
        and "marketable securities" in excerpt
    ):
        score += 20
    return score
