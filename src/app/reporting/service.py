"""Final report assembly and rendering."""

# ruff: noqa: E501

from __future__ import annotations

import html
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


def render_report_html(report: FinalReport) -> str:
    """Render the final report as a styled standalone HTML document."""

    key_risks = _html_note_cards(
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
    ) if not report.key_risks else _html_claim_cards(report.key_risks)
    key_opportunities = _html_claim_cards(_presentation_key_opportunities(report))
    watchpoints = _html_watchpoints(report)
    presentation_note = ""
    if _has_presentation_fill(report):
        presentation_note = """
        <div class="notice-banner">
          Some Key Risks, Key Opportunities, or Watchpoints entries are renderer-level
          fill heuristics for readability and are not canonical report objects.
        </div>
        """

    missing_channels = ""
    if report.coverage_summary.missing_channels:
        missing_channels = f"""
        <div class="meta-row">
          <span class="meta-label">Missing Channels</span>
          <span class="meta-value">{_html_filing_type_chips(report.coverage_summary.missing_channels)}</span>
        </div>
        """

    uncertainty_block = ""
    if report.final_investment_thesis.uncertainty_statement:
        uncertainty_block = (
            f'<p class="muted-note">{_html_text(report.final_investment_thesis.uncertainty_statement)}</p>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(report.report_id)}</title>
  <style>
    :root {{
      --paper: #f6f1e7;
      --paper-strong: #fffdf8;
      --ink: #1d2a24;
      --muted: #617067;
      --border: #d8d0c2;
      --accent: #214a3a;
      --accent-soft: #dce9e0;
      --gold: #9f7a2a;
      --risk: #8b3a2b;
      --opportunity: #1f6a48;
      --warning: #8a5a18;
      --shadow: 0 10px 30px rgba(29, 42, 36, 0.08);
      --radius: 20px;
      --radius-sm: 12px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(33, 74, 58, 0.08), transparent 28rem),
        linear-gradient(180deg, #efe8da 0%, var(--paper) 24rem);
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
      line-height: 1.65;
    }}
    .page {{
      width: min(1120px, calc(100vw - 32px));
      margin: 32px auto 56px;
    }}
    .hero {{
      background: linear-gradient(145deg, rgba(255, 253, 248, 0.98), rgba(245, 240, 231, 0.96));
      border: 1px solid rgba(216, 208, 194, 0.9);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 32px;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -120px -120px auto;
      width: 280px;
      height: 280px;
      background: radial-gradient(circle, rgba(33, 74, 58, 0.09), transparent 68%);
      pointer-events: none;
    }}
    .eyebrow {{
      font-size: 0.8rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 10px;
      font-family: "Helvetica Neue", Arial, sans-serif;
      font-weight: 700;
    }}
    h1, h2, h3, h4 {{
      margin: 0;
      color: #17211c;
      line-height: 1.2;
    }}
    .hero h1 {{
      font-size: clamp(2rem, 4vw, 3.5rem);
      max-width: 14ch;
      margin-bottom: 8px;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 1.05rem;
      max-width: 60ch;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 24px;
      margin-top: 24px;
    }}
    .meta-panel, .summary-panel {{
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(216, 208, 194, 0.7);
      border-radius: 18px;
      padding: 18px 20px;
      backdrop-filter: blur(10px);
    }}
    .summary-panel p {{
      margin: 0;
      font-size: 1.02rem;
    }}
    .meta-row {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 16px;
      padding: 9px 0;
      border-bottom: 1px solid rgba(216, 208, 194, 0.65);
      align-items: start;
    }}
    .meta-row:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .meta-label {{
      font: 700 0.78rem/1.2 "Helvetica Neue", Arial, sans-serif;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .meta-value {{
      min-width: 0;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip, .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      font: 700 0.75rem/1 "Helvetica Neue", Arial, sans-serif;
      letter-spacing: 0.02em;
      padding: 7px 10px;
    }}
    .chip {{
      background: #f2ece0;
      color: #304038;
      border: 1px solid rgba(216, 208, 194, 0.9);
    }}
    .badge {{
      border: 1px solid transparent;
      text-transform: capitalize;
    }}
    .badge.supported {{ background: #dcefe5; color: #184d34; border-color: #bdd8c8; }}
    .badge.partially-supported {{ background: #f4ead4; color: #7a5613; border-color: #e2cc9c; }}
    .badge.insufficient, .badge.low {{ background: #f8e0dc; color: #7e2e25; border-color: #e8b8b1; }}
    .badge.conflicting {{ background: #f3d6d0; color: #74251c; border-color: #dca59b; }}
    .badge.unavailable {{ background: #ece7dd; color: #5e665f; border-color: #d7cebe; }}
    .badge.high {{ background: #d8ece4; color: #154b37; border-color: #bdd8cc; }}
    .badge.medium {{ background: #efe3c6; color: #7e5c16; border-color: #e0cb96; }}
    .badge.presentation-fill {{
      background: rgba(139, 90, 24, 0.12);
      color: #7a5311;
      border-color: rgba(138, 90, 24, 0.35);
    }}
    .badge.monitoring-heuristic {{
      background: rgba(33, 74, 58, 0.12);
      color: #214a3a;
      border-color: rgba(33, 74, 58, 0.28);
    }}
    .badge.heuristic-badge {{
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.4);
    }}
    .notice-banner {{
      margin-top: 18px;
      background: linear-gradient(90deg, rgba(159, 122, 42, 0.12), rgba(159, 122, 42, 0.04));
      color: #6b541b;
      border: 1px solid rgba(159, 122, 42, 0.22);
      border-radius: 16px;
      padding: 14px 16px;
      font-size: 0.95rem;
    }}
    .section-grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 20px;
      margin-top: 24px;
    }}
    .section-card {{
      grid-column: span 12;
      background: rgba(255, 253, 248, 0.94);
      border: 1px solid rgba(216, 208, 194, 0.88);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 24px;
    }}
    .section-card.compact {{ padding: 20px; }}
    .section-card.half {{ grid-column: span 6; }}
    .section-card.third {{ grid-column: span 4; }}
    .section-kicker {{
      font: 700 0.77rem/1 "Helvetica Neue", Arial, sans-serif;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .section-card > h2 {{
      font-size: 1.6rem;
      margin-bottom: 14px;
    }}
    .section-card > p,
    .claim-copy,
    .muted-note {{
      margin: 0;
      font-size: 1rem;
    }}
    .muted-note {{
      color: var(--muted);
      margin-top: 10px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }}
    .stat-card {{
      background: rgba(255, 255, 255, 0.74);
      border: 1px solid rgba(216, 208, 194, 0.8);
      border-radius: 16px;
      padding: 14px;
    }}
    .stat-label {{
      font: 700 0.75rem/1.1 "Helvetica Neue", Arial, sans-serif;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .stat-value {{
      font-size: 1.6rem;
      font-weight: 700;
    }}
    .claim-stack {{
      display: grid;
      gap: 14px;
    }}
    .claim-card {{
      background: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(216, 208, 194, 0.82);
      border-radius: 18px;
      padding: 18px;
    }}
    .claim-card.fill-card {{
      background: linear-gradient(180deg, rgba(245, 239, 228, 0.96), rgba(255, 251, 243, 0.98));
      border-style: dashed;
      border-color: rgba(159, 122, 42, 0.38);
    }}
    .claim-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .card-eyebrow {{
      font: 700 0.7rem/1 "Helvetica Neue", Arial, sans-serif;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--warning);
      margin-bottom: 8px;
    }}
    .claim-title {{
      font-size: 1.08rem;
      font-weight: 700;
      margin: 0;
    }}
    .claim-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .claim-copy {{ margin-bottom: 10px; }}
    .supporting-copy {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.96rem;
    }}
    .evidence-list {{
      list-style: none;
      padding: 0;
      margin: 14px 0 0;
      display: grid;
      gap: 10px;
    }}
    .evidence-item {{
      background: #f7f2e7;
      border-radius: 14px;
      padding: 12px 14px;
      border: 1px solid rgba(216, 208, 194, 0.7);
    }}
    .evidence-id {{
      font: 700 0.72rem/1 "SFMono-Regular", Menlo, monospace;
      color: var(--muted);
      display: inline-block;
      margin-bottom: 8px;
    }}
    .source-list {{
      display: grid;
      gap: 12px;
    }}
    .source-card {{
      background: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(216, 208, 194, 0.82);
      border-radius: 16px;
      padding: 14px 16px;
    }}
    .source-card a {{
      color: var(--accent);
      text-decoration: none;
      word-break: break-word;
    }}
    .source-card a:hover {{ text-decoration: underline; }}
    .empty-state {{
      padding: 16px 18px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px dashed rgba(216, 208, 194, 0.92);
      color: var(--muted);
    }}
    @media (max-width: 920px) {{
      .hero-grid, .stats-grid {{ grid-template-columns: 1fr; }}
      .section-card.half, .section-card.third {{ grid-column: span 12; }}
      .meta-row {{ grid-template-columns: 1fr; gap: 6px; }}
      .claim-header {{ flex-direction: column; }}
      .claim-meta {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="eyebrow">Evidence-Grounded Equity Research</div>
      <h1>{_html_text(report.company.company_name or report.company.ticker)}</h1>
      <p class="subtitle">
        Valuation-free synthesis built from verified 10-K, 10-Q, and 8-K disclosures.
      </p>
      <div class="hero-grid">
        <section class="meta-panel">
          <div class="meta-row">
            <span class="meta-label">Ticker</span>
            <span class="meta-value">{_html_text(report.company.ticker)}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">Report ID</span>
            <span class="meta-value"><code>{_html_text(report.report_id)}</code></span>
          </div>
          <div class="meta-row">
            <span class="meta-label">Generated</span>
            <span class="meta-value">{_html_text(report.generated_at.isoformat())}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">Sources</span>
            <span class="meta-value">{_html_filing_type_chips(report.analysis_scope.allowed_sources)}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">Coverage</span>
            <span class="meta-value">{_html_badge(report.coverage_summary.coverage_label.value)}</span>
          </div>
          {missing_channels}
        </section>
        <section class="summary-panel">
          <div class="section-kicker">Executive Summary</div>
          <p>{_html_text(report.executive_summary.summary)}</p>
        </section>
      </div>
      {presentation_note}
    </header>

    <main class="section-grid">
      <section class="section-card half">
        <div class="section-kicker">Scope</div>
        <h2>Data Coverage and Confidence Summary</h2>
        <div class="meta-row">
          <span class="meta-label">Covered Channels</span>
          <span class="meta-value">{_html_filing_type_chips(report.coverage_summary.covered_channels)}</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Missing Topics</span>
          <span class="meta-value">{_html_text(', '.join(report.coverage_summary.missing_topics) or 'None')}</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Notes</span>
          <span class="meta-value">{_html_text(report.coverage_summary.notes or 'No additional coverage notes.')}</span>
        </div>
      </section>

      <section class="section-card half">
        <div class="section-kicker">Verification</div>
        <h2>Claim Verification Summary</h2>
        <div class="stats-grid">
          {_html_stat_card("Total Claims", str(report.verification_summary.total_claims))}
          {_html_stat_card("Supported", str(report.verification_summary.supported_claim_count))}
          {_html_stat_card("Partial", str(report.verification_summary.partially_supported_claim_count))}
          {_html_stat_card("High Confidence", str(report.verification_summary.high_confidence_claim_count))}
          {_html_stat_card("Medium Confidence", str(report.verification_summary.medium_confidence_claim_count))}
        </div>
      </section>

      <section class="section-card">
        <div class="section-kicker">Thesis</div>
        <h2>Final Investment Thesis</h2>
        <p>{_html_text(report.final_investment_thesis.thesis)}</p>
        {uncertainty_block}
      </section>

      <section class="section-card third">
        <div class="section-kicker">Risk View</div>
        <h2>Key Risks</h2>
        {_html_section_block(key_risks)}
      </section>

      <section class="section-card third">
        <div class="section-kicker">Upside View</div>
        <h2>Key Opportunities</h2>
        {_html_section_block(key_opportunities)}
      </section>

      <section class="section-card third">
        <div class="section-kicker">Monitoring</div>
        <h2>Watchpoints</h2>
        {_html_section_block(watchpoints)}
      </section>

      <section class="section-card half">
        <div class="section-kicker">Core Evidence</div>
        <h2>Business Overview</h2>
        {_html_section_block(_html_claim_cards(report.evidence_grounded_report.business_overview))}
      </section>

      <section class="section-card half">
        <div class="section-kicker">Quarterly Read</div>
        <h2>Recent Quarter Change</h2>
        {_html_section_block(_html_claim_cards(report.evidence_grounded_report.recent_quarter_change))}
      </section>

      <section class="section-card">
        <div class="section-kicker">Events</div>
        <h2>Material Events</h2>
        {_html_section_block(_html_claim_cards(report.evidence_grounded_report.material_events))}
      </section>

      <section class="section-card half compact">
        <div class="section-kicker">Appendix</div>
        <h2>Coverage Gaps</h2>
        {_html_section_block(_html_coverage_gaps(report.appendix.coverage_gaps))}
      </section>

      <section class="section-card half compact">
        <div class="section-kicker">Appendix</div>
        <h2>Excluded Claims</h2>
        {_html_section_block(_html_claim_cards(report.appendix.excluded_claims))}
      </section>

      <section class="section-card compact">
        <div class="section-kicker">Appendix</div>
        <h2>Source Documents</h2>
        {_html_section_block(_html_source_documents(report.appendix.source_documents))}
      </section>
    </main>
  </div>
</body>
</html>"""


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


def _presentation_key_opportunities(report: FinalReport) -> list[ReportClaim]:
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
    return selected


def _presentation_watchpoint_notes(report: FinalReport) -> list[dict[str, str]]:
    if report.watchpoints and any(
        claim.verification_label != VerificationLabel.SUPPORTED for claim in report.watchpoints
    ):
        return [
            {
                "title": claim.title,
                "claim": claim.claim,
                "note": (
                    f"verification={claim.verification_label.value} / "
                    f"confidence={claim.confidence.value}"
                ),
                "source": claim.trigger_to_monitor or "",
            }
            for claim in report.watchpoints
        ]
    if report.watchpoints:
        return _heuristic_watchpoint_notes_from_claims(report.watchpoints)
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
    return notes


def _html_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br />")


def _html_badge(text: str, *, extra_class: str = "") -> str:
    css_class = text.lower().replace(" ", "-").replace("_", "-")
    combined = f"{css_class} {extra_class}".strip()
    return f'<span class="badge {combined}">{_html_text(text)}</span>'


def _html_filing_type_chips(filing_types: Sequence[FilingType]) -> str:
    if not filing_types:
        return '<span class="chip">None</span>'
    return '<div class="chip-row">' + "".join(
        f'<span class="chip">{_html_text(item.value)}</span>' for item in filing_types
    ) + "</div>"


def _html_stat_card(label: str, value: str) -> str:
    return (
        '<div class="stat-card">'
        f'<div class="stat-label">{_html_text(label)}</div>'
        f'<div class="stat-value">{_html_text(value)}</div>'
        "</div>"
    )


def _html_claim_cards(claims: Sequence[ReportClaim]) -> str:
    if not claims:
        return '<div class="empty-state">None</div>'
    cards: list[str] = ['<div class="claim-stack">']
    for claim in claims:
        evidence_block = ""
        evidence_items = "".join(
            (
                '<li class="evidence-item">'
                f'<span class="evidence-id">{_html_text(evidence_ref.evidence_id)}</span>'
                f'<div>{_html_text(evidence_ref.excerpt[:220])}</div>'
                "</li>"
            )
            for evidence_ref in claim.evidence_refs[:3]
        )
        if evidence_items:
            evidence_block = f'<ul class="evidence-list">{evidence_items}</ul>'
        summary_block = ""
        if claim.summary and claim.summary != claim.claim:
            summary_block = f'<p class="supporting-copy">{_html_text(claim.summary)}</p>'
        trigger_block = ""
        if claim.trigger_to_monitor:
            trigger_block = (
                f'<p class="supporting-copy"><strong>Trigger to monitor:</strong> '
                f'{_html_text(claim.trigger_to_monitor)}</p>'
            )
        cards.append(
            '<article class="claim-card">'
            '<div class="claim-header">'
            f'<h3 class="claim-title">{_html_text(claim.title)}</h3>'
            '<div class="claim-meta">'
            f'{_html_badge(claim.verification_label.value)}'
            f'{_html_badge(claim.confidence.value)}'
            '</div></div>'
            f'<p class="claim-copy">{_html_text(claim.claim)}</p>'
            f'{summary_block}'
            f'{trigger_block}'
            f'{evidence_block}'
            '</article>'
        )
    cards.append("</div>")
    return "".join(cards)


def _html_note_cards(notes: Sequence[dict[str, str]]) -> str:
    if not notes:
        return '<div class="empty-state">None</div>'
    cards: list[str] = ['<div class="claim-stack">']
    for note in notes:
        note_kind = (note.get("note") or "").strip().lower()
        is_fill = note_kind in {"presentation fill", "monitoring heuristic"}
        card_class = "claim-card fill-card" if is_fill else "claim-card"
        note_badge = (
            f'<div class="claim-meta">{_html_badge(note["note"], extra_class="heuristic-badge" if is_fill else "")}</div>'
            if note.get("note")
            else ""
        )
        fill_label = (
            '<div class="card-eyebrow">Renderer-level fill</div>' if is_fill else ""
        )
        based_on = (
            f'<p class="supporting-copy"><strong>Based on:</strong> {_html_text(note["source"])}</p>'
            if note.get("source")
            else ""
        )
        cards.append(
            f'<article class="{card_class}">'
            '<div class="claim-header">'
            f'<div>{fill_label}<h3 class="claim-title">{_html_text(note["title"])}</h3></div>'
            f'{note_badge}'
            '</div>'
            f'<p class="claim-copy">{_html_text(note.get("claim", ""))}</p>'
            f'{based_on}'
            '</article>'
        )
    cards.append("</div>")
    return "".join(cards)


def _html_watchpoints(report: FinalReport) -> str:
    notes = _presentation_watchpoint_notes(report)
    return _html_note_cards(notes)


def _html_coverage_gaps(coverage_gaps: Sequence[CoverageGap]) -> str:
    if not coverage_gaps:
        return '<div class="empty-state">None</div>'
    cards: list[str] = ['<div class="claim-stack">']
    for gap in coverage_gaps:
        topic = getattr(gap, "topic", None) or "unknown"
        reason = getattr(gap, "reason", None)
        reason_block = (
            f'<p class="supporting-copy">{_html_text(reason)}</p>' if reason else ""
        )
        cards.append(
            '<article class="claim-card">'
            f'<h3 class="claim-title">{_html_text(topic)}</h3>'
            f'{reason_block}'
            '</article>'
        )
    cards.append("</div>")
    return "".join(cards)


def _html_source_documents(source_documents: Sequence[SourceDocumentRef]) -> str:
    if not source_documents:
        return '<div class="empty-state">None</div>'
    cards: list[str] = ['<div class="source-list">']
    for document in source_documents:
        source_link = ""
        if document.source_uri:
            escaped_uri = html.escape(document.source_uri, quote=True)
            source_link = (
                f'<div class="supporting-copy"><a href="{escaped_uri}" target="_blank" '
                f'rel="noreferrer noopener">{_html_text(document.source_uri)}</a></div>'
            )
        cards.append(
            '<article class="source-card">'
            f'<div class="claim-title">{_html_text(document.document_id)}</div>'
            f'<div class="supporting-copy">{_html_text(document.filing_type.value)}</div>'
            f'{source_link}'
            '</article>'
        )
    cards.append("</div>")
    return "".join(cards)


def _html_section_block(content: str) -> str:
    return content or '<div class="empty-state">None</div>'
