import json
from datetime import date

import pytest
from pydantic import ValidationError

from app.domain import (
    AgentFinding,
    AnalysisRequest,
    AnalysisScope,
    AppendixSection,
    ChunkRecord,
    Company,
    ConfidenceLabel,
    CoverageLabel,
    CoverageStatus,
    DataCoverageSummary,
    EvidenceGroundedReportSection,
    EvidenceRef,
    ExecutiveSummarySection,
    FilingMetadata,
    FilingType,
    FinalInvestmentThesisSection,
    FinalReport,
    GraphState,
    ParsedSection,
    RawDocument,
    TranscriptMetadata,
    VerificationLabel,
    VerificationStatus,
    VerificationSummary,
)


def build_company() -> Company:
    return Company(cik="0000789019", ticker="MSFT", company_name="Microsoft Corp.")


def test_raw_document_requires_matching_metadata() -> None:
    company = build_company()

    with pytest.raises(ValidationError):
        RawDocument(
            document_id="doc-1",
            company=company,
            document_type=FilingType.FORM_10K,
            content="Annual report",
        )

    filing_document = RawDocument(
        document_id="doc-2",
        company=company,
        document_type=FilingType.FORM_10K,
        content="Annual report",
        filing_metadata=FilingMetadata(
            filing_id="filing-1",
            company=company,
            filing_type=FilingType.FORM_10K,
            accession_number="0000789019-24-000001",
            filed_at=date(2024, 7, 30),
        ),
    )

    transcript_document = RawDocument(
        document_id="doc-3",
        company=company,
        document_type=FilingType.EARNINGS_CALL,
        content="Prepared remarks",
        transcript_metadata=TranscriptMetadata(
            transcript_id="tx-1",
            company=company,
            call_date=date(2024, 7, 30),
        ),
    )

    assert filing_document.filing_metadata is not None
    assert transcript_document.transcript_metadata is not None


def test_graph_state_inherits_request_limits() -> None:
    request = AnalysisRequest(
        request_id="req-1",
        company=build_company(),
        question="Summarize disclosure-grounded thesis.",
        include_transcript=True,
        max_reretrievals=2,
    )

    state = GraphState(request=request)

    assert state.max_reretrievals == 2
    assert FilingType.EARNINGS_CALL in request.requested_filing_types


def test_final_report_serializes_with_enum_values() -> None:
    company = build_company()
    section = ParsedSection(
        section_id="sec-1",
        document_id="doc-2",
        company=company,
        filing_type=FilingType.FORM_10Q,
        heading="Risk Factors",
        text="Risk factor text",
        ordinal=1,
    )
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id=section.document_id,
        section_id=section.section_id,
        company=company,
        filing_type=section.filing_type,
        section_name=section.heading,
        text=section.text,
        token_count=3,
    )
    finding = AgentFinding(
        finding_id="finding-1",
        agent_name="risk_agent",
        company=company,
        claim="Liquidity risk remains contained.",
        summary="Liquidity disclosures remain stable.",
        filing_types=[FilingType.FORM_10Q],
        evidence_refs=[
            EvidenceRef(
                evidence_id="evidence-1",
                document_id=chunk.document_id,
                section_id=chunk.section_id,
                chunk_id=chunk.chunk_id,
                filing_type=chunk.filing_type,
                excerpt=chunk.text,
            )
        ],
        coverage_status=CoverageStatus(
            label=CoverageLabel.COMPLETE,
            covered_topics=["liquidity"],
        ),
        verification_status=VerificationStatus(
            label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
    )
    report = FinalReport(
        report_id="report-1",
        company=company,
        analysis_scope=AnalysisScope(
            question="Summarize the quarter.",
            allowed_sources=[FilingType.FORM_10K, FilingType.FORM_10Q, FilingType.FORM_8K],
            transcript_included=False,
            max_reretrievals=2,
        ),
        coverage_summary=DataCoverageSummary(
            coverage_label=CoverageLabel.COMPLETE,
            covered_channels=[FilingType.FORM_10Q],
            covered_topics=["liquidity"],
        ),
        executive_summary=ExecutiveSummarySection(
            summary="Disclosure-grounded thesis summary.",
            evidence_refs=finding.evidence_refs,
            verification_label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
        final_investment_thesis=FinalInvestmentThesisSection(
            thesis="Liquidity risk remains contained.",
            stance="valuation_free_disclosure_thesis",
            verification_label=VerificationLabel.SUPPORTED,
            confidence=ConfidenceLabel.HIGH,
        ),
        evidence_grounded_report=EvidenceGroundedReportSection(
            recent_quarter_change=[],
        ),
        verification_summary=VerificationSummary(total_claims=1, supported_claim_count=1),
        appendix=AppendixSection(),
    )

    payload = json.loads(report.model_dump_json())

    assert payload["executive_summary"]["verification_label"] == "supported"
    assert payload["executive_summary"]["confidence"] == "high"
    assert payload["coverage_summary"]["coverage_label"] == "complete"
