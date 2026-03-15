from datetime import date
from pathlib import Path

from app.domain import (
    Company,
    ConfidenceLabel,
    DocumentSectionType,
    FilingMetadata,
    FilingType,
    RawDocument,
    TranscriptMetadata,
    TranscriptSegmentType,
)
from app.parsing import parse_document


def build_company() -> Company:
    return Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")


def build_filing_document(name: str, form_type: FilingType) -> str:
    return Path(f"tests/fixtures/{name}").read_text(encoding="utf-8")


def build_raw_document(name: str, form_type: FilingType) -> RawDocument:
    company = build_company()
    content = build_filing_document(name, form_type)
    if form_type == FilingType.EARNINGS_CALL:
        return RawDocument(
            document_id=f"doc-{name}",
            company=company,
            document_type=form_type,
            content=content,
            transcript_metadata=TranscriptMetadata(
                transcript_id=f"tx-{name}",
                company=company,
                call_date=date(2024, 11, 20),
                fiscal_quarter="Q3FY2025",
                speaker_separated=True,
            ),
        )
    return RawDocument(
        document_id=f"doc-{name}",
        company=company,
        document_type=form_type,
        content=content,
        filing_metadata=FilingMetadata(
            filing_id=f"filing-{name}",
            company=company,
            filing_type=form_type,
            accession_number="0001045810-24-000101",
            filed_at=date(2024, 11, 20),
        ),
    )


def test_ten_k_parser_extracts_expected_sections() -> None:
    sections = parse_document(build_raw_document("parser_10k.txt", FilingType.FORM_10K))

    assert [section.section_type for section in sections] == [
        DocumentSectionType.BUSINESS,
        DocumentSectionType.RISK_FACTORS,
        DocumentSectionType.MDA,
        DocumentSectionType.FINANCIAL_STATEMENTS,
        DocumentSectionType.NOTES,
    ]
    assert all(
        section.parse_confidence in {ConfidenceLabel.HIGH, ConfidenceLabel.MEDIUM}
        for section in sections
    )
    assert all(section.used_fallback is False for section in sections)


def test_ten_k_parser_strips_business_heading_artifacts_from_section_body() -> None:
    sections = parse_document(
        RawDocument(
            document_id="doc-10k-business-artifact",
            company=build_company(),
            document_type=FilingType.FORM_10K,
            content=(
                "Item 1. Business\n"
                "Our Businesses\n"
                "We report our business results in two segments.\n\n"
                "Item 1A. Risk Factors\n"
                "Risk factor text.\n"
            ),
            filing_metadata=FilingMetadata(
                filing_id="filing-10k-business-artifact",
                company=build_company(),
                filing_type=FilingType.FORM_10K,
                accession_number="0001045810-24-000101",
                filed_at=date(2024, 11, 20),
            ),
        )
    )

    assert sections[0].section_type == DocumentSectionType.BUSINESS
    assert sections[0].text == "We report our business results in two segments."


def test_ten_q_parser_extracts_liquidity_and_controls() -> None:
    sections = parse_document(build_raw_document("parser_10q.txt", FilingType.FORM_10Q))

    section_types = {section.section_type for section in sections}
    assert DocumentSectionType.MDA in section_types
    assert DocumentSectionType.LIQUIDITY in section_types
    assert DocumentSectionType.CONTROLS in section_types
    assert DocumentSectionType.FINANCIAL_STATEMENTS in section_types


def test_ten_q_parser_splits_market_risk_heading_from_liquidity_section() -> None:
    sections = parse_document(
        RawDocument(
            document_id="doc-10q-market-risk-boundary",
            company=build_company(),
            document_type=FilingType.FORM_10Q,
            content=(
                "Management's Discussion and Analysis of Financial Condition and Results "
                "of Operations\n"
                "Narrative intro.\n"
                "Liquidity and Capital Resources\n"
                "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, "
                "and marketable securities.\n"
                "Item 3. Quantitative and Qualitative Disclosures about Market Risk\n"
                "A hypothetical 10% decrease in fair value would reduce the balance.\n"
                "Item 4. Controls and Procedures\n"
                "Controls text.\n"
            ),
            filing_metadata=FilingMetadata(
                filing_id="filing-10q-market-risk-boundary",
                company=build_company(),
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filed_at=date(2025, 11, 19),
            ),
        )
    )

    liquidity_section = next(
        section for section in sections if section.section_type == DocumentSectionType.LIQUIDITY
    )
    controls_section = next(
        section for section in sections if section.section_type == DocumentSectionType.CONTROLS
    )

    assert "As of October 26, 2025" in liquidity_section.text
    assert (
        "Quantitative and Qualitative Disclosures about Market Risk"
        not in liquidity_section.text
    )
    assert "Controls text." in controls_section.text


def test_ten_q_parser_splits_accounting_pronouncements_from_liquidity_section() -> None:
    sections = parse_document(
        RawDocument(
            document_id="doc-10q-accounting-boundary",
            company=build_company(),
            document_type=FilingType.FORM_10Q,
            content=(
                "Management's Discussion and Analysis of Financial Condition and Results "
                "of Operations\n"
                "Narrative intro.\n"
                "Liquidity and Capital Resources\n"
                "As of October 26, 2025, we had $60.6 billion in cash, cash equivalents, "
                "and marketable securities.\n"
                "Adoption of New and Recently Issued Accounting Pronouncements\n"
                "There has been no adoption of any new accounting pronouncements.\n"
                "Item 4. Controls and Procedures\n"
                "Controls text.\n"
            ),
            filing_metadata=FilingMetadata(
                filing_id="filing-10q-accounting-boundary",
                company=build_company(),
                filing_type=FilingType.FORM_10Q,
                accession_number="0001045810-25-000230",
                filed_at=date(2025, 11, 19),
            ),
        )
    )

    liquidity_section = next(
        section for section in sections if section.section_type == DocumentSectionType.LIQUIDITY
    )
    notes_section = next(
        section for section in sections if section.section_type == DocumentSectionType.NOTES
    )

    assert "As of October 26, 2025" in liquidity_section.text
    assert "Accounting Pronouncements" not in liquidity_section.text
    assert "new accounting pronouncements" in notes_section.text


def test_eight_k_parser_emits_item_numbers() -> None:
    sections = parse_document(build_raw_document("parser_8k.txt", FilingType.FORM_8K))

    assert [section.item_number for section in sections] == ["1.01", "2.02"]
    assert all(section.section_type == DocumentSectionType.ITEM for section in sections)
    assert all(section.parse_confidence == ConfidenceLabel.HIGH for section in sections)


def test_eight_k_parser_accepts_dotted_item_headings() -> None:
    sections = parse_document(
        RawDocument(
            document_id="doc-8k-dotted",
            company=build_company(),
            document_type=FilingType.FORM_8K,
            content=(
                "Item 5.02. Departure of Directors or Certain Officers.\n\n"
                "The board approved a compensation update.\n"
            ),
            filing_metadata=FilingMetadata(
                filing_id="filing-8k-dotted",
                company=build_company(),
                filing_type=FilingType.FORM_8K,
                accession_number="0001045810-24-000101",
                filed_at=date(2024, 11, 20),
            ),
        )
    )

    assert len(sections) == 1
    assert sections[0].item_number == "5.02"
    assert sections[0].section_type == DocumentSectionType.ITEM


def test_transcript_parser_emits_segments_and_speaker_turns() -> None:
    sections = parse_document(build_raw_document("parser_transcript.txt", FilingType.EARNINGS_CALL))

    segment_types = [section.transcript_segment_type for section in sections]
    assert TranscriptSegmentType.PREPARED_REMARKS in segment_types
    assert TranscriptSegmentType.QA in segment_types
    assert TranscriptSegmentType.SPEAKER_TURN in segment_types
    assert any(section.speaker == "Jensen Huang" for section in sections if section.speaker)


def test_parser_fallback_marks_low_confidence() -> None:
    sections = parse_document(build_raw_document("parser_fallback.txt", FilingType.FORM_10K))

    assert len(sections) == 1
    assert sections[0].section_type == DocumentSectionType.FALLBACK
    assert sections[0].parse_confidence == ConfidenceLabel.LOW
    assert sections[0].used_fallback is True
    assert sections[0].fallback_reason is not None
