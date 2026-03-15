from __future__ import annotations

from app.domain import ChunkRecord, Company, FilingType
from app.domain.enums import DocumentSectionType
from app.graph.ten_q_heuristics import (
    build_ten_q_recent_change_candidate,
    is_ten_q_recent_change_candidate_sentence,
)


def _make_chunk(text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chunk-10q",
        document_id="doc-10q",
        section_id="section-10q",
        company=Company(cik="0001018724", ticker="AMZN", company_name="Amazon.com Inc."),
        filing_type=FilingType.FORM_10Q,
        section_name="Liquidity and Capital Resources — Overview",
        section_type=DocumentSectionType.MDA,
        chunk_index=1,
        text=text,
        token_count=len(text.split()),
    )


def test_ten_q_recent_change_candidate_rejects_foreign_exchange_only_sentence() -> None:
    sentence = (
        "Changes in foreign exchange rates did not significantly impact operating "
        "income for Q3 2025, but negatively impacted operating income by $131 million "
        "for the nine months ended September 30, 2025."
    )

    assert (
        is_ten_q_recent_change_candidate_sentence(
            sentence=sentence,
            section_name="Liquidity and Capital Resources — Overview",
            section_type=DocumentSectionType.MDA,
        )
        is False
    )


def test_ten_q_recent_change_candidate_prefers_operating_delta_over_fx_impact() -> None:
    chunk = _make_chunk(
        "The decrease in North America operating income in Q3 2025, compared to the "
        "comparable prior year period, is primarily due to increased fulfillment and "
        "shipping costs. Changes in foreign exchange rates did not significantly impact "
        "operating income for Q3 2025, but negatively impacted operating income by "
        "$131 million for the nine months ended September 30, 2025."
    )

    candidate = build_ten_q_recent_change_candidate(chunk)

    assert candidate is not None
    assert candidate.startswith("The decrease in North America operating income in Q3 2025")
