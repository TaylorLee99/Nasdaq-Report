"""Normalize fixture XBRL facts into canonical evidence rows."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.domain import Company, FilingType, XbrlCanonicalFact, XbrlFact

CANONICAL_CONCEPT_MAP: dict[str, XbrlCanonicalFact] = {
    "revenues": XbrlCanonicalFact.REVENUE,
    "salesrevenuenet": XbrlCanonicalFact.REVENUE,
    "operatingincomeloss": XbrlCanonicalFact.OPERATING_INCOME,
    "netincomeloss": XbrlCanonicalFact.NET_INCOME,
    "earningspersharediluted": XbrlCanonicalFact.DILUTED_EPS,
    "cashandcashequivalentsatcarryingvalue": XbrlCanonicalFact.CASH_AND_CASH_EQUIVALENTS,
    "netcashprovidedbyusedinoperatingactivities": XbrlCanonicalFact.OPERATING_CASH_FLOW,
    "assets": XbrlCanonicalFact.TOTAL_ASSETS,
    "liabilities": XbrlCanonicalFact.TOTAL_LIABILITIES,
}


def normalize_xbrl_fixture(
    payload: dict[str, Any],
    *,
    company: Company,
    filing_type: FilingType,
) -> list[XbrlFact]:
    """Normalize a fixture payload into canonical XBRL fact rows."""

    accession_number = str(payload["accession_number"])
    filing_date = payload.get("filing_date")
    fiscal_period = payload.get("fiscal_period")
    facts = payload.get("facts", [])
    normalized: list[XbrlFact] = []
    for index, fact in enumerate(facts, start=1):
        concept = str(fact["concept"]).strip()
        concept_key = concept.lower()
        if concept_key not in CANONICAL_CONCEPT_MAP:
            continue
        normalized.append(
            XbrlFact(
                fact_id=f"{accession_number}:{index}",
                accession_number=accession_number,
                company=company,
                filing_type=filing_type,
                canonical_fact=CANONICAL_CONCEPT_MAP[concept_key],
                original_concept=concept,
                numeric_value=Decimal(str(fact["value"])),
                unit=fact.get("unit"),
                period_start=fact.get("period_start"),
                period_end=fact.get("period_end"),
                frame=fact.get("frame"),
                context_ref=fact.get("context_ref"),
                decimals=fact.get("decimals"),
                filing_date=filing_date,
                fiscal_period=fiscal_period,
            )
        )
    return normalized
