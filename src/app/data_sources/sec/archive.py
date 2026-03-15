"""SEC archive URL and raw store path helpers."""

from __future__ import annotations

from app.domain import SecSubmissionMetadata


def build_archive_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Construct the canonical SEC archive URL for a filing document."""

    accession_without_dashes = accession_number.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_without_dashes}/{primary_document}"
    )


def build_raw_store_path(filing: SecSubmissionMetadata) -> str:
    """Build the relative raw document store path for a filing."""

    return (
        "sec/"
        f"{filing.ticker}/"
        f"{filing.form_type.value}/"
        f"{filing.filing_date.isoformat()}/"
        f"{filing.accession_number}/"
        f"{filing.primary_document}"
    )
