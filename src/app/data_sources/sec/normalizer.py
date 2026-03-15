"""Normalize SEC submissions payloads into canonical filing metadata."""

from __future__ import annotations

from datetime import date

from app.data_sources.sec.archive import build_archive_url
from app.domain import FilingType, SecSubmissionMetadata

SUPPORTED_FORMS: tuple[FilingType, ...] = (
    FilingType.FORM_10K,
    FilingType.FORM_10Q,
    FilingType.FORM_8K,
)


def coerce_form_type(raw_form: str) -> FilingType | None:
    """Convert an SEC form label into the canonical filing enum."""

    normalized = raw_form.strip().upper()
    if normalized == FilingType.FORM_10K.value:
        return FilingType.FORM_10K
    if normalized == FilingType.FORM_10Q.value:
        return FilingType.FORM_10Q
    if normalized == FilingType.FORM_8K.value:
        return FilingType.FORM_8K
    return None


def get_recent_list(payload: dict[str, object], key: str) -> list[str]:
    """Extract a recent filings list from the SEC payload."""

    filings = payload.get("filings")
    if not isinstance(filings, dict):
        return []
    recent = filings.get("recent")
    if not isinstance(recent, dict):
        return []
    values = recent.get(key)
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def normalize_submissions_payload(
    payload: dict[str, object],
    *,
    cik: str,
    ticker: str,
) -> list[SecSubmissionMetadata]:
    """Normalize the SEC submissions payload into filing metadata rows."""

    accession_numbers = get_recent_list(payload, "accessionNumber")
    filing_dates = get_recent_list(payload, "filingDate")
    forms = get_recent_list(payload, "form")
    primary_documents = get_recent_list(payload, "primaryDocument")
    report_dates = get_recent_list(payload, "reportDate")

    row_count = max(
        len(accession_numbers),
        len(filing_dates),
        len(forms),
        len(primary_documents),
        len(report_dates),
    )
    normalized: list[SecSubmissionMetadata] = []
    for index in range(row_count):
        raw_form = forms[index] if index < len(forms) else ""
        form_type = coerce_form_type(raw_form)
        if form_type is None:
            continue
        accession_number = accession_numbers[index] if index < len(accession_numbers) else ""
        filing_date = filing_dates[index] if index < len(filing_dates) else ""
        primary_document = primary_documents[index] if index < len(primary_documents) else ""
        if not accession_number or not filing_date or not primary_document:
            continue
        report_period = report_dates[index] if index < len(report_dates) else ""
        normalized.append(
            SecSubmissionMetadata(
                cik=cik,
                ticker=ticker,
                accession_number=accession_number,
                filing_date=date.fromisoformat(filing_date),
                form_type=form_type,
                primary_document=primary_document,
                report_period=date.fromisoformat(report_period) if report_period else None,
                source_url=build_archive_url(cik, accession_number, primary_document),
            )
        )
    return normalized


def filter_supported_forms(
    filings: list[SecSubmissionMetadata],
    forms: list[FilingType] | None,
) -> list[SecSubmissionMetadata]:
    """Filter normalized filing rows to the requested supported forms."""

    if forms is None:
        allowed_forms = set(SUPPORTED_FORMS)
    else:
        allowed_forms = {form for form in forms if form in SUPPORTED_FORMS}
    return [filing for filing in filings if filing.form_type in allowed_forms]
