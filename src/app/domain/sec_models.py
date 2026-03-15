"""SEC submissions metadata and raw document schemas."""

from datetime import UTC, date, datetime

from pydantic import Field

from app.domain.enums import DownloadStatusLabel, FilingType
from app.domain.models import CanonicalModel


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class SecSubmissionMetadata(CanonicalModel):
    """Normalized SEC filing metadata row."""

    cik: str
    ticker: str
    accession_number: str
    filing_date: date
    form_type: FilingType
    primary_document: str
    report_period: date | None = None
    source_url: str


class SecSubmissionsSyncResult(CanonicalModel):
    """Summary for a submissions fetch or sync run."""

    cik: str
    ticker: str
    requested_forms: list[FilingType] = Field(default_factory=list)
    fetched_rows: int
    stored_rows: int
    save_raw_json: bool = False


class SecRawDocumentRecord(CanonicalModel):
    """Raw filing download state stored in the metadata plane."""

    accession_number: str
    cik: str
    ticker: str
    form_type: FilingType
    source_url: str
    storage_path: str | None = None
    checksum_sha256: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    status: DownloadStatusLabel = DownloadStatusLabel.PENDING
    downloaded_at: datetime | None = None
    retry_count: int = 0
    last_error: str | None = None


class SecRawDownloadResult(CanonicalModel):
    """Summary of a raw filing download run."""

    ticker: str
    requested_forms: list[FilingType] = Field(default_factory=list)
    attempted_count: int
    downloaded_count: int
    skipped_count: int
    failed_count: int
    completed_at: datetime = Field(default_factory=utc_now)
