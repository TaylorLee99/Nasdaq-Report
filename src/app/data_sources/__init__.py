"""Source constraints and registry helpers."""

from app.data_sources.registry import ALLOWED_FORM_TYPES
from app.data_sources.sec import (
    HttpBinaryResponse,
    RateLimiter,
    RetryableHttpError,
    SecArchiveClient,
    SecMetadataService,
    SecRawFilingDownloadService,
    SecSubmissionsClient,
    UrllibBinaryHttpClient,
    UrllibJsonHttpClient,
)
from app.data_sources.transcripts import (
    FileTranscriptSourceAdapter,
    GenericHttpTranscriptSourceAdapter,
    TranscriptIngestionService,
)
from app.data_sources.xbrl import (
    FileXbrlFixtureLoader,
    XbrlIngestionService,
    normalize_xbrl_fixture,
)

__all__ = [
    "ALLOWED_FORM_TYPES",
    "HttpBinaryResponse",
    "RateLimiter",
    "RetryableHttpError",
    "SecArchiveClient",
    "SecMetadataService",
    "SecRawFilingDownloadService",
    "SecSubmissionsClient",
    "TranscriptIngestionService",
    "FileTranscriptSourceAdapter",
    "FileXbrlFixtureLoader",
    "GenericHttpTranscriptSourceAdapter",
    "UrllibBinaryHttpClient",
    "UrllibJsonHttpClient",
    "XbrlIngestionService",
    "normalize_xbrl_fixture",
]
