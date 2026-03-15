"""Document ingestion entrypoints."""

from app.data_sources import ALLOWED_FORM_TYPES
from app.domain import FilingType, RawDocument


def ingest_document(document: RawDocument) -> RawDocument:
    """Validate and pass through a normalized raw document."""

    if document.document_type.value not in ALLOWED_FORM_TYPES:
        msg = f"Unsupported form type: {document.document_type}"
        raise ValueError(msg)
    if document.document_type == FilingType.EARNINGS_CALL and document.transcript_metadata is None:
        msg = "Transcript ingestion requires transcript metadata"
        raise ValueError(msg)
    return document
