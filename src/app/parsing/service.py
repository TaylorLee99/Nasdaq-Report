"""Form-aware parsing service and parser dispatcher."""

from __future__ import annotations

from app.domain import FilingType, ParsedSection, RawDocument
from app.parsing.base import DocumentParser
from app.parsing.parsers import EarningsCallParser, EightKParser, TenKParser, TenQParser


class ParserDispatcher:
    """Dispatch raw documents to the matching form-aware parser."""

    def __init__(self) -> None:
        self._parsers: dict[FilingType, DocumentParser] = {
            FilingType.FORM_10K: TenKParser(),
            FilingType.FORM_10Q: TenQParser(),
            FilingType.FORM_8K: EightKParser(),
            FilingType.EARNINGS_CALL: EarningsCallParser(),
        }

    def parse(self, document: RawDocument) -> list[ParsedSection]:
        """Parse the document using the parser registered for its filing type."""

        return self._parsers[document.document_type].parse(document)


def parse_document(document: RawDocument) -> list[ParsedSection]:
    """Parse a raw document using the default parser dispatcher."""

    return ParserDispatcher().parse(document)
