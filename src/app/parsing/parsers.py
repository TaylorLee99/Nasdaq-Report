"""Rule-based and fallback parsers for supported document types."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain import (
    ConfidenceLabel,
    DocumentSectionType,
    ParsedSection,
    RawDocument,
    TranscriptSegmentType,
)
from app.parsing.base import DocumentParser


@dataclass(frozen=True)
class HeadingRule:
    """A heading rule mapped to a semantic section type."""

    pattern: re.Pattern[str]
    section_type: DocumentSectionType
    confidence: ConfidenceLabel


def normalize_whitespace(text: str) -> str:
    """Normalize dense whitespace while keeping section boundaries intact."""

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clean_leading_section_text(
    *,
    heading: str,
    text: str,
    section_type: DocumentSectionType,
) -> str:
    """Drop short heading-like artifacts that bleed into the section body."""

    cleaned = text.lstrip()
    if not cleaned:
        return cleaned

    first_line, separator, remainder = cleaned.partition("\n")
    normalized_first_line = " ".join(first_line.split()).strip()
    normalized_heading = " ".join(heading.split()).strip()
    if (
        separator
        and normalized_first_line
        and len(normalized_first_line) <= 120
        and normalized_first_line.lower() != normalized_heading.lower()
        and _looks_like_heading_fragment(normalized_first_line)
    ):
        cleaned = remainder.lstrip()

    if section_type == DocumentSectionType.BUSINESS:
        cleaned = re.sub(
            r"^(?:item\s+1\.\s*)?(?:our businesses|our business|business overview)\b[:\s-]*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).lstrip()
        cleaned = re.sub(
            r"^(?:item\s+1\.\s*)?business\b(?=\s+(?:NVIDIA|We|Our|The company)\b)[:\s-]*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).lstrip()

    return cleaned


def _looks_like_heading_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.match(r"^(?:item\s+\d+[a-z]?\.\s+)?[A-Z][A-Za-z0-9&/' -]{1,80}$", stripped):
        return True
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}$", stripped):
        return True
    return False


def build_section(
    *,
    document: RawDocument,
    ordinal: int,
    heading: str,
    text: str,
    char_start: int,
    char_end: int,
    section_type: DocumentSectionType,
    parse_confidence: ConfidenceLabel,
    used_fallback: bool = False,
    fallback_reason: str | None = None,
    item_number: str | None = None,
    transcript_segment_type: TranscriptSegmentType | None = None,
    speaker: str | None = None,
    parent_section_id: str | None = None,
) -> ParsedSection:
    """Create a parsed section with shared metadata."""

    cleaned_text = _clean_leading_section_text(
        heading=heading,
        text=text,
        section_type=section_type,
    )
    return ParsedSection(
        section_id=f"{document.document_id}:section:{ordinal}",
        document_id=document.document_id,
        company=document.company,
        filing_type=document.document_type,
        heading=heading,
        text=normalize_whitespace(cleaned_text),
        ordinal=ordinal,
        parent_section_id=parent_section_id,
        char_start=char_start,
        char_end=char_end,
        section_type=section_type,
        parse_confidence=parse_confidence,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        item_number=item_number,
        transcript_segment_type=transcript_segment_type,
        speaker=speaker,
    )


def fallback_section(document: RawDocument, reason: str) -> list[ParsedSection]:
    """Return a single fallback section covering the raw span."""

    content = normalize_whitespace(document.content)
    return [
        build_section(
            document=document,
            ordinal=1,
            heading=document.document_type.value,
            text=content,
            char_start=0,
            char_end=len(document.content),
            section_type=DocumentSectionType.FALLBACK,
            parse_confidence=ConfidenceLabel.LOW,
            used_fallback=True,
            fallback_reason=reason,
        )
    ]


class RegexHeadingParser(DocumentParser):
    """Base parser for filings with heading-like section markers."""

    rules: tuple[HeadingRule, ...] = ()

    def parse(self, document: RawDocument) -> list[ParsedSection]:
        matches: list[tuple[int, int, str, HeadingRule]] = []
        for rule in self.rules:
            for heading_match in rule.pattern.finditer(document.content):
                matches.append(
                    (
                        heading_match.start(),
                        heading_match.end(),
                        heading_match.group(0),
                        rule,
                    )
                )
        if not matches:
            return fallback_section(document, "No configured section headings matched.")

        deduped: dict[int, tuple[int, int, str, HeadingRule]] = {}
        for heading_tuple in sorted(matches, key=lambda item: item[0]):
            deduped.setdefault(heading_tuple[0], heading_tuple)
        ordered_matches = list(deduped.values())

        sections: list[ParsedSection] = []
        for index, (start, end, heading, rule) in enumerate(ordered_matches, start=1):
            next_start = (
                ordered_matches[index][0] if index < len(ordered_matches) else len(document.content)
            )
            body = document.content[end:next_start].strip()
            if not body:
                continue
            sections.append(
                build_section(
                    document=document,
                    ordinal=len(sections) + 1,
                    heading=heading.strip(),
                    text=body,
                    char_start=start,
                    char_end=next_start,
                    section_type=rule.section_type,
                    parse_confidence=rule.confidence,
                )
            )
        if not sections:
            return fallback_section(
                document,
                "Configured headings matched but no body spans were extracted.",
            )
        return sections


class TenKParser(RegexHeadingParser):
    """Rule-based parser for 10-K filings."""

    rules = (
        HeadingRule(
            re.compile(r"(?im)^item\s+1\.\s+business\b.*$"),
            DocumentSectionType.BUSINESS,
            ConfidenceLabel.HIGH,
        ),
        HeadingRule(
            re.compile(r"(?im)^item\s+1a\.\s+risk\s+factors\b.*$"),
            DocumentSectionType.RISK_FACTORS,
            ConfidenceLabel.HIGH,
        ),
        HeadingRule(
            re.compile(r"(?im)^item\s+7\.\s+management['’]s\s+discussion.*$"),
            DocumentSectionType.MDA,
            ConfidenceLabel.HIGH,
        ),
        HeadingRule(
            re.compile(r"(?im)^item\s+8\.\s+financial\s+statements.*$"),
            DocumentSectionType.FINANCIAL_STATEMENTS,
            ConfidenceLabel.HIGH,
        ),
        HeadingRule(
            re.compile(r"(?im)^notes\s+to\s+.*financial\s+statements.*$"),
            DocumentSectionType.NOTES,
            ConfidenceLabel.MEDIUM,
        ),
    )


class TenQParser(RegexHeadingParser):
    """Rule-based parser for 10-Q filings."""

    rules = (
        HeadingRule(
            re.compile(r"(?im)^management['’]s\s+discussion.*$"),
            DocumentSectionType.MDA,
            ConfidenceLabel.MEDIUM,
        ),
        HeadingRule(
            re.compile(r"(?im)^liquidity\s+and\s+capital\s+resources.*$"),
            DocumentSectionType.LIQUIDITY,
            ConfidenceLabel.HIGH,
        ),
        HeadingRule(
            re.compile(
                r"(?im)^adoption\s+of\s+new\s+and\s+recently\s+issued\s+accounting\s+pronouncements.*$"
            ),
            DocumentSectionType.NOTES,
            ConfidenceLabel.MEDIUM,
        ),
        HeadingRule(
            re.compile(r"(?im)^notes?\s+to\s+.*financial\s+statements.*$"),
            DocumentSectionType.NOTES,
            ConfidenceLabel.MEDIUM,
        ),
        HeadingRule(
            re.compile(
                r"(?im)^item\s+3\.\s+quantitative\s+and\s+qualitative\s+disclosures\s+about\s+market\s+risk.*$"
            ),
            DocumentSectionType.MDA,
            ConfidenceLabel.HIGH,
        ),
        HeadingRule(
            re.compile(r"(?im)^controls\s+and\s+procedures.*$"),
            DocumentSectionType.CONTROLS,
            ConfidenceLabel.HIGH,
        ),
        HeadingRule(
            re.compile(r"(?im)^item\s+4\.\s+controls\s+and\s+procedures.*$"),
            DocumentSectionType.CONTROLS,
            ConfidenceLabel.HIGH,
        ),
        HeadingRule(
            re.compile(r"(?im)^financial\s+statements.*$"),
            DocumentSectionType.FINANCIAL_STATEMENTS,
            ConfidenceLabel.MEDIUM,
        ),
    )


class EightKParser(DocumentParser):
    """Item-based parser for 8-K filings."""

    item_pattern = re.compile(r"(?im)^\s*(item\s+(\d+\.\d{2})\.?\s+.*)$")

    def parse(self, document: RawDocument) -> list[ParsedSection]:
        matches = list(self.item_pattern.finditer(document.content))
        if not matches:
            return fallback_section(document, "No 8-K item headings matched.")

        sections: list[ParsedSection] = []
        for index, match in enumerate(matches, start=1):
            start = match.start()
            next_start = matches[index].start() if index < len(matches) else len(document.content)
            body = document.content[match.end() : next_start].strip()
            if not body:
                continue
            sections.append(
                build_section(
                    document=document,
                    ordinal=len(sections) + 1,
                    heading=match.group(1).strip(),
                    text=body,
                    char_start=start,
                    char_end=next_start,
                    section_type=DocumentSectionType.ITEM,
                    parse_confidence=ConfidenceLabel.HIGH,
                    item_number=match.group(2),
                )
            )
        if not sections:
            return fallback_section(
                document,
                "8-K item headings matched but no body spans were extracted.",
            )
        return sections


class EarningsCallParser(DocumentParser):
    """Parser for earnings call transcripts with remarks, Q&A, and speaker turns."""

    qa_heading_pattern = re.compile(r"(?im)^question[- ]and[- ]answer(?:\s+session)?[:\s]*$")
    prepared_heading_pattern = re.compile(r"(?im)^prepared\s+remarks[:\s]*$")
    speaker_pattern = re.compile(r"(?m)^([A-Z][A-Za-z .'\-]+)\s*(?:--|:)\s*(.*)$")

    def parse(self, document: RawDocument) -> list[ParsedSection]:
        content = document.content
        qa_match = self.qa_heading_pattern.search(content)
        prepared_match = self.prepared_heading_pattern.search(content)

        boundaries: list[
            tuple[str, int, int, DocumentSectionType, TranscriptSegmentType, ConfidenceLabel]
        ] = []
        if prepared_match is not None:
            prepared_start = prepared_match.start()
            prepared_end = qa_match.start() if qa_match is not None else len(content)
            boundaries.append(
                (
                    prepared_match.group(0).strip(),
                    prepared_start,
                    prepared_end,
                    DocumentSectionType.PREPARED_REMARKS,
                    TranscriptSegmentType.PREPARED_REMARKS,
                    ConfidenceLabel.HIGH,
                )
            )
        if qa_match is not None:
            boundaries.append(
                (
                    qa_match.group(0).strip(),
                    qa_match.start(),
                    len(content),
                    DocumentSectionType.QA,
                    TranscriptSegmentType.QA,
                    ConfidenceLabel.HIGH,
                )
            )

        if not boundaries:
            if not self.speaker_pattern.search(content):
                return fallback_section(document, "No transcript segment heuristics matched.")
            boundaries.append(
                (
                    "Transcript",
                    0,
                    len(content),
                    DocumentSectionType.PREPARED_REMARKS,
                    TranscriptSegmentType.PREPARED_REMARKS,
                    ConfidenceLabel.LOW,
                )
            )

        sections: list[ParsedSection] = []
        for _boundary_index, (
            heading,
            start,
            end,
            section_type,
            segment_type,
            confidence,
        ) in enumerate(boundaries, start=1):
            heading_end = content.find("\n", start)
            if heading_end == -1 or heading_end > end:
                heading_end = start
            body_start = heading_end + 1 if heading_end >= start else start
            body = content[body_start:end].strip()
            parent = build_section(
                document=document,
                ordinal=len(sections) + 1,
                heading=heading,
                text=body,
                char_start=start,
                char_end=end,
                section_type=section_type,
                parse_confidence=confidence,
                transcript_segment_type=segment_type,
                used_fallback=confidence == ConfidenceLabel.LOW,
                fallback_reason=(
                    "Transcript segmented by speaker turns only."
                    if confidence == ConfidenceLabel.LOW
                    else None
                ),
            )
            sections.append(parent)
            sections.extend(self._speaker_turns(document, parent, body_start, end))
        return sections

    def _speaker_turns(
        self,
        document: RawDocument,
        parent: ParsedSection,
        start: int,
        end: int,
    ) -> list[ParsedSection]:
        speaker_matches = list(self.speaker_pattern.finditer(document.content, start, end))
        speaker_sections: list[ParsedSection] = []
        for index, match in enumerate(speaker_matches, start=1):
            turn_start = match.start()
            turn_end = speaker_matches[index].start() if index < len(speaker_matches) else end
            speaker_sections.append(
                build_section(
                    document=document,
                    ordinal=parent.ordinal + len(speaker_sections) + 1,
                    heading=match.group(1).strip(),
                    text=document.content[turn_start:turn_end].strip(),
                    char_start=turn_start,
                    char_end=turn_end,
                    section_type=DocumentSectionType.SPEAKER_TURN,
                    parse_confidence=ConfidenceLabel.MEDIUM,
                    transcript_segment_type=TranscriptSegmentType.SPEAKER_TURN,
                    speaker=match.group(1).strip(),
                    parent_section_id=parent.section_id,
                )
            )
        return speaker_sections
