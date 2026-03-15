"""Section-preserving chunking helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain import FilingType, ParsedSection


@dataclass(frozen=True)
class ChunkSpan:
    """A chunk of text carved out from a parsed section."""

    text: str
    char_start: int | None
    char_end: int | None
    chunk_index: int


@dataclass(frozen=True)
class ChunkingConfig:
    """Configurable chunking parameters for retrieval indexing."""

    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 150

    def __post_init__(self) -> None:
        if self.chunk_size_chars <= 0:
            msg = "chunk_size_chars must be positive"
            raise ValueError(msg)
        if self.chunk_overlap_chars < 0:
            msg = "chunk_overlap_chars must be non-negative"
            raise ValueError(msg)
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            msg = "chunk_overlap_chars must be smaller than chunk_size_chars"
            raise ValueError(msg)


class SectionChunker:
    """Split sections into retrieval chunks without crossing section boundaries."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    def chunk_section(self, section: ParsedSection) -> list[ChunkSpan]:
        text = section.text.strip()
        if not text:
            return []

        spans: list[ChunkSpan] = []
        start = 0
        chunk_index = 1
        text_length = len(text)
        while start < text_length:
            tentative_end = min(start + self._config.chunk_size_chars, text_length)
            end = self._find_split_point(text, start, tentative_end)
            raw_chunk = text[start:end].strip()
            if not raw_chunk:
                break
            if self._should_skip_chunk(section=section, chunk_text=raw_chunk):
                if end >= text_length:
                    break
                next_start = max(end - self._config.chunk_overlap_chars, start + 1)
                next_start = self._adjust_start_point(text, next_start, text_length)
                start = next_start
                chunk_index += 1
                continue

            leading_trim = len(text[start:end]) - len(text[start:end].lstrip())
            trailing_trim = len(text[start:end]) - len(text[start:end].rstrip())
            local_start = start + leading_trim
            local_end = end - trailing_trim
            spans.append(
                ChunkSpan(
                    text=raw_chunk,
                    char_start=(
                        (section.char_start or 0) + local_start
                        if section.char_start is not None
                        else local_start
                    ),
                    char_end=(
                        (section.char_start or 0) + local_end
                        if section.char_start is not None
                        else local_end
                    ),
                    chunk_index=chunk_index,
                )
            )
            if end >= text_length:
                break
            next_start = max(end - self._config.chunk_overlap_chars, start + 1)
            next_start = self._adjust_start_point(text, next_start, text_length)
            start = next_start
            chunk_index += 1
        return spans

    @staticmethod
    def _should_skip_chunk(*, section: ParsedSection, chunk_text: str) -> bool:
        if section.filing_type not in {FilingType.FORM_10K, FilingType.FORM_10Q}:
            return False

        normalized = " ".join(chunk_text.split()).strip()
        if not normalized:
            return True

        lowered = normalized.lower()
        if _looks_like_numeric_residue(normalized):
            return True
        if _looks_like_item_heading_fragment(normalized):
            return True
        if _looks_table_like_block(normalized):
            return True
        if _looks_like_accounting_pronouncement_fragment(normalized):
            return True

        generic_markers = (
            "the following table sets forth",
            "refer to item 1a",
            "pursuant to the requirements",
        )
        if any(marker in lowered for marker in generic_markers) and len(normalized) < 220:
            return True
        return False

    @staticmethod
    def _find_split_point(text: str, start: int, tentative_end: int) -> int:
        if tentative_end >= len(text):
            return len(text)

        minimum_break = start + max((tentative_end - start) // 2, 1)
        window = text[minimum_break:tentative_end]
        for pattern in (r"\n\s*\n", r"\n", r"\. ", r"; ", r", ", r" "):
            matches = list(re.finditer(pattern, window))
            if matches:
                return minimum_break + matches[-1].end()
        return tentative_end

    @staticmethod
    def _adjust_start_point(text: str, start: int, text_length: int) -> int:
        if start >= text_length:
            return text_length
        adjusted = start
        if 0 < adjusted < text_length and text[adjusted - 1].isalnum() and text[adjusted].isalnum():
            while adjusted < text_length and text[adjusted].isalnum():
                adjusted += 1
        while adjusted < text_length and text[adjusted].isspace():
            adjusted += 1
        return min(adjusted, text_length)


def _looks_like_item_heading_fragment(text: str) -> bool:
    lowered = text.lower()
    if re.match(r"^(item|part)\s+[0-9ivx]+", lowered) and len(text) <= 160:
        return True
    if re.match(r"^\d{1,2}\s+item\s+\d", lowered):
        return True
    if re.match(r"^[a-z0-9 ,.$()-]{1,40}\s+item\s+\d", lowered):
        return True
    return False


def _looks_like_numeric_residue(text: str) -> bool:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return True
    if re.fullmatch(r"\d{1,3}", normalized):
        return True
    tokens = normalized.split()
    if len(tokens) <= 2 and all(re.fullmatch(r"[\d.,$()%]+", token) for token in tokens):
        return True
    return False


def _looks_table_like_block(text: str) -> bool:
    number_tokens = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)
    dollar_tokens = re.findall(r"\$\s?\d", text)
    month_tokens = re.findall(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
        text.lower(),
    )
    if len(number_tokens) >= 10 and len(dollar_tokens) >= 2:
        return True
    if len(number_tokens) >= 12 and len(month_tokens) >= 1:
        return True
    if text.count("$") >= 3 and text.count(",") >= 8:
        return True
    return False


def _looks_like_accounting_pronouncement_fragment(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "adoption of new and recently issued accounting pronouncements",
        "recently issued accounting pronouncements",
        "new accounting pronouncements",
        "recent accounting pronouncements",
        "summary of significant accounting policies",
        "critical accounting estimates",
        "critical accounting policies",
        "forward-looking statements",
        "private securities litigation reform act",
    )
    return any(marker in lowered for marker in markers)
