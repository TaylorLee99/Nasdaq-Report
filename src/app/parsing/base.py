"""Shared parser interfaces and helper types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain import ParsedSection, RawDocument


class DocumentParser(Protocol):
    """Common interface for form-aware parsers."""

    def parse(self, document: RawDocument) -> list[ParsedSection]:
        """Parse a raw document into semantic sections."""


@dataclass(frozen=True)
class SectionBoundary:
    """A raw character span identified by a rule-based parser."""

    heading: str
    start: int
    end: int
