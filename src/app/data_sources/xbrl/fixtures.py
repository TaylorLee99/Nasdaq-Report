"""Fixture loaders for MVP XBRL ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class XbrlFixtureLoader(Protocol):
    """Abstract fixture loader for normalized XBRL import."""

    def load(self, file_path: Path) -> dict[str, Any]:
        """Load a fixture payload from a file path."""


class FileXbrlFixtureLoader:
    """JSON fixture loader for XBRL fact imports."""

    def load(self, file_path: Path) -> dict[str, Any]:
        """Read a JSON fixture from disk."""

        return json.loads(file_path.read_text(encoding="utf-8"))
