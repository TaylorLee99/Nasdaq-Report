"""Universe snapshot ingestion interfaces and services."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Protocol

from app.domain import UniverseLoadResult, UniverseSnapshot, UniverseSnapshotConstituent
from app.storage.repositories import UniverseRepository


class UniverseConstituentLoader(Protocol):
    """Loader interface for provider snapshots or fixture files."""

    def load(
        self,
        file_path: Path,
        *,
        snapshot_date_override: date | None = None,
    ) -> list[UniverseSnapshotConstituent]:
        """Load constituent rows from a source file."""


class DomesticFilerFilter(Protocol):
    """Filter interface for universe membership policies."""

    def apply(
        self,
        constituents: Sequence[UniverseSnapshotConstituent],
    ) -> list[UniverseSnapshotConstituent]:
        """Filter the provided constituents."""


def parse_bool(raw_value: str) -> bool:
    """Parse a CSV boolean field."""

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    msg = f"Unsupported boolean value: {raw_value}"
    raise ValueError(msg)


class CsvUniverseConstituentLoader:
    """CSV-backed loader for universe snapshot fixtures."""

    required_columns = {
        "ticker",
        "cik",
        "company_name",
        "exchange",
        "is_domestic_filer",
    }

    def load(
        self,
        file_path: Path,
        *,
        snapshot_date_override: date | None = None,
    ) -> list[UniverseSnapshotConstituent]:
        """Load constituent rows from a CSV file."""

        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                msg = f"CSV file has no header: {file_path}"
                raise ValueError(msg)
            missing = self.required_columns - set(reader.fieldnames)
            if missing:
                msg = f"CSV file missing required columns: {sorted(missing)}"
                raise ValueError(msg)
            if snapshot_date_override is None and "snapshot_date" not in reader.fieldnames:
                msg = "CSV file missing required column: snapshot_date"
                raise ValueError(msg)

            constituents: list[UniverseSnapshotConstituent] = []
            for row in reader:
                row_snapshot_date = snapshot_date_override or date.fromisoformat(
                    row["snapshot_date"]
                )
                constituents.append(
                    UniverseSnapshotConstituent(
                        snapshot_date=row_snapshot_date,
                        ticker=row["ticker"].strip().upper(),
                        cik=row["cik"].strip(),
                        company_name=row["company_name"].strip(),
                        exchange=row["exchange"].strip().upper(),
                        is_domestic_filer=parse_bool(row["is_domestic_filer"]),
                    )
                )
        return constituents


class DomesticFilerOnlyFilter:
    """Retain only domestic filers in the active universe."""

    def apply(
        self,
        constituents: Sequence[UniverseSnapshotConstituent],
    ) -> list[UniverseSnapshotConstituent]:
        """Return only domestic filer constituents."""

        return [constituent for constituent in constituents if constituent.is_domestic_filer]


class UniverseIngestionService:
    """Service that loads, filters, and persists universe snapshots."""

    def __init__(
        self,
        loader: UniverseConstituentLoader,
        filter_policy: DomesticFilerFilter,
        repository: UniverseRepository,
    ) -> None:
        self._loader = loader
        self._filter_policy = filter_policy
        self._repository = repository

    def load_snapshot(
        self,
        file_path: Path,
        *,
        snapshot_date_override: date | None = None,
    ) -> UniverseLoadResult:
        """Load a snapshot file into the company master and snapshot tables."""

        raw_constituents = self._loader.load(
            file_path,
            snapshot_date_override=snapshot_date_override,
        )
        if not raw_constituents:
            msg = f"No universe rows found in {file_path}"
            raise ValueError(msg)

        filtered_constituents = self._filter_policy.apply(raw_constituents)
        snapshot_date = snapshot_date_override or raw_constituents[0].snapshot_date
        snapshot = UniverseSnapshot(snapshot_date=snapshot_date, constituents=filtered_constituents)
        self._repository.replace_snapshot(snapshot)
        return UniverseLoadResult(
            snapshot_date=snapshot.snapshot_date,
            source_file=str(file_path),
            total_rows=len(raw_constituents),
            loaded_rows=len(filtered_constituents),
            filtered_out_rows=len(raw_constituents) - len(filtered_constituents),
        )
