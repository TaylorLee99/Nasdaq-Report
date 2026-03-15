"""Universe snapshot schemas."""

from datetime import date

from pydantic import Field, model_validator

from app.domain.models import CanonicalModel, Company


class UniverseSnapshotConstituent(CanonicalModel):
    """A single company membership record for a snapshot date."""

    snapshot_date: date
    ticker: str
    cik: str
    company_name: str
    exchange: str
    is_domestic_filer: bool

    def to_company(self) -> Company:
        """Convert a universe record into the canonical company model."""

        return Company(
            cik=self.cik,
            ticker=self.ticker,
            company_name=self.company_name,
            exchange=self.exchange,
            is_domestic_filer=self.is_domestic_filer,
            latest_snapshot_date=self.snapshot_date,
        )


class UniverseSnapshot(CanonicalModel):
    """A dated snapshot of the analysis universe."""

    snapshot_date: date
    constituents: list[UniverseSnapshotConstituent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_snapshot_dates(self) -> "UniverseSnapshot":
        """Ensure all constituent rows share the same snapshot date."""

        for constituent in self.constituents:
            if constituent.snapshot_date != self.snapshot_date:
                msg = "All constituents must match the parent snapshot_date"
                raise ValueError(msg)
        return self


class UniverseLoadResult(CanonicalModel):
    """Summary of a universe snapshot load."""

    snapshot_date: date
    source_file: str
    total_rows: int
    loaded_rows: int
    filtered_out_rows: int
