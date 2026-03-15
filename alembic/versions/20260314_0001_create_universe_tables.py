"""Create universe snapshot and company master tables."""

import sqlalchemy as sa

from alembic import op

revision = "20260314_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "company_master",
        sa.Column("cik", sa.String(length=32), primary_key=True),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("is_domestic_filer", sa.Boolean(), nullable=False),
        sa.Column("latest_snapshot_date", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_company_master_ticker", "company_master", ["ticker"], unique=False)

    op.create_table(
        "universe_snapshot_constituents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("cik", sa.String(length=32), sa.ForeignKey("company_master.cik"), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("is_domestic_filer", sa.Boolean(), nullable=False),
        sa.Column("loaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("snapshot_date", "cik", name="uq_universe_snapshot_constituents"),
    )
    op.create_index(
        "ix_universe_snapshot_constituents_snapshot_date",
        "universe_snapshot_constituents",
        ["snapshot_date"],
        unique=False,
    )
    op.create_index(
        "ix_universe_snapshot_constituents_ticker",
        "universe_snapshot_constituents",
        ["ticker"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_universe_snapshot_constituents_ticker",
        table_name="universe_snapshot_constituents",
    )
    op.drop_index(
        "ix_universe_snapshot_constituents_snapshot_date",
        table_name="universe_snapshot_constituents",
    )
    op.drop_table("universe_snapshot_constituents")
    op.drop_index("ix_company_master_ticker", table_name="company_master")
    op.drop_table("company_master")
