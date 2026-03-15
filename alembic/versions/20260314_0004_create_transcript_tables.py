"""Create transcript metadata and availability gap tables."""

import sqlalchemy as sa

from alembic import op

revision = "20260314_0004"
down_revision = "20260314_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "transcript_metadata",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("transcript_id", sa.String(length=128), nullable=False),
        sa.Column("cik", sa.String(length=32), sa.ForeignKey("company_master.cik"), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("call_date", sa.Date(), nullable=False),
        sa.Column("fiscal_quarter", sa.String(length=32), nullable=True),
        sa.Column("fiscal_year", sa.Integer(), nullable=True),
        sa.Column("source_url", sa.String(length=1024), nullable=True),
        sa.Column("speaker_separated", sa.Boolean(), nullable=False),
        sa.Column("source_reliability", sa.String(length=16), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("coverage_label", sa.String(length=16), nullable=False),
        sa.Column("covered_topics", sa.JSON(), nullable=False),
        sa.Column("missing_topics", sa.JSON(), nullable=False),
        sa.Column("coverage_notes", sa.Text(), nullable=True),
        sa.Column("storage_path", sa.String(length=1024), nullable=True),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("transcript_id", name="uq_transcript_metadata"),
    )
    op.create_index("ix_transcript_metadata_cik", "transcript_metadata", ["cik"], unique=False)
    op.create_index(
        "ix_transcript_metadata_call_date",
        "transcript_metadata",
        ["call_date"],
        unique=False,
    )
    op.create_index(
        "ix_transcript_metadata_ticker",
        "transcript_metadata",
        ["ticker"],
        unique=False,
    )

    op.create_table(
        "transcript_availability_gaps",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("gap_id", sa.String(length=128), nullable=False),
        sa.Column("cik", sa.String(length=32), sa.ForeignKey("company_master.cik"), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("source_url", sa.String(length=1024), nullable=True),
        sa.Column("coverage_label", sa.String(length=16), nullable=False),
        sa.Column("covered_topics", sa.JSON(), nullable=False),
        sa.Column("missing_topics", sa.JSON(), nullable=False),
        sa.Column("coverage_notes", sa.Text(), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("gap_id", name="uq_transcript_availability_gaps"),
    )
    op.create_index(
        "ix_transcript_availability_gaps_cik",
        "transcript_availability_gaps",
        ["cik"],
        unique=False,
    )
    op.create_index(
        "ix_transcript_availability_gaps_ticker",
        "transcript_availability_gaps",
        ["ticker"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_transcript_availability_gaps_ticker",
        table_name="transcript_availability_gaps",
    )
    op.drop_index(
        "ix_transcript_availability_gaps_cik",
        table_name="transcript_availability_gaps",
    )
    op.drop_table("transcript_availability_gaps")
    op.drop_index("ix_transcript_metadata_ticker", table_name="transcript_metadata")
    op.drop_index("ix_transcript_metadata_call_date", table_name="transcript_metadata")
    op.drop_index("ix_transcript_metadata_cik", table_name="transcript_metadata")
    op.drop_table("transcript_metadata")
