"""Create SEC submissions metadata tables."""

import sqlalchemy as sa

from alembic import op

revision = "20260314_0002"
down_revision = "20260314_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sec_submission_metadata",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cik", sa.String(length=32), sa.ForeignKey("company_master.cik"), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("accession_number", sa.String(length=32), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("form_type", sa.String(length=16), nullable=False),
        sa.Column("primary_document", sa.String(length=255), nullable=False),
        sa.Column("report_period", sa.Date(), nullable=True),
        sa.Column("source_url", sa.String(length=1024), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("accession_number", name="uq_sec_submission_metadata"),
    )
    op.create_index(
        "ix_sec_submission_metadata_cik",
        "sec_submission_metadata",
        ["cik"],
        unique=False,
    )
    op.create_index(
        "ix_sec_submission_metadata_filing_date",
        "sec_submission_metadata",
        ["filing_date"],
        unique=False,
    )
    op.create_index(
        "ix_sec_submission_metadata_form_type",
        "sec_submission_metadata",
        ["form_type"],
        unique=False,
    )
    op.create_index(
        "ix_sec_submission_metadata_ticker",
        "sec_submission_metadata",
        ["ticker"],
        unique=False,
    )

    op.create_table(
        "sec_submissions_raw",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cik", sa.String(length=32), sa.ForeignKey("company_master.cik"), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
    )
    op.create_index("ix_sec_submissions_raw_cik", "sec_submissions_raw", ["cik"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_sec_submissions_raw_cik", table_name="sec_submissions_raw")
    op.drop_table("sec_submissions_raw")
    op.drop_index("ix_sec_submission_metadata_ticker", table_name="sec_submission_metadata")
    op.drop_index("ix_sec_submission_metadata_form_type", table_name="sec_submission_metadata")
    op.drop_index("ix_sec_submission_metadata_filing_date", table_name="sec_submission_metadata")
    op.drop_index("ix_sec_submission_metadata_cik", table_name="sec_submission_metadata")
    op.drop_table("sec_submission_metadata")
