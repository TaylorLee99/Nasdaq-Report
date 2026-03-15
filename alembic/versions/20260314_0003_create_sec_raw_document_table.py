"""Create raw SEC filing download state table."""

import sqlalchemy as sa

from alembic import op

revision = "20260314_0003"
down_revision = "20260314_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sec_raw_documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "accession_number",
            sa.String(length=32),
            sa.ForeignKey("sec_submission_metadata.accession_number"),
            nullable=False,
        ),
        sa.Column("cik", sa.String(length=32), sa.ForeignKey("company_master.cik"), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("form_type", sa.String(length=16), nullable=False),
        sa.Column("source_url", sa.String(length=1024), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=True),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("downloaded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.UniqueConstraint("accession_number", name="uq_sec_raw_documents"),
    )
    op.create_index("ix_sec_raw_documents_cik", "sec_raw_documents", ["cik"], unique=False)
    op.create_index(
        "ix_sec_raw_documents_form_type",
        "sec_raw_documents",
        ["form_type"],
        unique=False,
    )
    op.create_index(
        "ix_sec_raw_documents_status",
        "sec_raw_documents",
        ["status"],
        unique=False,
    )
    op.create_index("ix_sec_raw_documents_ticker", "sec_raw_documents", ["ticker"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_sec_raw_documents_ticker", table_name="sec_raw_documents")
    op.drop_index("ix_sec_raw_documents_status", table_name="sec_raw_documents")
    op.drop_index("ix_sec_raw_documents_form_type", table_name="sec_raw_documents")
    op.drop_index("ix_sec_raw_documents_cik", table_name="sec_raw_documents")
    op.drop_table("sec_raw_documents")
