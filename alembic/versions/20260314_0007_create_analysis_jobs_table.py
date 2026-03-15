"""Create analysis job persistence table."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260314_0007"
down_revision = "20260314_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analysis_jobs",
        sa.Column("analysis_id", sa.String(length=128), primary_key=True),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("cik", sa.String(length=32), sa.ForeignKey("company_master.cik"), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("document_signals_json", sa.JSON(), nullable=False),
        sa.Column("request_payload_json", sa.JSON(), nullable=False),
        sa.Column("final_report_json", sa.JSON(), nullable=True),
        sa.Column("final_report_markdown", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_analysis_jobs_ticker", "analysis_jobs", ["ticker"], unique=False)
    op.create_index("ix_analysis_jobs_cik", "analysis_jobs", ["cik"], unique=False)
    op.create_index("ix_analysis_jobs_status", "analysis_jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_analysis_jobs_status", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_cik", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_ticker", table_name="analysis_jobs")
    op.drop_table("analysis_jobs")
