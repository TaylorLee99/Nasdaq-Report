"""Create canonical XBRL fact table."""

import sqlalchemy as sa

from alembic import op

revision = "20260314_0005"
down_revision = "20260314_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "xbrl_facts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("fact_id", sa.String(length=128), nullable=False),
        sa.Column(
            "accession_number",
            sa.String(length=32),
            sa.ForeignKey("sec_submission_metadata.accession_number"),
            nullable=False,
        ),
        sa.Column("cik", sa.String(length=32), sa.ForeignKey("company_master.cik"), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("filing_type", sa.String(length=16), nullable=False),
        sa.Column("canonical_fact", sa.String(length=64), nullable=False),
        sa.Column("original_concept", sa.String(length=255), nullable=False),
        sa.Column("numeric_value", sa.Numeric(20, 6), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=True),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("frame", sa.String(length=64), nullable=True),
        sa.Column("context_ref", sa.String(length=128), nullable=True),
        sa.Column("decimals", sa.Integer(), nullable=True),
        sa.Column("filing_date", sa.Date(), nullable=True),
        sa.Column("fiscal_period", sa.String(length=32), nullable=True),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("fact_id", name="uq_xbrl_facts"),
    )
    op.create_index(
        "ix_xbrl_facts_accession_number",
        "xbrl_facts",
        ["accession_number"],
        unique=False,
    )
    op.create_index("ix_xbrl_facts_cik", "xbrl_facts", ["cik"], unique=False)
    op.create_index("ix_xbrl_facts_ticker", "xbrl_facts", ["ticker"], unique=False)
    op.create_index("ix_xbrl_facts_filing_type", "xbrl_facts", ["filing_type"], unique=False)
    op.create_index("ix_xbrl_facts_canonical_fact", "xbrl_facts", ["canonical_fact"], unique=False)
    op.create_index("ix_xbrl_facts_period_end", "xbrl_facts", ["period_end"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_xbrl_facts_period_end", table_name="xbrl_facts")
    op.drop_index("ix_xbrl_facts_canonical_fact", table_name="xbrl_facts")
    op.drop_index("ix_xbrl_facts_filing_type", table_name="xbrl_facts")
    op.drop_index("ix_xbrl_facts_ticker", table_name="xbrl_facts")
    op.drop_index("ix_xbrl_facts_cik", table_name="xbrl_facts")
    op.drop_index("ix_xbrl_facts_accession_number", table_name="xbrl_facts")
    op.drop_table("xbrl_facts")
