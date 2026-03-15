"""Create retrieval indexing chunk table."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - optional dependency at runtime
    Vector = None

revision = "20260314_0006"
down_revision = "20260314_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    embedding_type: sa.TypeEngine[object] = sa.JSON()
    if bind.dialect.name == "postgresql" and Vector is not None:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        embedding_type = Vector(12)

    op.create_table(
        "indexed_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("chunk_id", sa.String(length=128), nullable=False),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("section_id", sa.String(length=128), nullable=False),
        sa.Column("cik", sa.String(length=32), sa.ForeignKey("company_master.cik"), nullable=False),
        sa.Column("ticker", sa.String(length=32), nullable=False),
        sa.Column("filing_type", sa.String(length=16), nullable=False),
        sa.Column("accession_number", sa.String(length=32), nullable=True),
        sa.Column("transcript_id", sa.String(length=128), nullable=True),
        sa.Column("filing_date", sa.Date(), nullable=True),
        sa.Column("report_period", sa.Date(), nullable=True),
        sa.Column("call_date", sa.Date(), nullable=True),
        sa.Column("fiscal_quarter", sa.String(length=32), nullable=True),
        sa.Column("section_name", sa.String(length=255), nullable=False),
        sa.Column("section_type", sa.String(length=64), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("item_number", sa.String(length=32), nullable=True),
        sa.Column("speaker", sa.String(length=255), nullable=True),
        sa.Column("transcript_segment_type", sa.String(length=32), nullable=True),
        sa.Column("parse_confidence", sa.String(length=16), nullable=False),
        sa.Column("used_fallback", sa.Boolean(), nullable=False),
        sa.Column("source_reliability", sa.String(length=16), nullable=True),
        sa.Column("source_url", sa.String(length=1024), nullable=True),
        sa.Column("embedding", embedding_type, nullable=False),
        sa.Column("embedding_key", sa.String(length=128), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("chunk_id", name="uq_indexed_chunks"),
    )
    op.create_index(
        "ix_indexed_chunks_document_id", "indexed_chunks", ["document_id"], unique=False
    )
    op.create_index("ix_indexed_chunks_section_id", "indexed_chunks", ["section_id"], unique=False)
    op.create_index("ix_indexed_chunks_cik", "indexed_chunks", ["cik"], unique=False)
    op.create_index("ix_indexed_chunks_ticker", "indexed_chunks", ["ticker"], unique=False)
    op.create_index(
        "ix_indexed_chunks_filing_type", "indexed_chunks", ["filing_type"], unique=False
    )
    op.create_index(
        "ix_indexed_chunks_accession_number",
        "indexed_chunks",
        ["accession_number"],
        unique=False,
    )
    op.create_index(
        "ix_indexed_chunks_transcript_id",
        "indexed_chunks",
        ["transcript_id"],
        unique=False,
    )
    op.create_index(
        "ix_indexed_chunks_filing_date", "indexed_chunks", ["filing_date"], unique=False
    )
    op.create_index("ix_indexed_chunks_call_date", "indexed_chunks", ["call_date"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_indexed_chunks_call_date", table_name="indexed_chunks")
    op.drop_index("ix_indexed_chunks_filing_date", table_name="indexed_chunks")
    op.drop_index("ix_indexed_chunks_transcript_id", table_name="indexed_chunks")
    op.drop_index("ix_indexed_chunks_accession_number", table_name="indexed_chunks")
    op.drop_index("ix_indexed_chunks_filing_type", table_name="indexed_chunks")
    op.drop_index("ix_indexed_chunks_ticker", table_name="indexed_chunks")
    op.drop_index("ix_indexed_chunks_cik", table_name="indexed_chunks")
    op.drop_index("ix_indexed_chunks_section_id", table_name="indexed_chunks")
    op.drop_index("ix_indexed_chunks_document_id", table_name="indexed_chunks")
    op.drop_table("indexed_chunks")
