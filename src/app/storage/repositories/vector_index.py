"""Vector index repository for retrieval-ready chunks."""

from __future__ import annotations

import math
from collections.abc import Callable

from sqlalchemy import Select, delete, select
from sqlalchemy.orm import Session

from app.domain import (
    ChunkRecord,
    Company,
    ConfidenceLabel,
    DocumentSectionType,
    FilingType,
    TranscriptSegmentType,
)
from app.indexing.models import ChunkSearchFilters, ChunkSearchResult
from app.storage.models import CompanyMasterModel, IndexedChunkModel


class PgVectorChunkRepository:
    """Persist and retrieve chunk vectors with pgvector-compatible storage."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def replace_document_chunks(self, *, document_id: str, chunks: list[ChunkRecord]) -> None:
        """Replace all indexed chunks for a source document."""

        with self._session_factory() as session:
            session.execute(
                delete(IndexedChunkModel).where(IndexedChunkModel.document_id == document_id)
            )
            for chunk in chunks:
                session.add(self._to_model(chunk))
            session.commit()

    def list_chunks(
        self,
        *,
        filters: ChunkSearchFilters | None = None,
        limit: int | None = None,
    ) -> list[ChunkRecord]:
        """List stored chunks using metadata filters only."""

        with self._session_factory() as session:
            statement = self._apply_filters(select(IndexedChunkModel), filters)
            statement = statement.order_by(
                IndexedChunkModel.document_id.asc(),
                IndexedChunkModel.chunk_index.asc(),
            )
            if limit is not None:
                statement = statement.limit(limit)
            rows = session.scalars(statement).all()
            return [self._to_domain(session, row) for row in rows]

    def search(
        self,
        *,
        query_embedding: list[float],
        filters: ChunkSearchFilters | None = None,
        limit: int = 5,
    ) -> list[ChunkSearchResult]:
        """Search stored chunks using metadata filters and cosine similarity."""

        with self._session_factory() as session:
            statement = self._apply_filters(select(IndexedChunkModel), filters)
            rows = session.scalars(statement).all()
            scored = [
                ChunkSearchResult(
                    chunk=self._to_domain(session, row),
                    score=self._cosine_similarity(query_embedding, row.embedding),
                )
                for row in rows
            ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    @staticmethod
    def _apply_filters(
        statement: Select[tuple[IndexedChunkModel]],
        filters: ChunkSearchFilters | None,
    ) -> Select[tuple[IndexedChunkModel]]:
        if filters is None:
            return statement
        if filters.ticker is not None:
            statement = statement.where(IndexedChunkModel.ticker == filters.ticker.upper())
        if filters.filing_types:
            statement = statement.where(
                IndexedChunkModel.filing_type.in_(
                    [filing_type.value for filing_type in filters.filing_types]
                )
            )
        if filters.accession_number is not None:
            statement = statement.where(
                IndexedChunkModel.accession_number == filters.accession_number
            )
        if filters.transcript_id is not None:
            statement = statement.where(IndexedChunkModel.transcript_id == filters.transcript_id)
        if filters.filing_date_on_or_after is not None:
            statement = statement.where(
                IndexedChunkModel.filing_date >= filters.filing_date_on_or_after
            )
        if filters.report_period_on_or_after is not None:
            statement = statement.where(
                IndexedChunkModel.report_period >= filters.report_period_on_or_after
            )
        if filters.section_name is not None:
            statement = statement.where(IndexedChunkModel.section_name == filters.section_name)
        if filters.item_number is not None:
            statement = statement.where(IndexedChunkModel.item_number == filters.item_number)
        if filters.speaker is not None:
            statement = statement.where(IndexedChunkModel.speaker == filters.speaker)
        if filters.transcript_segment_type is not None:
            statement = statement.where(
                IndexedChunkModel.transcript_segment_type == filters.transcript_segment_type.value
            )
        if filters.parse_confidence is not None:
            statement = statement.where(
                IndexedChunkModel.parse_confidence == filters.parse_confidence.value
            )
        if filters.source_reliability is not None:
            statement = statement.where(
                IndexedChunkModel.source_reliability == filters.source_reliability.value
            )
        return statement

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return -1.0
        numerator = sum(
            left_value * right_value for left_value, right_value in zip(left, right, strict=True)
        )
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return -1.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _to_model(chunk: ChunkRecord) -> IndexedChunkModel:
        return IndexedChunkModel(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            cik=chunk.company.cik,
            ticker=chunk.company.ticker,
            filing_type=chunk.filing_type.value,
            accession_number=chunk.accession_number,
            transcript_id=chunk.transcript_id,
            filing_date=chunk.filing_date,
            report_period=chunk.report_period,
            call_date=chunk.call_date,
            fiscal_quarter=chunk.fiscal_quarter,
            section_name=chunk.section_name,
            section_type=chunk.section_type.value if chunk.section_type is not None else None,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            token_count=chunk.token_count,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            item_number=chunk.item_number,
            speaker=chunk.speaker,
            transcript_segment_type=(
                chunk.transcript_segment_type.value
                if chunk.transcript_segment_type is not None
                else None
            ),
            parse_confidence=chunk.parse_confidence.value,
            used_fallback=chunk.used_fallback,
            source_reliability=(
                chunk.source_reliability.value if chunk.source_reliability is not None else None
            ),
            source_url=chunk.source_url,
            embedding=list(chunk.embedding),
            embedding_key=chunk.embedding_key,
        )

    @staticmethod
    def _to_domain(session: Session, row: IndexedChunkModel) -> ChunkRecord:
        company = session.get(CompanyMasterModel, row.cik)
        if company is None:
            company_domain = Company(cik=row.cik, ticker=row.ticker, company_name=row.ticker)
        else:
            company_domain = Company(
                cik=company.cik,
                ticker=company.ticker,
                company_name=company.company_name,
                exchange=company.exchange,
                is_domestic_filer=company.is_domestic_filer,
                latest_snapshot_date=company.latest_snapshot_date,
            )
        return ChunkRecord(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            section_id=row.section_id,
            company=company_domain,
            filing_type=FilingType(row.filing_type),
            accession_number=row.accession_number,
            transcript_id=row.transcript_id,
            filing_date=row.filing_date,
            report_period=row.report_period,
            call_date=row.call_date,
            fiscal_quarter=row.fiscal_quarter,
            section_name=row.section_name,
            section_type=(DocumentSectionType(row.section_type) if row.section_type else None),
            chunk_index=row.chunk_index,
            text=row.text,
            token_count=row.token_count,
            char_start=row.char_start,
            char_end=row.char_end,
            item_number=row.item_number,
            speaker=row.speaker,
            transcript_segment_type=(
                TranscriptSegmentType(row.transcript_segment_type)
                if row.transcript_segment_type
                else None
            ),
            parse_confidence=ConfidenceLabel(row.parse_confidence),
            used_fallback=row.used_fallback,
            source_reliability=(
                ConfidenceLabel(row.source_reliability) if row.source_reliability else None
            ),
            source_url=row.source_url,
            embedding=list(row.embedding),
            embedding_key=row.embedding_key,
        )
