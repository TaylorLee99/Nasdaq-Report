"""Retrieval indexing services."""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from pathlib import Path

from app.domain import (
    ChunkRecord,
    Company,
    ConfidenceLabel,
    DownloadStatusLabel,
    FilingMetadata,
    FilingType,
    ParsedSection,
    RawDocument,
)
from app.indexing.chunking import ChunkingConfig, SectionChunker
from app.indexing.embeddings import EmbeddingProvider
from app.indexing.models import ChunkSearchFilters, ChunkSearchResult, IndexingRunResult
from app.indexing.protocols import VectorIndexRepository
from app.parsing import parse_document
from app.storage.raw_store import RawDocumentStore
from app.storage.repositories import (
    SecMetadataRepository,
    SecRawDocumentRepository,
    TranscriptRepository,
    UniverseRepository,
)


class RetrievalIndexingService:
    """Build retrieval-ready chunks from stored filings and transcripts."""

    def __init__(
        self,
        *,
        universe_repository: UniverseRepository,
        sec_metadata_repository: SecMetadataRepository,
        sec_raw_document_repository: SecRawDocumentRepository,
        transcript_repository: TranscriptRepository,
        vector_repository: VectorIndexRepository,
        raw_document_store: RawDocumentStore,
        embedding_provider: EmbeddingProvider,
        chunking_config: ChunkingConfig,
    ) -> None:
        self._universe_repository = universe_repository
        self._sec_metadata_repository = sec_metadata_repository
        self._sec_raw_document_repository = sec_raw_document_repository
        self._transcript_repository = transcript_repository
        self._vector_repository = vector_repository
        self._raw_document_store = raw_document_store
        self._embedding_provider = embedding_provider
        self._chunker = SectionChunker(chunking_config)

    def index_filings(
        self,
        *,
        ticker: str,
        forms: list[FilingType] | None = None,
    ) -> IndexingRunResult:
        """Parse and index downloaded SEC filings for a ticker."""

        company = self._require_company(ticker)
        records = self._sec_raw_document_repository.query_download_records(
            ticker=company.ticker,
            forms=forms,
            status=DownloadStatusLabel.DOWNLOADED,
        )
        indexed_documents = 0
        indexed_chunks = 0
        for record in records:
            metadata = self._sec_metadata_repository.get_filing_by_accession(
                record.accession_number
            )
            if metadata is None or record.storage_path is None:
                continue
            raw_content = self._read_text(record.storage_path)
            document = RawDocument(
                document_id=record.accession_number,
                company=company,
                document_type=record.form_type,
                content=self._normalize_filing_text(raw_content),
                filing_metadata=FilingMetadata(
                    filing_id=record.accession_number,
                    company=company,
                    filing_type=record.form_type,
                    accession_number=record.accession_number,
                    filed_at=metadata.filing_date,
                    period_end_date=metadata.report_period,
                    source_uri=metadata.source_url,
                ),
                source_uri=metadata.source_url,
            )
            chunks = self._build_chunks(document)
            self._vector_repository.replace_document_chunks(
                document_id=document.document_id,
                chunks=chunks,
            )
            indexed_documents += 1
            indexed_chunks += len(chunks)
        return IndexingRunResult(
            ticker=company.ticker,
            source_label="filing",
            attempted_documents=len(records),
            indexed_documents=indexed_documents,
            indexed_chunks=indexed_chunks,
            skipped_documents=len(records) - indexed_documents,
            completed_at=datetime.now(UTC),
        )

    def index_transcripts(self, *, ticker: str) -> IndexingRunResult:
        """Parse and index stored earnings call transcripts for a ticker."""

        company = self._require_company(ticker)
        transcripts = self._transcript_repository.list_transcripts(ticker=company.ticker)
        indexed_documents = 0
        indexed_chunks = 0
        for metadata in transcripts:
            if metadata.storage_path is None:
                continue
            raw_content = self._read_text(metadata.storage_path)
            document = RawDocument(
                document_id=metadata.transcript_id,
                company=company,
                document_type=FilingType.EARNINGS_CALL,
                content=raw_content,
                transcript_metadata=metadata,
                source_uri=metadata.source_url,
            )
            chunks = self._build_chunks(document)
            self._vector_repository.replace_document_chunks(
                document_id=document.document_id,
                chunks=chunks,
            )
            indexed_documents += 1
            indexed_chunks += len(chunks)
        return IndexingRunResult(
            ticker=company.ticker,
            source_label="transcript",
            attempted_documents=len(transcripts),
            indexed_documents=indexed_documents,
            indexed_chunks=indexed_chunks,
            skipped_documents=len(transcripts) - indexed_documents,
            completed_at=datetime.now(UTC),
        )

    def search(
        self,
        *,
        query: str,
        filters: ChunkSearchFilters | None = None,
        limit: int = 5,
    ) -> list[ChunkSearchResult]:
        """Search indexed chunks using the configured embedding provider."""

        query_embedding = self._embedding_provider.embed_texts([query])[0]
        return self._vector_repository.search(
            query_embedding=query_embedding,
            filters=filters,
            limit=limit,
        )

    def _build_chunks(self, document: RawDocument) -> list[ChunkRecord]:
        sections = parse_document(document)
        chunk_inputs: list[ChunkRecord] = []
        for section in sections:
            spans = self._chunker.chunk_section(section)
            for span in spans:
                chunk_inputs.append(
                    ChunkRecord(
                        chunk_id=(
                            f"{document.document_id}:{section.section_id}:chunk:{span.chunk_index}"
                        ),
                        document_id=document.document_id,
                        section_id=section.section_id,
                        company=document.company,
                        filing_type=document.document_type,
                        accession_number=(
                            document.filing_metadata.accession_number
                            if document.filing_metadata is not None
                            else None
                        ),
                        transcript_id=(
                            document.transcript_metadata.transcript_id
                            if document.transcript_metadata is not None
                            else None
                        ),
                        filing_date=(
                            document.filing_metadata.filed_at
                            if document.filing_metadata is not None
                            else None
                        ),
                        report_period=(
                            document.filing_metadata.period_end_date
                            if document.filing_metadata is not None
                            else None
                        ),
                        call_date=(
                            document.transcript_metadata.call_date
                            if document.transcript_metadata is not None
                            else None
                        ),
                        fiscal_quarter=(
                            document.transcript_metadata.fiscal_quarter
                            if document.transcript_metadata is not None
                            else None
                        ),
                        section_name=section.heading,
                        section_type=section.section_type,
                        chunk_index=span.chunk_index,
                        text=span.text,
                        token_count=len(span.text.split()),
                        char_start=span.char_start,
                        char_end=span.char_end,
                        item_number=section.item_number,
                        speaker=section.speaker,
                        transcript_segment_type=section.transcript_segment_type,
                        parse_confidence=section.parse_confidence,
                        used_fallback=section.used_fallback,
                        source_reliability=self._source_reliability(document),
                        source_url=self._source_url(document),
                        embedding_key=self._embedding_provider.model_name,
                    )
                )
        if not chunk_inputs:
            return []
        embeddings = self._embedding_provider.embed_texts([chunk.text for chunk in chunk_inputs])
        return [
            chunk.model_copy(update={"embedding": embedding})
            for chunk, embedding in zip(chunk_inputs, embeddings, strict=True)
        ]

    def _require_company(self, ticker: str) -> Company:
        company = self._universe_repository.get_company_by_ticker(ticker)
        if company is None:
            msg = f"Ticker {ticker.upper()} was not found in company master"
            raise ValueError(msg)
        return company

    def _read_text(self, storage_path: str) -> str:
        path = Path(storage_path)
        if not path.is_absolute():
            path = self._raw_document_store.resolve(storage_path)
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _normalize_filing_text(raw_content: str) -> str:
        if "<html" not in raw_content.lower() and "<body" not in raw_content.lower():
            return raw_content
        text = re.sub(r"<[^>]+>", "\n", raw_content)
        text = html.unescape(text)
        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def _source_reliability(document: RawDocument) -> ConfidenceLabel | None:
        if document.transcript_metadata is not None:
            return document.transcript_metadata.source_reliability
        if document.filing_metadata is not None:
            return ConfidenceLabel.HIGH
        return None

    @staticmethod
    def _source_url(document: RawDocument) -> str | None:
        if document.transcript_metadata is not None:
            return document.transcript_metadata.source_url
        if document.filing_metadata is not None:
            return document.filing_metadata.source_uri
        return document.source_uri


def build_chunk_records(sections: list[ParsedSection]) -> list[ChunkRecord]:
    """Backward-compatible helper retained for older callers."""

    chunks: list[ChunkRecord] = []
    for index, section in enumerate(sections, start=1):
        chunks.append(
            ChunkRecord(
                chunk_id=f"{section.document_id}:chunk:{index}",
                document_id=section.document_id,
                section_id=section.section_id,
                company=section.company,
                filing_type=section.filing_type,
                section_name=section.heading,
                section_type=section.section_type,
                text=section.text,
                token_count=len(section.text.split()),
                char_start=section.char_start,
                char_end=section.char_end,
                item_number=section.item_number,
                speaker=section.speaker,
                transcript_segment_type=section.transcript_segment_type,
                parse_confidence=section.parse_confidence,
                used_fallback=section.used_fallback,
            )
        )
    return chunks
