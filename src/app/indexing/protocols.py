"""Protocols for retrieval backends."""

from __future__ import annotations

from typing import Protocol

from app.domain import ChunkRecord
from app.indexing.models import ChunkSearchFilters, ChunkSearchResult


class VectorIndexRepository(Protocol):
    """Persistence and retrieval contract for chunk vectors."""

    def replace_document_chunks(self, *, document_id: str, chunks: list[ChunkRecord]) -> None:
        """Replace stored chunks for a single source document."""

    def list_chunks(
        self,
        *,
        filters: ChunkSearchFilters | None = None,
        limit: int | None = None,
    ) -> list[ChunkRecord]:
        """List stored chunks using metadata filters only."""

    def search(
        self,
        *,
        query_embedding: list[float],
        filters: ChunkSearchFilters | None = None,
        limit: int = 5,
    ) -> list[ChunkSearchResult]:
        """Search stored chunks by vector similarity and metadata filters."""
