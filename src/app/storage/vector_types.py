"""Database types for vector storage."""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator, TypeEngine

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - optional dependency at runtime
    Vector = None


class PgVectorCompatible(TypeDecorator[list[float]]):
    """Use pgvector on PostgreSQL and JSON arrays everywhere else."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self._dimensions = dimensions

    def load_dialect_impl(self, dialect: Any) -> TypeEngine[Any]:
        if dialect.name == "postgresql" and Vector is not None:
            return dialect.type_descriptor(Vector(self._dimensions))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: list[float] | None, dialect: Any) -> list[float] | None:
        del dialect
        if value is None:
            return None
        return list(value)

    def process_result_value(self, value: Any, dialect: Any) -> list[float] | None:
        del dialect
        if value is None:
            return None
        return list(value)
