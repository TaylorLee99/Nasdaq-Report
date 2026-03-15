"""SQLAlchemy database setup."""

from collections.abc import Callable

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    """Base class for ORM models."""


def create_engine_from_settings(settings: Settings | None = None) -> Engine:
    """Create an engine from the configured database URL."""

    current_settings = settings or get_settings()
    return create_engine(current_settings.database_url, future=True)


def make_session_factory(settings: Settings | None = None) -> Callable[[], Session]:
    """Create a session factory bound to the configured engine."""

    engine = create_engine_from_settings(settings)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
