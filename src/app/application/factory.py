"""Factory helpers for application services."""

from __future__ import annotations

from app.application.service import AnalysisApplicationService
from app.config import Settings, get_settings
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.repositories import (
    AnalysisJobRepository,
    SecMetadataRepository,
    TranscriptRepository,
    UniverseRepository,
)


def build_analysis_application_service(
    settings: Settings | None = None,
) -> AnalysisApplicationService:
    """Build the synchronous analysis service with configured repositories."""

    current_settings = settings or get_settings()
    engine = create_engine_from_settings(current_settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(current_settings)
    return AnalysisApplicationService(
        universe_repository=UniverseRepository(session_factory),
        sec_metadata_repository=SecMetadataRepository(session_factory),
        transcript_repository=TranscriptRepository(session_factory),
        analysis_job_repository=AnalysisJobRepository(session_factory),
    )
