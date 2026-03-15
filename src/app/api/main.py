"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from app.application import AnalysisApplicationService, build_analysis_application_service
from app.config import get_settings
from app.domain import AnalysisJobRecord, AnalysisRunRequest, HealthStatus

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(
    title=settings.name,
    version="0.1.0",
    description=(
        "Synchronous analysis API for evidence-grounded equity research. "
        "Run one company analysis and fetch persisted results."
    ),
)


def get_analysis_service() -> AnalysisApplicationService:
    """Build the shared application service for analysis endpoints."""

    return build_analysis_application_service()


@app.get("/health", response_model=HealthStatus, tags=["system"])
def health_check() -> HealthStatus:
    """Return a simple health payload."""

    current_settings = get_settings()
    return HealthStatus(app_name=current_settings.name, environment=current_settings.environment)


@app.post(
    "/analysis/run",
    response_model=AnalysisJobRecord,
    tags=["analysis"],
    status_code=status.HTTP_200_OK,
)
def run_analysis(
    request: AnalysisRunRequest,
    service: Annotated[AnalysisApplicationService, Depends(get_analysis_service)],
) -> AnalysisJobRecord:
    """Run one company analysis synchronously and return the persisted job record."""

    try:
        return service.run_analysis(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API boundary
        logger.exception("Synchronous analysis execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis execution failed.",
        ) from exc


@app.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisJobRecord,
    tags=["analysis"],
)
def get_analysis(
    analysis_id: str,
    service: Annotated[AnalysisApplicationService, Depends(get_analysis_service)],
) -> AnalysisJobRecord:
    """Return one persisted analysis job."""

    job = service.get_analysis(analysis_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job {analysis_id} was not found.",
        )
    return job
