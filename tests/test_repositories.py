from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.config import get_settings
from app.domain import (
    AnalysisJobStatus,
    AnalysisScope,
    AppendixSection,
    Company,
    ConfidenceLabel,
    CoverageLabel,
    DataCoverageSummary,
    EvidenceGroundedReportSection,
    ExecutiveSummarySection,
    FinalInvestmentThesisSection,
    FinalReport,
    UniverseSnapshot,
    UniverseSnapshotConstituent,
    VerificationLabel,
    VerificationSummary,
)
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.repositories import AnalysisJobRepository, UniverseRepository


def configure_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "repositories.db"
    monkeypatch.setenv("APP_STORAGE__DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    get_settings.cache_clear()


def test_universe_repository_replaces_and_reads_snapshot_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repository = UniverseRepository(make_session_factory(settings))
    snapshot = UniverseSnapshot(
        snapshot_date=date(2024, 12, 31),
        constituents=[
            UniverseSnapshotConstituent(
                snapshot_date=date(2024, 12, 31),
                ticker="MSFT",
                cik="0000789019",
                company_name="Microsoft Corp.",
                exchange="NASDAQ",
                is_domestic_filer=True,
            )
        ],
    )

    repository.replace_snapshot(snapshot)

    latest_snapshot = repository.list_snapshot()
    company = repository.get_company_by_ticker("MSFT")

    assert latest_snapshot.snapshot_date == date(2024, 12, 31)
    assert len(latest_snapshot.constituents) == 1
    assert company is not None
    assert company.company_name == "Microsoft Corp."


def test_analysis_job_repository_persists_lifecycle_and_report_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_environment(monkeypatch, tmp_path)
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(settings)
    universe_repository = UniverseRepository(session_factory)
    universe_repository.replace_snapshot(
        UniverseSnapshot(
            snapshot_date=date(2024, 12, 31),
            constituents=[
                UniverseSnapshotConstituent(
                    snapshot_date=date(2024, 12, 31),
                    ticker="NVDA",
                    cik="0001045810",
                    company_name="NVIDIA Corp.",
                    exchange="NASDAQ",
                    is_domestic_filer=True,
                )
            ],
        )
    )
    repository = AnalysisJobRepository(session_factory)
    company = Company(cik="0001045810", ticker="NVDA", company_name="NVIDIA Corp.")
    report = FinalReport(
        report_id="report-1",
        company=company,
        analysis_scope=AnalysisScope(
            question="Build a thesis.",
            transcript_included=False,
            max_reretrievals=2,
        ),
        coverage_summary=DataCoverageSummary(
            coverage_label=CoverageLabel.PARTIAL,
            covered_topics=["verification"],
            missing_topics=["transcript"],
        ),
        executive_summary=ExecutiveSummarySection(
            summary="Verified disclosure summary.",
            verification_label=VerificationLabel.PARTIALLY_SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
        final_investment_thesis=FinalInvestmentThesisSection(
            thesis="Disclosures are mixed.",
            stance="valuation_free_disclosure_thesis",
            verification_label=VerificationLabel.PARTIALLY_SUPPORTED,
            confidence=ConfidenceLabel.MEDIUM,
        ),
        evidence_grounded_report=EvidenceGroundedReportSection(),
        verification_summary=VerificationSummary(total_claims=1),
        appendix=AppendixSection(),
    )

    repository.create_job(
        analysis_id="analysis-1",
        ticker="NVDA",
        cik="0001045810",
        question="Build a thesis.",
        request_payload={"ticker": "NVDA"},
    )
    repository.mark_completed(
        analysis_id="analysis-1",
        final_report=report,
        final_report_markdown="# Evidence-Grounded Report",
    )

    stored = repository.get_job("analysis-1")

    assert stored is not None
    assert stored.status == AnalysisJobStatus.COMPLETED
    assert stored.final_report is not None
    assert stored.final_report.report_id == "report-1"
    assert stored.final_report_markdown == "# Evidence-Grounded Report"
