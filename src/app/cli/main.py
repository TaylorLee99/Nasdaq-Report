"""Typer CLI entrypoint."""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from app.application import build_analysis_application_service
from app.config import get_settings
from app.data_sources import (
    FileXbrlFixtureLoader,
    RateLimiter,
    SecArchiveClient,
    SecMetadataService,
    SecRawFilingDownloadService,
    SecSubmissionsClient,
    TranscriptIngestionService,
    UrllibBinaryHttpClient,
    UrllibJsonHttpClient,
    XbrlIngestionService,
)
from app.domain import AnalysisRunRequest, ConfidenceLabel, FilingType, XbrlCanonicalFact
from app.graph import describe_graph
from app.indexing import (
    ChunkingConfig,
    ChunkSearchFilters,
    FakeEmbeddingProvider,
    RetrievalIndexingService,
)
from app.ingestion import (
    CsvUniverseConstituentLoader,
    DomesticFilerOnlyFilter,
    UniverseIngestionService,
)
from app.storage import Base, create_engine_from_settings, make_session_factory
from app.storage.raw_store import LocalRawDocumentStore
from app.storage.repositories import (
    PgVectorChunkRepository,
    SecMetadataRepository,
    SecRawDocumentRepository,
    TranscriptRepository,
    UniverseRepository,
    XbrlRepository,
)

app = typer.Typer(help="Developer CLI for the EMS research system.")
analyze_app = typer.Typer(help="Synchronous analysis commands.")
sec_app = typer.Typer(help="SEC metadata plane commands.")
transcript_app = typer.Typer(help="Transcript ingestion commands.")
universe_app = typer.Typer(help="Universe snapshot management commands.")
xbrl_app = typer.Typer(help="XBRL structured evidence commands.")
index_app = typer.Typer(help="Retrieval indexing commands.")
app.add_typer(analyze_app, name="analyze")
app.add_typer(sec_app, name="sec")
app.add_typer(transcript_app, name="transcript")
app.add_typer(universe_app, name="universe")
app.add_typer(xbrl_app, name="xbrl")
app.add_typer(index_app, name="index")

logging.basicConfig(level=logging.INFO)


def build_universe_repository() -> UniverseRepository:
    """Create a repository bound to the configured database."""

    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    return UniverseRepository(make_session_factory(settings))


def build_sec_metadata_repository() -> SecMetadataRepository:
    """Create a SEC metadata repository bound to the configured database."""

    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    return SecMetadataRepository(make_session_factory(settings))


def build_sec_raw_document_repository() -> SecRawDocumentRepository:
    """Create a SEC raw document repository bound to the configured database."""

    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    return SecRawDocumentRepository(make_session_factory(settings))


def build_transcript_repository() -> TranscriptRepository:
    """Create a transcript repository bound to the configured database."""

    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    return TranscriptRepository(make_session_factory(settings))


def build_xbrl_repository() -> XbrlRepository:
    """Create an XBRL repository bound to the configured database."""

    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    return XbrlRepository(make_session_factory(settings))


def build_vector_repository() -> PgVectorChunkRepository:
    """Create a vector index repository bound to the configured database."""

    settings = get_settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    return PgVectorChunkRepository(make_session_factory(settings))


def build_sec_service() -> SecMetadataService:
    """Create the SEC metadata service with configured rate limits and retries."""

    settings = get_settings()
    rate_limiter = RateLimiter(settings.sec.requests_per_second)
    submissions_client = SecSubmissionsClient(
        http_client=UrllibJsonHttpClient(),
        base_url=settings.sec.base_url,
        user_agent=settings.sec.user_agent,
        timeout_seconds=settings.sec.timeout_seconds,
        max_retries=settings.sec.max_retries,
        backoff_seconds=settings.sec.backoff_seconds,
        rate_limiter=rate_limiter,
    )
    return SecMetadataService(
        submissions_client=submissions_client,
        metadata_repository=build_sec_metadata_repository(),
        universe_repository=build_universe_repository(),
        save_raw_json=settings.sec.save_raw_json,
    )


def build_sec_download_service() -> SecRawFilingDownloadService:
    """Create the raw SEC filing download service."""

    settings = get_settings()
    rate_limiter = RateLimiter(settings.sec.requests_per_second)
    archive_client = SecArchiveClient(
        http_client=UrllibBinaryHttpClient(),
        user_agent=settings.sec.user_agent,
        timeout_seconds=settings.sec.timeout_seconds,
        max_retries=settings.sec.max_retries,
        backoff_seconds=settings.sec.backoff_seconds,
        rate_limiter=rate_limiter,
    )
    return SecRawFilingDownloadService(
        archive_client=archive_client,
        metadata_repository=build_sec_metadata_repository(),
        raw_document_repository=build_sec_raw_document_repository(),
        raw_document_store=LocalRawDocumentStore(settings.storage.raw_store_dir),
        skip_existing=settings.sec.skip_existing_downloads,
    )


def build_transcript_service() -> TranscriptIngestionService:
    """Create the transcript ingestion service."""

    settings = get_settings()
    return TranscriptIngestionService(
        transcript_repository=build_transcript_repository(),
        universe_repository=build_universe_repository(),
        raw_document_store=LocalRawDocumentStore(settings.storage.raw_store_dir),
        skip_existing=settings.transcript.skip_existing_imports,
    )


def build_xbrl_service() -> XbrlIngestionService:
    """Create the XBRL ingestion service."""

    return XbrlIngestionService(
        fixture_loader=FileXbrlFixtureLoader(),
        xbrl_repository=build_xbrl_repository(),
        universe_repository=build_universe_repository(),
        sec_metadata_repository=build_sec_metadata_repository(),
    )


def build_indexing_service() -> RetrievalIndexingService:
    """Create the retrieval indexing service."""

    settings = get_settings()
    return RetrievalIndexingService(
        universe_repository=build_universe_repository(),
        sec_metadata_repository=build_sec_metadata_repository(),
        sec_raw_document_repository=build_sec_raw_document_repository(),
        transcript_repository=build_transcript_repository(),
        vector_repository=build_vector_repository(),
        raw_document_store=LocalRawDocumentStore(settings.storage.raw_store_dir),
        embedding_provider=FakeEmbeddingProvider(dimensions=settings.indexing.embedding_dimensions),
        chunking_config=ChunkingConfig(
            chunk_size_chars=settings.indexing.chunk_size_chars,
            chunk_overlap_chars=settings.indexing.chunk_overlap_chars,
        ),
    )


def parse_snapshot_date(raw_value: str | None) -> date | None:
    """Parse an optional ISO snapshot date argument."""

    if raw_value is None:
        return None
    return date.fromisoformat(raw_value)


def parse_forms(raw_value: str | None) -> list[FilingType] | None:
    """Parse a comma-separated forms argument."""

    if raw_value is None:
        return None
    form_map = {
        "10-K": FilingType.FORM_10K,
        "10-Q": FilingType.FORM_10Q,
        "8-K": FilingType.FORM_8K,
    }
    forms: list[FilingType] = []
    for item in raw_value.split(","):
        normalized = item.strip().upper()
        if normalized not in form_map:
            msg = f"Unsupported form type: {item}"
            raise typer.BadParameter(msg)
        forms.append(form_map[normalized])
    return forms


def parse_confidence_label(raw_value: str) -> ConfidenceLabel:
    """Parse a transcript source reliability label."""

    return ConfidenceLabel(raw_value.strip().lower())


def parse_xbrl_fact_names(raw_value: str | None) -> list[XbrlCanonicalFact] | None:
    """Parse a comma-separated canonical XBRL fact filter."""

    if raw_value is None:
        return None
    return [
        XbrlCanonicalFact(item.strip().lower()) for item in raw_value.split(",") if item.strip()
    ]


@analyze_app.command("company")
def analyze_company(
    ticker: Annotated[str, typer.Option(...)],
    question: Annotated[str, typer.Option(...)],
    include_transcript: Annotated[bool, typer.Option()] = False,
    as_of_date: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Run a synchronous analysis job for one company."""

    service = build_analysis_application_service()
    result = service.run_analysis(
        AnalysisRunRequest(
            ticker=ticker,
            question=question,
            include_transcript=include_transcript,
            as_of_date=parse_snapshot_date(as_of_date),
        )
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))


@app.command()
def info() -> None:
    """Print minimal runtime information."""

    settings = get_settings()
    typer.echo(
        json.dumps(
            {
                "app_name": settings.name,
                "environment": settings.environment,
                "database_url": settings.database_url,
            },
            indent=2,
            default=str,
        )
    )


@app.command("graph-summary")
def graph_summary() -> None:
    """Print the registered orchestration nodes."""

    for node_name in describe_graph():
        typer.echo(node_name)


@universe_app.command("load")
def universe_load(
    file: Annotated[
        Path,
        typer.Option(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    snapshot_date: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Load a domestic-filer-filtered snapshot from a CSV fixture."""

    repository = build_universe_repository()
    service = UniverseIngestionService(
        loader=CsvUniverseConstituentLoader(),
        filter_policy=DomesticFilerOnlyFilter(),
        repository=repository,
    )
    result = service.load_snapshot(file, snapshot_date_override=parse_snapshot_date(snapshot_date))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@universe_app.command("list")
def universe_list(snapshot_date: Annotated[str | None, typer.Option()] = None) -> None:
    """List the latest stored snapshot, or a specific snapshot date."""

    repository = build_universe_repository()
    snapshot = repository.list_snapshot(parse_snapshot_date(snapshot_date))
    typer.echo(json.dumps(snapshot.model_dump(mode="json"), indent=2))


@sec_app.command("fetch-submissions")
def sec_fetch_submissions(
    ticker: Annotated[str, typer.Option(...)],
    save_raw_json: Annotated[bool | None, typer.Option()] = None,
) -> None:
    """Fetch and store SEC submissions metadata for all supported forms."""

    service = build_sec_service()
    result = service.fetch_submissions(ticker, save_raw_json=save_raw_json)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@sec_app.command("sync-filings")
def sec_sync_filings(
    ticker: Annotated[str, typer.Option(...)],
    forms: Annotated[str, typer.Option(...)] = "10-K,10-Q,8-K",
    save_raw_json: Annotated[bool | None, typer.Option()] = None,
) -> None:
    """Fetch, filter, and store SEC filing metadata for the requested forms."""

    service = build_sec_service()
    result = service.fetch_submissions(
        ticker,
        forms=parse_forms(forms),
        save_raw_json=save_raw_json,
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@sec_app.command("download-filings")
def sec_download_filings(
    ticker: Annotated[str, typer.Option(...)],
    forms: Annotated[str, typer.Option(...)] = "10-K,10-Q,8-K",
) -> None:
    """Download raw filing documents for stored SEC metadata rows."""

    service = build_sec_download_service()
    result = service.download_filings(ticker=ticker, forms=parse_forms(forms))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))


@transcript_app.command("import")
def transcript_import(
    ticker: Annotated[str, typer.Option(...)],
    file: Annotated[
        Path,
        typer.Option(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    call_date: Annotated[str | None, typer.Option()] = None,
    fiscal_quarter: Annotated[str | None, typer.Option()] = None,
    speaker_separated: Annotated[bool, typer.Option()] = False,
    source_url: Annotated[str | None, typer.Option()] = None,
    source_reliability: Annotated[str, typer.Option()] = "high",
) -> None:
    """Import a transcript file into metadata and raw storage."""

    service = build_transcript_service()
    effective_call_date = parse_snapshot_date(call_date) or date.today()
    result = service.import_transcript(
        ticker=ticker,
        file_path=file,
        call_date=effective_call_date,
        fiscal_quarter=fiscal_quarter,
        source_url=source_url,
        speaker_separated=speaker_separated,
        source_reliability=parse_confidence_label(source_reliability),
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))


@transcript_app.command("list")
def transcript_list(ticker: Annotated[str, typer.Option(...)]) -> None:
    """List transcript metadata and availability gaps for a ticker."""

    service = build_transcript_service()
    result = service.list_transcripts(ticker=ticker)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))


@xbrl_app.command("import-fixture")
def xbrl_import_fixture(
    ticker: Annotated[str, typer.Option(...)],
    file: Annotated[
        Path,
        typer.Option(..., exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
) -> None:
    """Import canonical XBRL facts from a fixture file."""

    service = build_xbrl_service()
    facts = service.import_fixture(ticker=ticker, file_path=file)
    typer.echo(json.dumps([fact.model_dump(mode="json") for fact in facts], indent=2, default=str))


@xbrl_app.command("facts")
def xbrl_facts(
    ticker: Annotated[str, typer.Option(...)],
    facts: Annotated[str | None, typer.Option()] = None,
    forms: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Query stored canonical XBRL facts for a ticker."""

    service = build_xbrl_service()
    result = service.query_facts(
        ticker=ticker,
        canonical_facts=parse_xbrl_fact_names(facts),
        filing_types=parse_forms(forms),
    )
    typer.echo(json.dumps([fact.model_dump(mode="json") for fact in result], indent=2, default=str))


@index_app.command("filing")
def index_filing(
    ticker: Annotated[str, typer.Option(...)],
    forms: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Parse and index downloaded filing documents for a ticker."""

    service = build_indexing_service()
    result = service.index_filings(ticker=ticker, forms=parse_forms(forms))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))


@index_app.command("transcript")
def index_transcript(ticker: Annotated[str, typer.Option(...)]) -> None:
    """Parse and index stored transcript documents for a ticker."""

    service = build_indexing_service()
    result = service.index_transcripts(ticker=ticker)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))


@index_app.command("build")
def index_build(
    ticker: Annotated[str, typer.Option(...)],
    forms: Annotated[str | None, typer.Option()] = None,
    include_transcript: Annotated[bool, typer.Option()] = False,
) -> None:
    """Build the retrieval index for filings and, optionally, transcripts."""

    service = build_indexing_service()
    filing_result = service.index_filings(ticker=ticker, forms=parse_forms(forms))
    transcript_result = service.index_transcripts(ticker=ticker) if include_transcript else None
    typer.echo(
        json.dumps(
            {
                "filings": filing_result.model_dump(mode="json"),
                "transcripts": (
                    transcript_result.model_dump(mode="json")
                    if transcript_result is not None
                    else None
                ),
            },
            indent=2,
            default=str,
        )
    )


@index_app.command("inspect-8k")
def index_inspect_8k(
    ticker: Annotated[str, typer.Option(...)],
    item_number: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 20,
    latest_only: Annotated[bool, typer.Option()] = True,
) -> None:
    """Inspect indexed 8-K chunks, optionally narrowed to the latest filing and one item."""

    repository = build_vector_repository()
    chunks = repository.list_chunks(
        filters=ChunkSearchFilters(
            ticker=ticker,
            filing_types=[FilingType.FORM_8K],
            item_number=item_number,
        ),
        limit=None,
    )
    if latest_only and chunks:
        latest_date = max(
            (chunk.filing_date for chunk in chunks if chunk.filing_date is not None),
            default=None,
        )
        if latest_date is not None:
            latest_accessions = {
                chunk.accession_number
                for chunk in chunks
                if chunk.filing_date == latest_date and chunk.accession_number is not None
            }
            if latest_accessions:
                chunks = [chunk for chunk in chunks if chunk.accession_number in latest_accessions]
            else:
                chunks = [chunk for chunk in chunks if chunk.filing_date == latest_date]
    chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.item_number or "", chunk.document_id, chunk.chunk_index),
    )
    item_counts: dict[str, int] = {}
    for chunk in chunks:
        key = chunk.item_number or "unlabeled"
        item_counts[key] = item_counts.get(key, 0) + 1
    payload = {
        "ticker": ticker.upper(),
        "latest_only": latest_only,
        "item_number": item_number,
        "chunk_count": len(chunks),
        "item_counts": item_counts,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "accession_number": chunk.accession_number,
                "filing_date": chunk.filing_date,
                "item_number": chunk.item_number,
                "section_name": chunk.section_name,
                "parse_confidence": chunk.parse_confidence.value,
                "excerpt": chunk.text[:240],
            }
            for chunk in chunks[:limit]
        ],
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    app()
