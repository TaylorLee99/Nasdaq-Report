# AGENTS.md

This repository implements a filing-form-aware agentic RAG system for evidence-grounded equity research. The repo is currently minimal; use the layout and rules below as the canonical target state for all future work.

## 1) Repo layout

Use this structure unless there is a strong reason to change it:

```text
src/ems/
  agents/           # agent logic; outputs only structured evidence packets
  orchestration/    # LangGraph graphs, nodes, routing, bounded retry/reretrieval
  etl/              # plain Python ingestion, parsing, chunking, normalization jobs
  retrieval/        # retrievers, rankers, filing-form-aware query logic
  verifier/         # confidence-calibrated checking, not formal proof
  reporting/        # thesis report assembly from verified evidence packets
  schemas/          # pydantic v2 models shared across modules
  domain/           # domain entities, enums, business rules
  db/               # SQLAlchemy 2.x models, session management, repositories
  cli/              # runnable entrypoints

tests/
  unit/
  integration/
  e2e/

scripts/            # local utilities and one-off maintenance scripts
configs/            # yaml/toml runtime config
data/
  raw/              # SEC filings and optional transcripts only
  processed/
  indexes/
```

Keep imports under `src/`. Do not mix orchestration code into `etl/`.

## 2) Dev environment and run

- Python `3.11+` only.
- Create env: `python3.11 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -U pip && pip install -e ".[dev]"`
- Local app config should come from environment variables or files under `configs/`; never hardcode secrets.
- Run ETL jobs as plain Python modules, for example: `python -m ems.etl.<job_name>`
- Run orchestration separately, for example: `python -m ems.cli.run_graph`

## 3) Build / test / lint commands

These commands should stay valid as the repo evolves:

- Build/package: `python -m build`
- Run tests: `pytest`
- Fast tests: `pytest tests/unit -q`
- Integration tests: `pytest tests/integration`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type-check: `mypy src`

If a command is introduced or changed, update this file in the same change.

## 4) Architecture constraints

- Allowed inputs: `10-K`, `10-Q`, `8-K`, and optional earnings call transcripts only.
- Universe: Nasdaq-100 snapshot, domestic filers only.
- Forbidden sources: news, blogs, social media, analyst reports, and any other non-filing evidence.
- LangGraph is orchestration only. ETL, parsing, chunking, indexing, and persistence stay in normal Python services.
- Final output is a valuation-free, disclosure-grounded thesis report.
- Every agent output must be a structured evidence packet with source metadata, claim, support span(s), and confidence.
- Verifier is a confidence-calibrated checker; do not represent it as formal proof or certainty.
- Reretrieval must be bounded. No open-ended reflection/retry loops.
- Transcript handling is optional and must not block filing-only flows.
- Prefer deterministic, auditable transformations over opaque agent side effects.

## 5) Coding conventions

- Use type hints everywhere in production code.
- Use `pydantic v2` for external I/O schemas and agent/evidence packet contracts.
- Use `SQLAlchemy 2.x` typed ORM/Core patterns only.
- Keep functions small and explicit; push side effects to edges.
- Prefer dataclass-like domain models or pydantic models over untyped dict passing.
- Log with enough context to trace ticker, filing form, accession/date, and run id.
- Keep modules single-purpose; avoid cross-layer imports that skip boundaries.
- Write tests for business logic and schema validation before wiring complex graphs.

## 6) Done criteria

A task is done only if:

- code follows the layer boundaries above;
- prohibited data sources are not introduced;
- outputs remain structured evidence packets or artifacts derived from them;
- tests for new behavior are added or updated;
- `ruff check .`, `mypy src`, and `pytest` pass locally for touched areas;
- docs or config references are updated when commands, schemas, or flows change.

## 7) Do-not rules

- Do not ingest or cite news, blogs, social posts, analyst notes, or market data for thesis generation.
- Do not put LangGraph into ETL or parsing paths.
- Do not emit free-form agent text as an intermediate contract when a schema should exist.
- Do not add unbounded retry, rerank, or reretrieval loops.
- Do not make transcript availability mandatory.
- Do not introduce valuation targets, price targets, or recommendation labels into the final report.
- Do not bypass type checking with pervasive `Any`, raw dict payloads, or stringly-typed state.
