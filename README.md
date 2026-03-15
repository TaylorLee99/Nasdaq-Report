# Nasdaq Report


Minimal Python monorepo skeleton for a filing-form-aware agentic RAG system.

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

Run the API:

```bash
uv run uvicorn app.api.main:app --reload
```

Run the CLI:

```bash
uv run app info
uv run app graph-summary
uv run app universe load --file tests/fixtures/universe_snapshot.csv
uv run app universe list
uv run app sec fetch-submissions --ticker NVDA
uv run app sec sync-filings --ticker NVDA --forms 10-K,10-Q,8-K
uv run app sec download-filings --ticker NVDA --forms 10-K,10-Q
uv run app index filing --ticker NVDA
uv run app analyze company --ticker NVDA --question "What changed recently?"
```

Run the full NVDA filing-only pipeline in one command:

```bash
bash scripts/run_nvda_pipeline.sh
```

Override ticker or question when needed:

```bash
bash scripts/run_nvda_pipeline.sh NVDA "Build a valuation-free disclosure-grounded thesis using 10-K, 10-Q, and 8-K only."
```

Quality checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy src tests
```

## Test Strategy

- The test suite is offline-first. Most tests run without external network access and rely on local fixtures plus fake components.
- Fake components are the default for speed and determinism:
  - fake retriever / fake LLM logic in specialized-agent tests
  - fake embedding provider in indexing and retriever tests
  - mocked HTTP clients for SEC metadata/download tests
- Core pytest coverage is split by layer:
  - parser unit tests: `tests/test_parsing.py`
  - repository tests: `tests/test_repositories.py`
  - retriever/vector repository tests: `tests/test_retriever.py`
  - planner/router tests: `tests/test_graph.py`
  - specialized agent loop tests: `tests/test_specialized_agents.py`
  - verifier tests: `tests/test_verification.py`
  - synthetic report tests: `tests/test_reporting.py`
  - end-to-end smoke tests: `tests/test_e2e_smoke.py`, `tests/test_api.py`, `tests/test_cli.py`
- Canonical local fixtures live in `tests/fixtures`:
  - `sample_10k.txt`
  - `sample_10q.txt`
  - `sample_8k.txt`
  - `sample_transcript.txt`
  - `sample_xbrl_facts.json`
- Additional source-specific fixtures remain for targeted tests, for example SEC submissions JSON and raw NVDA HTML.
- Recommended local loop:

```bash
uv run pytest -q
uv run pytest tests/test_e2e_smoke.py -q
uv run ruff check .
uv run mypy src tests
```

## Layout

- `src/app`: application packages by layer
- `tests`: unit and integration tests
- `alembic`: future database migrations
- `scripts`: developer utilities
- `data`: local development data

## Universe MVP

- Load a snapshot CSV into the company master and universe snapshot tables with `uv run app universe load --file <path>`
- List the latest domestic-filer snapshot with `uv run app universe list`
- Apply migrations with `uv run alembic upgrade head`
- Index retrieval-ready chunks from downloaded filings with `uv run app index filing --ticker <ticker>`
- Index optional transcript chunks with `uv run app index transcript --ticker <ticker>`

## Live Gemini reasoning

- The filing agents default to the offline stub model.
- To enable live Gemini 2.5 Flash reasoning for `10-K`, `10-Q`, and `8-K` agents, set these in `.env`:

```bash
APP_LLM__ENABLED=true
APP_LLM__PROVIDER=gemini
APP_LLM__MODEL=gemini-2.5-flash
APP_LLM__API_KEY=your_key
```

- The test suite stays offline even if live Gemini is enabled in `.env`.

## PDF export

- For higher-quality PDF export, use the pandoc wrapper:

```bash
python3.11 scripts/render_report_pdf.py \
  output/nvda_latest_report.md \
  --output output/nvda_latest_report.pdf
```

- Requirements:
  - `pandoc`
  - one PDF engine: `weasyprint`, `wkhtmltopdf`, `xelatex`, `lualatex`, or `pdflatex`
