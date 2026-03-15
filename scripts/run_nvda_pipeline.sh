#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src"

TICKER="${1:-NVDA}"
QUESTION="${2:-Build a valuation-free disclosure-grounded thesis using 10-K, 10-Q, and 8-K only.}"
FORMS="${FORMS:-10-K,10-Q,8-K}"
UNIVERSE_FIXTURE="${UNIVERSE_FIXTURE:-${ROOT_DIR}/tests/fixtures/universe_snapshot.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output}"
export TICKER QUESTION OUTPUT_DIR

echo "[1/6] Applying migrations"
if command -v alembic >/dev/null 2>&1; then
  alembic upgrade head
else
  echo "  - alembic not found; skipping migration step and relying on create_all()"
fi

echo "[2/6] Loading universe snapshot"
python3.11 -m app.cli.main universe load --file "${UNIVERSE_FIXTURE}" >/tmp/ems-universe-load.json

echo "[3/6] Syncing SEC metadata for ${TICKER}"
python3.11 -m app.cli.main sec sync-filings --ticker "${TICKER}" --forms "${FORMS}" >/tmp/ems-sec-sync.json

echo "[4/6] Downloading raw filings for ${TICKER}"
python3.11 -m app.cli.main sec download-filings --ticker "${TICKER}" --forms "${FORMS}" >/tmp/ems-sec-download.json

echo "[5/6] Building retrieval index for ${TICKER}"
python3.11 -m app.cli.main index filing --ticker "${TICKER}" --forms "${FORMS}" >/tmp/ems-index.json

echo "[6/6] Running analysis and exporting outputs"
python3.11 - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

from app.application import build_analysis_application_service
from app.domain import AnalysisRunRequest
from app.reporting import render_report_html, render_report_markdown, serialize_report_json

OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
TICKER = os.environ["TICKER"]
QUESTION = os.environ["QUESTION"]

service = build_analysis_application_service()
job = service.run_analysis(
    AnalysisRunRequest(
        ticker=TICKER,
        question=QUESTION,
        include_transcript=False,
    )
)
report = job.final_report
if report is None:
    raise RuntimeError("Analysis completed without a final report")

markdown_text = job.final_report_markdown or render_report_markdown(report)
json_text = serialize_report_json(report)
response_payload = job.model_dump(mode="json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / f"{TICKER.lower()}_latest_report.md").write_text(markdown_text, encoding="utf-8")
(OUTPUT_DIR / f"{TICKER.lower()}_latest_report.json").write_text(json_text, encoding="utf-8")
(OUTPUT_DIR / f"{TICKER.lower()}_latest_analysis_response.json").write_text(
    json.dumps(response_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

html_text = render_report_html(report)
(OUTPUT_DIR / f"{TICKER.lower()}_latest_report.html").write_text(html_text, encoding="utf-8")

print(json.dumps(
    {
        "analysis_id": job.analysis_id,
        "status": job.status.value,
        "ticker": TICKER,
        "markdown_path": str(OUTPUT_DIR / f"{TICKER.lower()}_latest_report.md"),
        "json_path": str(OUTPUT_DIR / f"{TICKER.lower()}_latest_report.json"),
        "html_path": str(OUTPUT_DIR / f"{TICKER.lower()}_latest_report.html"),
    },
    ensure_ascii=False,
    indent=2,
))
PY
