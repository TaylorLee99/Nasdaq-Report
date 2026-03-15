"""edgartools-backed SEC filing fetcher for raw disclosure acquisition."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependency may not be installed in all environments
    from edgar import Company as EdgarCompany
    from edgar import set_identity
except ImportError:  # pragma: no cover - handled at runtime
    EdgarCompany = None
    set_identity = None


@dataclass(slots=True)
class FilingRecord:
    """Normalized raw filing output produced by the edgartools adapter."""

    ticker: str
    company_name: str | None
    form_type: str
    filing_date: str | None
    accession_number: str | None
    cik: str | None
    raw_text_path: str | None
    metadata_path: str | None
    has_xbrl: bool
    xbrl_summary: dict[str, Any] | None


class EdgarSecClient:
    """Fetch 10-K, 10-Q, and 8-K filings through edgartools.

    This adapter is intentionally file-oriented so it can coexist with the
    existing DB-backed SEC metadata plane already present in the project.
    """

    def __init__(
        self,
        *,
        identity: str | None = None,
        raw_base_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self._identity = identity or settings.sec.edgar_identity
        self._raw_base_dir = raw_base_dir or settings.sec.raw_sec_dir

        if EdgarCompany is None or set_identity is None:
            msg = "edgartools is not installed. Install the package before using " "EdgarSecClient."
            raise RuntimeError(msg)
        if not self._identity:
            msg = (
                "SEC identity is required. Set APP_SEC__EDGAR_IDENTITY in "
                ".env or your environment."
            )
            raise ValueError(msg)

        set_identity(self._identity)
        self._raw_base_dir.mkdir(parents=True, exist_ok=True)

    def get_company(self, ticker: str) -> Any:
        """Return an edgartools company object for one ticker."""

        return EdgarCompany(ticker.upper())

    def fetch_latest_filings_for_company(
        self,
        ticker: str,
        *,
        ten_k_n: int = 1,
        ten_q_n: int = 3,
        eight_k_n: int = 5,
        save_raw_text: bool = True,
    ) -> list[FilingRecord]:
        """Fetch the latest filing sets for one company."""

        company = self.get_company(ticker)
        records: list[FilingRecord] = []
        records.extend(
            self._collect_latest(
                ticker=ticker,
                company=company,
                form_type="10-K",
                n=ten_k_n,
                save_raw_text=save_raw_text,
            )
        )
        records.extend(
            self._collect_latest(
                ticker=ticker,
                company=company,
                form_type="10-Q",
                n=ten_q_n,
                save_raw_text=save_raw_text,
            )
        )
        records.extend(
            self._collect_latest(
                ticker=ticker,
                company=company,
                form_type="8-K",
                n=eight_k_n,
                save_raw_text=save_raw_text,
            )
        )
        return records

    def _collect_latest(
        self,
        *,
        ticker: str,
        company: Any,
        form_type: str,
        n: int,
        save_raw_text: bool,
    ) -> list[FilingRecord]:
        result = company.latest(form_type, n) if n > 1 else company.latest(form_type)
        filings = result if isinstance(result, list) else [result]

        collected: list[FilingRecord] = []
        for filing in filings:
            if filing is None:
                continue
            try:
                collected.append(
                    self._materialize_filing(
                        ticker=ticker,
                        filing=filing,
                        save_raw_text=save_raw_text,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed to process %s filing for %s: %s",
                    form_type,
                    ticker.upper(),
                    exc,
                )
        return collected

    def _materialize_filing(
        self,
        *,
        ticker: str,
        filing: Any,
        save_raw_text: bool,
    ) -> FilingRecord:
        company_name = self._safe_getattr(filing, "company")
        form_type = self._safe_getattr(filing, "form") or "UNKNOWN"
        filing_date = self._normalize_date(self._safe_getattr(filing, "filing_date"))
        accession_number = self._safe_getattr(filing, "accession_no")
        cik = self._safe_getattr(filing, "cik")

        raw_text_path: str | None = None
        if save_raw_text:
            raw_text_path = self._save_raw_text(
                ticker=ticker,
                form_type=form_type,
                filing_date=filing_date,
                accession_number=accession_number,
                text=self._safe_text(filing),
            )

        xbrl_summary: dict[str, Any] | None = None
        has_xbrl = False
        if form_type in {"10-K", "10-Q"}:
            xbrl_summary = self._inspect_xbrl(filing)
            has_xbrl = xbrl_summary is not None

        metadata = {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "form_type": form_type,
            "filing_date": filing_date,
            "accession_number": accession_number,
            "cik": cik,
            "has_xbrl": has_xbrl,
            "xbrl_summary": xbrl_summary,
        }
        metadata_path = self._save_metadata(
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            accession_number=accession_number,
            payload=metadata,
        )

        return FilingRecord(
            ticker=ticker.upper(),
            company_name=company_name,
            form_type=form_type,
            filing_date=filing_date,
            accession_number=accession_number,
            cik=str(cik) if cik is not None else None,
            raw_text_path=raw_text_path,
            metadata_path=metadata_path,
            has_xbrl=has_xbrl,
            xbrl_summary=xbrl_summary,
        )

    def _inspect_xbrl(self, filing: Any) -> dict[str, Any] | None:
        """Conservatively inspect whether statement-level XBRL exists."""

        try:
            xbrl = filing.xbrl()
            if not xbrl:
                return None
            statements = getattr(xbrl, "statements", None)
            if not statements:
                return {"available": True, "statements": None}

            summary: dict[str, Any] = {"available": True}
            for name, method_name in {
                "income_statement": "income_statement",
                "balance_sheet": "balance_sheet",
                "cash_flow_statement": "cash_flow_statement",
            }.items():
                exists = False
                method = getattr(statements, method_name, None)
                if callable(method):
                    try:
                        exists = method() is not None
                    except Exception:  # noqa: BLE001
                        exists = False
                summary[name] = exists
            return summary
        except Exception:  # noqa: BLE001
            return None

    def _save_raw_text(
        self,
        *,
        ticker: str,
        form_type: str,
        filing_date: str | None,
        accession_number: str | None,
        text: str | None,
    ) -> str | None:
        if not text:
            return None

        out_dir = self._build_output_dir(ticker=ticker, form_type=form_type)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self._build_filename(
            filing_date=filing_date,
            accession_number=accession_number,
            suffix=".txt",
        )
        out_path.write_text(text, encoding="utf-8")
        return str(out_path)

    def _save_metadata(
        self,
        *,
        ticker: str,
        form_type: str,
        filing_date: str | None,
        accession_number: str | None,
        payload: dict[str, Any],
    ) -> str:
        out_dir = self._build_output_dir(ticker=ticker, form_type=form_type)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self._build_filename(
            filing_date=filing_date,
            accession_number=accession_number,
            suffix=".json",
        )
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(out_path)

    def _build_output_dir(self, *, ticker: str, form_type: str) -> Path:
        return self._raw_base_dir / ticker.upper() / form_type

    @staticmethod
    def _build_filename(
        *,
        filing_date: str | None,
        accession_number: str | None,
        suffix: str,
    ) -> str:
        safe_date = filing_date or "unknown_date"
        safe_accession = (accession_number or "unknown_accession").replace("/", "_")
        return f"{safe_date}__{safe_accession}{suffix}"

    @staticmethod
    def _normalize_date(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _safe_getattr(obj: Any, attr: str) -> Any:
        try:
            return getattr(obj, attr, None)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _safe_text(filing: Any) -> str | None:
        try:
            text = filing.text()
            if text is None:
                return None
            return str(text)
        except Exception:  # noqa: BLE001
            return None


def filings_to_dicts(records: Iterable[FilingRecord]) -> list[dict[str, Any]]:
    """Convert filing records into JSON-serializable dicts."""

    return [asdict(record) for record in records]
