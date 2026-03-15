"""Shared heuristics for 10-Q recent-quarter extraction."""

from __future__ import annotations

import re
from typing import Any

from app.domain import ChunkRecord, FilingType


def build_ten_q_recent_change_candidate(chunk: ChunkRecord) -> str | None:
    """Return the best recent-quarter sentence we can recover from a 10-Q chunk."""

    if chunk.filing_type != FilingType.FORM_10Q:
        return None

    normalized = " ".join(chunk.text.split()).strip()
    if not normalized:
        return None

    candidates: list[tuple[int, str]] = []
    for raw_sentence in re.split(r"(?<=[.!?])\s+", normalized):
        sentence = _clean_ten_q_sentence(raw_sentence.strip())
        if not sentence:
            continue
        if is_ten_q_recent_change_candidate_sentence(
            sentence=sentence,
            section_name=chunk.section_name,
            section_type=chunk.section_type,
        ):
            candidates.append((_score_ten_q_candidate(sentence), sentence))

    synthetic = _synthesize_ten_q_table_row_sentence(normalized)
    if synthetic is not None:
        candidates.append((_score_ten_q_candidate(synthetic) + 10, synthetic))

    if not candidates:
        return None
    sentence = max(candidates, key=lambda item: item[0])[1]
    if not sentence.endswith("."):
        sentence = f"{sentence}."
    return sentence


def is_ten_q_recent_change_candidate_sentence(
    *,
    sentence: str,
    section_name: str,
    section_type: Any | None = None,
) -> bool:
    """Check whether a sentence is a usable recent-quarter delta statement."""

    normalized = _clean_ten_q_sentence(sentence)
    if not normalized:
        return False
    lowered = normalized.lower()
    section_lower = section_name.lower()
    section_type_lower = _normalize_section_type(section_type)

    if _looks_like_low_signal_fragment(normalized):
        return False
    if _looks_like_truncated_sentence(normalized):
        return False
    if normalized.startswith("See Item "):
        return False
    if _looks_table_like(normalized):
        return False
    if lowered.startswith("changes in foreign exchange rates"):
        return False

    disallowed_markers = (
        "historically experienced higher net sales",
        "compared to other quarters in its fiscal year",
        "seasonal holiday demand",
        "product and service introductions can significantly impact",
        "legal proceedings",
        "subject to significant uncertainty",
        "updated products",
        "sales and marketing efforts",
        "sales and marketing costs",
        "technology and infrastructure costs",
        "segment information",
        "see item 1 of part i",
        "note 8",
        "note 1",
        "critical accounting",
        "accounting pronouncements",
        "forward-looking statements",
        "contractual obligations and commitments",
        "commitments and contingencies",
        "operating lease commitments",
        "future capital requirements",
        "future arrangements",
        "share repurchase program remained available",
        "estimated life of the products",
        "anthropic",
        "subject to closing conditions",
        "invest up to",
        "changes in foreign exchange rates did not significantly impact",
    )
    if any(marker in lowered for marker in disallowed_markers):
        return False

    if section_type_lower in {"controls", "notes", "financial_statements", "risk_factors"}:
        return False

    if any(
        marker in section_lower
        for marker in (
            "sales and marketing",
            "legal proceedings",
            "segment information",
            "recent accounting pronouncements",
            "critical accounting",
        )
    ):
        return False

    balance_markers = ("cash", "cash equivalents", "marketable securities")
    delta_metric_markers = (
        "revenue",
        "net sales",
        "gross margin",
        "gross margin percentage",
        "operating income",
        "operating margin",
        "net income",
        "cash flows from operating activities",
        "cash flow from operating activities",
        "customer count",
        "customers",
        "average revenue",
    )
    delta_markers = (
        " increased ",
        " decrease ",
        " decreased ",
        " growth ",
        " grew ",
        " positively impacted ",
        " negatively impacted ",
        " materially impacted ",
        "declined",
        " rose ",
        " relatively flat ",
        " compared to the prior year period",
        " compared with the prior year period",
        " compared to the comparable prior year period",
        " compared with the comparable prior year period",
        " compared to the same quarter",
        " compared with the same quarter",
        " compared to the same period",
        " compared with the same period",
        " year-over-year ",
        " quarter-over-quarter ",
        " sequentially ",
        " trailing twelve months ",
    )

    if normalized.startswith("As of ") and any(marker in lowered for marker in balance_markers):
        return bool(re.search(r"\$\s?\d", normalized))

    if _looks_like_ten_q_customer_growth_sentence(lowered):
        return True

    if (
        "operating income" in lowered
        and re.search(r"\$\s?\d", normalized)
        and any(
            marker in lowered
            for marker in ("period ended", "quarter ended", "q1 ", "q2 ", "q3 ", "q4 ")
        )
        and any(
            marker in lowered
            for marker in ("positively impacted", "negatively impacted", "materially impacted")
        )
    ):
        return True

    has_metric = any(marker in lowered for marker in delta_metric_markers)
    has_delta = any(marker in f" {lowered} " for marker in delta_markers) or bool(
        re.search(r"\b\d+%\b", lowered)
    )
    if not (has_metric and has_delta):
        return False

    if any(
        marker in lowered
        for marker in (
            "sales and marketing costs",
            "technology and infrastructure costs",
            "fulfillment costs",
            "advertising expenses",
            "severance costs",
            "macroeconomic conditions",
        )
    ):
        return False

    return True


def _clean_ten_q_sentence(sentence: str) -> str:
    cleaned = " ".join(sentence.split()).strip()
    if not cleaned:
        return cleaned
    cleaned = cleaned.lstrip(" ,;:-")
    if not cleaned:
        return ""
    cleaned = re.sub(r"^([A-Z][A-Za-z]+)\s+\1\b", r"\1", cleaned)
    cleaned = re.sub(
        r"^(Products Gross Margin)\s+Products gross margin\b",
        "Products gross margin",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(Services Gross Margin)\s+Services gross margin\b",
        "Services gross margin",
        cleaned,
        flags=re.IGNORECASE,
    )
    if cleaned and cleaned[0].islower():
        return ""
    return cleaned


def _score_ten_q_candidate(sentence: str) -> int:
    lowered = sentence.lower()
    score = 0
    if sentence[0].isupper():
        score += 10
    if 50 <= len(sentence) <= 260:
        score += 10
    if sentence.startswith("As of "):
        score += 35
    if re.match(
        r"^the (increase|decrease) in [a-z0-9 ,&-]+"
        r"(revenue|net sales|operating income|gross margin|operating cash flow)",
        lowered,
    ):
        score += 20
    if any(
        marker in lowered
        for marker in (
            "revenue",
            "net sales",
            "gross margin",
            "operating income",
            "cash and cash equivalents",
            "marketable securities",
        )
    ):
        score += 20
    if any(
        marker in lowered
        for marker in (
            "north america operating income",
            "international operating income",
            "aws operating income",
            "operating cash flow",
        )
    ):
        score += 12
    if any(
        marker in lowered
        for marker in (
            "compared to the same quarter",
            "compared with the same quarter",
            "compared to the comparable prior year period",
            "compared with the comparable prior year period",
            "compared to the prior year period",
            "compared with the prior year period",
            "first quarter",
            "second quarter",
            "third quarter",
            "fourth quarter",
            "q1 ",
            "q2 ",
            "q3 ",
            "q4 ",
        )
    ):
        score += 15
    if re.search(r"\b\d+%\b", lowered):
        score += 15
    if re.search(r"\$\s?\d", sentence):
        score += 10
    if _looks_like_ten_q_customer_growth_sentence(lowered):
        score += 12
    if any(
        marker in lowered
        for marker in (
            "historically experienced higher net sales",
            "compared to other quarters in its fiscal year",
            "seasonal holiday demand",
            "product and service introductions can significantly impact",
        )
    ):
        score -= 25
    if any(
        marker in lowered
        for marker in (
            "changes in foreign exchange rates",
            "foreign exchange rates did not significantly impact",
            "negatively impacted operating income by",
        )
    ):
        score -= 35
    return score


def _looks_like_ten_q_customer_growth_sentence(lowered: str) -> bool:
    if "customers" in lowered and re.search(r"\b\d{2,4}\b", lowered):
        if any(
            marker in lowered
            for marker in (
                "during the period ended",
                "trailing twelve months ended",
                "grew ",
                "increased ",
            )
        ):
            return True
    if "top twenty customers" in lowered and re.search(r"\b\d+%\b", lowered):
        return True
    return False


def _synthesize_ten_q_table_row_sentence(text: str) -> str | None:
    row_patterns = (
        (
            re.compile(
                r"(?P<label>Productivity and Business Processes|Intelligent Cloud|"
                r"More Personal Computing|North America|International|AWS|Products|"
                r"Services|Total)\s+Revenue\s+\$\s*(?P<current>[\d,]+)\s+\$\s*"
                r"(?P<prior>[\d,]+)\s+(?P<pct>\(?-?\d+\)?)%",
                flags=re.IGNORECASE,
            ),
            "{label} revenue {direction} {pct}% compared with the prior-year period.",
        ),
        (
            re.compile(
                r"Operating income\s+\$\s*(?P<current>[\d,]+)\s+\$\s*"
                r"(?P<prior>[\d,]+)\s+(?P<pct>\(?-?\d+\)?)%",
                flags=re.IGNORECASE,
            ),
            "Operating income {direction} {pct}% compared with the prior-year period.",
        ),
        (
            re.compile(
                r"Total gross margin percentage\s+(?P<current>\d+(?:\.\d+)?)\s*%\s*"
                r"(?P<prior>\d+(?:\.\d+)?)\s*%",
                flags=re.IGNORECASE,
            ),
            "Total gross margin percentage increased to {current}% from {prior}% in the quarter.",
        ),
    )

    normalized = " ".join(text.split()).strip()
    for pattern, template in row_patterns:
        match = pattern.search(normalized)
        if match is None:
            continue
        groups = match.groupdict()
        if "pct" in groups:
            pct_raw = groups["pct"]
            direction, pct = _pct_direction_and_value(pct_raw)
            label = groups.get("label")
            sentence = template.format(
                label=label,
                direction=direction,
                pct=pct,
                current=groups.get("current"),
                prior=groups.get("prior"),
            )
        else:
            sentence = template.format(
                current=groups.get("current"),
                prior=groups.get("prior"),
            )
        sentence = sentence.replace("  ", " ").strip()
        if sentence:
            return sentence
    return None


def _pct_direction_and_value(raw_value: str) -> tuple[str, str]:
    cleaned = raw_value.strip()
    negative = cleaned.startswith("(") or cleaned.startswith("-")
    pct = cleaned.strip("()%").lstrip("-")
    if not pct:
        pct = "0"
    return ("decreased", pct) if negative else ("increased", pct)


def _normalize_section_type(section_type: Any | None) -> str:
    if section_type is None:
        return ""
    value = getattr(section_type, "value", section_type)
    return str(value).lower()


def _looks_like_low_signal_fragment(text: str) -> bool:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return True
    if re.fullmatch(r"\d{1,3}[.]?", normalized):
        return True
    tokens = normalized.split()
    if len(tokens) <= 2 and all(re.fullmatch(r"[\d.,$()%]+", token) for token in tokens):
        return True
    if len(tokens) <= 4 and not any(
        token.lower()
        in {
            "is",
            "are",
            "was",
            "were",
            "provide",
            "provides",
            "offer",
            "offers",
            "design",
            "designs",
            "manufacture",
            "manufactures",
            "market",
            "markets",
            "sell",
            "sells",
            "report",
            "reports",
            "serve",
            "serves",
            "increased",
            "decreased",
            "grew",
        }
        for token in tokens
    ):
        return True
    if re.match(r"^(item|part)\s+\d", normalized, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_truncated_sentence(sentence: str) -> bool:
    normalized = " ".join(sentence.split()).strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    if normalized.endswith(","):
        return True
    if lowered.endswith((" and", " or", " as well as", " including")):
        return True
    if normalized.count(",") >= 4 and not re.search(r"[.!?]$", normalized):
        return True
    return False


def _looks_table_like(text: str) -> bool:
    number_tokens = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)
    money_like_tokens = re.findall(r"\$\s?\d", text)
    short_lines = [line.strip() for line in text.splitlines() if line.strip()]
    month_headers = {
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    }

    if len(number_tokens) >= 8 and len(money_like_tokens) >= 2:
        return True
    if short_lines:
        header_tokens = set(re.findall(r"[A-Za-z0-9_]+", " ".join(short_lines[:2]).lower()))
        if len(number_tokens) >= 8 and header_tokens & month_headers:
            return True
    if text.count("$") >= 3 and text.count(",") >= 8:
        return True
    return False
