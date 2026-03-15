"""Standalone demo for fetching NVDA filings with edgartools."""

from __future__ import annotations

import json
import logging

from app.data_sources.sec import EdgarSecClient, filings_to_dicts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    """Fetch the latest NVDA 10-K, 10-Q, and 8-K filings and print saved artifacts."""

    client = EdgarSecClient()
    records = client.fetch_latest_filings_for_company(
        ticker="NVDA",
        ten_k_n=1,
        ten_q_n=3,
        eight_k_n=5,
        save_raw_text=True,
    )
    print(json.dumps(filings_to_dicts(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
