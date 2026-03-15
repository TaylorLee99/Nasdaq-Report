from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_10k_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_10k.txt"


@pytest.fixture
def sample_10q_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_10q.txt"


@pytest.fixture
def sample_8k_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_8k.txt"


@pytest.fixture
def sample_transcript_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_transcript.txt"


@pytest.fixture
def sample_xbrl_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_xbrl_facts.json"
