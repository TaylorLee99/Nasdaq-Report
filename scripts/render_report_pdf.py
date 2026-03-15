#!/usr/bin/env python3
"""Render a markdown report to PDF via pandoc."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HTML_ENGINES = {"weasyprint", "wkhtmltopdf"}
DEFAULT_ENGINE_ORDER = ("weasyprint", "wkhtmltopdf", "xelatex", "lualatex", "pdflatex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a markdown report to PDF with pandoc.",
    )
    parser.add_argument("input", type=Path, help="Markdown input path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="PDF output path. Defaults to the input path with a .pdf suffix.",
    )
    parser.add_argument(
        "--pdf-engine",
        dest="pdf_engine",
        help="Explicit pandoc PDF engine to use.",
    )
    return parser.parse_args()


def resolve_pdf_engine(explicit_engine: str | None) -> str:
    if explicit_engine:
        if shutil.which(explicit_engine) is None:
            raise RuntimeError(
                f"Requested PDF engine `{explicit_engine}` is not installed or not on PATH."
            )
        return explicit_engine

    for engine in DEFAULT_ENGINE_ORDER:
        if shutil.which(engine) is not None:
            return engine

    raise RuntimeError(
        "No supported PDF engine was found. Install one of: "
        "weasyprint, wkhtmltopdf, xelatex, lualatex, pdflatex."
    )


def build_pandoc_command(input_path: Path, output_path: Path, pdf_engine: str) -> list[str]:
    repo_root = Path(__file__).resolve().parent.parent
    css_path = repo_root / "scripts" / "report_pdf.css"

    command = [
        "pandoc",
        str(input_path),
        "--from",
        "gfm",
        "--standalone",
        "--toc",
        "--pdf-engine",
        pdf_engine,
        "--output",
        str(output_path),
        "-V",
        "geometry:margin=0.9in",
        "-V",
        "fontsize=10pt",
        "-V",
        "linestretch=1.15",
    ]
    if pdf_engine in HTML_ENGINES and css_path.exists():
        command.extend(["--css", str(css_path)])
    return command


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Input markdown file not found: {input_path}", file=sys.stderr)
        return 1

    if shutil.which("pandoc") is None:
        print(
            "pandoc is not installed. Install pandoc and rerun this script.",
            file=sys.stderr,
        )
        return 1

    output_path = (args.output or input_path.with_suffix(".pdf")).resolve()

    try:
        pdf_engine = resolve_pdf_engine(args.pdf_engine)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    command = build_pandoc_command(input_path, output_path, pdf_engine)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"pandoc PDF render failed with exit code {exc.returncode}.",
            file=sys.stderr,
        )
        return exc.returncode

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
