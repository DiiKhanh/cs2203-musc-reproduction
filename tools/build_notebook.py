#!/usr/bin/env python3
"""Build a Colab notebook from the project's fenced-code Markdown runbook."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def make_markdown_cell(lines: list[str]) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "".join(lines),
    }


def make_code_cell(language: str, lines: list[str]) -> dict[str, object]:
    source = list(lines)
    if language == "bash" and (not source or not source[0].startswith("%%bash")):
        source.insert(0, "%%bash\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "".join(source),
    }


def convert(markdown: str) -> dict[str, object]:
    lines = markdown.splitlines(keepends=True)
    cells: list[dict[str, object]] = []
    markdown_buffer: list[str] = []
    code_buffer: list[str] = []
    code_language: str | None = None

    for line in lines:
        if code_language is None:
            match = re.fullmatch(r"```(python|bash)\s*\n?", line)
            if match:
                if markdown_buffer:
                    cells.append(make_markdown_cell(markdown_buffer))
                    markdown_buffer = []
                code_language = match.group(1)
                code_buffer = []
            else:
                markdown_buffer.append(line)
        elif re.fullmatch(r"```\s*\n?", line):
            cells.append(make_code_cell(code_language, code_buffer))
            code_language = None
            code_buffer = []
        else:
            code_buffer.append(line)

    if code_language is not None:
        raise ValueError(f"Unclosed {code_language} code fence")
    if markdown_buffer:
        cells.append(make_markdown_cell(markdown_buffer))

    if not any(cell["cell_type"] == "code" for cell in cells):
        raise ValueError("The runbook produced no executable cells")

    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "E00_mvtec_bottle_smoke_test.ipynb"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    notebook = convert(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
