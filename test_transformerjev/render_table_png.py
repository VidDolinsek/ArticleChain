#!/usr/bin/env python3
"""Render a nice PNG table from transformer_table.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

HIGHLIGHT_SUBSTRING = "paraphrase-multilingual-MiniLM-L12-v2"
HIGHLIGHT_BG = "#FFE08A"
HIGHLIGHT_TEXT = "#1E1E1E"


def read_rows(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows


def render_table(rows: List[List[str]], out_path: Path) -> None:
    if not rows:
        raise ValueError("Input CSV is empty.")

    header, data = rows[0], rows[1:]

    fig_height = max(2.5, 0.38 * (len(data) + 1))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.set_axis_off()

    table = ax.table(
        cellText=data,
        colLabels=header,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)

    # Style header
    for col in range(len(header)):
        cell = table[0, col]
        cell.set_facecolor("#2D6A4F")
        cell.set_text_props(color="white", weight="bold")

    # Zebra striping + highlight chosen model rows
    for row in range(1, len(data) + 1):
        is_highlight = False
        if data[row - 1] and HIGHLIGHT_SUBSTRING in data[row - 1][0]:
            is_highlight = True

        bg = HIGHLIGHT_BG if is_highlight else ("#F5F7FA" if row % 2 == 0 else "#FFFFFF")
        for col in range(len(header)):
            cell = table[row, col]
            cell.set_facecolor(bg)
            if is_highlight:
                cell.set_text_props(color=HIGHLIGHT_TEXT, weight="bold")

    # Subtle column width tweaks
    for col in range(len(header)):
        table.auto_set_column_width(col=col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a PNG table from transformer_table.csv."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="vizualizacije/test_transformerjev_results.csv",
        help="Path to the input CSV",
    )
    parser.add_argument(
        "--out",
        default="vizualizacije/transformer_table_test2.png",
        help="Path to the output PNG",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    rows = read_rows(in_path)
    render_table(rows, out_path)


if __name__ == "__main__":
    main()
