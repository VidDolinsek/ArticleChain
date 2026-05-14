#!/usr/bin/env python3
"""Create a CSV table from test_transformerjev_results.txt."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

MODEL_RE = re.compile(r"^Model:\s*(.+)$")
PURITY_RE = re.compile(r"Topic neighbor purity:\s*([0-9]*\.?[0-9]+)%")
MEAN_RE = re.compile(r"^mean_coherence\s*=\s*([+-]?[0-9]*\.?[0-9]+)")


@dataclass
class Row:
    model: str
    purity: Optional[float] = None
    mean: Optional[float] = None


def parse_rows(lines: List[str]) -> List[Row]:
    rows: List[Row] = []
    current: Optional[Row] = None

    for line in lines:
        model_match = MODEL_RE.match(line)
        if model_match:
            if current:
                rows.append(current)
            current = Row(model=model_match.group(1).strip())
            continue

        if current is None:
            continue

        purity_match = PURITY_RE.search(line)
        if purity_match:
            current.purity = float(purity_match.group(1))
            continue

        mean_match = MEAN_RE.match(line)
        if mean_match:
            current.mean = float(mean_match.group(1))
            continue

    if current:
        rows.append(current)

    return rows


def write_csv(rows: List[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "transformer",
            "neighborhood_purity_pct",
            "mean_coherence",
        ])
        for row in rows:
            purity = f"{row.purity:.2f}" if row.purity is not None else ""
            mean = f"{row.mean:.5f}" if row.mean is not None else ""
            writer.writerow([row.model, purity, mean])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a CSV table from a transformer results text file."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="testiranje.txt",
        help="Path to the input text file",
    )
    parser.add_argument(
        "--out",
        default="test_transformerjev.csv",
        help="Path to the output CSV",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    rows = parse_rows(in_path.read_text(encoding="utf-8").splitlines())
    write_csv(rows, out_path)


if __name__ == "__main__":
    main()
