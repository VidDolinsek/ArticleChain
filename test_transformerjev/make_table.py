#!/usr/bin/env python3
"""Build a clean table from transformer test output."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

MODEL_RE = re.compile(r"^Model:\s*(.+)$")
PURITY_RE = re.compile(r"Topic neighbor purity:\s*([0-9]*\.?[0-9]+)%")
K_RE = re.compile(r"k=(\d+)")
COHERENCE_RE = re.compile(r"coherence=([+-]?[0-9]*\.?[0-9]+)")
MEAN_RE = re.compile(r"^mean_coherence\s*=\s*([+-]?[0-9]*\.?[0-9]+)")


@dataclass
class Block:
    model: str
    purity: Optional[float] = None
    k: Optional[int] = None
    mean: Optional[float] = None
    coherences: List[float] = field(default_factory=list)

    def finalize(self) -> None:
        if self.mean is None and self.coherences:
            self.mean = sum(self.coherences) / len(self.coherences)


def parse_blocks(lines: List[str]) -> List[Block]:
    blocks: List[Block] = []
    current: Optional[Block] = None

    for line in lines:
        model_match = MODEL_RE.match(line)
        if model_match:
            if current:
                current.finalize()
                blocks.append(current)
            current = Block(model=model_match.group(1).strip())
            continue

        if current is None:
            continue

        purity_match = PURITY_RE.search(line)
        if purity_match:
            current.purity = float(purity_match.group(1))
            k_match = K_RE.search(line)
            if k_match:
                current.k = int(k_match.group(1))
            continue

        mean_match = MEAN_RE.match(line)
        if mean_match:
            current.mean = float(mean_match.group(1))
            continue

        coh_match = COHERENCE_RE.search(line)
        if coh_match:
            current.coherences.append(float(coh_match.group(1)))

    if current:
        current.finalize()
        blocks.append(current)

    return blocks


def write_table(blocks: List[Block], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "transformer",
            "neighborhood_purity_pct",
            "mean_coherence",
        ])
        for block in blocks:
            model = block.model
            if block.k is not None:
                model = f"{model} (k={block.k})"
            purity = f"{block.purity:.2f}" if block.purity is not None else ""
            mean = f"{block.mean:.5f}" if block.mean is not None else ""
            writer.writerow([model, purity, mean])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a CSV table from transformer test output."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="testiranje.txt",
        help="Path to the input report",
    )
    parser.add_argument(
        "--out",
        default="vizualizacije/transformer_table.csv",
        help="Path to the output CSV",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    blocks = parse_blocks(in_path.read_text(encoding="utf-8").splitlines())
    write_table(blocks, out_path)


if __name__ == "__main__":
    main()
