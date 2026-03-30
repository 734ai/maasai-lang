#!/usr/bin/env python3
"""
Merge open-source Maasai supplement records into data/final_v3.

This keeps the existing split files intact, appends only new unique pairs,
and uses deterministic ids plus deterministic split assignment so reruns are
stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = (
    "data/raw/public_domain_hollis_proverbs.jsonl",
    "data/raw/public_domain_hinde_vocabulary.jsonl",
    "data/raw/open_asjp_maasai_wordlist.jsonl",
)
DEFAULT_OUTPUT_DIR = "data/final_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment final_v3 with open-source supplement rows")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["source_lang"]),
        str(row["target_lang"]),
        str(row["source_text"]),
        str(row["target_text"]),
    )


def stable_hex(row: dict[str, Any]) -> str:
    raw = "||".join(record_key(row))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def choose_split(row: dict[str, Any]) -> str:
    ratio = int(stable_hex(row)[:8], 16) / 0xFFFFFFFF
    if ratio < 0.85:
        return "train"
    if ratio < 0.925:
        return "valid"
    return "test"


def make_final_row(row: dict[str, Any]) -> dict[str, Any]:
    digest = stable_hex(row)
    source_name = str(row.get("source_name", "open_source")).lower().replace(" ", "_")
    source_lang = str(row["source_lang"]).lower()
    target_lang = str(row["target_lang"]).lower()
    quality_score = float(row.get("quality_score", 0.9))
    tier = "gold" if quality_score >= 0.95 else "silver"

    return {
        "id": f"open-{source_name}-{source_lang}2{target_lang}-{digest[:10]}",
        "source_text": row["source_text"],
        "target_text": row["target_text"],
        "source_lang": source_lang,
        "target_lang": target_lang,
        "domain": row.get("domain", "open_source"),
        "source_name": source_name,
        "quality_score": quality_score,
        "notes": row.get("notes", ""),
        "tier": tier,
        "confidence": quality_score,
        "quality_assessment": {
            "overall_score": quality_score,
            "tier": tier,
            "issues": [],
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    splits = {
        split: load_jsonl(output_dir / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }

    existing_keys = {
        record_key(row)
        for split_rows in splits.values()
        for row in split_rows
    }

    appended = {"train": 0, "valid": 0, "test": 0}
    for input_file in args.inputs:
        for row in load_jsonl(Path(input_file)):
            key = record_key(row)
            if key in existing_keys:
                continue

            split_name = choose_split(row)
            splits[split_name].append(make_final_row(row))
            existing_keys.add(key)
            appended[split_name] += 1

    for split_name, rows in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)

    total_added = sum(appended.values())
    print(f"Added {total_added} new rows to {output_dir}")
    for split_name in ("train", "valid", "test"):
        print(f"  {split_name}: +{appended[split_name]}")


if __name__ == "__main__":
    main()
