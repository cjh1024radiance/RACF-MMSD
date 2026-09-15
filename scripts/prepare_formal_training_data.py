from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from racf.io import read_jsonl, write_jsonl


def prepare(rows: list[dict], labels: tuple[str, ...]) -> list[dict]:
    records = []
    for row in rows:
        target_label = str(row.get("label_name", row.get("label")))
        if target_label not in labels:
            raise ValueError(f"Unknown label {target_label!r} for {row.get('sample_id')}")
        records.append({**row, "target_label": target_label, "counterfactuals": []})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare factual-only MMSD2.0 records for the formal Qwen trainer.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=["vanilla"], required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    args = parser.parse_args()

    labels = tuple(str(label) for label in args.labels)
    train_records = prepare(read_jsonl(args.train), labels)
    valid_records = prepare(read_jsonl(args.valid), labels)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train_records)
    write_jsonl(output / "valid.jsonl", valid_records)
    metadata = {"method": "vanilla", "labels": list(labels), "train": len(train_records), "valid": len(valid_records)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
