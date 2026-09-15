from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, record: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def completed_keys(path: str | Path) -> set[tuple[str, str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return set()
    return {
        (str(record["sample_id"]), str(record["condition"]), str(record["cf_id"]))
        for record in read_jsonl(file_path)
    }
