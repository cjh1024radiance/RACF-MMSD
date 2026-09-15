from __future__ import annotations

import argparse
import random
from pathlib import Path

from racf.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed stratified MMSD train subset.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if args.limit >= len(rows):
        selected = rows
    else:
        rng = random.Random(args.seed)
        groups = {}
        for row in rows:
            groups.setdefault(str(row.get("label_name", row.get("label"))), []).append(row)
        quotas = {label: int(args.limit * len(group) / len(rows)) for label, group in groups.items()}
        remainder = args.limit - sum(quotas.values())
        ranked = sorted(groups, key=lambda label: len(groups[label]) * args.limit / len(rows) - quotas[label], reverse=True)
        for label in ranked[:remainder]:
            quotas[label] += 1
        selected = []
        for label, group in groups.items():
            selected.extend(rng.sample(group, quotas[label]))
        selected.sort(key=lambda row: str(row["sample_id"]))
    derived_fields = {"target_label", "counterfactuals", "category"}
    selected = [{key: value for key, value in row.items() if key not in derived_fields} for row in selected]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    print(f"Wrote {len(selected)} sampled MMSD rows: {output}")


if __name__ == "__main__":
    main()
