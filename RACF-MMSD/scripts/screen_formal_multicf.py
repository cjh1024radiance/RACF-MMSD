from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racf.io import read_jsonl, write_jsonl


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decision(row: dict) -> tuple[str, str]:
    if row.get("generation_status") != "success":
        return "FAIL", "generation_not_successful"
    value = row.get("cf")
    intervention = str(row.get("intervention_type"))
    if intervention == "text":
        text, original = str(value or "").strip(), str(row.get("original_text") or "").strip()
        if not text:
            return "FAIL", "empty_text_cf"
        if text == original:
            return "FAIL", "unchanged_text"
        if "[MASK]" not in text:
            return "FAIL", "no_text_mask"
        return "PASS", "structural_text_gate"
    if intervention == "image":
        path, original = Path(str(value or "")), Path(str(row.get("original_image") or ""))
        if not path.is_file():
            return "FAIL", "missing_image_cf"
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            return "FAIL", "unreadable_image_cf"
        if original.is_file() and image_digest(path) == image_digest(original):
            return "FAIL", "unchanged_image"
        return "PASS", "structural_image_gate"
    return "FAIL", "unknown_intervention"


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the fixed non-subjective PASS/FAIL gate to generated K=3 CFs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    outputs, reasons = [], Counter()
    for row in read_jsonl(args.input):
        existing_status = str(row.get("audit_status", "")).strip().upper()
        if existing_status in {"YES", "NO"}:
            status, reason = ("PASS", "human_yes") if existing_status == "YES" else ("FAIL", "human_no")
            protocol = "preserved_human_audit"
        else:
            status, reason = decision(row)
            protocol = "fixed_structural_gate_v1"
        outputs.append({**row, "audit_status": status, "audit_rule": reason, "audit_protocol": protocol})
        reasons[f"{status}:{reason}"] += 1
    output = Path(args.output)
    write_jsonl(output, outputs)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {"input": str(Path(args.input).resolve()), "output": str(output.resolve()), "count": len(outputs), "status_counts": dict(Counter(row["audit_status"] for row in outputs)), "rule_counts": dict(reasons), "screened_cache_hash": digest, "protocol": "fixed_structural_gate_v1"}
    output.with_name("screening_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
