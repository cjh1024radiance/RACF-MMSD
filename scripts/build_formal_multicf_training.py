from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racf.io import read_jsonl, write_jsonl

LABELS = ("non-sarcastic", "sarcastic")


def reverse_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Unknown binary MMSD label: {label}")
    return LABELS[1] if label == LABELS[0] else LABELS[0]


def normalized(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values] if total > 0 else [1.0 / len(values)] * len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable K=3 formal training records from one PASS-only shared cache.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--train-manifest")
    parser.add_argument("--valid-manifest")
    parser.add_argument("--cf-cache", required=True)
    parser.add_argument("--reliability", required=True)
    parser.add_argument("--method", choices=["uniform", "magnitude", "racf", "shuffled_racf"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_path = Path(args.train_manifest) if args.train_manifest else Path(args.manifest_dir) / "train.jsonl"
    valid_path = Path(args.valid_manifest) if args.valid_manifest else Path(args.manifest_dir) / "valid.jsonl"
    manifest = {str(row["sample_id"]): row for row in read_jsonl(train_path)}
    manifest.update({str(row["sample_id"]): row for row in read_jsonl(valid_path)})
    cache = defaultdict(lambda: defaultdict(list))
    for cf in read_jsonl(args.cf_cache):
        if str(cf.get("audit_status", "")).upper() != "PASS" or cf.get("generation_status") != "success" or not cf.get("cf"):
            continue
        cache[str(cf["sample_id"])][str(cf["intervention_type"])].append(cf)
    reliability = {str(row["sample_id"]): row for row in read_jsonl(args.reliability)}
    outputs = {"train": [], "valid": []}
    excluded = []
    for sample_id, sample in manifest.items():
        split = str(sample["split"])
        record = {**sample, "target_label": sample["label_name"], "counterfactuals": []}
        selected = ("text",)
        if split == "valid":
            outputs[split].append(record)
            continue
        metadata = reliability.get(sample_id, {}).get("interventions", {})
        failed = False
        for intervention in selected:
            group = sorted(cache[sample_id].get(intervention, []), key=lambda row: str(row["cf_id"]))
            if len(group) < args.k:
                failed = True
                break
            group = group[:args.k]
            details = {str(item["cf_id"]): item for item in metadata.get(intervention, {}).get("counterfactuals", [])}
            if len(details) < args.k:
                failed = True
                break
            if args.method == "uniform":
                weights = [1.0 / args.k] * args.k
            elif args.method == "magnitude":
                weights = normalized([max(0.0, float(details[str(cf["cf_id"])]["ce_js"])) for cf in group])
            else:
                weights = [float(details[str(cf["cf_id"])]["weight"]) for cf in group]
                if args.method == "shuffled_racf":
                    random.Random(f"{args.seed}:{sample_id}:{intervention}").shuffle(weights)
                weights = normalized(weights)
            for cf, weight in zip(group, weights):
                text = str(cf["cf"]) if intervention == "text" else ""
                image_path = str(cf["cf"]) if intervention == "image" else None
                record["counterfactuals"].append({"cf_id": cf["cf_id"], "intervention_type": intervention, "text": text, "image_path": image_path, "target_label": reverse_label(sample["label_name"]), "loss_weight": float(weight), "prompt_template": 'You are provided with a text from a tweet: "{text}". Is the text sarcastic? Answer with exactly one label: non-sarcastic or sarcastic.' if intervention == "text" else 'You are provided with an image from a tweet. Is the image sarcastic? Answer with exactly one label: non-sarcastic or sarcastic.'})
        if failed:
            excluded.append({"sample_id": sample_id, "reason": "missing_passed_k3_cache_or_reliability"})
        else:
            outputs[split].append(record)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    cache_hash = hashlib.sha256(Path(args.cf_cache).read_bytes()).hexdigest()
    metadata = {"method": args.method, "seed": args.seed, "k": args.k, "cf_cache": str(Path(args.cf_cache).resolve()), "cf_cache_hash": cache_hash, "train": len(outputs["train"]), "valid": len(outputs["valid"]), "excluded": len(excluded)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_jsonl(output / "excluded_samples.jsonl", excluded)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
