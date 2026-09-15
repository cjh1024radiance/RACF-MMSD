from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racf.io import append_jsonl, read_jsonl
from racf.models import create_probability_runner


ORIGINAL_PROMPT = 'You are provided with an image from a tweet with the associated text: "{text}". Is the image-text pair sarcastic? Answer with exactly one label: non-sarcastic or sarcastic.'
TEXT_PROMPT = 'You are provided with a text from a tweet: "{text}". Is the text sarcastic? Answer with exactly one label: non-sarcastic or sarcastic.'
IMAGE_PROMPT = 'You are provided with an image from a tweet. Is the image sarcastic? Answer with exactly one label: non-sarcastic or sarcastic.'


def normalized(values: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in values.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError(f"Invalid probability distribution: {values}")
    result = {key: float(value) / total for key, value in values.items()}
    if any(value < 0 or not math.isfinite(value) for value in result.values()) or abs(sum(result.values()) - 1.0) > 1e-5:
        raise ValueError(f"Probability sanity failed: {values}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe original and same-intervention K-CFs with frozen base Qwen.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cf-cache", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intervention", choices=["text", "image", "both"], default="text")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in read_jsonl(args.cf_cache):
        if args.intervention != "both" and str(row.get("intervention_type")) != args.intervention:
            continue
        groups[str(row["sample_id"])].append(row)
    manifest = {str(row["sample_id"]): row for row in read_jsonl(args.manifest)}
    completed = {str(row["sample_id"]) for row in read_jsonl(output)} if output.exists() else set()
    runner = create_probability_runner({"name": "Qwen2-VL-7B-Instruct", "backend": "qwen", "path": args.model_path, "labels": ["non-sarcastic", "sarcastic"]}, ORIGINAL_PROMPT, {"do_sample": False, "max_new_tokens": 8})
    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)
    runner.model.eval()
    if any(parameter.requires_grad for parameter in runner.model.parameters()) or runner.model.training:
        raise SystemExit("Probe model is not frozen/eval.")
    with torch.inference_mode():
        for sample_id, cf_rows in tqdm(groups.items(), desc="frozen_multicf_probe"):
            if sample_id in completed:
                continue
            source = manifest[sample_id]
            original = runner.predict(source["text"], [source["image_path"]])
            p0 = normalized(original["probability_distribution"])
            responses = []
            for cf in sorted(cf_rows, key=lambda row: str(row["cf_id"])):
                intervention = str(cf["intervention_type"])
                if intervention == "text":
                    prediction = runner.predict(str(cf["cf"]), [], TEXT_PROMPT)
                elif intervention == "image":
                    prediction = runner.predict("", [str(cf["cf"])], IMAGE_PROMPT)
                else:
                    raise ValueError(f"Unknown intervention: {intervention}")
                probability = normalized(prediction["probability_distribution"])
                responses.append({"cf_id": cf["cf_id"], "intervention_type": intervention, "seed": cf["seed"], "probability": probability, "predicted_label": prediction["predicted_label"], "confidence": max(probability.values()), "raw_response": prediction["raw_response"]})
            append_jsonl(output, {"sample_id": sample_id, "split": source.get("split"), "label": source.get("label"), "label_name": source.get("label_name"), "p0": p0, "pred0": original["predicted_label"], "confidence0": max(p0.values()), "raw_response0": original["raw_response"], "responses": responses, "probe_model": "Qwen2-VL-7B-Instruct", "probe_model_revision": original["model_revision"], "probe_frozen": True})
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name("probe_sanity.json").write_text(json.dumps({"passed": True, "probe_frozen": True, "count": len(read_jsonl(output)), "probe_hash": digest, "base_model_path": str(Path(args.model_path).resolve())}, indent=2), encoding="utf-8")
    print(f"Wrote frozen multi-CF probe: {output}")


if __name__ == "__main__":
    main()
