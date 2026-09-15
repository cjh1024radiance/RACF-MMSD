from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racf.evaluation.classification import evaluate_classification
from racf.io import append_jsonl, completed_keys, read_jsonl
from racf.models import create_probability_runner

PROMPT = 'You are provided with an image from a tweet with the associated text: "{text}". Is the image-text pair sarcastic? Answer with exactly one label: non-sarcastic or sarcastic.'


def best_adapter(run_dir: Path) -> Path:
    candidate = run_dir / "best" / "adapter"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Best validation checkpoint missing: {candidate}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed Qwen formal run exactly once on MMSD test.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-manifest", default="data_manifests/mmsd2/test.jsonl")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    result_path = run_dir / "metrics_test.json"
    if result_path.exists():
        print(f"[test] existing result preserved: {result_path}")
        return
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    adapter = best_adapter(run_dir)
    print(f"[TEST] evaluating best validation checkpoint: {adapter}", flush=True)
    runner = create_probability_runner({**config["model"], "adapter_checkpoint": str(adapter), "labels": config["dataset"]["labels"]}, PROMPT, {"do_sample": False, "max_new_tokens": 8})
    predictions = run_dir / "predictions_test.jsonl"
    completed = completed_keys(predictions)
    for row in tqdm(read_jsonl(args.test_manifest), desc="test"):
        key = (str(row["sample_id"]), "original", "original")
        if key in completed:
            continue
        prediction = runner.predict(str(row["text"]), [str(row["image_path"])])
        append_jsonl(predictions, {"sample_id": row["sample_id"], "condition": "original", "cf_id": "original", "label": row["label"], "label_name": row["label_name"], "prediction": prediction["predicted_label"], "probability_distribution": prediction["probability_distribution"], "raw_response": prediction["raw_response"]})
    metrics = evaluate_classification(read_jsonl(predictions), tuple(config["dataset"]["labels"]))
    metrics.update({"checkpoint": str(adapter), "selection": "minimum_validation_loss", "test_evaluated_once": True})
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[TEST] " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
