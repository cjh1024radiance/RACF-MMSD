from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racf.causal.response_consistency import consistency
from racf.io import read_jsonl, write_jsonl
from racf.metrics import compute_effects, normalize_distribution

LABELS = ("non-sarcastic", "sarcastic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CE and same-intervention CEC/RACF weights.")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--intervention", choices=["text", "image", "both"], default="text")
    args = parser.parse_args()
    outputs = []
    for row in read_jsonl(args.probe):
        p0 = normalize_distribution(row["p0"], LABELS)
        grouped = defaultdict(list)
        for response in row["responses"]:
            if args.intervention != "both" and str(response.get("intervention_type")) != args.intervention:
                continue
            grouped[str(response["intervention_type"])].append(response)
        intervention_results = {}
        for intervention, responses in grouped.items():
            responses = sorted(responses, key=lambda item: str(item["cf_id"]))
            if len(responses) < args.k:
                continue
            probabilities = [normalize_distribution(item["probability"], LABELS) for item in responses[:args.k]]
            vectors = [probability - p0 for probability in probabilities]
            response_consistency = consistency(vectors, method="median", distance="l2", tau=args.tau)
            raw_reliability = np.asarray(response_consistency["reliability"], dtype=float)
            weights = (raw_reliability / raw_reliability.sum()).tolist()
            effects = compute_effects(p0, probabilities)
            intervention_results[intervention] = {"consistency_valid": True, "k": args.k, "cec": float(1.0 / (1.0 + np.mean(response_consistency["deviation"]))), "response_consistency": response_consistency, "weights": weights, "counterfactuals": [{"cf_id": item["cf_id"], "ce_js": float(effect["ce_js"]), "ce_l1": float(effect["ce_l1"]), "reliability": float(reliability), "weight": float(weight)} for item, effect, reliability, weight in zip(responses[:args.k], effects, raw_reliability, weights)]}
        if intervention_results:
            outputs.append({"sample_id": row["sample_id"], "split": row.get("split"), "probe_model": row["probe_model"], "probe_model_revision": row["probe_model_revision"], "interventions": intervention_results, "weights_detached": True})
    write_jsonl(args.output, outputs)
    print(f"Wrote precomputed same-intervention reliability for {len(outputs)} samples: {args.output}")


if __name__ == "__main__":
    main()
