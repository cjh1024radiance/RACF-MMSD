from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np


METHOD_LABELS = {
    "vanilla": "Vanilla",
    "uniform": "Multi-CF Uniform",
    "magnitude": "CE-Magnitude",
    "racf": "RACF (ours proposed)",
    "shuffled_racf": "Shuffled-RACF",
}

EXPECTED_MATRIX = {
    ("vanilla", 42), ("vanilla", 43), ("uniform", 42),
    ("racf", 42), ("racf", 43), ("magnitude", 42),
    ("shuffled_racf", 42),
}


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None, "p25": None, "p75": None, "p95": None}
    array = np.asarray(values, dtype=float)
    return {"count": int(array.size), "mean": float(array.mean()), "std": float(array.std()), "median": float(np.median(array)), "p25": float(np.percentile(array, 25)), "p75": float(np.percentile(array, 75)), "p95": float(np.percentile(array, 95))}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready MMSD result tables and response analyses from completed formal runs.")
    parser.add_argument("--runs-root", default="results/formal/mmsd/qwen")
    parser.add_argument("--reliability", required=True)
    parser.add_argument("--output", default="results/formal/final_paper_results")
    args = parser.parse_args()
    runs_root, output = Path(args.runs_root), Path(args.output)
    rows = []
    for method, label in METHOD_LABELS.items():
        directory = runs_root / method
        if not directory.is_dir():
            continue
        for run in sorted(directory.iterdir()):
            metrics_path, valid_path = run / "metrics_test.json", run / "metrics_valid.json"
            if not metrics_path.is_file() or not valid_path.is_file():
                continue
            test, valid = json.loads(metrics_path.read_text(encoding="utf-8")), json.loads(valid_path.read_text(encoding="utf-8"))
            seed = json.loads((run / "config.json").read_text(encoding="utf-8"))["seed"]
            if (method, int(seed)) in EXPECTED_MATRIX:
                rows.append({"model": "Qwen2-VL-7B", "method": method, "method_label": label, "seed": seed, "accuracy": test["accuracy"], "precision": test["precision"], "recall": test["recall"], "f1": test["f1"], "best_val_loss": valid["best_val_loss"], "status": "completed", "run_dir": str(run)})
    deduplicated = {}
    for row in rows:
        key = (row["method"], int(row["seed"]))
        current = deduplicated.get(key)
        if current is None or Path(row["run_dir"]).stat().st_mtime > Path(current["run_dir"]).stat().st_mtime:
            deduplicated[key] = row
    rows = [deduplicated[key] for key in sorted(deduplicated)]
    write_csv(runs_root / "final_summary.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    averages = []
    for method, values in grouped.items():
        summary = {"method": method, "method_label": METHOD_LABELS[method], "n_seeds": len(values)}
        for metric in ("accuracy", "precision", "recall", "f1", "best_val_loss"):
            metric_values = [float(value[metric]) for value in values if value[metric] is not None]
            summary[f"{metric}_mean"] = mean(metric_values) if metric_values else None
            summary[f"{metric}_std"] = pstdev(metric_values) if len(metric_values) > 1 else 0.0 if metric_values else None
        averages.append(summary)
    write_csv(runs_root / "final_mean_std.csv", averages)
    output.mkdir(parents=True, exist_ok=True)
    main_methods = {"vanilla", "racf"}
    ablation_methods = {"uniform", "magnitude", "racf", "shuffled_racf"}
    write_csv(output / "table_main.csv", [row for row in averages if row["method"] in main_methods])
    write_csv(output / "table_ablation.csv", [row for row in averages if row["method"] in ablation_methods])
    reliability_rows = [row for row in (json.loads(line) for line in Path(args.reliability).read_text(encoding="utf-8").splitlines() if line.strip())]
    by_intervention = defaultdict(lambda: defaultdict(list))
    representatives = []
    for row in reliability_rows:
        for intervention, value in row.get("interventions", {}).items():
            counterfactuals = value["counterfactuals"]
            deviations = value["response_consistency"]["deviation"]
            weights = value["weights"]
            entropy = -sum(weight * np.log(max(weight, 1e-12)) for weight in weights)
            by_intervention[intervention]["ce"].extend(item["ce_js"] for item in counterfactuals)
            by_intervention[intervention]["cec"].append(value["cec"])
            by_intervention[intervention]["reliability"].extend(item["reliability"] for item in counterfactuals)
            by_intervention[intervention]["weight"].extend(weights)
            by_intervention[intervention]["distance"].extend(deviations)
            by_intervention[intervention]["entropy"].append(entropy)
            if len(representatives) < 20:
                representatives.append({"sample_id": row["sample_id"], "intervention": intervention, "counterfactuals": counterfactuals, "weights": weights, "cec": value["cec"]})
    analysis = {name: {metric: describe(values) for metric, values in metrics.items()} for name, metrics in by_intervention.items()}
    analysis_dir = runs_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    mapping = {"response_variance.json": "distance", "pairwise_response_distance.json": "distance", "ce_distribution.json": "ce", "cec_distribution.json": "cec", "reliability_distribution.json": "reliability", "weight_distribution.json": "weight", "weight_entropy.json": "entropy"}
    for filename, metric in mapping.items():
        (analysis_dir / filename).write_text(json.dumps({intervention: values.get(metric, {}) for intervention, values in analysis.items()}, indent=2), encoding="utf-8")
    (output / "figure2_data.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (output / "figure3_data.json").write_text(json.dumps({"representative_samples": representatives}, indent=2), encoding="utf-8")
    manifests = [json.loads((Path(row["run_dir"]) / "experiment_manifest.json").read_text(encoding="utf-8")) for row in rows]
    (output / "all_runs_manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    completed_keys = {(row["method"], int(row["seed"])) for row in rows}
    expected = len(EXPECTED_MATRIX)
    ready = completed_keys == EXPECTED_MATRIX
    (output / "READY_FOR_PAPER_RESULTS.txt").write_text("YES\n" if ready else "NO\n", encoding="utf-8")
    print(json.dumps({"completed_runs": len(rows), "expected_runs": expected, "READY_FOR_PAPER_RESULTS": "YES" if ready else "NO"}, indent=2))


if __name__ == "__main__":
    main()
