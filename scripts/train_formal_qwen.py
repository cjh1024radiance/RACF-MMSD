from __future__ import annotations

import argparse
import json
import platform
import hashlib
import subprocess
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racf.io import read_jsonl
from racf.training.formal_qwen import FormalQwenTrainer


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a formal Qwen RACF or control LoRA adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if config["model"].get("backend") != "qwen":
        raise SystemExit("This compact release supports the Qwen backend only.")
    if args.max_steps is not None:
        config["training"]["max_steps"] = int(args.max_steps)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.max_steps is not None and args.epochs is None:
        config["training"]["formal_mode"] = False
    else:
        config["training"]["formal_mode"] = True
    config["training"]["status_interval"] = max(1, int(args.log_interval))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "seed.txt").write_text(str(config["seed"]), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({"python": platform.python_version(), "platform": platform.platform()}, indent=2), encoding="utf-8")
    train_path, valid_path = Path(args.data_dir) / "train.jsonl", Path(args.data_dir) / "valid.jsonl"
    train_records, valid_records = read_jsonl(train_path), read_jsonl(valid_path)
    print(f"[setup] model={config['model']['name']} dataset={config['dataset']['name']} method={config['method']} seed={config['seed']}", flush=True)
    print(f"[data] train_size={len(train_records)} valid_size={len(valid_records)} epochs={config['training']['epochs']} batch_size={config['training']['batch_size']} gradient_accumulation={config['training']['gradient_accumulation']}", flush=True)
    train_ids = {str(row["sample_id"]) for row in train_records}
    valid_ids = {str(row["sample_id"]) for row in valid_records}
    if train_ids & valid_ids:
        raise SystemExit("Formal training preflight failed: train/valid overlap.")
    test_path = Path("data_manifests/mmsd2/test.jsonl")
    if test_path.is_file():
        test_ids = {str(row["sample_id"]) for row in read_jsonl(test_path)}
        if (train_ids | valid_ids) & test_ids:
            raise SystemExit("Formal training preflight failed: training data overlaps test split.")
    if config["method"] == "vanilla" and any(row.get("counterfactuals") for row in train_records):
        raise SystemExit("Vanilla preflight failed: counterfactual loss must be absent.")
    if config["method"] != "vanilla":
        for row in train_records:
            for cf in row.get("counterfactuals", []):
                if cf["target_label"] == row["target_label"]:
                    raise SystemExit(f"Counterfactual target is not reversed for {row['sample_id']}")
    root = Path(__file__).resolve().parents[1]
    manifest = {"dataset": config["dataset"]["name"], "model": config["model"]["name"], "method": config["method"], "seed": config["seed"], "config": str(Path(args.config).resolve()), "config_hash": file_hash(Path(args.config)), "training_data": str(Path(args.data_dir).resolve()), "training_data_hash": {"train": file_hash(train_path), "valid": file_hash(valid_path)}, "test_manifest": str(test_path.resolve()) if test_path.is_file() else None, "test_manifest_hash": file_hash(test_path) if test_path.is_file() else None, "git_commit": git_commit(root), "git_commit_available": git_commit(root) is not None, "run_kind": "debug" if args.max_steps is not None or args.epochs is not None else "formal", "status": "started"}
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    trainer = FormalQwenTrainer(config, run_dir)
    best = trainer.train(train_records, valid_records, args.resume)
    checkpoint_info = json.loads((run_dir / "checkpoint_info.json").read_text(encoding="utf-8"))
    (run_dir / "metrics_valid.json").write_text(json.dumps({"best_val_loss": checkpoint_info["best_validation_loss"], "checkpoint": checkpoint_info["best"], "selection": "minimum_validation_loss"}, indent=2), encoding="utf-8")
    manifest.update({"status": "completed", "best_checkpoint": str(best)})
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
