from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FormalQwenTrainer:
    """Independent, validation-aware Qwen LoRA trainer for formal experiments."""

    def __init__(self, config: dict, run_dir: str | Path) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        set_seed(int(config["seed"]))
        model_config = config["model"]
        training = config["training"]
        dataset_labels = config.get("dataset", {}).get("labels")
        model_labels = model_config.get("labels")
        fallback_labels = ("non-sarcastic", "sarcastic")
        resolved_labels = dataset_labels or model_labels or fallback_labels
        self.labels = tuple(str(label) for label in resolved_labels)
        if len(self.labels) < 2:
            raise ValueError("FormalQwenTrainer requires at least two labels.")
        config.setdefault("dataset", {}).setdefault("labels", list(self.labels))
        model_config.setdefault("labels", list(self.labels))
        revision = model_config.get("revision")
        revision_kwargs = {} if revision in (None, "", "local_checkpoint") else {"revision": revision}
        print(f"[model] loading processor from {model_config['path']} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_config["path"], **revision_kwargs)
        print(f"[model] loading Qwen2-VL weights from {model_config['path']} ...", flush=True)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            model_config["path"], **revision_kwargs,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
        )
        base.config.use_cache = False
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
        lora = training["lora"]
        adapter = LoraConfig(
            r=int(lora["r"]), lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]), target_modules=list(lora["target_modules"]),
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, adapter)
        print("[model] LoRA adapter initialized", flush=True)
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.train()
        self.device = next(self.model.parameters()).device
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
        )
        self.max_frames = int(training.get("max_frames", 1))
        self.batch_size = int(training["batch_size"])
        self.grad_accumulation = int(training["gradient_accumulation"])
        self.samples_per_update = self.batch_size * self.grad_accumulation
        self.checkpoint_interval = int(training["checkpoint_interval"])
        self.status_interval = int(training.get("status_interval", 10))

    def _prompt(self, text: str, image_path: str | None, template: str | None = None) -> tuple[str, list[dict]]:
        content = []
        if image_path:
            content.append({"type": "image", "image": str(Path(image_path).resolve())})
        content.append({"type": "text", "text": (template or self.config["prompt"]["user_template"]).format(text=text)})
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), messages

    def _loss(self, text: str, target: str, image_path: str | None, template: str | None = None):
        from qwen_vl_utils import process_vision_info

        prompt, messages = self._prompt(text, image_path, template)
        image_inputs, video_inputs = process_vision_info(messages)
        encoded = self.processor(text=[prompt + target], images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.device)
        prefix = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt").input_ids[0]
        if encoded.input_ids.shape[1] <= prefix.numel() or not self.torch.equal(encoded.input_ids[0, :prefix.numel()].cpu(), prefix.cpu()):
            raise RuntimeError("Supervised target does not extend the inference prompt exactly.")
        labels = encoded.input_ids.clone()
        labels[:, :prefix.numel()] = -100
        with self.torch.autocast(device_type="cuda", dtype=self.torch.float16, enabled=self.torch.cuda.is_available()):
            return self.model(**encoded, labels=labels).loss

    @staticmethod
    def _image(record: dict, counterfactual: dict | None = None) -> str | None:
        value = (counterfactual or {}).get("image_path") or record.get("image_path") or record.get("image")
        if value is not None and not Path(value).is_file():
            raise FileNotFoundError(f"Missing image for {record.get('sample_id')}: {value}")
        return str(value) if value else None

    def _record_loss(self, record: dict) -> tuple[object, dict]:
        original = self._loss(record["text"], record["target_label"], self._image(record))
        total = original
        cf_values, weighted_cf = [], 0.0
        for cf in record.get("counterfactuals", []):
            intervention_type = str(cf.get("intervention_type", cf.get("modality", "text"))).lower()
            cf_image = self._image(record, cf) if intervention_type.startswith("image") else None
            value = self._loss(cf["text"], cf["target_label"], cf_image, cf.get("prompt_template"))
            weight = float(cf.get("loss_weight", 0.0))
            total = total + float(self.config["training"].get("lambda_cf", 1.0)) * weight * value
            cf_values.append(float(value.detach()))
            weighted_cf += weight * cf_values[-1]
        counterfactuals = record.get("counterfactuals", [])
        weights = [float(cf.get("loss_weight", 0.0)) for cf in counterfactuals]
        text_weight_sum = sum(weight for weight, cf in zip(weights, counterfactuals) if str(cf.get("intervention_type", cf.get("modality", ""))).startswith("text") or cf.get("modality") == "text")
        image_weight_sum = sum(weight for weight, cf in zip(weights, counterfactuals) if str(cf.get("intervention_type", cf.get("modality", ""))).startswith("image") or cf.get("modality") == "image")
        return total, {"original": float(original.detach()), "counterfactual": weighted_cf, "cf_losses": cf_values, "num_cf_used": sum(weight > 0 for weight in weights), "mean_cf_weight": float(np.mean(weights)) if weights else 0.0, "max_cf_weight": max(weights, default=0.0), "min_cf_weight": min(weights, default=0.0), "text_cf_weight_sum": text_weight_sum, "image_cf_weight_sum": image_weight_sum, "category": record.get("category")}

    def _save(self, step: int, name: str | None = None) -> Path:
        folder = name or f"checkpoint-{step:06d}"
        path = self.run_dir / folder
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "adapter")
        self.torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        (path / "state.json").write_text(json.dumps({"step": step, "seed": self.config["seed"]}, indent=2), encoding="utf-8")
        return path

    def _validate_metrics(self, records: list[dict]) -> dict:
        self.model.eval()
        values = []
        predictions, targets = [], []
        with self.torch.inference_mode():
            for record in records:
                target = str(record["target_label"])
                values.append(float(self._loss(record["text"], target, self._image(record))))
                candidate_losses = [
                    float(self._loss(record["text"], label, self._image(record)))
                    for label in self.labels
                ]
                predictions.append(self.labels[int(np.argmin(candidate_losses))])
                targets.append(target)
        self.model.train()
        accuracy = float(np.mean([pred == target for pred, target in zip(predictions, targets)])) if targets else 0.0
        positive = self.labels[-1]
        true_positive = sum(pred == positive and target == positive for pred, target in zip(predictions, targets))
        false_positive = sum(pred == positive and target != positive for pred, target in zip(predictions, targets))
        false_negative = sum(pred != positive and target == positive for pred, target in zip(predictions, targets))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {"loss": float(np.mean(values)) if values else float("inf"), "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def _validate(self, records: list[dict]) -> float:
        return self._validate_metrics(records)["loss"]

    def train(self, train_records: list[dict], valid_records: list[dict], resume: str | None = None) -> Path:
        if not train_records or not valid_records:
            raise ValueError("Both train and validation records are required.")
        training = self.config["training"]
        method = str(self.config.get("method", "vanilla"))
        cf_records = [cf for record in train_records for cf in record.get("counterfactuals", [])]
        positive_cf_records = [cf for cf in cf_records if float(cf.get("loss_weight", 0.0)) > 0.0]
        if method == "vanilla" and cf_records:
            raise ValueError("Vanilla training received counterfactual records; refusing mixed training data.")
        if method != "vanilla" and not positive_cf_records:
            raise ValueError(f"{method} training has no positive-weight counterfactuals; CF join failed.")
        if method != "vanilla":
            print(f"[cf-preflight] records={len(cf_records)} positive_records={len(positive_cf_records)} mean_weight={np.mean([float(cf.get('loss_weight', 0.0)) for cf in positive_cf_records]):.6f}", flush=True)
        epochs = int(training["epochs"])
        max_steps = training.get("max_steps")
        max_steps = int(max_steps) if max_steps else None
        start_optimizer_step = 0
        if resume:
            checkpoint = Path(resume)
            required = (checkpoint / "adapter", checkpoint / "optimizer.pt", checkpoint / "state.json")
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Incomplete resume checkpoint: {missing}")
            self.model.load_adapter(str(checkpoint / "adapter"), adapter_name="resume", is_trainable=True)
            self.model.set_adapter("resume")
            if "default" in getattr(self.model, "peft_config", {}) and hasattr(self.model, "delete_adapter"):
                self.model.delete_adapter("default")
            self.optimizer = self.torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad),
                lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
            )
            self.optimizer.load_state_dict(self.torch.load(checkpoint / "optimizer.pt", map_location=self.device, weights_only=True))
            start_optimizer_step = int(json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))["step"])
        log_path = self.run_dir / "training_log.jsonl"
        best_loss, best_path = float("inf"), None
        batches_per_epoch = (len(train_records) + self.batch_size - 1) // self.batch_size
        optimizer_steps_per_epoch = (batches_per_epoch + self.grad_accumulation - 1) // self.grad_accumulation
        if resume:
            if start_optimizer_step % optimizer_steps_per_epoch != 0:
                raise ValueError("Resume checkpoint is not at an epoch boundary; refusing ambiguous continuation.")
            completed_epochs = start_optimizer_step // optimizer_steps_per_epoch
            if completed_epochs >= epochs:
                raise ValueError(f"Resume checkpoint already covers {completed_epochs} epochs; target is {epochs}.")
        else:
            completed_epochs = 0
        start_micro_step = completed_epochs * batches_per_epoch
        micro_step = start_micro_step
        optimizer_step = start_optimizer_step
        real_optimizer_steps = start_optimizer_step
        if resume and (self.run_dir / "checkpoint_info.json").is_file():
            previous = json.loads((self.run_dir / "checkpoint_info.json").read_text(encoding="utf-8"))
            best_loss = float(previous.get("best_validation_loss", float("inf")))
            previous_best = previous.get("best")
            if previous_best and Path(previous_best).is_dir():
                best_path = Path(previous_best)
        total_micro_steps = min(epochs * batches_per_epoch, max_steps) if max_steps else epochs * batches_per_epoch
        total_optimizer_steps = sum(
            (min(batches_per_epoch, max(0, total_micro_steps - epoch * batches_per_epoch)) + self.grad_accumulation - 1) // self.grad_accumulation
            for epoch in range(epochs)
            if epoch * batches_per_epoch < total_micro_steps
        )
        started = time.monotonic()
        print(f"[train] start epochs={epochs} train={len(train_records)} valid={len(valid_records)} batch_size={self.batch_size} gradient_accumulation={self.grad_accumulation} effective_batch={self.samples_per_update} total_micro_steps={total_micro_steps} total_optimizer_steps={total_optimizer_steps}", flush=True)
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(completed_epochs, epochs):
            print(f"[train] epoch={epoch + 1}/{epochs}", flush=True)
            order = list(range(len(train_records)))
            random.Random(int(self.config["seed"]) + epoch).shuffle(order)
            epoch_loss = []
            for batch_start in range(0, len(order), self.batch_size):
                if max_steps and micro_step >= max_steps:
                    break
                batch_indices = order[batch_start:batch_start + self.batch_size]
                batch_loss_values, batch_stats = [], []
                for index in batch_indices:
                    loss, stats = self._record_loss(train_records[index])
                    batch_loss_values.append(float(loss.detach()))
                    batch_stats.append(stats)
                    # Backpropagate each sample immediately so a batch does not
                    # retain eight Qwen multimodal graphs at once.
                    (loss / (len(batch_indices) * self.grad_accumulation)).backward()
                loss_value = float(np.mean(batch_loss_values))
                stats = {key: float(np.mean([item[key] for item in batch_stats if isinstance(item.get(key), (int, float))])) for key in ("original", "counterfactual", "num_cf_used", "mean_cf_weight", "max_cf_weight", "min_cf_weight", "text_cf_weight_sum", "image_cf_weight_sum")}
                stats["category"] = None
                epoch_loss.append(loss_value)
                micro_step += 1
                update = micro_step % self.grad_accumulation == 0 or batch_start + self.batch_size >= len(order)
                grad_norm = None
                if update:
                    grad_norm = float(self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    real_optimizer_steps += 1
                elapsed = time.monotonic() - started
                throughput = micro_step / max(elapsed, 1e-8)
                remaining = max(0, total_micro_steps - micro_step)
                row = {"epoch": epoch + 1, "global_step": micro_step, "micro_step": micro_step, "optimizer_step": optimizer_step, "real_optimizer_steps": real_optimizer_steps, "total_micro_steps": total_micro_steps, "total_optimizer_steps": total_optimizer_steps, "sample_id": train_records[batch_indices[-1]]["sample_id"], "loss_total": loss_value, "loss_original": stats["original"], "loss_cf": stats["counterfactual"], **{key: stats[key] for key in ("num_cf_used", "mean_cf_weight", "max_cf_weight", "min_cf_weight", "text_cf_weight_sum", "image_cf_weight_sum", "category")}, "grad_norm": grad_norm, "learning_rate": self.optimizer.param_groups[0]["lr"], "update": update, "processed_samples": (batch_start + len(batch_indices)), "elapsed": elapsed, "throughput": throughput, "remaining_steps": remaining, "eta_seconds": remaining / max(throughput, 1e-8)}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if micro_step == 1 or micro_step % self.status_interval == 0:
                    print(
                        f"[train] epoch={epoch + 1}/{epochs} micro_step={micro_step}/{total_micro_steps} optimizer_step={optimizer_step}/{total_optimizer_steps} "
                        f"loss={loss_value:.4f} task_loss={stats['original']:.4f} cf_loss={stats['counterfactual']:.4f} "
                        f"cf_count={stats['num_cf_used']} mean_weight={stats['mean_cf_weight']:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e} "
                        f"grad_norm={grad_norm} elapsed={elapsed:.1f}s ETA={row['eta_seconds']:.1f}s",
                        flush=True,
                    )
                if update and optimizer_step % self.checkpoint_interval == 0:
                    self._save(optimizer_step)
                    print(f"[train] saved checkpoint at optimizer_step={optimizer_step}", flush=True)
            if max_steps and micro_step >= max_steps:
                break
            print(f"[train] epoch={epoch + 1}/{epochs} complete train_loss={float(np.mean(epoch_loss)):.4f}", flush=True)
            print(f"[validation] epoch={epoch + 1}/{epochs}", flush=True)
            validation = self._validate_metrics(valid_records)
            validation_loss = validation["loss"]
            with (self.run_dir / "validation_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch + 1, "micro_step": micro_step, "optimizer_step": optimizer_step, "train_loss_mean": float(np.mean(epoch_loss)), "val_loss": validation_loss, "val_accuracy": validation["accuracy"], "val_precision": validation["precision"], "val_recall": validation["recall"], "val_f1": validation["f1"], "best_so_far": validation_loss < best_loss, "checkpoint_path": None}) + "\n")
            (self.run_dir / "epoch_metrics.json").write_text(json.dumps({"epoch": epoch + 1, "train_loss_mean": float(np.mean(epoch_loss)), "val_loss": validation_loss, **{key: validation[key] for key in ("accuracy", "precision", "recall", "f1")}, "micro_step": micro_step, "optimizer_step": optimizer_step}, indent=2), encoding="utf-8")
            print(f"[validation] epoch={epoch + 1}/{epochs} train_loss={float(np.mean(epoch_loss)):.4f} val_loss={validation_loss:.4f} val_accuracy={validation['accuracy']:.4f} val_f1={validation['f1']:.4f}", flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_path = self._save(optimizer_step, "best")
                print(f"[checkpoint] saved best={best_path} validation_loss={best_loss:.4f}", flush=True)
        if best_path is None:
            validation_loss = self._validate_metrics(valid_records)["loss"]
            with (self.run_dir / "validation_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epochs, "micro_step": micro_step, "optimizer_step": optimizer_step, "validation_loss": validation_loss}) + "\n")
            best_loss = validation_loss
            best_path = self._save(optimizer_step, "best")
        final = self._save(optimizer_step, "last")
        if best_path is None:
            best_path = final
        (self.run_dir / "checkpoint_info.json").write_text(json.dumps({"last": str(final), "best": str(best_path), "best_validation_loss": best_loss, "micro_steps": micro_step, "optimizer_steps": optimizer_step, "real_optimizer_steps": real_optimizer_steps, "total_micro_steps": total_micro_steps, "total_optimizer_steps": total_optimizer_steps}, indent=2), encoding="utf-8")
        print(f"[train] done best={best_path} last={final} micro_steps={micro_step} optimizer_steps={optimizer_step} real_optimizer_steps={real_optimizer_steps}", flush=True)
        return best_path
