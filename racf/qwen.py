from __future__ import annotations

from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


LABELS = ("negative", "neutral", "positive")


class QwenProbabilityRunner:
    def __init__(self, model_path: str, prompt_template: str, inference: dict | None = None) -> None:
        self.inference = inference or {}
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        ).eval()
        adapter_path = self.inference.get("adapter_checkpoint")
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.prompt_template = prompt_template
        self.labels = tuple(self.inference.get("labels", LABELS))
        self.model_revision = getattr(self.model.config, "_commit_hash", None) or "local_checkpoint"

    def predict(self, text: str, frame_paths: list[str] | None = None, prompt_template: str | None = None) -> dict:
        content = [{"type": "image", "image": str(Path(path).resolve())} for path in (frame_paths or [])]
        content.append({"type": "text", "text": (prompt_template or self.prompt_template).format(text=text)})
        messages = [{"role": "user", "content": content}]
        rendered_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        prompt_inputs = self.processor(text=[rendered_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt").to("cuda")
        prompt_ids = prompt_inputs.input_ids[0]
        log_likelihoods = []
        with torch.inference_mode():
            for label in self.labels:
                label_inputs = self.processor(
                    text=[rendered_prompt + label], images=image_inputs, videos=video_inputs, return_tensors="pt"
                ).to("cuda")
                label_ids = label_inputs.input_ids[0]
                target_ids = label_ids[prompt_ids.shape[0] :]
                if not len(target_ids) or not torch.equal(label_ids[: prompt_ids.shape[0]], prompt_ids):
                    raise RuntimeError("Label likelihood prompt prefix mismatch.")
                logits = self.model(**label_inputs).logits[0, prompt_ids.shape[0] - 1 : prompt_ids.shape[0] - 1 + len(target_ids)]
                log_likelihoods.append(torch.log_softmax(logits, dim=-1).gather(1, target_ids.unsqueeze(1)).sum())
            distribution = torch.softmax(torch.stack(log_likelihoods), dim=0).float().cpu().tolist()
            generated = self.model.generate(
                **prompt_inputs,
                max_new_tokens=int(self.inference.get("max_new_tokens", 8)),
                do_sample=bool(self.inference.get("do_sample", False)),
            )
        probabilities = {label: value for label, value in zip(self.labels, distribution)}
        raw_response = self.processor.batch_decode(generated[:, prompt_ids.shape[0] :], skip_special_tokens=True)[0].strip()
        return {
            "raw_response": raw_response,
            "probability_distribution": probabilities,
            "predicted_label": max(probabilities, key=probabilities.get),
            "model_revision": self.model_revision,
        }
