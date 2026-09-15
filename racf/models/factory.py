from __future__ import annotations


def create_probability_runner(model: dict, prompt_template: str, inference: dict):
    backend = model["backend"]
    if backend == "qwen":
        from racf.qwen import QwenProbabilityRunner

        options = dict(inference)
        options.setdefault("labels", tuple(model.get("labels", ("negative", "neutral", "positive"))))
        if model.get("adapter_checkpoint"):
            options["adapter_checkpoint"] = model["adapter_checkpoint"]
        return QwenProbabilityRunner(model["path"], prompt_template, options)
    raise ValueError(f"This compact release supports only the Qwen backend, not {backend!r}.")
