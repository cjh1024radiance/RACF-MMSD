from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline"))

from racf.io import append_jsonl, read_jsonl


PROMPT_VERSION = "beyond_text_same_intervention_v3"
PROMPT = (
    "Identify ONE semantic intervention target in the original tweet. First choose the most reliable anchor span, then provide up to three "
    "natural textual realizations of that SAME intervention. Do NOT select three independent important words. The realizations may be identical, "
    "partially overlap, be nested, or differ only in contextual boundaries. Do not force lexical diversity or choose unrelated semantic targets. "
    "If only one reliable target exists, return it once; the program may repeat it. Every span must be an exact contiguous substring copied verbatim from the tweet. "
    "Return only JSON in this exact form: "
    '{{"intervention_intent": "very short semantic description", "anchor_span": "most reliable exact span", "realizations": ["span 1", "span 2", "span 3"]}}. '
    "Tweet: {text}"
)


def cache_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_candidates(response: str, source: str) -> tuple[str, str, list[str], str]:
    payload = None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            payload, _ = decoder.raw_decode(response[match.start():])
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        return "", "", [], "no_json_object"
    if not isinstance(payload, dict):
        return "", "", [], "not_object"
    intervention_intent = str(payload.get("intervention_intent", payload.get("core_intervention", ""))).strip()
    anchor_span = str(payload.get("anchor_span", payload.get("core_intervention", ""))).strip()
    values = payload.get("realizations")
    if not intervention_intent or not isinstance(values, list):
        return "", "", [], "missing_intent_or_realizations"
    if not anchor_span and values:
        anchor_span = str(values[0]).strip()
    if not anchor_span or anchor_span not in source:
        anchor_span = ""
    candidates = []
    for value in values:
        span = str(value).strip()
        if span and span in source:
            candidates.append(span)
    if not candidates:
        return "", "", [], "no_source_realizations"
    if not anchor_span:
        anchor_span = max(candidates, key=len)
    while len(candidates) < 3:
        candidates.append(candidates[-1])
    return intervention_intent, anchor_span, candidates[:3], "ok"


def mask_one_exact_occurrence(source: str, span: str) -> str:
    exact_matches = list(re.finditer(re.escape(span), source))
    if not exact_matches:
        raise ValueError(f"semantic span is not present verbatim: {span!r}")
    token_matches = [
        match for match in exact_matches
        if (match.start() == 0 or not source[match.start() - 1].isalnum())
        and (match.end() == len(source) or not source[match.end()].isalnum())
    ]
    match = token_matches[0] if token_matches else exact_matches[0]
    return source[:match.start()] + "[MASK]" + source[match.end():]


def make_record(row: dict, sample_id: str, seed: int, response: str, intervention_intent: str, anchor_span: str, selected: list[str], realization_id: int, parse_status: str) -> dict:
    span = selected[realization_id - 1]
    cf_text = mask_one_exact_occurrence(row["text"], span)
    distinct = len(set(selected))
    return {"sample_id": sample_id, "split": row.get("split"), "label": row.get("label"), "label_name": row.get("label_name"), "original_text": row["text"], "original_image": row.get("image_path"), "intervention_type": "text", "cf_id": f"{sample_id}_text_{realization_id:02d}", "cf": cf_text, "cf_text": cf_text, "semantic_text": [span], "semantic_span": span, "intervention_intent": intervention_intent, "anchor_span": anchor_span, "core_intervention": intervention_intent, "realizations": selected, "realization_family": selected, "realization_diversity": distinct, "realization_id": realization_id, "seed": seed, "generation_seed": seed, "generation_model": "Qwen2-VL-7B-Instruct-text-only", "prompt_version": PROMPT_VERSION, "model_revision": "local_checkpoint", "generation_status": "success", "alignment_status": "passed", "audit_status": "pending", "raw_response": response, "parse_status": parse_status, "num_candidates": len(selected), "num_unique_spans": distinct, "num_unique_cf": len(set(selected)), "num_realizations": len(selected), "clue_parse_method": "model_same_intervention_family"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate distinct Beyond-style Text-CF realizations using Qwen text-only extraction.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="data/formal/mmsd/text_multispan_v3.generated.jsonl")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=4200)
    parser.add_argument("--interventions", choices=["text"], default="text")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.k != 3:
        raise SystemExit("Formal RACF uses exactly --k 3.")
    output = Path(args.output)
    if output.name == "text_multispan.generated.jsonl":
        raise SystemExit(
            "The v3 generator must not reuse the legacy cache. Use "
            "data/formal/mmsd/text_multispan_v3.generated.jsonl."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=True)
    completed = {str(row["cf_id"]) for row in read_jsonl(output)}
    failures = output.with_name(f"{output.stem}.failures.jsonl")
    rows = [row for row in read_jsonl(args.manifest) if any(f"{row['sample_id']}_text_{index:02d}" not in completed for index in range(1, args.k + 1))]
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    selection_stats = {"aaa": 0, "aab": 0, "abb": 0, "abc": 0, "final_k3": 0}
    diagnostics = {"parse_failures": 0, "validation_failures": 0, "skipped_samples": 0, "first_raw_responses": [], "first_parser_errors": []}
    print(
        f"[setup] generation_model=Qwen2-VL-7B-Instruct text_only=true "
        f"prompt_version={PROMPT_VERSION} cache={output.resolve()} pending_samples={len(rows)}",
        flush=True,
    )
    processed_samples = 0
    generated_rows = len(completed)
    for index, row in enumerate(tqdm(rows, desc="formal_text_multispan_cf"), start=1):
        sample_id, seed = str(row["sample_id"]), args.base_seed + index
        try:
            set_seed(seed)
            prompt = PROMPT.format(text=row["text"])
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[rendered], padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            intervention_intent, anchor_span, candidates, parse_status = parse_candidates(response, row["text"])
            if len(candidates) < args.k:
                diagnostics["parse_failures"] += 1
                diagnostics["skipped_samples"] += 1
                if len(diagnostics["first_raw_responses"]) < 3:
                    diagnostics["first_raw_responses"].append(response)
                if len(diagnostics["first_parser_errors"]) < 3:
                    diagnostics["first_parser_errors"].append(parse_status)
                append_jsonl(failures, {"sample_id": sample_id, "generation_status": "insufficient_realizations", "parse_status": parse_status, "num_candidates": len(candidates), "raw_response": response})
                continue
            selected = candidates[:args.k]
            distinct = len(set(selected))
            if distinct == 1:
                selection_stats["aaa"] += 1
            elif distinct == 2:
                pattern = "aab" if selected[0] == selected[1] else "abb" if selected[1] == selected[2] else "aab"
                selection_stats[pattern] += 1
            else:
                selection_stats["abc"] += 1
            selection_stats["final_k3"] += 1
            for realization_id, span in enumerate(candidates[:args.k], start=1):
                cf_id = f"{sample_id}_text_{realization_id:02d}"
                if cf_id in completed:
                    continue
                append_jsonl(output, make_record(row, sample_id, seed, response, intervention_intent, anchor_span, selected, realization_id, parse_status))
                completed.add(cf_id)
                generated_rows += 1
        except Exception as error:
            diagnostics["validation_failures"] += 1
            if len(diagnostics["first_parser_errors"]) < 3:
                diagnostics["first_parser_errors"].append(f"{type(error).__name__}: {error}")
            append_jsonl(failures, {"sample_id": sample_id, "generation_status": "failed", "error": f"{type(error).__name__}: {error}", "raw_response": locals().get("response", "")})
        processed_samples += 1
        elapsed = max(time.perf_counter() - started, 1e-6)
        if index % 20 == 0 or index == len(rows):
            print(f"[perf] prompt_version={PROMPT_VERSION} processed_samples={processed_samples}/{len(rows)} generated_rows={generated_rows} samples_per_sec={index / elapsed:.3f} eta_sec={(len(rows) - index) / max(index / elapsed, 1e-6):.0f}", flush=True)
            if index >= 20 and generated_rows == 0:
                print(json.dumps({"fail_fast": True, **diagnostics}, ensure_ascii=False, indent=2), flush=True)
                raise SystemExit("No generated rows after 20 processed samples; stopped to avoid wasting inference time.")
    metadata = {"k": args.k, "intervention": "text", "generation_model": "Qwen2-VL-7B-Instruct-text-only", "prompt_version": PROMPT_VERSION, "cache": str(output.resolve()), "count": len(read_jsonl(output)), "cf_cache_hash": cache_hash(output), "selection_stats": selection_stats, "diagnostics": diagnostics, "total_samples": len(read_jsonl(args.manifest)), "processed_samples": processed_samples, "generated_rows": generated_rows, "selection_rule": "K=3 exact realizations from one intervention intent; AAA allowed"}
    output.with_name(f"{output.stem}.metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
