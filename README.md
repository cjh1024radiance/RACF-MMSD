# Reliability-Calibrated Counterfactual Supervision (RACF)

This repository is the compact release accompanying the RACF paper.  It contains only the Qwen2-VL/MMSD2.0 pipeline used to construct frozen response-consistency weights and to train/evaluate the controlled counterfactual objectives.  Legacy MOSI, InternVL, MCTD, exploratory scripts, model checkpoints, data copies, caches, and experiment logs are deliberately excluded.

## Included scope

- one-call Qwen generation of a same-intervention, three-realization Text-CF family;
- deterministic screening of generated counterfactual records;
- frozen-probe inference and RACF response-consistency computation;
- preparation of fixed-subset factual and three-realization counterfactual records;
- Qwen2-VL LoRA training for `vanilla`, `uniform`, `magnitude`, `racf`, and `shuffled_racf`;
- validation-selected, once-only held-out test evaluation and result summarization;

## Installation

Use Python 3.10+ with a CUDA-enabled PyTorch installation compatible with your system, then install the remaining dependencies:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

Place a local Qwen2-VL-7B-Instruct checkpoint at `models/Qwen2-VL-7B-Instruct`, or update the `model.path` field in `configs/formal/*.yaml`.

## Required private/downloaded artifacts

No dataset images, MMSD2.0 manifests, model weights, generated counterfactuals, frozen-probe outputs, or training outputs are redistributed here.  Obtain and use them only under their original licenses and terms.  The expected paths are:

```text
data_manifests/mmsd2/{train,valid,test}.jsonl
data/formal/mmsd/formal_train_8k.jsonl
data/formal/mmsd/text_multispan_v3.screened.jsonl
data/formal/mmsd/qwen/probe/text_multispan_v3_reliability.jsonl
models/Qwen2-VL-7B-Instruct/
```

Manifest rows must retain the original `sample_id`, `label`, `label_name`, `text`, and valid `image_path` fields.  Do not upload dataset images, model weights, generated caches, or results unless their licenses explicitly permit redistribution.

## End-to-end workflow

The workflow has six ordered stages.  Stages 1--4 are performed once; stages 5--6 are repeated only for the controlled training objectives.  Each stage writes a distinct JSONL artifact, so an interrupted run can resume without silently overwriting a completed artifact.

```bash
# 1. Generate K=3 realizations for one semantic intervention per source sample.
python scripts/generate_formal_mmsd_multicf.py \
  --manifest data_manifests/mmsd2/train.jsonl \
  --model-path models/Qwen2-VL-7B-Instruct \
  --output data/formal/mmsd/text_multispan_v3.generated.jsonl --k 3

# 2. Apply the deterministic formal screening contract.
python scripts/screen_formal_multicf.py \
  --input data/formal/mmsd/text_multispan_v3.generated.jsonl \
  --output data/formal/mmsd/text_multispan_v3.screened.jsonl

# 3. Freeze Qwen2-VL responses for the factual samples and screened CFs.
python scripts/probe_formal_mmsd_multicf.py \
  --manifest data_manifests/mmsd2/train.jsonl \
  --cf-cache data/formal/mmsd/text_multispan_v3.screened.jsonl \
  --model-path models/Qwen2-VL-7B-Instruct \
  --output data/formal/mmsd/qwen/probe/text_multispan_v3_probe.jsonl \
  --intervention text

# 4. Convert frozen response shifts into detached RACF weights.
python scripts/build_formal_multicf_reliability.py \
  --probe data/formal/mmsd/qwen/probe/text_multispan_v3_probe.jsonl \
  --output data/formal/mmsd/qwen/probe/text_multispan_v3_reliability.jsonl \
  --tau 1.0 --k 3 --intervention text
```

## Train and evaluate

All methods must share the same fixed training subset, validation manifest, screened CF cache, and frozen reliability cache.  The launcher prepares immutable run data, selects the best checkpoint by validation loss, and evaluates that checkpoint once on the test split.

```bash
export FORMAL_TRAIN_MANIFEST=data/formal/mmsd/formal_train_8k.jsonl
export FORMAL_VALID_MANIFEST=data_manifests/mmsd2/valid.jsonl
export CF=data/formal/mmsd/text_multispan_v3.screened.jsonl
export REL=data/formal/mmsd/qwen/probe/text_multispan_v3_reliability.jsonl
export SEED=42

bash scripts/run_formal_qwen_mmsd.sh racf racf_seed42 "$CF" "$REL"
```

For the other controls, replace `racf` with `vanilla`, `uniform`, `magnitude`, or `shuffled_racf`.  The `vanilla` command does not require the last two arguments.  Each run writes configuration, split hashes, validation logs, checkpoints, predictions, and final test metrics under `results/formal/mmsd/qwen/`.

## Result summary

```bash
python scripts/summarize_formal_mmsd.py \
  --runs-root results/formal/mmsd/qwen \
  --reliability data/formal/mmsd/qwen/probe/text_multispan_v3_reliability.jsonl \
  --output results/formal/final_paper_results
```

## Repository layout

```text
configs/formal/  Qwen2-VL objective configurations
racf/            model, trainer, metrics, evaluation, and RACF-weight code
scripts/         generation, screening, probing, preparation, training, evaluation, summary
```

## Citation and release

Use `CITATION.cff` when citing this codebase.  The repository intentionally does not assign a software license on behalf of the authors; select and add the appropriate license before making a public release.
