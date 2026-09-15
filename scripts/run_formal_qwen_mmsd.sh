#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

METHOD="${1:?Usage: $0 vanilla|uniform|magnitude|racf|shuffled_racf [run_name] [screened_cf] [reliability]}"
RUN_NAME="${2:-qwen_mmsd_${METHOD}_$(date -u +%Y%m%d_%H%M%S)}"
CF_CACHE="${3:-}"
RELIABILITY="${4:-}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

case "$METHOD" in
  vanilla|uniform|magnitude|racf|shuffled_racf) ;;
  *) echo "Unsupported method: $METHOD" >&2; exit 1 ;;
esac

CONFIG="configs/formal/qwen_mmsd_${METHOD}.yaml"
TRAIN_MANIFEST="${FORMAL_TRAIN_MANIFEST:-data/formal/mmsd/formal_train_8k.jsonl}"
VALID_MANIFEST="${FORMAL_VALID_MANIFEST:-data_manifests/mmsd2/valid.jsonl}"
RUN_DIR="results/formal/mmsd/qwen/${METHOD}/${RUN_NAME}"
DATA_DIR="$RUN_DIR/data"

test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 1; }
test -f data_manifests/mmsd2/train.jsonl || { echo "Missing MMSD2.0 train manifest." >&2; exit 1; }
test -f "$TRAIN_MANIFEST" || { echo "Missing formal train subset: $TRAIN_MANIFEST" >&2; exit 1; }
test -f "$VALID_MANIFEST" || { echo "Missing validation manifest: $VALID_MANIFEST" >&2; exit 1; }
test -f data_manifests/mmsd2/test.jsonl || { echo "Missing test manifest." >&2; exit 1; }

echo "[setup] method=$METHOD seed=$SEED run_dir=$RUN_DIR"
if [[ "$METHOD" == "vanilla" ]]; then
  python -u scripts/prepare_formal_training_data.py \
    --train "$TRAIN_MANIFEST" --valid "$VALID_MANIFEST" --output-dir "$DATA_DIR" \
    --method vanilla --labels non-sarcastic sarcastic
else
  test -f "$CF_CACHE" || { echo "A screened CF cache is required." >&2; exit 1; }
  test -f "$RELIABILITY" || { echo "A frozen reliability cache is required." >&2; exit 1; }
  python -u scripts/build_formal_multicf_training.py \
    --manifest-dir data_manifests/mmsd2 --train-manifest "$TRAIN_MANIFEST" --valid-manifest "$VALID_MANIFEST" \
    --cf-cache "$CF_CACHE" --reliability "$RELIABILITY" --method "$METHOD" \
    --output-dir "$DATA_DIR" --seed "$SEED" --k 3
fi

python -u scripts/train_formal_qwen.py \
  --config "$CONFIG" --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
  --seed "$SEED" --log-interval "$LOG_INTERVAL"
python -u scripts/evaluate_formal_qwen.py --run-dir "$RUN_DIR"
echo "[done] $RUN_DIR"
