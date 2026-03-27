#!/bin/bash
# Run QLoRA training for Maasai-English translation
# Usage: bash training/run_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Maasai-English Translation QLoRA Training ==="
echo "Project root: $PROJECT_ROOT"
echo "Start time: $(date)"
echo ""

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/final_v3/train.jsonl}"
VALID_FILE="${VALID_FILE:-data/final_v3/valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/maasai-en-mt-qlora}"

echo "Python: $PYTHON_BIN"
echo "Model: $MODEL_NAME"
echo "Train file: $TRAIN_FILE"
echo "Valid file: $VALID_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""

"$PYTHON_BIN" scripts/train_qlora.py \
  --model_name "$MODEL_NAME" \
  --train_file "$TRAIN_FILE" \
  --valid_file "$VALID_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --max_length 512 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --seed 42 \
  "$@"

echo ""
echo "=== Training complete ==="
echo "End time: $(date)"
