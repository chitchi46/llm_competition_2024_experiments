#!/usr/bin/env bash
set -euo pipefail

# Example:
#   ADAPTER_ID=user/repo MODEL_ID=Qwen/Qwen3-1.7B-Base \
#   bash scripts/run_infer_lora.sh data/elyza-tasks-100-TV_0.jsonl outputs/qwen3_out.jsonl

INPUT_PATH=${1:-data/sample_inputs.jsonl}
OUTPUT_PATH=${2:-outputs/sample_outputs_lora.jsonl}

python -m src.infer \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --model-id "${MODEL_ID:-Qwen/Qwen3-1.7B-Base}" \
  --adapter-id "${ADAPTER_ID:-}" \
  ${USE_UNSLOTH:+--use-unsloth}


