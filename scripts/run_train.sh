#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_train.sh \
#     TRAIN_MODEL_ID=Qwen/Qwen3-1.7B-Base \
#     TRAIN_FILE_LOCAL=data/train.jsonl \
#     TRAIN_OUTPUT_DIR=outputs/train/qwen3-sft

export XFORMERS_DISABLED=1
export USE_FLASH_ATTENTION=0

python -m src.train


