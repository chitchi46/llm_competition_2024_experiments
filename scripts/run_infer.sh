#!/usr/bin/env bash
set -euo pipefail

python -m src.infer \
  --input data/sample_inputs.jsonl \
  --output outputs/sample_outputs.jsonl

