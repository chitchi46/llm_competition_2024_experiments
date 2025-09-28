#!/usr/bin/env bash
set -euo pipefail

# Example:
#   GEMINI_API_KEY=... bash scripts/run_eval_gemini.sh \
#     outputs/qwen3_out.jsonl eval/gemini_eval.jsonl 100

INPUT_PATH=${1:-outputs/sample_outputs.jsonl}
OUTPUT_PATH=${2:-eval/gemini_eval.jsonl}
MAX_RECORDS=${3:-}

CMD=(python -m src.eval_gemini --input "$INPUT_PATH" --output "$OUTPUT_PATH")
if [[ -n "${MAX_RECORDS}" ]]; then
  CMD+=(--max-records "$MAX_RECORDS")
fi

"${CMD[@]}"


