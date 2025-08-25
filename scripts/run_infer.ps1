param(
  [string]$InputPath = "data/sample_inputs.jsonl",
  [string]$OutputPath = "outputs/sample_outputs.jsonl",
  [string]$ModelId = $env:MODEL_ID
)

python -m src.infer -i $InputPath -o $OutputPath -m $ModelId

