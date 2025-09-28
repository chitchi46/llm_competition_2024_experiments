Param(
  [string]$InputPath = "data/sample_inputs.jsonl",
  [string]$OutputPath = "outputs/sample_outputs_lora.jsonl"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

python -m src.infer `
  --input $InputPath `
  --output $OutputPath `
  --model-id ($env:MODEL_ID | ForEach-Object { if ($_){$_} else {"Qwen/Qwen3-1.7B-Base"} }) `
  --adapter-id ($env:ADAPTER_ID) `
  $(if ($env:USE_UNSLOTH -and $env:USE_UNSLOTH.ToLower() -in @('1','true','yes')) { "--use-unsloth" })


