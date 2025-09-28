Param(
  [string]$InputPath = "outputs/sample_outputs.jsonl",
  [string]$OutputPath = "eval/gemini_eval.jsonl",
  [int]$MaxRecords
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$argsList = @('-m','src.eval_gemini','--input', $InputPath, '--output', $OutputPath)
if ($PSBoundParameters.ContainsKey('MaxRecords')) {
  $argsList += @('--max-records', $MaxRecords)
}

python @argsList


