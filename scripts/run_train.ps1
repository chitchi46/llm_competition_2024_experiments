Param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$env:XFORMERS_DISABLED = '1'
$env:USE_FLASH_ATTENTION = '0'

python -m src.train


