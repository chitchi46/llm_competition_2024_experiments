Param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Activate venv if present (Windows PowerShell host)
$venvActivate = Join-Path -Path $PSScriptRoot -ChildPath '..\.venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    . $venvActivate
}

uvicorn src.app:app --reload --host 0.0.0.0 --port 8000


