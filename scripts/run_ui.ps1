Param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

uvicorn src.app:app --reload --host 0.0.0.0 --port 8000


