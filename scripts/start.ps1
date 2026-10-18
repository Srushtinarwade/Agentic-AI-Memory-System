# After completing README setup, start the local API (Ctrl+C for graceful shutdown).
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Error 'Python environment missing. Follow README.md Quick start first.'
    exit 1
}
& .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1 --timeout-graceful-shutdown 30
