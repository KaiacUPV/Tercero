param(
    [string]$Python = "py"
)

$ErrorActionPreference = "Stop"

Write-Host "[V3D] Creando/activando entorno virtual (.venv)..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    & $Python -3 -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install -U pip
Write-Host "[V3D] Instalando dependencias (requirements.txt)..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "[V3D] OK. Para ejecutar:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe main.py" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe main_id.py" -ForegroundColor Green

