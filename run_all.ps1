param(
  [int]$epochs = 400,
  [int[]]$seeds = @(1,2),
  [string]$variant = "Proposed"
)

$ErrorActionPreference = "Stop"
Write-Host ">> Activate venv (current session)"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& .\.venv\Scripts\Activate.ps1

Write-Host ">> Run simulate"
.\.venv\Scripts\python.exe -m src.dt_gate.simulate --epochs $epochs --seeds $seeds --variant $variant

Write-Host ">> Make figures"
.\.venv\Scripts\python.exe scripts\make_figs.py --inputs data\outputs\epochs_log.csv --outdir data\outputs

Write-Host ">> Git commit & push"
git add -f data/outputs/epochs_log.csv data/outputs/table_main.csv data/outputs/*.png
git commit -m "results: epochs=$epochs seeds=$($seeds -join ',') variant=$variant" 2>$null
git push

Write-Host "? Done."
