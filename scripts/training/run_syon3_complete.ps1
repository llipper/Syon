# Syon 3 — pipeline local completo + export
param(
    [string]$Config = "training/configs/syon3.yaml",
    [int]$MinSamples = 10000,
    [string]$ModelName = "syon-3"
)

$ErrorActionPreference = "Stop"
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  SYON 3 — TREINAMENTO LOCAL" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

python -m training.pipeline --config $Config --min-samples $MinSamples

$src = "models/pretrained/syon-3"
if (-not (Test-Path $src)) { $src = "training/checkpoints/phase4/best" }
python scripts/training/export_model.py --source $src --name $ModelName

python -m evaluation.benchmarks.benchmark_runner --output training/logs/syon3_eval.json 2>$null

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  SYON 3 FINALIZADO" -ForegroundColor Green
Write-Host "  Modelo: models/pretrained/$ModelName" -ForegroundColor Green
Write-Host "  Logs:   training/logs/syon3_summary.json" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green