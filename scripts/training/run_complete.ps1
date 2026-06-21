# SYON — PIPELINE COMPLETO DE TREINAMENTO MASTER
# Um comando: dados → processamento → 3 fases → export → eval
param(
    [string]$Config = "training/configs/master_complete.yaml",
    [int]$MinSamples = 10000,
    [string]$ModelName = "syon-master-lora"
)

$ErrorActionPreference = "Stop"
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  SYON — TREINAMENTO MASTER COMPLETO" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "[1/6] Gerando curriculum master (min $MinSamples amostras)..." -ForegroundColor Yellow
python scripts/data/build_master_curriculum.py --min-samples $MinSamples

Write-Host "[2/6] Processando splits train/val/test..." -ForegroundColor Yellow
python scripts/data/process_pipeline.py

Write-Host "[3/6] Fase 1 — Foundation..." -ForegroundColor Yellow
python -m training.master_trainer --config $Config --phase 1 --augment 1

Write-Host "[4/6] Fase 2 — Security Mastery..." -ForegroundColor Yellow
python -m training.master_trainer --config $Config --phase 2 --resume training/checkpoints/phase1/best --augment 1

Write-Host "[5/6] Fase 3 — Architecture Mastery..." -ForegroundColor Yellow
python -m training.master_trainer --config $Config --phase 3 --resume training/checkpoints/phase2/best --augment 1

Write-Host "[6/6] Exportando modelo..." -ForegroundColor Yellow
$src = "models/pretrained/syon-master-lora"
if (-not (Test-Path $src)) { $src = "training/checkpoints/phase3/best" }
python scripts/training/export_model.py --source $src --name $ModelName

Write-Host "[+] Avaliacao..." -ForegroundColor Yellow
python -m evaluation.benchmarks.benchmark_runner --output training/logs/master_eval.json 2>$null

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  TREINAMENTO COMPLETO FINALIZADO" -ForegroundColor Green
Write-Host "  Modelo: models/pretrained/$ModelName" -ForegroundColor Green
Write-Host "  Logs:   training/logs/master_summary.json" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green