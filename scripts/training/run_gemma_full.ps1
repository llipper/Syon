# Syon + Gemma 2 - pipeline FULL local
# Uso: .\scripts\training\run_gemma_full.ps1
#      .\scripts\training\run_gemma_full.ps1 -TrainOnly

param(
    [switch]$TrainOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root

if (-not $TrainOnly) {
    Write-Host "=== [1/4] Curriculum 10.000+ ===" -ForegroundColor Cyan
    python scripts/data/build_master_curriculum.py --output data/raw --min-samples 10000 --augment 5

    Write-Host "=== [2/4] Conversacao PT (local ou HuggingFace) ===" -ForegroundColor Cyan
    python scripts/data/import_conversation_datasets.py --output data/raw/conversation --aira-limit 50000 --ultrachat-limit 100000

    Write-Host "=== [3/4] Dataset Gemma SFT FULL ===" -ForegroundColor Cyan
    python scripts/data/build_gemma_sft_dataset.py --raw data/raw --output data/gemma_sft --min-curriculum 10000 --no-conversation-import
} else {
    Write-Host "=== TrainOnly: pulando preparacao de dados ===" -ForegroundColor Yellow
}

Write-Host "=== [4/4] Treino Gemma 5 fases ===" -ForegroundColor Cyan
$env:CUDA_VISIBLE_DEVICES = "0"
python scripts/kaggle/kaggle_gemma_full_train.py --config training/configs/syon_gemma_full_local.yaml

Write-Host ""
Write-Host "Concluido -> models/pretrained/syon-gemma-full/syon-gemma-full" -ForegroundColor Green