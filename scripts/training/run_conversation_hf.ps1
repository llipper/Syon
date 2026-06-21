# Syon 3 — conversar + raciocinar (HF só infraestrutura)
param(
    [string]$Config = "training/configs/syon3_hf.yaml",
    [string]$Resume = "kl",
    [switch]$ImportData
)

$ErrorActionPreference = "Stop"

if ($ImportData) {
    Write-Host "[Syon 3] Importando conversacao PT..." -ForegroundColor Yellow
    python scripts/data/import_conversation_datasets.py --aira-limit 50000 --ultrachat-limit 100000
}

Write-Host "[Syon 3/HF] Treino conversa + raciocinio..." -ForegroundColor Cyan
$trainArgs = @("-m", "training.hf.syon3_hf_trainer", "--config", $Config, "--resume", $Resume)
python @trainArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Modelo: models/pretrained/syon-3" -ForegroundColor Green