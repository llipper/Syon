param(
    [string]$Config = "training/configs/syon3.yaml",
    [int]$Phase = 0,
    [int]$Augment = 3
)

$ErrorActionPreference = "Stop"
Write-Host "[Syon 3] Gerando curriculum..." -ForegroundColor Yellow
python scripts/data/build_syon3_curriculum.py --augment $Augment

$args = @("-m", "training.syon3_trainer", "--config", $Config, "--augment", "$Augment")
if ($Phase -gt 0) { $args += @("--phase", "$Phase") }
python @args

Write-Host "[Syon 3] Concluido. Verifique training/logs/syon3_summary.json" -ForegroundColor Green