param(
    [string]$Config = "training/configs/master_full.yaml",
    [int]$Phase = 0,
    [int]$Augment = 50,
    [string]$Resume = ""
)

Write-Host "=== Syon Master Training ===" -ForegroundColor Cyan

Write-Host "[1/3] Gerando curriculum master/senior..."
python scripts/data/build_master_curriculum.py --augment $Augment

$args = @("-m", "training.master_trainer", "--config", $Config, "--augment", $Augment)
if ($Phase -gt 0) { $args += @("--phase", $Phase) }
if ($Resume) { $args += @("--resume", $Resume) }

Write-Host "[2/3] Iniciando treino (fases)..."
python @args

Write-Host "[3/3] Concluido. Verifique training/logs/master_summary.json"