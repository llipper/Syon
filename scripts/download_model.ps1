param(
    [string]$Model = "syon-7b",
    [string]$Dest = "./models"
)

$file = switch ($Model) {
    "syon-7b"  { "syon-7b.gguf" }
    "syon-13b" { "syon-13b.gguf" }
    "syon-70b" { "syon-70b.gguf" }
    default { throw "Modelo desconhecido: $Model" }
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Write-Host "Download: https://huggingface.co/syon-ai/$Model -> $Dest/$file"
Write-Host "Use: huggingface-cli download syon-ai/$Model $file --local-dir $Dest"