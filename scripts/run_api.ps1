$env:SYON_API_HOST = if ($env:SYON_API_HOST) { $env:SYON_API_HOST } else { "0.0.0.0" }
$env:SYON_API_PORT = if ($env:SYON_API_PORT) { $env:SYON_API_PORT } else { "8000" }
$env:SYON_MODELS_DIR = if ($env:SYON_MODELS_DIR) { $env:SYON_MODELS_DIR } else { "./models" }

pip install -e ".[dev]" -q
syon-api