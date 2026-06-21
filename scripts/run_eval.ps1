param([string]$Output = "")

if ($Output) {
    python -m eval.runner --output $Output
} else {
    python -m eval.runner
}