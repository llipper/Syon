param(
    [string]$Config = "configs/training/parallel_sft.yaml",
    [string]$DataDir = "data/raw"
)

python -m training.parallel_trainer --config $Config --data-dir $DataDir