"""Core training engine."""

from training.core.distributed import DistributedTrainer, gather_results, setup_ddp, sync_gradients
from training.core.gradient_accumulation import accumulate_gradients, effective_batch_size, update_with_accumulation
from training.core.loss_functions import (
    ContrastiveLoss,
    SecurityAwareLoss,
    code_quality_loss,
    cross_entropy_loss,
    custom_loss_weighted,
    perplexity_calculation,
    security_aware_loss,
)
from training.core.mixed_precision import MixedPrecisionTrainer
from training.core.optimization import AdamWScheduler, cosine_annealing, linear_warmup, update_lr
from training.core.trainer import Trainer, load_training_config

__all__ = [
    "Trainer",
    "load_training_config",
    "DistributedTrainer",
    "setup_ddp",
    "sync_gradients",
    "gather_results",
    "AdamWScheduler",
    "linear_warmup",
    "cosine_annealing",
    "update_lr",
    "cross_entropy_loss",
    "security_aware_loss",
    "code_quality_loss",
    "custom_loss_weighted",
    "perplexity_calculation",
    "SecurityAwareLoss",
    "ContrastiveLoss",
    "MixedPrecisionTrainer",
    "accumulate_gradients",
    "effective_batch_size",
    "update_with_accumulation",
]