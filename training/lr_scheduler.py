"""
WarmupCosineScheduler: linear warmup followed by cosine annealing.

During warmup epochs, LR grows linearly from (lr / warmup_epochs) to lr.
After warmup, CosineAnnealingLR decays LR to eta_min over the remaining epochs.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 1e-6,
) -> SequentialLR:
    """
    Args:
        optimizer:      The wrapped optimizer
        warmup_epochs:  Number of linear warmup epochs (e.g. 5)
        total_epochs:   Total training epochs (e.g. 100)
        eta_min:        Minimum LR at end of cosine decay

    Returns:
        A SequentialLR that applies warmup then cosine annealing.
    """
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs, 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
