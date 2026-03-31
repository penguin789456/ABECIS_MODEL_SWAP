"""
BCEDiceLoss: weighted combination of Binary Cross-Entropy and soft Dice loss.

    L = bce_weight * BCE(logits, target) + dice_weight * (1 - Dice)

Both components operate on raw logits; sigmoid is applied internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Args:
        bce_weight:  Contribution of BCE term (default 0.5)
        dice_weight: Contribution of Dice term (default 0.5)
        smooth:      Laplace smoothing for Dice denominator (default 1.0)
        pos_weight:  Scalar weight for positive (crack) pixels in BCE loss.
                     Values > 1 penalise false negatives more, improving Recall.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self._pos_weight_val = pos_weight  # store scalar; tensor created lazily in forward
        self.bce = nn.BCEWithLogitsLoss()  # fallback (no pos_weight) for CPU / no-pw case

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 1, H, W) raw model output
            targets: (B, 1, H, W) binary ground-truth mask (0 or 1 float)

        Returns:
            Scalar loss value
        """
        if self._pos_weight_val is not None:
            # Create pos_weight on same device/dtype as logits to avoid device mismatch
            pw = logits.new_tensor([self._pos_weight_val])
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        else:
            bce_loss = self.bce(logits, targets)
        dice_loss = self._soft_dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

    def _soft_dice(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()
