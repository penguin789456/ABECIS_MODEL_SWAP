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


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Soft Dice for class-imbalanced binary segmentation.

    Focal Loss down-weights easy (high-confidence) examples via (1-p_t)^gamma,
    focusing training on hard/uncertain pixels. Unlike pos_weight, it reduces
    the influence of easy negatives rather than amplifying positives, avoiding
    degenerate all-positive predictions.

    Args:
        gamma:       Focus parameter — higher values down-weight easy examples more (default 2.0)
        alpha:       Weight for the positive class in [0, 1] (default 0.25)
        dice_weight: Contribution of Dice term; focal weight = 1 - dice_weight (default 0.5)
        smooth:      Laplace smoothing for Dice denominator (default 1.0)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.dice_weight = dice_weight
        self.focal_weight = 1.0 - dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Per-pixel class weight: alpha for crack, (1-alpha) for background
        alpha_t = torch.where(
            targets >= 0.5,
            logits.new_tensor(self.alpha),
            logits.new_tensor(1.0 - self.alpha),
        )
        # p_t: model confidence in the correct class
        p_t = torch.where(targets >= 0.5, probs, 1.0 - probs)
        focal_factor = (1.0 - p_t) ** self.gamma

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_loss = (alpha_t * focal_factor * bce).mean()

        dice_loss = self._soft_dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

    def _soft_dice(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pf = probs.view(probs.size(0), -1)
        tf = targets.view(targets.size(0), -1)
        intersection = (pf * tf).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pf.sum(dim=1) + tf.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for imbalanced binary segmentation.

    Tversky Index = TP / (TP + alpha·FP + beta·FN)
    FTL = (1 - TI)^gamma

    With beta > alpha the loss directly penalises False Negatives (missed cracks)
    more than False Positives, driving Recall up without degenerate all-positive
    collapse.

    Reference: Salehi et al. 2017, "Tversky loss function for image segmentation
    using 3D fully convolutional deep networks"

    Args:
        alpha:  FP penalty weight (default 0.3 — tolerate some false alarms)
        beta:   FN penalty weight (default 0.7 — strongly penalise missed cracks)
        gamma:  Focal exponent; focuses gradient on hard examples (default 0.75)
        smooth: Laplace smoothing for denominator stability (default 1.0)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pf = probs.view(probs.size(0), -1)
        tf = targets.view(targets.size(0), -1)

        tp = (pf * tf).sum(dim=1)
        fp = (pf * (1.0 - tf)).sum(dim=1)
        fn = ((1.0 - pf) * tf).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal_tversky = (1.0 - tversky) ** self.gamma
        return focal_tversky.mean()
