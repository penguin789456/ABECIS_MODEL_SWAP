"""
Pixel-level binary segmentation metrics.

All functions accept flat 1-D boolean (or 0/1 uint8) numpy arrays.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-7,
) -> dict[str, float]:
    """
    Compute IoU, Dice, Precision, and Recall from flat binary arrays.

    Args:
        pred: Predicted binary mask (bool or 0/1), shape (N,)
        gt:   Ground-truth binary mask (bool or 0/1), shape (N,)
        eps:  Small constant to avoid division by zero

    Returns:
        Dictionary with keys: iou, dice, precision, recall
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def compute_metrics_per_image(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    eps: float = 1e-7,
) -> dict[str, float]:
    """
    Compute metrics averaged per image (macro-average).

    Args:
        preds: List of 2-D binary mask arrays (H, W)
        gts:   List of 2-D binary ground-truth arrays (H, W)

    Returns:
        Dictionary with mean iou, dice, precision, recall across images
    """
    results = [compute_metrics(p.ravel(), g.ravel(), eps) for p, g in zip(preds, gts)]
    keys = ["iou", "dice", "precision", "recall"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
