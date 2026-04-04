"""
Pixel-level binary segmentation metrics.

All functions accept flat 1-D boolean (or 0/1 uint8) numpy arrays.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


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


def compute_metrics_2d(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-7,
) -> dict[str, float]:
    """
    Compute all metrics including clDice from 2-D binary mask arrays.

    Args:
        pred: 2-D predicted binary mask (H, W)
        gt:   2-D ground-truth binary mask (H, W)

    Returns:
        Dictionary with keys: iou, dice, precision, recall, cldice
    """
    metrics = compute_metrics(pred.ravel(), gt.ravel(), eps)
    metrics["cldice"] = compute_cldice(pred, gt, eps)
    return metrics


def compute_cldice(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Compute clDice (centerline Dice) for thin tubular structures like cracks.

    clDice skeletonizes both prediction and ground truth, then measures how well
    the predicted centerline is covered by the GT mask, and vice versa.
    This is more sensitive to connectivity than pixel-level Dice.

    Reference: Shit et al. (2021) "clDice - a Novel Topology-Preserving Loss
    Function for Tubular Structure Segmentation"

    Args:
        pred: 2-D binary mask (H, W), bool or 0/1
        gt:   2-D binary mask (H, W), bool or 0/1
        eps:  Small constant to avoid division by zero

    Returns:
        clDice score in [0, 1]
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # Edge case: both empty → perfect score; one empty → zero
    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0

    skel_pred = skeletonize(pred)
    skel_gt   = skeletonize(gt)

    # Tprec: fraction of predicted skeleton covered by GT mask
    tprec = (skel_pred & gt).sum() / (skel_pred.sum() + eps)
    # Tsens: fraction of GT skeleton covered by predicted mask
    tsens = (skel_gt & pred).sum() / (skel_gt.sum() + eps)

    return float(2 * tprec * tsens / (tprec + tsens + eps))


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
    results = [compute_metrics_2d(p, g, eps) for p, g in zip(preds, gts)]
    keys = ["iou", "dice", "precision", "recall", "cldice"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
