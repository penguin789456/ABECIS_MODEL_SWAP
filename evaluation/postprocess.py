"""
Post-processing utilities for crack segmentation masks.

- Skeletonization (scikit-image)
- Crack length estimation (skeleton pixel count)
- Continuity score (largest connected component ratio)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def postprocess_mask(
    binary_mask: np.ndarray,
    min_crack_length_px: int = 50,
) -> dict:
    """
    Analyse a binary segmentation mask.

    Args:
        binary_mask:         2-D boolean or 0/1 array (H, W)
        min_crack_length_px: Minimum skeleton pixels to count as a crack segment

    Returns:
        Dictionary with keys:
            skeleton          (H, W) bool array
            crack_length_px   total skeleton pixels (proxy for crack length)
            num_components    number of connected components in skeleton
            continuity_score  fraction of skeleton in the largest component
            long_crack_px     skeleton pixels in segments >= min_crack_length_px
    """
    binary_mask = binary_mask.astype(bool)
    skeleton: np.ndarray = skeletonize(binary_mask)

    total_px = int(skeleton.sum())

    labeled, num_components = ndimage.label(skeleton)

    if num_components == 0:
        return {
            "skeleton": skeleton,
            "crack_length_px": 0,
            "num_components": 0,
            "continuity_score": 0.0,
            "long_crack_px": 0,
        }

    component_sizes = np.bincount(labeled.ravel())[1:]  # exclude background
    largest = int(component_sizes.max())
    continuity = float(largest / (total_px + 1e-7))
    long_crack_px = int(component_sizes[component_sizes >= min_crack_length_px].sum())

    return {
        "skeleton": skeleton,
        "crack_length_px": total_px,
        "num_components": int(num_components),
        "continuity_score": continuity,
        "long_crack_px": long_crack_px,
    }
