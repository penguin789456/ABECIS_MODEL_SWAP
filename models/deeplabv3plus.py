"""
DeepLabV3+ wrapper for binary crack segmentation.

Wraps torchvision.models.segmentation.deeplabv3_resnet101 and replaces the
final classification head with a single-channel (binary) output layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3Plus(nn.Module):
    """
    Binary-output DeepLabV3+ using ResNet-101 backbone.

    Args:
        pretrained:     Load ImageNet-pretrained backbone weights
        output_stride:  Atrous rate (8 or 16); torchvision default is 16
    """

    def __init__(self, pretrained: bool = True, output_stride: int = 16):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.model: DeepLabV3 = deeplabv3_resnet101(weights=weights)

        # Replace the default classification head with a 1-class head.
        # ResNet-101 backbone outputs 2048 channels; read from the backbone
        # to be robust to future weight variants.
        in_channels = self.model.backbone.layer4[-1].conv3.out_channels  # 2048
        self.model.classifier = DeepLabHead(in_channels, num_classes=1)

        # Remove the auxiliary classifier (not needed for inference / our loss)
        self.model.aux_classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float tensor, normalised with ImageNet stats

        Returns:
            logits: (B, 1, H, W) — raw logits (not sigmoid-activated)
                    Apply sigmoid for probability, threshold at 0.5 for binary mask
        """
        out = self.model(x)
        return out["out"]  # shape: (B, 1, H, W)
