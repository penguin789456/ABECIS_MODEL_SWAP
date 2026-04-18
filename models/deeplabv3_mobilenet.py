"""
DeepLabV3 wrapper for binary crack segmentation.

Supports multiple torchvision backbones via the `backbone` argument:
    - "resnet50"           → deeplabv3_resnet50    (~42M params)
    - "resnet101"          → deeplabv3_resnet101   (~61M params)
    - "mobilenet_v3_large" → deeplabv3_mobilenet_v3_large (~11M params)  ← default

Change the backbone in config only; no code changes required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# Registry: backbone name → (factory function name, in_channels for DeepLabHead)
_BACKBONE_REGISTRY: dict[str, tuple[str, int]] = {
    "resnet50":           ("deeplabv3_resnet50",           2048),
    "resnet101":          ("deeplabv3_resnet101",          2048),
    "mobilenet_v3_large": ("deeplabv3_mobilenet_v3_large", 960),
}


class DeepLabV3Mobilenet(nn.Module):
    """
    Binary-output DeepLabV3 with selectable torchvision backbone.

    Default: MobileNetV3-Large (~11M params).
    Also supports resnet50 (~42M) and resnet101 (~61M) via the `backbone` arg.
    """

    def __init__(self, pretrained: bool = True, backbone: str = "mobilenet_v3_large"):
        super().__init__()
        backbone = backbone.lower()
        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone {backbone!r}. Choose from: {list(_BACKBONE_REGISTRY)}"
            )

        factory_name, in_channels = _BACKBONE_REGISTRY[backbone]
        import torchvision.models.segmentation as _seg
        factory_fn = getattr(_seg, factory_name)

        weights = "DEFAULT" if pretrained else None
        self.model = factory_fn(weights=weights)

        # Replace the default multi-class head with a single-channel binary head.
        self.model.classifier = DeepLabHead(in_channels, num_classes=1)

        # Auxiliary classifier is unused (our BCEDice loss trains on main output only).
        self.model.aux_classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float tensor, normalised with ImageNet stats.
        Returns:
            logits: (B, 1, H, W) — raw logits (apply sigmoid for probability).
        """
        return self.model(x)["out"]
