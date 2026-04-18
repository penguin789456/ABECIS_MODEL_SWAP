"""
CrackInstanceDataset — full-image instance segmentation dataset for Mask R-CNN.

Each connected component in the binary BW mask becomes one instance.
Returns torchvision-compatible targets:
    {
        "boxes":   FloatTensor[N, 4]  (x1, y1, x2, y2)
        "labels":  IntTensor[N]        all 1 (crack)
        "masks":   BoolTensor[N, H, W]
        "image_id": Int64Tensor[1]
        "area":    FloatTensor[N]
        "iscrowd": IntTensor[N]       all 0
    }

For images with no crack pixels a single dummy "empty" annotation is returned
so the collate function stays consistent (filtered out in the loss).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CrackInstanceDataset(Dataset):
    """Full-image dataset for instance segmentation (Mask R-CNN)."""

    def __init__(
        self,
        split_file: str | Path,
        dataset_root: str | Path,
        train: bool = False,
        min_area: int = 16,          # minimum crack-instance area (pixels)
    ) -> None:
        self.root = Path(dataset_root)
        self.rgb_dir = self.root / "rgb"
        self.bw_dir = self.root / "BW"
        self.train = train
        self.min_area = min_area

        with open(split_file) as f:
            self.stems = [ln.strip() for ln in f if ln.strip()]

        # Build case-insensitive index
        self._rgb_index = {
            p.stem.lower(): p
            for p in self.rgb_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        }
        self._bw_index = {
            p.stem.lower(): p
            for p in self.bw_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        }

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        rgb_path = self._rgb_index.get(stem.lower())
        bw_path = self._bw_index.get(stem.lower())

        img = Image.open(rgb_path).convert("RGB")
        bw = Image.open(bw_path).convert("L")

        # Guard against EXIF rotation mismatch (RGB and BW may differ in orientation).
        # PIL.Image.size = (W, H); numpy array shape = (H, W).
        if img.size != bw.size:
            if img.size == (bw.height, bw.width):
                # Dimensions are transposed — rotate BW 90° to match RGB orientation
                bw_np = np.array(bw)
                bw = Image.fromarray(np.rot90(bw_np, k=1))   # k=1 → 90° CCW
            else:
                bw = bw.resize(img.size, Image.NEAREST)

        # Simple augmentation during training
        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                bw = TF.hflip(bw)
            if random.random() < 0.5:
                img = TF.vflip(img)
                bw = TF.vflip(bw)

        img_tensor = TF.to_tensor(img)  # (3, H, W) float [0,1]

        mask_np = np.array(bw) > 127          # (H, W) bool

        # Label connected components → instances
        labeled, n_instances = ndimage.label(mask_np)
        boxes, labels, masks, areas = [], [], [], []

        for inst_id in range(1, n_instances + 1):
            inst_mask = labeled == inst_id       # (H, W) bool
            area = int(inst_mask.sum())
            if area < self.min_area:
                continue
            rows = np.where(inst_mask.any(axis=1))[0]
            cols = np.where(inst_mask.any(axis=0))[0]
            x1, y1 = int(cols[0]), int(rows[0])
            x2, y2 = int(cols[-1]) + 1, int(rows[-1]) + 1
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(1)
            masks.append(inst_mask)
            areas.append(float(area))

        image_id = torch.tensor([idx], dtype=torch.int64)

        if len(boxes) == 0:
            # No valid instances — return empty target
            target = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,),   dtype=torch.int64),
                "masks":    torch.zeros((0, mask_np.shape[0], mask_np.shape[1]), dtype=torch.bool),
                "image_id": image_id,
                "area":     torch.zeros((0,),   dtype=torch.float32),
                "iscrowd":  torch.zeros((0,),   dtype=torch.int64),
            }
        else:
            target = {
                "boxes":    torch.tensor(boxes,  dtype=torch.float32),
                "labels":   torch.tensor(labels, dtype=torch.int64),
                "masks":    torch.tensor(np.stack(masks), dtype=torch.bool),
                "image_id": image_id,
                "area":     torch.tensor(areas,  dtype=torch.float32),
                "iscrowd":  torch.zeros((len(boxes),), dtype=torch.int64),
            }

        return img_tensor, target
