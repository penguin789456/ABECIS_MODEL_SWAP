"""
CrackDataset: PyTorch Dataset for concrete crack segmentation.

- Loads image/mask pairs listed in a split .txt file
- Extracts 512×512 patches with configurable overlap (default 128 px)
- Handles mixed .jpg / .JPG extensions in the rgb/ folder
- Lightweight LRU cache to avoid re-reading the same image for every patch
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    """
    Args:
        dataset_root: Path to concreteCrackSegmentationDataset/
        split_file:   Path to a .txt file with one image stem per line (no extension)
        patch_size:   Side length of extracted square patches (default 512)
        overlap:      Overlap between adjacent patches in pixels (default 128)
        transform:    Albumentations Compose transform applied to (image, mask) pairs
        cache_size:   Number of full images to keep in memory (default 8)
    """

    def __init__(
        self,
        dataset_root: str,
        split_file: str,
        patch_size: int = 512,
        overlap: int = 128,
        transform=None,
        cache_size: int = 8,
    ):
        self.rgb_dir = Path(dataset_root) / "rgb"
        self.bw_dir = Path(dataset_root) / "BW"
        self.patch_size = patch_size
        self.stride = patch_size - overlap
        self.transform = transform
        self.cache_size = cache_size

        # Build case-insensitive stem → rgb path index
        rgb_index: dict[str, Path] = {}
        for p in self.rgb_dir.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rgb_index[p.stem.lower()] = p

        # Load split file
        with open(split_file) as f:
            stems = [ln.strip() for ln in f if ln.strip()]

        # Match rgb + BW pairs
        self.image_pairs: list[tuple[Path, Path]] = []
        for stem in stems:
            rgb_p = rgb_index.get(stem.lower())
            bw_p = self.bw_dir / f"{stem}.jpg"
            if not bw_p.exists():
                bw_p = self.bw_dir / f"{stem}.JPG"
            if rgb_p is not None and bw_p.exists():
                self.image_pairs.append((rgb_p, bw_p))

        # Pre-compute patch grid: list of (image_index, y_start, x_start)
        self.patches: list[tuple[int, int, int]] = []
        for img_idx, (rgb_p, _) in enumerate(self.image_pairs):
            W, H = Image.open(rgb_p).size
            ys = list(range(0, max(H - patch_size, 0) + 1, self.stride))
            xs = list(range(0, max(W - patch_size, 0) + 1, self.stride))
            # Ensure the last patch reaches the image boundary
            if not ys or ys[-1] + patch_size < H:
                ys.append(max(H - patch_size, 0))
            if not xs or xs[-1] + patch_size < W:
                xs.append(max(W - patch_size, 0))
            seen: set[tuple[int, int]] = set()
            for y in ys:
                for x in xs:
                    if (y, x) not in seen:
                        seen.add((y, x))
                        self.patches.append((img_idx, y, x))

        # LRU image cache
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_idx, y, x = self.patches[idx]
        rgb_img, bw_mask = self._load(img_idx)

        ps = self.patch_size
        img_patch = rgb_img[y : y + ps, x : x + ps].copy()
        mask_patch = bw_mask[y : y + ps, x : x + ps].copy()

        # Pad if image is smaller than patch_size (rare edge case)
        if img_patch.shape[0] < ps or img_patch.shape[1] < ps:
            img_patch = _pad(img_patch, ps, mode="reflect")
            mask_patch = _pad(mask_patch, ps, mode="constant")

        # Binarise: white (>127) = crack = 1
        mask_patch = (mask_patch > 127).astype(np.uint8)

        if self.transform:
            aug = self.transform(image=img_patch, mask=mask_patch)
            img_patch = aug["image"]
            mask_patch = aug["mask"]

        # Convert to tensors if albumentations did not (i.e. no ToTensorV2)
        if isinstance(img_patch, np.ndarray):
            img_t = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_patch).float().unsqueeze(0)
        else:
            img_t = img_patch.float()
            mask_t = mask_patch.float().unsqueeze(0) if mask_patch.ndim == 2 else mask_patch.float()

        return img_t, mask_t

    # ------------------------------------------------------------------
    def _load(self, img_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if img_idx in self._cache:
            self._cache.move_to_end(img_idx)
            return self._cache[img_idx]

        rgb_p, bw_p = self.image_pairs[img_idx]
        rgb = np.array(Image.open(rgb_p).convert("RGB"))
        bw = np.array(Image.open(bw_p).convert("L"))
        # Some BW masks were saved with H/W swapped relative to the RGB image.
        # Resize mask to match RGB spatial dimensions.
        if rgb.shape[:2] != bw.shape[:2]:
            bw = np.array(Image.fromarray(bw).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST))
        self._cache[img_idx] = (rgb, bw)
        self._cache.move_to_end(img_idx)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return rgb, bw


# ------------------------------------------------------------------
class PrecomputedCrackDataset(Dataset):
    """
    Loads pre-computed patches from .npy files (uint8 HWC / HW).
    Generated by scripts/precompute_patches.py.

    Args:
        patches_dir: Path to data/patches/{split}/
        transform:   Albumentations Compose applied to each patch
    """

    def __init__(self, patches_dir: str, transform=None):
        base = Path(patches_dir)
        self._rgb_files = sorted((base / "rgb").glob("*.npy"))
        self._mask_dir = base / "mask"
        self.transform = transform

        if not self._rgb_files:
            raise FileNotFoundError(
                f"No .npy files found in {base / 'rgb'}.\n"
                "Run: python scripts/precompute_patches.py --config configs/ppliteseg.yaml"
            )

    def __len__(self) -> int:
        return len(self._rgb_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_path = self._rgb_files[idx]
        mask_path = self._mask_dir / rgb_path.name

        img_patch = np.load(rgb_path)    # uint8 HWC
        mask_patch = np.load(mask_path)  # uint8 HW, already binarised

        if self.transform:
            aug = self.transform(image=img_patch, mask=mask_patch)
            img_patch = aug["image"]
            mask_patch = aug["mask"]

        if isinstance(img_patch, np.ndarray):
            img_t = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_patch).float().unsqueeze(0)
        else:
            img_t = img_patch.float()
            mask_t = (
                mask_patch.float().unsqueeze(0)
                if mask_patch.ndim == 2
                else mask_patch.float()
            )

        return img_t, mask_t


# ------------------------------------------------------------------
def _pad(arr: np.ndarray, target: int, mode: str = "reflect") -> np.ndarray:
    h, w = arr.shape[:2]
    ph = max(target - h, 0)
    pw = max(target - w, 0)
    if arr.ndim == 3:
        return np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode=mode)
    return np.pad(arr, ((0, ph), (0, pw)), mode=mode)


# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "concreteCrackSegmentationDataset"
    split = sys.argv[2] if len(sys.argv) > 2 else "data/splits/train.txt"

    ds = CrackDataset(root, split)
    print(f"Pairs loaded : {len(ds.image_pairs)}")
    print(f"Total patches: {len(ds)}")
    img, mask = ds[0]
    print(f"Image shape  : {tuple(img.shape)}")
    print(f"Mask shape   : {tuple(mask.shape)}")
    print(f"Mask unique  : {mask.unique().tolist()}")
