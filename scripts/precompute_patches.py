"""
Pre-compute all patches and save as .npy files.

Run once before training:
    python scripts/precompute_patches.py --config configs/ppliteseg.yaml

Output structure:
    data/patches/
        train/rgb/<stem>_<y>_<x>.npy   (uint8 HWC)
              mask/<stem>_<y>_<x>.npy  (uint8 HW, 0/1 binary)
        val/  ...
        test/ ...
"""

from __future__ import annotations

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import _pad


def precompute(
    dataset_root: str,
    splits_dir: str,
    out_dir: str,
    patch_size: int = 512,
    overlap: int = 128,
) -> None:
    root = Path(dataset_root)
    out = Path(out_dir)
    rgb_dir = root / "rgb"
    bw_dir = root / "BW"
    stride = patch_size - overlap

    rgb_index: dict[str, Path] = {
        p.stem.lower(): p
        for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    }

    for split in ("train", "val", "test"):
        split_file = Path(splits_dir) / f"{split}.txt"
        if not split_file.exists():
            print(f"[skip] {split} — no split file at {split_file}")
            continue

        stems = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rgb_out = out / split / "rgb"
        mask_out = out / split / "mask"
        rgb_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        total = 0
        for stem in tqdm(stems, desc=f"{split:5s}"):
            rgb_p = rgb_index.get(stem.lower())
            bw_p = bw_dir / f"{stem}.jpg"
            if not bw_p.exists():
                bw_p = bw_dir / f"{stem}.JPG"
            if rgb_p is None or not bw_p.exists():
                continue

            rgb = np.array(Image.open(rgb_p).convert("RGB"))
            bw = np.array(Image.open(bw_p).convert("L"))
            if rgb.shape[:2] != bw.shape[:2] and rgb.shape[:2] == bw.shape[:2][::-1]:
                bw = bw.T

            H, W = rgb.shape[:2]
            ys = list(range(0, max(H - patch_size, 0) + 1, stride))
            xs = list(range(0, max(W - patch_size, 0) + 1, stride))
            if not ys or ys[-1] + patch_size < H:
                ys.append(max(H - patch_size, 0))
            if not xs or xs[-1] + patch_size < W:
                xs.append(max(W - patch_size, 0))

            seen: set[tuple[int, int]] = set()
            for y in ys:
                for x in xs:
                    if (y, x) in seen:
                        continue
                    seen.add((y, x))

                    img_p = rgb[y : y + patch_size, x : x + patch_size].copy()
                    msk_p = bw[y : y + patch_size, x : x + patch_size].copy()

                    if img_p.shape[0] < patch_size or img_p.shape[1] < patch_size:
                        img_p = _pad(img_p, patch_size, mode="reflect")
                        msk_p = _pad(msk_p, patch_size, mode="constant")

                    msk_p = (msk_p > 127).astype(np.uint8)

                    name = f"{stem}_{y:04d}_{x:04d}.npy"
                    np.save(rgb_out / name, img_p)
                    np.save(mask_out / name, msk_p)
                    total += 1

        print(f"  [{split}] {total} patches → {out / split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppliteseg.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = cfg["dataset"]
    precompute(
        dataset_root=ds["root"],
        splits_dir=ds["splits_dir"],
        out_dir=ds.get("precomputed_dir", "data/patches"),
        patch_size=ds["patch_size"],
        overlap=ds["overlap"],
    )
