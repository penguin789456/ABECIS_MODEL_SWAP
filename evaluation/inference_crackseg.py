"""
Inference script for CrackSeg models (DeepLabV3+, PP-LiteSeg, PIDNet).

Runs full-image patch inference with overlap averaging (stitching), then
saves one binary PNG mask per test image to outputs/predictions/<model>/.

Usage (inside CrackSeg conda env):
    python evaluation/inference_crackseg.py --config configs/deeplabv3plus.yaml
    python evaluation/inference_crackseg.py --config configs/ppliteseg.yaml
    python evaluation/inference_crackseg.py --config configs/pidnet.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import albumentations as A
from data.transforms import get_test_transforms
from training.train_crackseg import build_model


def stitch_patches(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    transform: A.Compose,
    patch_size: int = 512,
    overlap: int = 128,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Slide a window over the full image, run inference on each patch, and
    average overlapping logits before thresholding.

    Returns a (H, W) binary uint8 mask.
    """
    H, W = image.shape[:2]
    stride = patch_size - overlap
    logit_sum = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # Build patch grid (same logic as CrackDataset)
    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))
    if not ys or ys[-1] + patch_size < H:
        ys.append(max(H - patch_size, 0))
    if not xs or xs[-1] + patch_size < W:
        xs.append(max(W - patch_size, 0))

    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = image[y : y + patch_size, x : x + patch_size]
                # Pad if needed
                ph = patch_size - patch.shape[0]
                pw = patch_size - patch.shape[1]
                if ph > 0 or pw > 0:
                    patch = np.pad(patch, ((0, ph), (0, pw), (0, 0)), mode="reflect")

                aug = transform(image=patch)
                tensor = aug["image"].unsqueeze(0).float().to(device)
                logit = model(tensor).squeeze().cpu().numpy()  # (H_p, W_p)

                actual_h = min(patch_size, H - y)
                actual_w = min(patch_size, W - x)
                logit_sum[y : y + actual_h, x : x + actual_w] += logit[:actual_h, :actual_w]
                count_map[y : y + actual_h, x : x + actual_w] += 1.0

    avg_logit = logit_sum / np.maximum(count_map, 1.0)
    prob = 1.0 / (1.0 + np.exp(-avg_logit))  # sigmoid
    return (prob > threshold).astype(np.uint8) * 255


def run_inference(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)

    ckpt_path = Path(cfg["checkpoint"]["save_dir"]) / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"Loaded: {ckpt_path}")

    transform = get_test_transforms()
    ds_cfg = cfg["dataset"]
    rgb_dir = Path(ds_cfg["root"]) / "rgb"
    split_file = Path(ds_cfg["splits_dir"]) / "test.txt"

    with open(split_file) as f:
        stems = [ln.strip() for ln in f if ln.strip()]

    # Case-insensitive rgb lookup
    rgb_index = {p.stem.lower(): p for p in rgb_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in tqdm(stems, desc="Inference"):
        rgb_p = rgb_index.get(stem.lower())
        if rgb_p is None:
            print(f"WARNING: image not found for stem '{stem}'")
            continue
        image = np.array(Image.open(rgb_p).convert("RGB"))
        mask = stitch_patches(
            image, model, device, transform,
            patch_size=ds_cfg["patch_size"],
            overlap=ds_cfg["overlap"],
        )
        Image.fromarray(mask).save(out_dir / f"{stem}.png")

    print(f"Saved {len(stems)} masks to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for CrackSeg models")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg)
