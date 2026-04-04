"""
Sweep prediction threshold on the validation set to find the best IoU.

Usage:
    conda activate CrackSeg
    python scripts/threshold_sweep.py --config configs/ppliteseg.yaml \
        --checkpoint outputs/checkpoints/ppliteseg/best.pth

Output: prints per-threshold table and highlights best IoU threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import PrecomputedCrackDataset, CrackDataset
from data.transforms import get_val_transforms


def build_model(model_cfg: dict) -> torch.nn.Module:
    name = model_cfg["name"].lower()

    if name == "deeplabv3plus":
        from models.deeplabv3plus import DeepLabV3Plus
        return DeepLabV3Plus(pretrained=False)

    if name == "ppliteseg":
        zh320_root = (Path(__file__).resolve().parent.parent
                      / "realtime-semantic-segmentation-pytorch")
        sys.path.insert(0, str(zh320_root))
        import importlib; importlib.invalidate_caches()
        _saved = {k: v for k, v in list(sys.modules.items())
                  if k == "models" or k.startswith("models.")}
        for k in _saved:
            del sys.modules[k]
        try:
            from models.pp_liteseg import PPLiteSeg  # type: ignore[import]
        finally:
            sys.modules.update(_saved)
        return PPLiteSeg(num_class=1, encoder_type=model_cfg.get("backbone", "STDC1").lower())

    if name in ("pidnet", "ddrnet"):
        import importlib
        zh320_root = (Path(__file__).resolve().parent.parent
                      / "realtime-semantic-segmentation-pytorch")
        if str(zh320_root) not in sys.path:
            sys.path.insert(0, str(zh320_root))
        importlib.invalidate_caches()
        _saved = {k: v for k, v in list(sys.modules.items())
                  if k == "models" or k.startswith("models.")}
        for k in _saved:
            del sys.modules[k]
        try:
            from models.ddrnet import DDRNet  # type: ignore[import]
        finally:
            sys.modules.update(_saved)
        arch_type = model_cfg.get("arch_type", "DDRNet-23-slim")
        return DDRNet(num_class=1, arch_type=arch_type)

    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def collect_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on entire loader, return flattened probs and labels."""
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    model.eval()
    for imgs, masks in tqdm(loader, desc="inference"):
        imgs = imgs.to(device)
        logits = model(imgs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)
        labels = masks.numpy()                        # (B, 1, H, W)
        all_probs.append(probs.ravel())
        all_labels.append(labels.ravel())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def sweep(probs: np.ndarray, labels: np.ndarray,
          thresholds: list[float]) -> list[dict]:
    gt = labels.astype(bool)
    results = []
    for t in thresholds:
        pred = probs >= t
        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        eps = 1e-7
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        results.append({"threshold": t, "iou": iou, "dice": dice,
                         "precision": precision, "recall": recall})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppliteseg.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint. Defaults to save_dir/best.pth")
    parser.add_argument("--min_t", type=float, default=0.10)
    parser.add_argument("--max_t", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = str(Path(cfg["checkpoint"]["save_dir"]) / "best.pth")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[sweep] checkpoint: {ckpt_path}")

    # Build dataset
    ds_cfg = cfg["dataset"]
    precomputed_dir = ds_cfg.get("precomputed_dir")
    val_dir = Path(precomputed_dir) / "val" if precomputed_dir else None
    if val_dir and (val_dir / "rgb").exists():
        val_ds = PrecomputedCrackDataset(str(val_dir), transform=get_val_transforms())
    else:
        val_ds = CrackDataset(
            ds_cfg["root"],
            f"{ds_cfg['splits_dir']}/val.txt",
            patch_size=ds_cfg["patch_size"],
            overlap=ds_cfg["overlap"],
            transform=get_val_transforms(),
        )
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=ds_cfg["num_workers"],
                            pin_memory=False)
    print(f"[sweep] val patches: {len(val_ds)}")

    # Build model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # Training saves weights under "model"; fall back to bare dict for compatibility
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    print(f"[sweep] model loaded on {device}")

    # Collect probabilities
    probs, labels = collect_probs(model, val_loader, device)

    # Sweep thresholds
    thresholds = [round(t, 2) for t in
                  np.arange(args.min_t, args.max_t + args.step / 2, args.step)]
    results = sweep(probs, labels, thresholds)

    # Print table
    header = f"{'threshold':>10} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8}"
    print("\n" + header)
    print("-" * len(header))
    best = max(results, key=lambda r: r["iou"])
    for r in results:
        mark = " << BEST" if r["threshold"] == best["threshold"] else ""
        print(f"{r['threshold']:>10.2f} {r['iou']:>8.4f} {r['dice']:>8.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f}{mark}")

    print(f"\nBest threshold: {best['threshold']:.2f}  →  IoU={best['iou']:.4f}, "
          f"P={best['precision']:.4f}, R={best['recall']:.4f}")


if __name__ == "__main__":
    main()
