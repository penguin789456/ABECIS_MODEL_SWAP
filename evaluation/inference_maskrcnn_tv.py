"""
Inference script for torchvision Mask R-CNN.

Loads a trained checkpoint from training/train_maskrcnn_tv.py, runs full-image
inference on the test split, merges instance masks into binary semantic masks,
and saves PNG predictions to outputs/predictions/maskrcnn/.

Usage (CrackSeg env):
    python evaluation/inference_maskrcnn_tv.py --config configs/maskrcnn_tv.yaml
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
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_maskrcnn(num_classes: int = 2):
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def run_inference(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_maskrcnn(num_classes=2)
    ckpt_path = Path(cfg["checkpoint"]["save_dir"]) / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device)
    model.eval()
    print(f"Loaded: {ckpt_path}")

    ds_cfg = cfg["dataset"]
    eval_cfg = cfg.get("evaluation", {})
    threshold = eval_cfg.get("threshold", 0.5)
    score_threshold = eval_cfg.get("score_threshold", 0.3)

    rgb_dir = Path(ds_cfg["root"]) / "rgb"
    split_file = Path(ds_cfg["splits_dir"]) / "test.txt"
    with open(split_file) as f:
        stems = [ln.strip() for ln in f if ln.strip()]

    rgb_index = {
        p.stem.lower(): p
        for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    }

    out_dir = Path(eval_cfg.get("predictions_dir", "outputs/predictions/maskrcnn"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for stem in tqdm(stems, desc="Inference"):
            rgb_path = rgb_index.get(stem.lower())
            if rgb_path is None:
                print(f"WARNING: image not found for stem '{stem}'")
                continue

            img = Image.open(rgb_path).convert("RGB")
            img_tensor = TF.to_tensor(img).to(device)

            outputs = model([img_tensor])[0]

            H, W = img_tensor.shape[-2:]
            pred_mask = torch.zeros((H, W), dtype=torch.bool, device=device)

            if len(outputs["scores"]) > 0:
                keep = outputs["scores"] >= score_threshold
                if keep.any():
                    inst_masks = outputs["masks"][keep, 0] > threshold
                    pred_mask = inst_masks.any(dim=0)

            mask_np = pred_mask.cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(mask_np).save(out_dir / f"{stem}.png")

    print(f"Saved {len(stems)} masks to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for torchvision Mask R-CNN")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg)
