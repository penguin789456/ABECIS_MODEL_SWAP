"""
Inference script for Mask R-CNN (Detectron2).

Runs inside the CrackPre conda environment.
Saves one binary PNG mask per test image to outputs/predictions/maskrcnn/.

Usage:
    conda activate CrackPre
    python evaluation/inference_maskrcnn.py --config configs/maskrcnn.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_inference(cfg: dict) -> None:
    import cv2
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultPredictor
    from detectron2.model_zoo import model_zoo

    ds_cfg = cfg["dataset"]
    ckpt_cfg = cfg["checkpoint"]

    # Register test dataset
    coco_dir = Path(ds_cfg["coco_annotations_dir"])
    rgb_dir = str(Path(ds_cfg["root"]) / "rgb")
    register_coco_instances("crack_test", {}, str(coco_dir / "test.json"), rgb_dir)

    # Build Detectron2 config
    d2 = get_cfg()
    d2.merge_from_file(model_zoo.get_config_file(cfg["model"]["config_file"]))
    d2.MODEL.ROI_HEADS.NUM_CLASSES = 1
    d2.MODEL.WEIGHTS = str(Path(ckpt_cfg["save_dir"]) / "model_final.pth")
    d2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(d2)

    split_file = Path(ds_cfg["splits_dir"]) / "test.txt"
    with open(split_file) as f:
        stems = [ln.strip() for ln in f if ln.strip()]

    rgb_index = {
        p.stem.lower(): p
        for p in Path(ds_cfg["root"], "rgb").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    }

    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in tqdm(stems, desc="Mask R-CNN inference"):
        rgb_p = rgb_index.get(stem.lower())
        if rgb_p is None:
            print(f"WARNING: image not found for stem '{stem}'")
            continue

        bgr_img = cv2.imread(str(rgb_p))
        H, W = bgr_img.shape[:2]
        outputs = predictor(bgr_img)
        instances = outputs["instances"].to("cpu")

        # Merge all instance masks into a single binary mask
        combined = np.zeros((H, W), dtype=np.uint8)
        if len(instances) > 0:
            masks = instances.pred_masks.numpy()  # (N, H, W) bool
            for m in masks:
                combined[m] = 255

        Image.fromarray(combined).save(out_dir / f"{stem}.png")

    print(f"Saved {len(stems)} masks to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mask R-CNN inference")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg)
