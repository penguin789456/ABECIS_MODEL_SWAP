"""
Unified evaluation script — environment agnostic (numpy / PIL only).

Reads saved PNG binary masks from outputs/predictions/<model>/ for all
four models, compares against ground-truth BW masks, and writes a
summary CSV to outputs/results/metrics_summary.csv.

Usage (either conda env):
    python evaluation/evaluate.py
    python evaluation/evaluate.py --output_dir outputs/results \
        --test_split data/splits/test.txt \
        --dataset_root concreteCrackSegmentationDataset \
        --predictions_dir outputs/predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_metrics, compute_metrics_2d
from evaluation.postprocess import postprocess_mask

MODELS = ["deeplabv3_mobilenet", "ppliteseg", "ddrnet", "maskrcnn"]


def evaluate_model(
    model_name: str,
    test_stems: list[str],
    gt_dir: Path,
    pred_dir: Path,
) -> dict[str, float]:
    all_preds, all_gts = [], []
    all_preds_2d, all_gts_2d = [], []
    missing = 0

    for stem in test_stems:
        pred_path = pred_dir / model_name / f"{stem}.png"
        gt_path_jpg = gt_dir / f"{stem}.jpg"
        gt_path_JPG = gt_dir / f"{stem}.JPG"
        gt_path = gt_path_jpg if gt_path_jpg.exists() else gt_path_JPG

        if not pred_path.exists():
            missing += 1
            continue
        if not gt_path.exists():
            print(f"WARNING: GT not found for {stem}")
            continue

        pred = np.array(Image.open(pred_path).convert("L")) > 127
        gt = np.array(Image.open(gt_path).convert("L")) > 127
        # Guard against EXIF-rotation mismatch (pred stitched from RGB, GT loaded raw)
        if pred.shape != gt.shape:
            if pred.shape == gt.shape[::-1]:
                pred = pred.T   # transpose to match GT orientation
            else:
                pred = np.array(
                    Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
                )
        all_preds.append(pred.ravel())
        all_gts.append(gt.ravel())
        all_preds_2d.append(pred)
        all_gts_2d.append(gt)

    if missing > 0:
        print(f"  [{model_name}] {missing}/{len(test_stems)} predictions missing")

    if not all_preds:
        print(f"  [{model_name}] No predictions found — skipping")
        return {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "cldice": 0.0}

    all_preds_arr = np.concatenate(all_preds)
    all_gts_arr = np.concatenate(all_gts)
    metrics = compute_metrics(all_preds_arr, all_gts_arr)

    # clDice: average per image (skeleton-based, must operate on 2D arrays)
    from evaluation.metrics import compute_cldice
    cldice_scores = [compute_cldice(p, g) for p, g in zip(all_preds_2d, all_gts_2d)]
    metrics["cldice"] = float(np.mean(cldice_scores))
    return metrics


def main(
    output_dir: str = "outputs/results",
    test_split: str = "data/splits/test.txt",
    dataset_root: str = "concreteCrackSegmentationDataset",
    predictions_dir: str = "outputs/predictions",
) -> None:
    with open(test_split) as f:
        stems = [ln.strip() for ln in f if ln.strip()]
    print(f"Evaluating on {len(stems)} test images\n")

    gt_dir = Path(dataset_root) / "BW"
    pred_dir = Path(predictions_dir)
    records = []

    for model in MODELS:
        print(f"[{model}]")
        metrics = evaluate_model(model, stems, gt_dir, pred_dir)
        metrics["model"] = model
        records.append(metrics)
        print(
            f"  IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  "
            f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
            f"clDice={metrics.get('cldice', 0.0):.4f}\n"
        )

    df = pd.DataFrame(records).set_index("model")[["iou", "dice", "precision", "recall", "cldice"]]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "metrics_summary.csv"
    df.to_csv(out_path)
    print(f"Results saved to {out_path}")
    print(df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all segmentation models")
    parser.add_argument("--output_dir", default="outputs/results")
    parser.add_argument("--test_split", default="data/splits/test.txt")
    parser.add_argument("--dataset_root", default="concreteCrackSegmentationDataset")
    parser.add_argument("--predictions_dir", default="outputs/predictions")
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        test_split=args.test_split,
        dataset_root=args.dataset_root,
        predictions_dir=args.predictions_dir,
    )
