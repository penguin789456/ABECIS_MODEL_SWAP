# -*- coding: utf-8 -*-
"""
Ground Truth 評估腳本（含 clDice）— CFD 或 Mendeley 資料集

用法：
  python evaluate_gt_with_cldice.py --model ppliteseg_cfd         --dataset cfd    --device gpu
  python evaluate_gt_with_cldice.py --model ppliteseg_mendeley    --dataset mendeley --device gpu
  python evaluate_gt_with_cldice.py --model abecis_detectron2_whole_cfd --dataset cfd --device cpu

輸出：
  logs/{model}/{device}/gt_metrics_cldice_{timestamp}.txt
  logs/{model}/{device}/gt_per_image_cldice_{timestamp}.csv
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

# ── 路徑 ──────────────────────────────────────────────────────────────────────
CFD_GT_DIR      = Path(r"H:\ChihleeMaster\CrackK500\CFD\seg_gt")
CFD_IMAGE_DIR   = Path(r"H:\ChihleeMaster\CrackK500\CFD\cfd_image")
MEN_GT_DIR      = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW")
MEN_IMAGE_DIR   = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\rgb")
MEN_TEST_SPLIT  = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt")
OUT_BASE        = Path(r"H:\ChihleeMaster\dev\final_outputs")


# ── clDice ────────────────────────────────────────────────────────────────────
def compute_cldice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0
    skel_pred = skeletonize(pred)
    skel_gt   = skeletonize(gt)
    tprec = float((skel_pred & gt).sum()) / (float(skel_pred.sum()) + eps)
    tsens = float((skel_gt & pred).sum()) / (float(skel_gt.sum()) + eps)
    return float(2 * tprec * tsens / (tprec + tsens + eps))


def compute_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> dict:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp = int(np.logical_and(pred,  gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred,  gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    total = tp + fp + fn + tn
    cldice = compute_cldice(pred, gt, eps)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou"      : tp / (tp + fp + fn + eps),
        "dice"     : 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall"   : tp / (tp + fn + eps),
        "accuracy" : (tp + tn) / (total + eps),
        "cldice"   : cldice,
        "pred_px"  : int(pred.sum()),
        "gt_px"    : int(gt.sum()),
        "total_px" : total,
    }


def build_index(folder: Path) -> dict:
    return {
        p.stem.lower(): p
        for p in folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    }


def read_binary(path: Path, target_wh=None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if target_wh is not None and img.size != target_wh:
        img = img.resize(target_wh, Image.NEAREST)
    return np.array(img) > 127


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True)
    parser.add_argument("--dataset", choices=["cfd", "mendeley"], default="cfd")
    parser.add_argument("--device",  choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()

    model_name = args.model
    device     = args.device
    dataset    = args.dataset

    # ── 選資料集路徑 ──────────────────────────────────────────────────────────
    if dataset == "cfd":
        gt_dir    = CFD_GT_DIR
        img_dir   = CFD_IMAGE_DIR
        gt_index  = build_index(gt_dir)
        img_index = build_index(img_dir)
        stems     = sorted(s for s in img_index if s in gt_index)
        dataset_label = "CFD（Crack Forest Dataset）完全未見資料集"
    else:
        gt_dir   = MEN_GT_DIR
        img_dir  = MEN_IMAGE_DIR
        gt_index = build_index(gt_dir)
        stems_raw = [l.strip() for l in MEN_TEST_SPLIT.read_text(encoding="utf-8").splitlines() if l.strip()]
        # 處理 leading-zero（test.txt 中 "059" → 對應 BW/059.jpg）
        stems = [s.zfill(3) if s.isdigit() else s for s in stems_raw]
        # 過濾掉 GT 中找不到的（理論上不應有）
        stems = [s for s in stems if s.lower() in gt_index]
        dataset_label = "Mendeley CCSD 凍結測試集（70 張）"

    pred_masks_dir = OUT_BASE / "pred_masks" / model_name / device
    pred_index     = build_index(pred_masks_dir) if pred_masks_dir.exists() else {}

    print(f"模型         : {model_name}")
    print(f"資料集       : {dataset_label}")
    print(f"裝置         : {device}")
    print(f"pred_masks   : {pred_masks_dir}")
    print(f"評估影像數   : {len(stems)} 張")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = OUT_BASE / "logs" / model_name / device
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"gt_per_image_cldice_{ts}.csv"

    # ── 逐張計算 ─────────────────────────────────────────────────────────────
    rows = []
    total_tp = total_fp = total_fn = total_tn = 0
    ok_iou, ok_dice, ok_prec, ok_rec, ok_acc, ok_cldice = [], [], [], [], [], []
    missing_pred = missing_gt = 0

    for stem in stems:
        key    = stem.lower()
        gt_p   = gt_index.get(key)
        pred_p = pred_index.get(key)

        if gt_p is None:
            missing_gt += 1
            rows.append({"image_id": stem, "status": "missing GT"})
            continue
        if pred_p is None:
            missing_pred += 1
            rows.append({"image_id": stem, "status": "missing prediction"})
            continue

        gt_img = Image.open(gt_p).convert("L")
        gt     = np.array(gt_img) > 127
        pred   = read_binary(pred_p, target_wh=gt_img.size)

        m = compute_metrics(pred, gt)
        total_tp += m["tp"]; total_fp += m["fp"]
        total_fn += m["fn"]; total_tn += m["tn"]
        ok_iou.append(m["iou"]); ok_dice.append(m["dice"])
        ok_prec.append(m["precision"]); ok_rec.append(m["recall"])
        ok_acc.append(m["accuracy"]); ok_cldice.append(m["cldice"])

        rows.append({
            "image_id"    : stem,
            "status"      : "OK",
            "iou"         : round(m["iou"],        6),
            "dice"        : round(m["dice"],       6),
            "precision"   : round(m["precision"],  6),
            "recall"      : round(m["recall"],     6),
            "accuracy"    : round(m["accuracy"],   6),
            "cldice"      : round(m["cldice"],     6),
            "tp"          : m["tp"], "fp": m["fp"],
            "fn"          : m["fn"], "tn": m["tn"],
            "pred_px"     : m["pred_px"],
            "gt_px"       : m["gt_px"],
        })

    # CSV
    fields = ["image_id","status","iou","dice","precision","recall","accuracy","cldice",
              "tp","fp","fn","tn","pred_px","gt_px"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    eps = 1e-7
    n = len(ok_iou)
    micro_iou  = total_tp / (total_tp + total_fp + total_fn + eps)
    micro_prec = total_tp / (total_tp + total_fp + eps)
    micro_rec  = total_tp / (total_tp + total_fn + eps)
    micro_dice = 2*total_tp / (2*total_tp + total_fp + total_fn + eps)
    micro_acc  = (total_tp+total_tn)/(total_tp+total_fp+total_fn+total_tn+eps)

    sep  = "=" * 90
    dash = "-" * 90
    lines = [
        sep,
        f"  Ground Truth 評估結果（含 clDice）— {dataset_label}",
        sep,
        f"  評估日期    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  模型名稱    : {model_name}",
        f"  裝置        : {device}",
        f"  資料集      : {dataset_label}",
        f"  GT 來源     : {gt_dir}",
        f"  pred_masks  : {pred_masks_dir}",
        f"  評估影像數  : {len(stems)} 張",
        dash,
        f"  評估成功張數: {n}（缺預測 {missing_pred}，缺GT {missing_gt}）",
        "",
        "  ── Micro metrics（所有像素累積後計算）",
        f"    IoU       : {micro_iou:.4f}",
        f"    Dice      : {micro_dice:.4f}",
        f"    Precision : {micro_prec:.4f}",
        f"    Recall    : {micro_rec:.4f}",
        f"    Accuracy  : {micro_acc:.4f}",
        "",
        "  ── Macro metrics（逐張平均）",
        f"    IoU       : {np.mean(ok_iou):.4f}" if ok_iou else "    IoU       : N/A",
        f"    Dice      : {np.mean(ok_dice):.4f}" if ok_dice else "    Dice      : N/A",
        f"    Precision : {np.mean(ok_prec):.4f}" if ok_prec else "    Precision : N/A",
        f"    Recall    : {np.mean(ok_rec):.4f}"  if ok_rec  else "    Recall    : N/A",
        f"    Accuracy  : {np.mean(ok_acc):.4f}"  if ok_acc  else "    Accuracy  : N/A",
        f"    clDice    : {np.mean(ok_cldice):.4f}" if ok_cldice else "    clDice    : N/A",
        "",
        f"  TP={total_tp}  FP={total_fp}  FN={total_fn}  TN={total_tn}",
        sep,
    ]

    log_path = log_dir / f"gt_metrics_cldice_{ts}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n✅ 完成")
    print(f"   gt_metrics  → {log_path}")
    print(f"   gt_per_image→ {csv_path}")


if __name__ == "__main__":
    main()
