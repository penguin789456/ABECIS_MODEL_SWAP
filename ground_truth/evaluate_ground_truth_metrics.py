"""
Ground Truth 評估腳本 — CFD 資料集（單一模型版）

用途：使用 CFD ground truth mask 評估指定模型的預測結果

資料結構：
  CFD GT mask  : H:\ChihleeMaster\CrackK500\CFD\seg_gt\
  pred_masks   : H:\ChihleeMaster\dev\final_outputs\pred_masks\{model_name}\{device}\

啟動方式（任何有 numpy/PIL 的環境皆可）：
  python evaluate_ground_truth_metrics.py --model ppliteseg_cfd                    --device gpu
  python evaluate_ground_truth_metrics.py --model ddrnet_cfd                       --device gpu
  python evaluate_ground_truth_metrics.py --model deeplabv3_cfd                    --device gpu
  python evaluate_ground_truth_metrics.py --model maskrcnn_cfd                     --device gpu
  python evaluate_ground_truth_metrics.py --model ddrnet_ftl_torchscript_whole_cfd --device gpu
  python evaluate_ground_truth_metrics.py --model abecis_detectron2_whole_cfd      --device cpu

輸出（每次執行都會產生新的帶時間戳記檔案）：
  H:\ChihleeMaster\dev\final_outputs\logs\{model_name}\{device}\gt_metrics_{timestamp}.txt
  H:\ChihleeMaster\dev\final_outputs\logs\{model_name}\{device}\gt_per_image_{timestamp}.csv
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
CFD_IMAGE_DIR = Path(r"H:\ChihleeMaster\CrackK500\CFD\cfd_image")
CFD_GT_DIR    = Path(r"H:\ChihleeMaster\CrackK500\CFD\seg_gt")
OUT_BASE      = Path(r"H:\ChihleeMaster\dev\final_outputs")


def build_index(folder: Path) -> dict:
    """建立 stem.lower() → Path 的索引"""
    return {
        p.stem.lower(): p
        for p in folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    }


def read_binary(path: Path, target_wh=None, threshold: int = 127) -> np.ndarray:
    img = Image.open(path).convert("L")
    if target_wh is not None and img.size != target_wh:
        img = img.resize(target_wh, Image.NEAREST)
    return np.array(img) > threshold


def compute_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> dict:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp = int(np.logical_and(pred,  gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred,  gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    total = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou"      : tp / (tp + fp + fn + eps),
        "dice"     : 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall"   : tp / (tp + fn + eps),
        "accuracy" : (tp + tn) / (total + eps),
        "pred_px"  : int(pred.sum()),
        "gt_px"    : int(gt.sum()),
        "total_px" : total,
    }


def evaluate_model(
    model_name: str,
    pred_masks_dir: Path,
    gt_index: dict,
    stems: list,
    csv_path: Path,
) -> dict:
    pred_index = build_index(pred_masks_dir) if pred_masks_dir.exists() else {}

    rows = []
    total_tp = total_fp = total_fn = total_tn = 0
    ok_iou, ok_prec, ok_rec, ok_dice, ok_acc = [], [], [], [], []
    missing_pred = missing_gt = 0

    for stem in stems:
        key   = stem.lower()
        gt_p  = gt_index.get(key)
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
        ok_iou.append(m["iou"]);  ok_prec.append(m["precision"])
        ok_rec.append(m["recall"]); ok_dice.append(m["dice"]); ok_acc.append(m["accuracy"])

        rows.append({
            "image_id"      : stem,
            "status"        : "OK",
            "pred_path"     : str(pred_p),
            "gt_path"       : str(gt_p),
            "tp"            : m["tp"], "fp": m["fp"],
            "fn"            : m["fn"], "tn": m["tn"],
            "iou"           : round(m["iou"],       6),
            "dice"          : round(m["dice"],      6),
            "precision"     : round(m["precision"], 6),
            "recall"        : round(m["recall"],    6),
            "accuracy"      : round(m["accuracy"],  6),
            "pred_px"       : m["pred_px"],
            "gt_px"         : m["gt_px"],
            "pred_cover_%"  : round(m["pred_px"] / m["total_px"] * 100, 4),
            "gt_cover_%"    : round(m["gt_px"]   / m["total_px"] * 100, 4),
        })

    # 寫 per-image CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id", "status", "pred_path", "gt_path",
        "tp", "fp", "fn", "tn",
        "iou", "dice", "precision", "recall", "accuracy",
        "pred_px", "gt_px", "pred_cover_%", "gt_cover_%",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    eps = 1e-7
    n = len(ok_iou)
    micro_iou  = total_tp / (total_tp + total_fp + total_fn + eps)
    micro_prec = total_tp / (total_tp + total_fp + eps)
    micro_rec  = total_tp / (total_tp + total_fn + eps)
    micro_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)
    micro_acc  = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)

    return {
        "model"        : model_name,
        "n_evaluated"  : n,
        "missing_pred" : missing_pred,
        "missing_gt"   : missing_gt,
        "micro_iou"    : micro_iou,
        "micro_prec"   : micro_prec,
        "micro_rec"    : micro_rec,
        "micro_dice"   : micro_dice,
        "micro_acc"    : micro_acc,
        "macro_iou"    : float(np.mean(ok_iou))  if ok_iou  else 0.0,
        "macro_prec"   : float(np.mean(ok_prec)) if ok_prec else 0.0,
        "macro_rec"    : float(np.mean(ok_rec))  if ok_rec  else 0.0,
        "macro_dice"   : float(np.mean(ok_dice)) if ok_dice else 0.0,
        "macro_acc"    : float(np.mean(ok_acc))  if ok_acc  else 0.0,
        "total_tp"     : total_tp,
        "total_fp"     : total_fp,
        "total_fn"     : total_fn,
        "total_tn"     : total_tn,
        "csv"          : csv_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        help="模型名稱（對應 pred_masks/{model_name}/ 資料夾），例如：ppliteseg_cfd"
    )
    parser.add_argument(
        "--device", choices=["cpu", "gpu"], default="gpu",
        help="推論時使用的裝置（決定讀取哪個 pred_masks 子目錄）"
    )
    args = parser.parse_args()

    model_name = args.model
    device     = args.device

    pred_masks_dir = OUT_BASE / "pred_masks" / model_name / device

    if not CFD_GT_DIR.exists():
        print(f"❌ CFD GT 資料夾不存在：{CFD_GT_DIR}")
        sys.exit(1)

    gt_index    = build_index(CFD_GT_DIR)
    image_index = build_index(CFD_IMAGE_DIR)
    stems       = sorted(s for s in image_index if s in gt_index)

    print(f"模型         : {model_name}")
    print(f"裝置         : {device}")
    print(f"pred_masks   : {pred_masks_dir}")
    print(f"CFD GT 目錄  : {CFD_GT_DIR}")
    print(f"GT mask 數   : {len(gt_index)} 張")
    print(f"評估影像數   : {len(stems)} 張（有 GT 者）")
    print()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = OUT_BASE / "logs" / model_name / device
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / f"gt_per_image_{ts}.csv"

    result = evaluate_model(
        model_name     = model_name,
        pred_masks_dir = pred_masks_dir,
        gt_index       = gt_index,
        stems          = stems,
        csv_path       = csv_path,
    )

    sep  = "=" * 90
    dash = "-" * 90

    lines = [
        sep,
        f"  Ground Truth 評估結果 — CFD（Crack Forest Dataset）完全未見資料集",
        sep,
        f"  評估日期    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  模型名稱    : {model_name}",
        f"  裝置        : {device}",
        f"  GT 來源     : {CFD_GT_DIR}",
        f"  pred_masks  : {pred_masks_dir}",
        f"  評估影像數  : {len(stems)} 張",
        dash,
        f"  評估成功張數: {result['n_evaluated']}（缺預測 {result['missing_pred']}，缺GT {result['missing_gt']}）",
        "",
        "  ── Micro metrics（所有像素累積後計算）",
        f"    IoU       : {result['micro_iou']:.4f}",
        f"    Dice      : {result['micro_dice']:.4f}",
        f"    Precision : {result['micro_prec']:.4f}",
        f"    Recall    : {result['micro_rec']:.4f}",
        f"    Accuracy  : {result['micro_acc']:.4f}",
        "",
        "  ── Macro metrics（逐張平均）",
        f"    IoU       : {result['macro_iou']:.4f}",
        f"    Dice      : {result['macro_dice']:.4f}",
        f"    Precision : {result['macro_prec']:.4f}",
        f"    Recall    : {result['macro_rec']:.4f}",
        f"    Accuracy  : {result['macro_acc']:.4f}",
        "",
        f"  TP={result['total_tp']}  FP={result['total_fp']}  FN={result['total_fn']}  TN={result['total_tn']}",
        sep,
    ]

    log_path = log_dir / f"gt_metrics_{ts}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n✅ 評估完成")
    print(f"   gt_metrics  → {log_path}")
    print(f"   gt_per_image→ {csv_path}")


if __name__ == "__main__":
    main()
