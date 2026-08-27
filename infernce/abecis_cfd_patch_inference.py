"""
ABECIS Detectron2 CFD Patch 版推論腳本
原始解析度推論（不 resize 成 1024×1024）

CFD 影像約 640×480，本身尺寸已小；
以原始解析度推論等效於「patch 管線」的精神（避免上/下縮放失真）。

ABECIS 為偵測式模型，無法做滑動視窗 NMS，因此不切 patch；
改為取消強制 resize，讓 Detectron2 在原生解析度下推論。

啟動方式（ABECIS conda env）：
  conda activate ABECIS
  cd H:\\ChihleeMaster\\dev\\final_outputs\\infernce
  python abecis_cfd_patch_inference.py
  python abecis_cfd_patch_inference.py --threshold 0.8 --device cpu

輸出：
  pred_masks/abecis_cfd_patch/{device}/
  logs/abecis_cfd_patch/{device}/inference_log_{ts}.txt
"""

import sys, os, argparse, shutil, time, platform, subprocess, csv
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from PIL import Image

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()

# ── 設定 ────────────────────────────────────────────────────────────────────
RGB_DIR    = Path(r'H:\ChihleeMaster\CrackK500\CFD\cfd_image')
OUT_BASE   = Path(r'H:\ChihleeMaster\dev\final_outputs')
MODEL_NAME = 'abecis_cfd_patch'
CKPT_PATH  = r'H:\ChihleeMaster\CrackPreVer3.5.3\ABECIS-main\output\model_final.pth'


def get_cpu_name() -> str:
    try:
        if platform.system().lower() == "windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                stderr=subprocess.DEVNULL, text=True
            )
            lines = [x.strip() for x in out.splitlines()
                     if x.strip() and x.strip().lower() != "name"]
            if lines:
                return lines[0]
    except Exception:
        pass
    return platform.processor()


def select_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("⚠️ 指定 cuda，但目前 CUDA 不可用，改用 CPU。")
        return "cpu"
    return device_arg


def overlay_mask(rgb: np.ndarray, mask: np.ndarray,
                 color=(220, 50, 50), alpha=0.5) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    crack = mask >= 128
    for c, v in enumerate(color):
        out[:, :, c] = np.where(
            crack,
            out[:, :, c] * (1 - alpha) + v * alpha,
            out[:, :, c]
        )
    return out.clip(0, 255).astype(np.uint8)


def main():
    program_start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    threshold     = args.threshold
    device        = select_device(args.device)
    device_suffix = 'gpu' if device == 'cuda' else 'cpu'

    ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir        = OUT_BASE / "logs"       / MODEL_NAME / device_suffix
    before_dir     = OUT_BASE / "before"     / MODEL_NAME / device_suffix
    after_dir      = OUT_BASE / "after"      / MODEL_NAME / device_suffix
    pred_masks_dir = OUT_BASE / "pred_masks" / MODEL_NAME / device_suffix
    for d in [log_dir, before_dir, after_dir, pred_masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 載入模型（原始解析度推論：MIN_SIZE_TEST=480, MAX_SIZE_TEST=640）──────
    print(f"Loading ABECIS Detectron2 (SCORE_THRESH={threshold})...")
    print(f"Device     : {device}")
    print(f"Mode       : native resolution（不 resize，CFD 原始尺寸 ~640×480）")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = CKPT_PATH
    cfg.MODEL.DEVICE  = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    # 關鍵：不強制 resize，讓 Detectron2 在原生解析度下推論
    cfg.INPUT.MIN_SIZE_TEST = 480
    cfg.INPUT.MAX_SIZE_TEST = 640

    predictor = DefaultPredictor(cfg)
    print("Model loaded.\n")

    # ── 讀取所有 CFD 影像 ───────────────────────────────────────────────────
    rgb_files = sorted([
        p for p in RGB_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    n_total = len(rgb_files)
    print(f"影像數 : {n_total} 張（全部 CFD cfd_image，原始解析度）")

    log_lines = [
        f"ABECIS Detectron2 CFD Patch（原始解析度）推論記錄",
        f"時間        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint  : {CKPT_PATH}",
        f"閾值        : {threshold}",
        f"推論方式    : 原始解析度（CFD ~640×480，不 resize）",
        f"影像數      : {n_total}",
        f"資料集      : CFD（Crack Forest Dataset）— 完全未見資料集",
        f"RGB 來源    : {RGB_DIR}",
        f"裝置        : {device}",
        f"CPU         : {get_cpu_name()}",
        f"CUDA 可用   : {torch.cuda.is_available()}",
        f"PyTorch     : {torch.__version__}",
    ]

    csv_path = log_dir / f"per_image_{ts}.csv"
    csv_rows = []

    success_cnt  = 0
    crack_cnt    = 0
    total_inst   = 0
    elapsed_list = []

    inference_start = time.perf_counter()

    for i, img_path in enumerate(rgb_files):
        stem = img_path.stem
        t0   = time.perf_counter()

        im_bgr = cv2.imread(str(img_path))
        if im_bgr is None:
            csv_rows.append({"stem": stem, "status": "read_failed"})
            continue

        orig_h, orig_w = im_bgr.shape[:2]

        outputs   = predictor(im_bgr)
        instances = outputs["instances"].to("cpu")
        n_inst    = len(instances)

        combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if n_inst > 0 and instances.has("pred_masks"):
            for m in instances.pred_masks.numpy():
                combined = np.where(m, 255, combined)

        elapsed  = time.perf_counter() - t0
        crack_px = int((combined >= 128).sum())
        has_crack = crack_px > 0

        if has_crack:
            crack_cnt += 1
        elapsed_list.append(elapsed)
        success_cnt += 1
        total_inst  += n_inst

        # 儲存輸出
        shutil.copy(str(img_path), before_dir / f"{stem}.png")
        Image.fromarray(combined).save(pred_masks_dir / f"{stem}.png")

        rgb_np  = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        overlay = overlay_mask(rgb_np, combined)
        Image.fromarray(overlay).save(after_dir / f"{stem}.png")

        csv_rows.append({
            "stem"       : stem,
            "status"     : "OK",
            "latency_s"  : round(elapsed, 6),
            "instances"  : n_inst,
            "has_crack"  : int(has_crack),
            "crack_px"   : crack_px,
            "total_px"   : orig_h * orig_w,
            "cover_pct"  : round(crack_px / (orig_h * orig_w) * 100, 4),
            "orig_w"     : orig_w,
            "orig_h"     : orig_h,
        })

        sys.stdout.write(
            f"\r  [{i+1}/{n_total}] {stem}  inst={n_inst}  time={elapsed:.3f}s"
        )
        sys.stdout.flush()

    inference_elapsed = time.perf_counter() - inference_start

    # ── 統計 ────────────────────────────────────────────────────────────────
    avg_t = float(np.mean(elapsed_list)) if elapsed_list else 0
    fps   = 1.0 / avg_t if avg_t > 0 else 0
    sep   = "-" * 100

    log_lines += [
        sep,
        "整體統計",
        f"成功處理    : {success_cnt} / {n_total} 張",
        f"有裂縫判定  : {crack_cnt} 張",
        f"總實例數    : {total_inst} 個",
        f"推論平均    : {avg_t*1000:.3f} ms/張",
        f"FPS         : {fps:.2f}",
        f"總耗時      : {inference_elapsed:.3f} 秒",
        sep,
        f"pred_masks → {pred_masks_dir}",
        f"before     → {before_dir}",
        f"after      → {after_dir}",
        f"CSV        → {csv_path}",
    ]

    # CSV
    fields = ["stem","status","latency_s","instances","has_crack","crack_px",
              "total_px","cover_pct","orig_w","orig_h"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(csv_rows)

    log_path = log_dir / f"inference_log_{ts}.txt"
    report   = "\n".join(log_lines)
    print(f"\n\n{report}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 完成 → {log_path}")


if __name__ == "__main__":
    main()
