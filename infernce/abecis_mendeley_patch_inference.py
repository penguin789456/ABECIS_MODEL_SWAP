"""
ABECIS Detectron2 Mendeley 凍結測試集 — 512×512 Patch 滑動視窗推論腳本

Mendeley 影像為 4032×3024，必須切 patch 才能匹配訓練尺度
（與 4 個語意分割模型的 mendeley_patch_inference.py 完全對齊）。

對偵測式模型（Mask R-CNN）的處理方式：
  - 在每個 512×512 patch 各自推論
  - 將該 patch 內所有實例遮罩「聯集」寫回全圖對應位置
  - 重疊區域以 OR 合併（二值語意遮罩，無 NMS 跨邊界問題）

啟動方式（ABECIS conda env）：
  conda activate ABECIS
  cd H:\\ChihleeMaster\\dev\\final_outputs\\infernce
  python abecis_mendeley_patch_inference.py --threshold 0.8 --device cpu

輸出（不覆蓋全圖結果）：
  pred_masks/abecis_mendeley_patch/{device}/
  logs/abecis_mendeley_patch/{device}/inference_log_{ts}.txt
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
RGB_DIR    = Path(r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\rgb')
TEST_SPLIT = Path(r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt')
OUT_BASE   = Path(r'H:\ChihleeMaster\dev\final_outputs')
MODEL_NAME = 'abecis_mendeley_patch'
CKPT_PATH  = r'H:\ChihleeMaster\CrackPreVer3.5.3\ABECIS-main\output\model_final.pth'

PATCH_SIZE = 512
OVERLAP    = 128
STRIDE     = PATCH_SIZE - OVERLAP   # 384


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


def get_starts(length: int) -> list:
    """產生起始座標列表（確保覆蓋右 / 下邊緣）"""
    starts = list(range(0, max(length - PATCH_SIZE, 0) + 1, STRIDE))
    last = max(length - PATCH_SIZE, 0)
    if starts and starts[-1] != last:
        starts.append(last)
    if not starts:
        starts = [0]
    return starts


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


def infer_patch_sliding(predictor, im_bgr: np.ndarray) -> tuple:
    """
    512×512 滑動視窗推論，將各 patch 實例遮罩聯集回全圖。

    Returns:
        combined (np.uint8 全圖二值遮罩 0/255), total_instances (int)
    """
    orig_h, orig_w = im_bgr.shape[:2]
    combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    total_inst = 0

    ys = get_starts(orig_h) if orig_h >= PATCH_SIZE else [0]
    xs = get_starts(orig_w) if orig_w >= PATCH_SIZE else [0]

    for y1 in ys:
        y2 = min(y1 + PATCH_SIZE, orig_h)
        ph = y2 - y1
        for x1 in xs:
            x2 = min(x1 + PATCH_SIZE, orig_w)
            pw = x2 - x1

            # 裁取 patch；不足 512 則 pad 到 512
            patch = im_bgr[y1:y2, x1:x2]
            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                pad = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=im_bgr.dtype)
                pad[:ph, :pw] = patch
                patch = pad

            outputs   = predictor(patch)
            instances = outputs["instances"].to("cpu")
            n_inst    = len(instances)
            total_inst += n_inst

            if n_inst > 0 and instances.has("pred_masks"):
                patch_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                for m in instances.pred_masks.numpy():
                    patch_mask = np.where(m, 255, patch_mask)
                # 只寫回有效區域，OR 合併
                valid = patch_mask[:ph, :pw]
                combined[y1:y2, x1:x2] = np.maximum(combined[y1:y2, x1:x2], valid)

    return combined, total_inst


def main():
    program_start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true",
                        help="跳過 pred_masks 已存在的影像（斷點續傳）")
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

    # ── 載入模型（patch 推論：MIN/MAX_SIZE_TEST=512）─────────────────────────
    print(f"Loading ABECIS Detectron2 (SCORE_THRESH={threshold})...")
    print(f"Device : {device}")
    print(f"Mode   : 512×512 滑動視窗，overlap={OVERLAP}，stride={STRIDE}")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = CKPT_PATH
    cfg.MODEL.DEVICE  = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    # patch 固定 512，不再額外 resize
    cfg.INPUT.MIN_SIZE_TEST = PATCH_SIZE
    cfg.INPUT.MAX_SIZE_TEST = PATCH_SIZE

    predictor = DefaultPredictor(cfg)
    print("Model loaded.\n")

    # ── 讀 test split ───────────────────────────────────────────────────────
    stems_raw = [l.strip() for l in
                 TEST_SPLIT.read_text(encoding="utf-8").splitlines() if l.strip()]
    stems = [s.zfill(3) if s.isdigit() else s for s in stems_raw]
    rgb_index = {p.stem.lower(): p for p in RGB_DIR.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    image_paths = [(s, rgb_index[s.lower()]) for s in stems if s.lower() in rgb_index]

    # ── 斷點續傳：跳過 pred_masks 已存在者 ──────────────────────────────────
    if args.resume:
        before = len(image_paths)
        image_paths = [(s, p) for (s, p) in image_paths
                       if not (pred_masks_dir / f"{s}.png").exists()]
        skipped = before - len(image_paths)
        print(f"斷點續傳：跳過 {skipped} 張已完成，剩 {len(image_paths)} 張")

    n_total = len(image_paths)
    print(f"影像數 : {n_total} 張（test.txt 凍結測試集，512 patch 滑動視窗）")

    log_lines = [
        f"ABECIS Detectron2 Mendeley Patch 推論記錄",
        f"時間        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint  : {CKPT_PATH}",
        f"閾值        : {threshold}",
        f"推論方式    : 512×512 滑動視窗，overlap={OVERLAP}，stride={STRIDE}",
        f"影像數      : {n_total}",
        f"資料集      : Mendeley CCSD 凍結測試集（70 張）",
        f"RGB 來源    : {RGB_DIR}（test.txt）",
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

    for i, (stem, img_path) in enumerate(image_paths):
        t0     = time.perf_counter()
        im_bgr = cv2.imread(str(img_path))
        if im_bgr is None:
            csv_rows.append({"stem": stem, "status": "read_failed"})
            continue

        orig_h, orig_w = im_bgr.shape[:2]
        combined, n_inst = infer_patch_sliding(predictor, im_bgr)
        elapsed = time.perf_counter() - t0

        crack_px  = int((combined >= 128).sum())
        has_crack = crack_px > 0
        if has_crack:
            crack_cnt += 1
        elapsed_list.append(elapsed)
        success_cnt += 1
        total_inst  += n_inst

        shutil.copy(str(img_path), before_dir / f"{stem}.png")
        Image.fromarray(combined).save(pred_masks_dir / f"{stem}.png")
        rgb_np  = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        overlay = overlay_mask(rgb_np, combined)
        Image.fromarray(overlay).save(after_dir / f"{stem}.png")

        csv_rows.append({
            "stem"      : stem,
            "status"    : "OK",
            "latency_s" : round(elapsed, 6),
            "instances" : n_inst,
            "has_crack" : int(has_crack),
            "crack_px"  : crack_px,
            "total_px"  : orig_h * orig_w,
            "cover_pct" : round(crack_px / (orig_h * orig_w) * 100, 4),
            "orig_w"    : orig_w,
            "orig_h"    : orig_h,
        })

        sys.stdout.write(
            f"\r  [{i+1}/{n_total}] {stem}  inst={n_inst}  time={elapsed:.2f}s"
        )
        sys.stdout.flush()

    inference_elapsed = time.perf_counter() - inference_start

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
