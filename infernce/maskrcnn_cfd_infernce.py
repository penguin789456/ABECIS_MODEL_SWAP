"""
Mask R-CNN (torchvision) Whole-image 推論腳本 — CFD 資料集

功能：
  1. 載入 torchvision Mask R-CNN R50-FPN checkpoint（.pth）
  2. 讀取 CFD 完全未見資料集所有圖片
  3. whole-image inference：resize 1024x1024，合併 instance masks → 二元 mask
  4. 輸出 before / after 疊圖、純二值 pred_mask
  5. log / csv 檔名加 timestamp 避免覆蓋

注意：Mask R-CNN 輸出為 list[dict]，不支援 torch.jit.trace，故直接載入 .pth。

啟動方式：
  conda activate CrackSeg
  cd H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP
  python "H:\ChihleeMaster\dev\final_outputs\infernce\maskrcnn_cfd_infernce.py" --device cuda
  python "H:\ChihleeMaster\dev\final_outputs\infernce\maskrcnn_cfd_infernce.py" --device cpu

輸出：
  H:\ChihleeMaster\dev\final_outputs\
    logs\maskrcnn_cfd\{gpu|cpu}\inference_log_{timestamp}.txt
    logs\maskrcnn_cfd\{gpu|cpu}\per_image_{timestamp}.csv
    before\maskrcnn_cfd\{gpu|cpu}\*.png
    after\maskrcnn_cfd\{gpu|cpu}\*.png
    pred_masks\maskrcnn_cfd\{gpu|cpu}\*.png
"""

import os
import sys
import csv
import time
import shutil
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ── 路徑設定 ────────────────────────────────────────────────────────────────
CKPT_PATH  = Path(
    r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\checkpoints\maskrcnn\best.pth"
)
RGB_DIR    = Path(r"H:\ChihleeMaster\CrackK500\CFD\cfd_image")
OUT_BASE   = Path(r"H:\ChihleeMaster\dev\final_outputs")
MODEL_NAME = "maskrcnn_cfd"


# ── psutil 可選 ─────────────────────────────────────────────────────────────
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False


def bytes_to_mb(v: int) -> float:
    return v / (1024 ** 2)


def get_cpu_name() -> str:
    try:
        if platform.system().lower() == "windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                stderr=subprocess.DEVNULL, text=True,
            )
            lines = [x.strip() for x in out.splitlines()
                     if x.strip() and x.strip().lower() != "name"]
            if lines:
                return lines[0]
    except Exception:
        pass
    cpu = platform.processor()
    return cpu if cpu else platform.machine()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("⚠️ 指定 cuda，但目前 CUDA 不可用，改用 CPU。")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_device_info(device: torch.device) -> dict:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cpu_name": get_cpu_name(),
        "gpu_name": "N/A",
        "gpu_total_memory_gb": "N/A",
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_total_memory_gb"] = f"{prop.total_memory / (1024 ** 3):.2f} GB"
    return info


def reset_gpu_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_resource_snapshot(device: torch.device) -> dict:
    s = {
        "process_ram_mb": 0.0, "process_cpu_percent": 0.0,
        "gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0,
        "gpu_max_allocated_mb": 0.0, "gpu_max_reserved_mb": 0.0,
    }
    if PSUTIL_AVAILABLE:
        proc = psutil.Process(os.getpid())
        s["process_ram_mb"] = bytes_to_mb(proc.memory_info().rss)
        s["process_cpu_percent"] = proc.cpu_percent(interval=None)
    if device.type == "cuda" and torch.cuda.is_available():
        s["gpu_allocated_mb"] = bytes_to_mb(torch.cuda.memory_allocated())
        s["gpu_reserved_mb"] = bytes_to_mb(torch.cuda.memory_reserved())
        s["gpu_max_allocated_mb"] = bytes_to_mb(torch.cuda.max_memory_allocated())
        s["gpu_max_reserved_mb"] = bytes_to_mb(torch.cuda.max_memory_reserved())
    return s


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


def overlay_mask(rgb: np.ndarray, mask: np.ndarray,
                 color=(220, 50, 50), alpha=0.5) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    crack = mask >= 128
    for c, v in enumerate(color):
        out[:, :, c] = np.where(crack, out[:, :, c] * (1 - alpha) + v * alpha, out[:, :, c])
    return out.clip(0, 255).astype(np.uint8)


def infer_maskrcnn(image: np.ndarray, model, device: torch.device,
                   threshold: float, score_threshold: float,
                   target_size: tuple) -> np.ndarray:
    """
    全圖推論：resize → torchvision MaskRCNN → 合併 instance masks → resize back
    threshold      : mask pixel threshold（instance mask confidence）
    score_threshold: instance score threshold（default 0.8）
    """
    original_h, original_w = image.shape[:2]
    target_w, target_h = target_size

    pil_resized = Image.fromarray(image).resize((target_w, target_h), Image.BILINEAR)
    tensor = TF.to_tensor(pil_resized).to(device)

    with torch.no_grad():
        preds = model([tensor])[0]

    binary = np.zeros((target_h, target_w), np.uint8)
    scores = preds["scores"].cpu().numpy()
    masks  = preds["masks"]   # shape (N, 1, H, W), values in [0,1]

    for i, score in enumerate(scores):
        if score >= score_threshold:
            m = (masks[i, 0].cpu().numpy() > threshold).astype(np.uint8)
            binary = np.maximum(binary, m * 255)

    # resize back to original size
    binary = np.array(
        Image.fromarray(binary).resize((original_w, original_h), Image.NEAREST)
    )
    return binary


def list_rgb_files(rgb_dir: Path) -> list:
    if not rgb_dir.exists():
        raise FileNotFoundError(f"找不到 RGB 資料夾：{rgb_dir}")
    files = sorted([p for p in rgb_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        raise FileNotFoundError(f"找不到圖片：{rgb_dir}")
    return files


def main():
    program_start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path",      type=str,   default=str(CKPT_PATH))
    parser.add_argument("--rgb-dir",        type=str,   default=str(RGB_DIR))
    parser.add_argument("--threshold",      type=float, default=0.5,
                        help="Instance mask pixel threshold（default 0.5）")
    parser.add_argument("--score-threshold",type=float, default=0.8,
                        help="Instance score threshold（default 0.8）")
    parser.add_argument("--target-size",    type=int,   nargs=2, default=[1024, 1024],
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--device",         choices=["auto","cpu","cuda"], default="auto")
    args = parser.parse_args()

    ckpt_path     = Path(args.ckpt_path)
    rgb_dir       = Path(args.rgb_dir)
    threshold     = args.threshold
    score_thresh  = args.score_threshold
    target_size   = tuple(args.target_size)
    device        = select_device(args.device)
    device_info   = get_device_info(device)
    device_suffix = "gpu" if device.type == "cuda" else "cpu"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir        = OUT_BASE / "logs"       / MODEL_NAME / device_suffix
    before_dir     = OUT_BASE / "before"     / MODEL_NAME / device_suffix
    after_dir      = OUT_BASE / "after"      / MODEL_NAME / device_suffix
    pred_masks_dir = OUT_BASE / "pred_masks" / MODEL_NAME / device_suffix

    for d in [log_dir, before_dir, after_dir, pred_masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"inference_log_{ts}.txt"
    csv_path = log_dir / f"per_image_{ts}.csv"

    print(f"Checkpoint   : {ckpt_path}")
    print(f"RGB dir      : {rgb_dir}")
    print(f"Device       : {device}")
    print(f"Score thresh : {score_thresh}")
    print(f"Mask thresh  : {threshold}")
    print(f"Target size  : {target_size[0]}x{target_size[1]}")

    # 載入模型
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")
    model = build_maskrcnn(num_classes=2)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()
    print("Model loaded.\n")

    reset_gpu_peak_memory(device)
    resource_start = get_resource_snapshot(device)

    rgb_files = list_rgb_files(rgb_dir)

    log_lines = [
        "Mask R-CNN R50-FPN (torchvision) Whole-image 推論記錄 — CFD",
        f"時間         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"模型名稱     : {MODEL_NAME}",
        f"Checkpoint   : {ckpt_path}",
        f"測試圖片來源 : {rgb_dir}",
        "資料集       : CFD（Crack Forest Dataset）— 完全未見資料集，無資料洩漏",
        f"Score 閾值   : {score_thresh}",
        f"Mask 閾值    : {threshold}",
        f"輸入尺寸     : {target_size[0]}x{target_size[1]}",
        f"影像數       : {len(rgb_files)}",
        "-" * 100,
        "硬體資訊",
        f"運算裝置     : {device_info['device']}",
        f"CPU          : {device_info['cpu_name']}",
        f"CUDA 可用    : {device_info['cuda_available']}",
        f"GPU          : {device_info['gpu_name']}",
        f"GPU 記憶體   : {device_info['gpu_total_memory_gb']}",
        f"PyTorch      : {device_info['torch_version']}",
        f"psutil 可用  : {PSUTIL_AVAILABLE}",
        "-" * 100,
    ]

    csv_rows = []
    success_count = fail_count = crack_image_count = no_crack_image_count = total_crack_px = 0
    coverage_list, elapsed_list = [], []
    ram_list, cpu_percent_list = [], []
    gpu_allocated_list, gpu_reserved_list = [], []

    print(f"Processing {len(rgb_files)} images...")
    inference_start = time.perf_counter()

    for rgb_p in tqdm(rgb_files, desc=MODEL_NAME):
        stem = rgb_p.stem
        try:
            item_start = time.perf_counter()
            rgb = np.array(Image.open(rgb_p).convert("RGB"))

            mask = infer_maskrcnn(
                image=rgb, model=model, device=device,
                threshold=threshold, score_threshold=score_thresh,
                target_size=target_size,
            )

            shutil.copy(str(rgb_p), before_dir / f"{stem}.png")
            Image.fromarray(mask).save(pred_masks_dir / f"{stem}.png")
            overlay = overlay_mask(rgb, mask)
            Image.fromarray(overlay).save(after_dir / f"{stem}.png")

            crack_px = int((mask >= 128).sum())
            coverage = crack_px / (mask.shape[0] * mask.shape[1]) * 100.0
            judged   = "有裂縫" if crack_px > 0 else "無裂縫"
            elapsed  = time.perf_counter() - item_start
            resource_now = get_resource_snapshot(device)

            success_count += 1
            total_crack_px += crack_px
            coverage_list.append(coverage)
            elapsed_list.append(elapsed)
            ram_list.append(resource_now["process_ram_mb"])
            cpu_percent_list.append(resource_now["process_cpu_percent"])
            gpu_allocated_list.append(resource_now["gpu_allocated_mb"])
            gpu_reserved_list.append(resource_now["gpu_reserved_mb"])

            if crack_px > 0:
                crack_image_count += 1
            else:
                no_crack_image_count += 1

            csv_rows.append({
                "image_id": stem, "filename": rgb_p.name, "status": "OK",
                "crack_pixels": crack_px,
                "coverage_percent": round(coverage, 6),
                "judgement": judged,
                "elapsed_sec": round(elapsed, 6),
                "process_ram_mb": round(resource_now["process_ram_mb"], 3),
                "process_cpu_percent": round(resource_now["process_cpu_percent"], 3),
                "gpu_allocated_mb": round(resource_now["gpu_allocated_mb"], 3),
                "gpu_reserved_mb": round(resource_now["gpu_reserved_mb"], 3),
            })

        except Exception as e:
            fail_count += 1
            csv_rows.append({
                "image_id": stem, "filename": rgb_p.name,
                "status": f"FAILED: {type(e).__name__}: {e}",
                "crack_pixels": "", "coverage_percent": "", "judgement": "",
                "elapsed_sec": "", "process_ram_mb": "", "process_cpu_percent": "",
                "gpu_allocated_mb": "", "gpu_reserved_mb": "",
            })

    inference_elapsed = time.perf_counter() - inference_start
    program_elapsed   = time.perf_counter() - program_start
    resource_end      = get_resource_snapshot(device)

    judge_rate     = (crack_image_count / success_count * 100.0) if success_count > 0 else 0.0
    avg_coverage   = float(np.mean(coverage_list))    if coverage_list    else 0.0
    avg_time       = float(np.mean(elapsed_list))     if elapsed_list     else 0.0
    min_time       = float(np.min(elapsed_list))      if elapsed_list     else 0.0
    max_time       = float(np.max(elapsed_list))      if elapsed_list     else 0.0
    avg_ram        = float(np.mean(ram_list))         if ram_list         else 0.0
    peak_ram       = float(np.max(ram_list))          if ram_list         else 0.0
    avg_cpu        = float(np.mean(cpu_percent_list)) if cpu_percent_list else 0.0
    peak_cpu       = float(np.max(cpu_percent_list))  if cpu_percent_list else 0.0
    avg_gpu_alloc  = float(np.mean(gpu_allocated_list)) if gpu_allocated_list else 0.0
    peak_gpu_alloc = float(np.max(gpu_allocated_list))  if gpu_allocated_list else 0.0
    avg_gpu_res    = float(np.mean(gpu_reserved_list))  if gpu_reserved_list  else 0.0
    peak_gpu_res   = float(np.max(gpu_reserved_list))   if gpu_reserved_list  else 0.0

    log_lines += [
        "整體統計",
        f"總影像數          : {len(rgb_files)} 張",
        f"成功處理影像數    : {success_count} 張",
        f"失敗影像數        : {fail_count} 張",
        f"判定有裂縫影像數  : {crack_image_count} 張",
        f"判定無裂縫影像數  : {no_crack_image_count} 張",
        f"判斷率            : {judge_rate:.2f}%",
        f"總裂縫像素數      : {total_crack_px}px",
        f"平均覆蓋率        : {avg_coverage:.6f}%",
        f"推論處理總耗時    : {inference_elapsed:.3f} 秒",
        f"程式總耗時        : {program_elapsed:.3f} 秒",
        f"平均每張耗時      : {avg_time:.6f} 秒",
        f"最快單張耗時      : {min_time:.6f} 秒",
        f"最慢單張耗時      : {max_time:.6f} 秒",
        "-" * 100,
        "計算資源消耗",
        f"開始 Process RAM   : {resource_start['process_ram_mb']:.2f} MB",
        f"結束 Process RAM   : {resource_end['process_ram_mb']:.2f} MB",
        f"平均 Process RAM   : {avg_ram:.2f} MB",
        f"峰值 Process RAM   : {peak_ram:.2f} MB",
        f"平均 Process CPU   : {avg_cpu:.2f}%",
        f"峰值 Process CPU   : {peak_cpu:.2f}%",
        f"平均 GPU allocated : {avg_gpu_alloc:.2f} MB",
        f"峰值 GPU allocated : {peak_gpu_alloc:.2f} MB",
        f"平均 GPU reserved  : {avg_gpu_res:.2f} MB",
        f"峰值 GPU reserved  : {peak_gpu_res:.2f} MB",
        f"PyTorch GPU peak allocated : {resource_end['gpu_max_allocated_mb']:.2f} MB",
        f"PyTorch GPU peak reserved  : {resource_end['gpu_max_reserved_mb']:.2f} MB",
        "-" * 100,
        f"before 目錄    : {before_dir}",
        f"after  目錄    : {after_dir}",
        f"pred_masks 目錄: {pred_masks_dir}",
        f"CSV 檔案       : {csv_path}",
    ]

    csv_fields = [
        "image_id", "filename", "status", "crack_pixels", "coverage_percent",
        "judgement", "elapsed_sec", "process_ram_mb", "process_cpu_percent",
        "gpu_allocated_mb", "gpu_reserved_mb",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("\n✅ 完成")
    print(f"   log       → {log_path}")
    print(f"   csv       → {csv_path}")
    print(f"   pred_masks→ {pred_masks_dir}")


if __name__ == "__main__":
    main()
