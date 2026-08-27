"""
DDRNet FTL TorchScript Whole-image 推論腳本
公開套件部署版：不依賴 training.train_crackseg.build_model / data.transforms.get_test_transforms

功能：
  1. 直接載入 TorchScript 模型：ddrnet_ftl_torchscript.pt
  2. 直接讀取整個 RGB 資料夾內所有 .jpg/.jpeg/.png 圖片
  3. whole-image inference：不切 patch
  4. 統一 resize 到指定 target-size，例如 1024x1024
  5. 輸出 before / after 疊圖
  6. 輸出整體 log
  7. 每張圖片詳細結果輸出 CSV
  8. log / csv 檔名會標示 cpu 或 gpu
  9. 記錄 CPU / RAM / GPU VRAM 資源消耗

啟動方式：
  conda activate CrackSeg
  cd H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP
  python ddrnet_cfd_infernce.py --device cpu

GPU：
  python ddrnet_cfd_infernce.py --device cuda

指定輸入尺寸與閾值：
  python ddrnet_cfd_infernce.py --target-size 1024 1024 --threshold 0.8 --device cpu
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
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


# ── 路徑設定 ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(
    r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\checkpoints\ddrnet_ftl\ddrnet_ftl_torchscript.pt"
)

# ⚠️ 使用完全未見資料集（CFD），避免資料洩漏
# CFD 資料集從未用於 DDRNet FTL 訓練（訓練僅使用 v0 Mendeley 資料集）
RGB_DIR = Path(
    r"H:\ChihleeMaster\CrackK500\CFD\cfd_image"
)

OUT_BASE = Path(
    r"H:\ChihleeMaster\dev\final_outputs"
)

MODEL_NAME = "ddrnet_ftl_torchscript_whole_cfd"


# ── Normalize 設定 ──────────────────────────────────────────────────────────
# 你的原本 data/transforms.py 使用 ImageNet mean/std：
# _MEAN = (0.485, 0.456, 0.406)
# _STD  = (0.229, 0.224, 0.225)
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


# ── psutil 可選資源監控 ─────────────────────────────────────────────────────
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False


def bytes_to_mb(value: int) -> float:
    return value / (1024 ** 2)


def get_cpu_name() -> str:
    """盡量取得 CPU 型號。Windows 優先使用 wmic，失敗則回傳 platform 資訊。"""
    try:
        if platform.system().lower() == "windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            lines = [
                x.strip()
                for x in out.splitlines()
                if x.strip() and x.strip().lower() != "name"
            ]
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
    snapshot = {
        "process_ram_mb": 0.0,
        "process_cpu_percent": 0.0,
        "gpu_allocated_mb": 0.0,
        "gpu_reserved_mb": 0.0,
        "gpu_max_allocated_mb": 0.0,
        "gpu_max_reserved_mb": 0.0,
    }

    if PSUTIL_AVAILABLE:
        proc = psutil.Process(os.getpid())
        snapshot["process_ram_mb"] = bytes_to_mb(proc.memory_info().rss)
        snapshot["process_cpu_percent"] = proc.cpu_percent(interval=None)

    if device.type == "cuda" and torch.cuda.is_available():
        snapshot["gpu_allocated_mb"] = bytes_to_mb(torch.cuda.memory_allocated())
        snapshot["gpu_reserved_mb"] = bytes_to_mb(torch.cuda.memory_reserved())
        snapshot["gpu_max_allocated_mb"] = bytes_to_mb(torch.cuda.max_memory_allocated())
        snapshot["gpu_max_reserved_mb"] = bytes_to_mb(torch.cuda.max_memory_reserved())

    return snapshot


def build_preprocess(mean, std):
    """
    使用 torchvision 公開套件取代原本 get_test_transforms()。
    原本流程等價於：
      Normalize(ImageNet mean/std) + ToTensor
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color=(220, 50, 50),
    alpha=0.5,
) -> np.ndarray:
    """mask 白色區域疊合紅色到原始影像上"""
    out = rgb.copy().astype(np.float32)
    crack = mask >= 128

    for c, v in enumerate(color):
        out[:, :, c] = np.where(
            crack,
            out[:, :, c] * (1 - alpha) + v * alpha,
            out[:, :, c],
        )

    return out.clip(0, 255).astype(np.uint8)


def normalize_model_output(output: torch.Tensor) -> torch.Tensor:
    """
    將 TorchScript 模型輸出整理成 [H, W] logit。

    你的匯出測試輸出是：
      (1, 1, 1024, 1024)

    因此正常會走：
      [B, C, H, W] -> [H, W]
    """
    if isinstance(output, (tuple, list)):
        output = output[0]

    if isinstance(output, dict):
        for key in ["out", "logits", "pred", "mask"]:
            if key in output:
                output = output[key]
                break
        else:
            raise ValueError(f"模型輸出為 dict，但找不到可用 key：{list(output.keys())}")

    if output.dim() == 4:
        # [B, C, H, W]
        if output.shape[1] == 1:
            output = output[:, 0, :, :]
        else:
            # 若未來是多類別，這裡可改成 output[:, crack_class_id, :, :]
            output = output[:, 0, :, :]

    if output.dim() == 3:
        # [B, H, W]
        output = output[0]

    if output.dim() != 2:
        raise ValueError(f"不支援的模型輸出 shape：{tuple(output.shape)}")

    return output


def load_torchscript_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 TorchScript 模型：{model_path}")

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def infer_whole_image(
    image: np.ndarray,
    model,
    device: torch.device,
    preprocess,
    threshold: float,
    target_size: tuple[int, int],
) -> np.ndarray:
    """
    整張影像推論，不切 patch。

    流程：
      1. 原圖 resize 到 target_size
      2. ToTensor + Normalize
      3. TorchScript DDRNet 推論
      4. sigmoid + threshold
      5. mask resize 回原圖大小
    """
    original_h, original_w = image.shape[:2]
    target_w, target_h = target_size

    pil_resized = Image.fromarray(image).resize(
        (target_w, target_h),
        Image.BILINEAR,
    )

    tensor = preprocess(pil_resized).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(tensor)
        logit = normalize_model_output(output)

        if tuple(logit.shape[-2:]) != (target_h, target_w):
            logit = torch.nn.functional.interpolate(
                logit.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        prob = torch.sigmoid(logit)
        mask = (prob > threshold).to(torch.uint8).cpu().numpy() * 255

    mask = Image.fromarray(mask).resize(
        (original_w, original_h),
        Image.NEAREST,
    )

    return np.array(mask)


def list_rgb_files(rgb_dir: Path) -> list[Path]:
    if not rgb_dir.exists():
        raise FileNotFoundError(f"找不到 RGB 資料夾：{rgb_dir}")

    files = sorted([
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    if not files:
        raise FileNotFoundError(f"RGB 資料夾內找不到 jpg/jpeg/png 圖片：{rgb_dir}")

    return files


def main():
    program_start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="TorchScript DDRNet 模型路徑",
    )
    parser.add_argument(
        "--rgb-dir",
        type=str,
        default=str(RGB_DIR),
        help="RGB 圖片資料夾，會直接讀取整個資料夾",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="裂縫判定 threshold，預設 0.8",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        metavar=("WIDTH", "HEIGHT"),
        help="whole-image 推論前 resize 尺寸，例如 --target-size 1024 1024",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="運算裝置：auto / cpu / cuda",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=DEFAULT_MEAN,
        help="Normalize mean，預設 ImageNet mean",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=DEFAULT_STD,
        help="Normalize std，預設 ImageNet std",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    rgb_dir = Path(args.rgb_dir)
    threshold = args.threshold
    target_size = tuple(args.target_size)
    device = select_device(args.device)
    device_info = get_device_info(device)
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

    print(f"Model      : {model_path}")
    print(f"RGB dir    : {rgb_dir}")
    print(f"Device     : {device}")
    print(f"Threshold  : {threshold}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")

    preprocess = build_preprocess(args.mean, args.std)
    model = load_torchscript_model(model_path, device)

    reset_gpu_peak_memory(device)
    resource_start = get_resource_snapshot(device)

    rgb_files = list_rgb_files(rgb_dir)

    log_lines = [
        "DDRNet FTL TorchScript Whole-image 推論記錄",
        f"時間        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"模型檔案    : {model_path}",
        f"測試圖片來源 : {rgb_dir}",
        "資料集      : CFD（Crack Forest Dataset）— 完全未見資料集，無資料洩漏",
        "讀取方式    : 直接讀取資料夾內所有 jpg/jpeg/png 圖片",
        f"閾值        : {threshold}",
        "信心度設定  : 0.8（預設；可用 --threshold 覆蓋）",
        f"輸入尺寸    : {target_size[0]}x{target_size[1]}",
        f"影像數      : {len(rgb_files)}",
        f"輸出目錄    : {OUT_BASE}",
        f"逐張圖片 CSV: {csv_path}",
        "-" * 100,
        "硬體資訊",
        f"運算裝置    : {device_info['device']}",
        f"CPU         : {device_info['cpu_name']}",
        f"CUDA 可用   : {device_info['cuda_available']}",
        f"GPU         : {device_info['gpu_name']}",
        f"GPU 記憶體  : {device_info['gpu_total_memory_gb']}",
        f"PyTorch     : {device_info['torch_version']}",
        f"psutil 可用 : {PSUTIL_AVAILABLE}",
        "-" * 100,
    ]

    csv_rows = []

    success_count = 0
    fail_count = 0
    crack_image_count = 0
    no_crack_image_count = 0
    total_crack_px = 0

    coverage_list = []
    elapsed_list = []
    ram_list = []
    cpu_percent_list = []
    gpu_allocated_list = []
    gpu_reserved_list = []

    print(f"Processing {len(rgb_files)} images...")
    inference_start = time.perf_counter()

    for rgb_p in tqdm(rgb_files, desc="DDRNet TorchScript Whole"):
        stem = rgb_p.stem

        try:
            item_start = time.perf_counter()

            rgb = np.array(Image.open(rgb_p).convert("RGB"))

            mask = infer_whole_image(
                image=rgb,
                model=model,
                device=device,
                preprocess=preprocess,
                threshold=threshold,
                target_size=target_size,
            )

            shutil.copy(str(rgb_p), before_dir / f"{stem}.png")

            # 儲存純二值 mask（供 GT 評估使用）
            Image.fromarray(mask).save(pred_masks_dir / f"{stem}.png")

            overlay = overlay_mask(rgb, mask)
            Image.fromarray(overlay).save(after_dir / f"{stem}.png")

            crack_px = int((mask >= 128).sum())
            coverage = crack_px / (mask.shape[0] * mask.shape[1]) * 100.0
            judged = "有裂縫" if crack_px > 0 else "無裂縫"

            elapsed = time.perf_counter() - item_start
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
                "image_id": stem,
                "filename": rgb_p.name,
                "status": "OK",
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
                "image_id": stem,
                "filename": rgb_p.name,
                "status": f"FAILED: {type(e).__name__}: {e}",
                "crack_pixels": "",
                "coverage_percent": "",
                "judgement": "",
                "elapsed_sec": "",
                "process_ram_mb": "",
                "process_cpu_percent": "",
                "gpu_allocated_mb": "",
                "gpu_reserved_mb": "",
            })

    inference_elapsed = time.perf_counter() - inference_start
    program_elapsed = time.perf_counter() - program_start
    resource_end = get_resource_snapshot(device)

    judge_rate = (crack_image_count / success_count * 100.0) if success_count > 0 else 0.0
    avg_coverage = float(np.mean(coverage_list)) if coverage_list else 0.0
    avg_time = float(np.mean(elapsed_list)) if elapsed_list else 0.0
    min_time = float(np.min(elapsed_list)) if elapsed_list else 0.0
    max_time = float(np.max(elapsed_list)) if elapsed_list else 0.0

    avg_ram = float(np.mean(ram_list)) if ram_list else 0.0
    peak_ram = float(np.max(ram_list)) if ram_list else 0.0
    avg_cpu = float(np.mean(cpu_percent_list)) if cpu_percent_list else 0.0
    peak_cpu = float(np.max(cpu_percent_list)) if cpu_percent_list else 0.0
    avg_gpu_allocated = float(np.mean(gpu_allocated_list)) if gpu_allocated_list else 0.0
    peak_gpu_allocated = float(np.max(gpu_allocated_list)) if gpu_allocated_list else 0.0
    avg_gpu_reserved = float(np.mean(gpu_reserved_list)) if gpu_reserved_list else 0.0
    peak_gpu_reserved = float(np.max(gpu_reserved_list)) if gpu_reserved_list else 0.0

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
        f"平均 GPU allocated : {avg_gpu_allocated:.2f} MB",
        f"峰值 GPU allocated : {peak_gpu_allocated:.2f} MB",
        f"平均 GPU reserved  : {avg_gpu_reserved:.2f} MB",
        f"峰值 GPU reserved  : {peak_gpu_reserved:.2f} MB",
        f"PyTorch GPU peak allocated : {resource_end['gpu_max_allocated_mb']:.2f} MB",
        f"PyTorch GPU peak reserved  : {resource_end['gpu_max_reserved_mb']:.2f} MB",
        "-" * 100,
        f"before 目錄: {before_dir}",
        f"after  目錄: {after_dir}",
        f"CSV 檔案   : {csv_path}",
    ]

    csv_fields = [
        "image_id",
        "filename",
        "status",
        "crack_pixels",
        "coverage_percent",
        "judgement",
        "elapsed_sec",
        "process_ram_mb",
        "process_cpu_percent",
        "gpu_allocated_mb",
        "gpu_reserved_mb",
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("\n✅ 完成")
    print(f"   logs      → {log_path}")
    print(f"   csv       → {csv_path}")
    print(f"   before    → {before_dir}")
    print(f"   after     → {after_dir}")
    print(f"   pred_masks→ {pred_masks_dir}")


if __name__ == "__main__":
    main()
