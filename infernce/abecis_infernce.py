"""
ABECIS Detectron2 whole-image 推論腳本 — 統一比較版
輸出目錄：H:\ChihleeMaster\dev\final_outputs\
  logs/   → abecis_detectron2_whole_log.txt
  before/ → 原始 RGB 影像
  after/  → 原始影像 + 紅色裂縫遮罩疊合圖

目的：
  與 DDRNet 採用相同 whole-image inference 條件比較。
  本腳本會先將完整影像 resize 到固定尺寸後推論，不切 patch。

啟動方式（ABECIS conda env）：
  conda activate ABECIS
  cd H:\ChihleeMaster\CrackPreVer3.5.3\ABECIS-main
  python run_abecis_whole_final.py

指定設定：
  python run_abecis_whole_final.py --threshold 0.8 --target-size 1024 1024 --device auto
"""

import sys, os, argparse, shutil, time, platform, subprocess, csv
from pathlib import Path
from datetime import datetime

ABECIS_DIR = Path(__file__).resolve().parent
os.chdir(str(ABECIS_DIR))
sys.path.insert(0, str(ABECIS_DIR))

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
# ⚠️ 使用完全未見資料集（CFD），避免資料洩漏
# CFD 資料集從未用於 ABECIS Detectron2 訓練
RGB_DIR    = r'H:\ChihleeMaster\CrackK500\CFD\cfd_image'
SPLITS     = None  # 不使用 test.txt，直接讀取整個 RGB_DIR 資料夾
OUT_BASE   = Path(r'H:\ChihleeMaster\dev\final_outputs')
MODEL_NAME = 'abecis_detectron2_whole_cfd'


def get_cpu_name() -> str:
    """盡量取得 CPU 型號。Windows 優先使用 wmic，失敗則回傳 platform 資訊。"""
    try:
        if platform.system().lower() == "windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                stderr=subprocess.DEVNULL,
                text=True
            )
            lines = [x.strip() for x in out.splitlines() if x.strip() and x.strip().lower() != "name"]
            if lines:
                return lines[0]
    except Exception:
        pass

    cpu = platform.processor()
    return cpu if cpu else platform.machine()


def select_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("⚠️ 指定 cuda，但目前 CUDA 不可用，改用 CPU。")
        return "cpu"
    return device_arg


def get_device_info(device: str) -> dict:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "cpu_name": get_cpu_name(),
        "gpu_name": "N/A",
        "gpu_total_memory_gb": "N/A",
    }

    if device == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_total_memory_gb"] = f"{prop.total_memory / (1024 ** 3):.2f} GB"

    return info



# ── 資源監控工具 ─────────────────────────────────────────────────────────────
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    _PSUTIL_AVAILABLE = False


def bytes_to_mb(value: int) -> float:
    return value / (1024 ** 2)


def bytes_to_gb(value: int) -> float:
    return value / (1024 ** 3)


def get_process_resource_snapshot(device=None) -> dict:
    """
    取得目前程式的資源使用狀況。
    - RSS RAM：目前 Python process 使用的實體記憶體
    - CPU %：process CPU 使用率，需搭配前一次 cpu_percent 呼叫才準
    - GPU allocated/reserved：PyTorch 管理的 GPU 記憶體
    """
    snapshot = {
        "psutil_available": _PSUTIL_AVAILABLE,
        "process_ram_mb": 0.0,
        "process_cpu_percent": 0.0,
        "gpu_allocated_mb": 0.0,
        "gpu_reserved_mb": 0.0,
        "gpu_max_allocated_mb": 0.0,
        "gpu_max_reserved_mb": 0.0,
    }

    if _PSUTIL_AVAILABLE:
        proc = psutil.Process(os.getpid())
        snapshot["process_ram_mb"] = bytes_to_mb(proc.memory_info().rss)
        snapshot["process_cpu_percent"] = proc.cpu_percent(interval=None)

    if device is not None:
        device_type = device if isinstance(device, str) else getattr(device, "type", str(device))
        if device_type == "cuda" and torch.cuda.is_available():
            snapshot["gpu_allocated_mb"] = bytes_to_mb(torch.cuda.memory_allocated())
            snapshot["gpu_reserved_mb"] = bytes_to_mb(torch.cuda.memory_reserved())
            snapshot["gpu_max_allocated_mb"] = bytes_to_mb(torch.cuda.max_memory_allocated())
            snapshot["gpu_max_reserved_mb"] = bytes_to_mb(torch.cuda.max_memory_reserved())

    return snapshot


def reset_gpu_peak_memory(device=None):
    device_type = device if isinstance(device, str) else getattr(device, "type", str(device))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def format_resource_line(prefix: str, snapshot: dict) -> list:
    return [
        f"{prefix} Process RAM       : {snapshot['process_ram_mb']:.2f} MB",
        f"{prefix} Process CPU       : {snapshot['process_cpu_percent']:.2f}%",
        f"{prefix} GPU allocated     : {snapshot['gpu_allocated_mb']:.2f} MB",
        f"{prefix} GPU reserved      : {snapshot['gpu_reserved_mb']:.2f} MB",
        f"{prefix} GPU max allocated : {snapshot['gpu_max_allocated_mb']:.2f} MB",
        f"{prefix} GPU max reserved  : {snapshot['gpu_max_reserved_mb']:.2f} MB",
    ]


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(220, 50, 50), alpha=0.5) -> np.ndarray:
    """mask 白色區域疊合紅色到原始影像上"""
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
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Detectron2 SCORE_THRESH_TEST（預設 0.8）")
    parser.add_argument("--target-size", type=int, nargs=2, default=[1024, 1024],
                        metavar=("WIDTH", "HEIGHT"),
                        help="whole-image 推論前統一 resize 尺寸，例如 --target-size 1024 1024")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="運算裝置：auto / cpu / cuda")
    args = parser.parse_args()

    threshold = args.threshold
    target_w, target_h = args.target_size
    device = select_device(args.device)
    device_info = get_device_info(device)
    device_suffix = 'gpu' if device == 'cuda' else 'cpu'

    # 建立輸出目錄
    ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir        = OUT_BASE / "logs"       / MODEL_NAME / device_suffix
    before_dir     = OUT_BASE / "before"     / MODEL_NAME / device_suffix
    after_dir      = OUT_BASE / "after"      / MODEL_NAME / device_suffix
    pred_masks_dir = OUT_BASE / "pred_masks" / MODEL_NAME / device_suffix
    for d in [log_dir, before_dir, after_dir, pred_masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 載入模型
    print(f"Loading ABECIS Detectron2 (SCORE_THRESH={threshold})...")
    print(f"Device     : {device}")
    print(f"Target size: {target_w}x{target_h}")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS = r"H:\ChihleeMaster\CrackPreVer3.5.3\ABECIS-main\output\model_final.pth"
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    # 重要：
    # Detectron2 本身也可能做 resize。
    # 這裡把 MIN_SIZE_TEST / MAX_SIZE_TEST 設為 target 尺寸，盡量讓輸入條件一致。
    cfg.INPUT.MIN_SIZE_TEST = target_h
    cfg.INPUT.MAX_SIZE_TEST = max(target_w, target_h)

    predictor = DefaultPredictor(cfg)
    print("Model loaded.\n")

    reset_gpu_peak_memory(device)
    resource_start = get_process_resource_snapshot(device)

    rgb_files = sorted([
        p for p in Path(RGB_DIR).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    stems = [p.stem for p in rgb_files]

    rgb_index = {Path(p).stem.lower(): str(p) for p in rgb_files}

    log_path = log_dir / f"inference_log_{ts}.txt"
    csv_path = log_dir / f"per_image_{ts}.csv"
    csv_rows = []
    log_lines = [
        "ABECIS Detectron2 Whole-image 推論記錄",
        f"時間        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Checkpoint  : H:\\ChihleeMaster\\CrackPreVer3.5.3\\ABECIS-main\\output\\model_final.pth",
        f"閾值        : {threshold}",
        "信心度設定  : 0.8（預設；可用 --threshold 覆蓋）",
        f"輸入尺寸    : {target_w}x{target_h}",
        f"影像數      : {len(stems)}",
        f"測試 RGB 來源 : {RGB_DIR}",
        "資料集      : CFD（Crack Forest Dataset）— 完全未見資料集，無資料洩漏",
        f"讀取方式    : 直接讀取資料夾內所有 jpg/jpeg/png 圖片",
        f"輸出目錄    : {OUT_BASE}",
        f"運算裝置    : {device_info['device']}",
        f"CPU         : {device_info['cpu_name']}",
        f"CUDA 可用   : {device_info['cuda_available']}",
        f"GPU         : {device_info['gpu_name']}",
        f"GPU 記憶體  : {device_info['gpu_total_memory_gb']}",
        f"PyTorch     : {device_info['torch_version']}",
    ]

    success_count = 0
    missing_count = 0
    crack_image_count = 0
    no_crack_image_count = 0
    total_instances = 0
    total_crack_px = 0
    coverage_list = []
    elapsed_list = []
    ram_list = []
    cpu_percent_list = []
    gpu_allocated_list = []
    gpu_reserved_list = []

    print(f"Processing {len(stems)} images...")
    inference_start = time.perf_counter()

    for i, stem in enumerate(stems):
        rgb_p = rgb_index.get(stem.lower())
        if rgb_p is None:
            missing_count += 1
            csv_rows.append({
                "image_id": stem,
                "status": "RGB not found",
                "instances": "",
                "crack_pixels": "",
                "coverage_percent": "",
                "judgement": "",
                "elapsed_sec": "",
                "process_ram_mb": "",
                "process_cpu_percent": "",
                "gpu_allocated_mb": "",
                "gpu_reserved_mb": "",
            })
            continue

        item_start = time.perf_counter()

        im_original = cv2.imread(rgb_p)
        if im_original is None:
            missing_count += 1
            csv_rows.append({
                "image_id": stem,
                "status": "RGB read failed",
                "instances": "",
                "crack_pixels": "",
                "coverage_percent": "",
                "judgement": "",
                "elapsed_sec": "",
                "process_ram_mb": "",
                "process_cpu_percent": "",
                "gpu_allocated_mb": "",
                "gpu_reserved_mb": "",
            })
            continue

        original_h, original_w = im_original.shape[:2]

        # whole-image + 統一尺寸，不切 patch
        im_resized = cv2.resize(
            im_original,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )

        outputs = predictor(im_resized)
        instances = outputs["instances"].to("cpu")

        combined_resized = np.zeros((target_h, target_w), dtype=np.uint8)
        n_inst = len(instances)

        if n_inst > 0 and instances.has("pred_masks"):
            for m in instances.pred_masks.numpy():
                combined_resized = np.where(m, 255, combined_resized)

        # resize 回原圖大小，方便疊圖與以原始尺寸統計
        combined = cv2.resize(
            combined_resized,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        rgb_original = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)

        shutil.copy(rgb_p, before_dir / f"{stem}.png")

        # 儲存純二值 mask（供 GT 評估使用）
        Image.fromarray(combined).save(pred_masks_dir / f"{stem}.png")

        overlay = overlay_mask(rgb_original, combined)
        Image.fromarray(overlay).save(after_dir / f"{stem}.png")

        crack_px = int((combined >= 128).sum())
        coverage = crack_px / (original_h * original_w) * 100
        judged = "有裂縫" if crack_px > 0 else "無裂縫"

        if crack_px > 0:
            crack_image_count += 1
        else:
            no_crack_image_count += 1

        elapsed = time.perf_counter() - item_start
        resource_now = get_process_resource_snapshot(device)

        success_count += 1
        total_instances += n_inst
        total_crack_px += crack_px
        coverage_list.append(coverage)
        elapsed_list.append(elapsed)
        ram_list.append(resource_now['process_ram_mb'])
        cpu_percent_list.append(resource_now['process_cpu_percent'])
        gpu_allocated_list.append(resource_now['gpu_allocated_mb'])
        gpu_reserved_list.append(resource_now['gpu_reserved_mb'])

        csv_rows.append({
            "image_id": stem,
            "status": "OK",
            "instances": n_inst,
            "crack_pixels": crack_px,
            "coverage_percent": round(coverage, 6),
            "judgement": judged,
            "elapsed_sec": round(elapsed, 6),
            "process_ram_mb": round(resource_now["process_ram_mb"], 3),
            "process_cpu_percent": round(resource_now["process_cpu_percent"], 3),
            "gpu_allocated_mb": round(resource_now["gpu_allocated_mb"], 3),
            "gpu_reserved_mb": round(resource_now["gpu_reserved_mb"], 3),
        })

        sys.stdout.write(f"\r  [{i+1}/{len(stems)}] {stem}  inst={n_inst}  time={elapsed:.3f}s")
        sys.stdout.flush()

    inference_elapsed = time.perf_counter() - inference_start
    program_elapsed = time.perf_counter() - program_start

    judge_rate = (crack_image_count / success_count * 100) if success_count > 0 else 0.0
    avg_coverage = float(np.mean(coverage_list)) if coverage_list else 0.0
    avg_time = float(np.mean(elapsed_list)) if elapsed_list else 0.0
    min_time = float(np.min(elapsed_list)) if elapsed_list else 0.0
    max_time = float(np.max(elapsed_list)) if elapsed_list else 0.0
    avg_instances = (total_instances / success_count) if success_count > 0 else 0.0
    resource_end = get_process_resource_snapshot(device)
    avg_ram = float(np.mean(ram_list)) if ram_list else 0.0
    peak_ram = float(np.max(ram_list)) if ram_list else 0.0
    avg_cpu_percent = float(np.mean(cpu_percent_list)) if cpu_percent_list else 0.0
    peak_cpu_percent = float(np.max(cpu_percent_list)) if cpu_percent_list else 0.0
    avg_gpu_allocated = float(np.mean(gpu_allocated_list)) if gpu_allocated_list else 0.0
    peak_gpu_allocated = float(np.max(gpu_allocated_list)) if gpu_allocated_list else 0.0
    avg_gpu_reserved = float(np.mean(gpu_reserved_list)) if gpu_reserved_list else 0.0
    peak_gpu_reserved = float(np.max(gpu_reserved_list)) if gpu_reserved_list else 0.0

    log_lines += [
        "",
        "-" * 112,
        "整體統計",
        f"總影像數          : {len(stems)} 張",
        f"成功處理影像數    : {success_count} 張",
        f"缺少 RGB 影像數   : {missing_count} 張",
        f"判定有裂縫影像數  : {crack_image_count} 張",
        f"判定無裂縫影像數  : {no_crack_image_count} 張",
        f"判斷率            : {judge_rate:.2f}%",
        f"總實例數          : {total_instances} 個",
        f"平均每張實例數    : {avg_instances:.2f} 個",
        f"總裂縫像素數      : {total_crack_px}px",
        f"平均覆蓋率        : {avg_coverage:.4f}%",
        f"推論處理總耗時    : {inference_elapsed:.3f} 秒",
        f"程式總耗時        : {program_elapsed:.3f} 秒",
        f"平均每張耗時      : {avg_time:.3f} 秒",
        f"最快單張耗時      : {min_time:.3f} 秒",
        f"最慢單張耗時      : {max_time:.3f} 秒",
        "-" * 112,
        "計算資源消耗",
        f"psutil 可用        : {_PSUTIL_AVAILABLE}",
        f"開始 Process RAM   : {resource_start['process_ram_mb']:.2f} MB",
        f"結束 Process RAM   : {resource_end['process_ram_mb']:.2f} MB",
        f"平均 Process RAM   : {avg_ram:.2f} MB",
        f"峰值 Process RAM   : {peak_ram:.2f} MB",
        f"平均 Process CPU   : {avg_cpu_percent:.2f}%",
        f"峰值 Process CPU   : {peak_cpu_percent:.2f}%",
        f"平均 GPU allocated : {avg_gpu_allocated:.2f} MB",
        f"峰值 GPU allocated : {peak_gpu_allocated:.2f} MB",
        f"平均 GPU reserved  : {avg_gpu_reserved:.2f} MB",
        f"峰值 GPU reserved  : {peak_gpu_reserved:.2f} MB",
        f"PyTorch GPU peak allocated : {resource_end['gpu_max_allocated_mb']:.2f} MB",
        f"PyTorch GPU peak reserved  : {resource_end['gpu_max_reserved_mb']:.2f} MB",
        "-" * 112,
        "硬體資訊",
        f"運算裝置          : {device_info['device']}",
        f"CPU               : {device_info['cpu_name']}",
        f"CUDA 可用         : {device_info['cuda_available']}",
        f"GPU               : {device_info['gpu_name']}",
        f"GPU 記憶體        : {device_info['gpu_total_memory_gb']}",
        f"PyTorch           : {device_info['torch_version']}",
        "-" * 112,
        f"完成：{success_count} / {len(stems)} 張",
        f"before 目錄: {before_dir}",
        f"after  目錄: {after_dir}",
    ]

    csv_fields = [
        "image_id",
        "status",
        "instances",
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

    log_lines.append(f"逐張圖片 CSV: {csv_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("\n\n✅ 完成")
    print(f"   logs  → {log_path}")
    print(f"   csv   → {csv_path}")
    print(f"   before    → {before_dir}")
    print(f"   after     → {after_dir}")
    print(f"   pred_masks→ {pred_masks_dir}")


if __name__ == "__main__":
    main()
