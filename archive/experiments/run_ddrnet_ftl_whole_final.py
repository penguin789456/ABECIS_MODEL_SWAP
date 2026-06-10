"""
DDRNet FTL whole-image 推論腳本 — 統一比較版
輸出目錄：H:\ChihleeMaster\dev\final_outputs\
  logs/   → ddrnet_ftl_whole_log.txt
  before/ → 原始 RGB 影像
  after/  → 原始影像 + 紅色裂縫遮罩疊合圖

目的：
  與 ABECIS / Detectron2 採用相同 whole-image inference 條件比較。
  本腳本不切 patch，會先將完整影像 resize 到固定尺寸後推論。

啟動方式（CrackSeg conda env）：
  conda activate CrackSeg
  cd H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP
  python run_ddrnet_ftl_whole_final.py

指定設定：
  python run_ddrnet_ftl_whole_final.py --threshold 0.30 --target-size 1024 1024 --device auto
"""

import sys, os, argparse, shutil, time, platform, subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'realtime-semantic-segmentation-pytorch'))

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from data.transforms import get_test_transforms
from training.train_crackseg import build_model

# ── 設定 ────────────────────────────────────────────────────────────────────
CFG_PATH   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ddrnet_ftl.yaml'
CKPT_PATH  = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\checkpoints\ddrnet_ftl\20260501_181112\best.pth'
OUT_BASE   = Path(r'H:\ChihleeMaster\dev\final_outputs')
MODEL_NAME = 'ddrnet_ftl_whole'


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


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("⚠️ 指定 cuda，但目前 CUDA 不可用，改用 CPU。")
        return torch.device("cpu")
    return torch.device(device_arg)


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


def infer_whole_image(
    image: np.ndarray,
    model,
    device: torch.device,
    transform,
    threshold: float = 0.30,
    target_size=(1024, 1024)
) -> np.ndarray:
    """
    整張影像推論，不切 patch。
    流程：
      1. 原圖 resize 到 target_size
      2. 模型推論
      3. threshold 產生二值 mask
      4. mask resize 回原圖大小，方便疊圖與統計
    """
    original_h, original_w = image.shape[:2]
    target_w, target_h = target_size

    resized = np.array(
        Image.fromarray(image).resize((target_w, target_h), Image.BILINEAR)
    )

    model.eval()
    with torch.no_grad():
        aug = transform(image=resized)
        tensor = aug["image"].unsqueeze(0).float().to(device)

        logit = model(tensor)

        # 兼容部分模型輸出 tuple/list 的狀況
        if isinstance(logit, (tuple, list)):
            logit = logit[0]

        logit = logit.squeeze().detach().cpu().numpy()

        # 如果輸出尺寸與 target size 不一致，resize 回 target size
        if logit.shape != (target_h, target_w):
            logit_img = Image.fromarray(logit.astype(np.float32))
            logit = np.array(logit_img.resize((target_w, target_h), Image.BILINEAR))

        prob = 1.0 / (1.0 + np.exp(-logit))
        mask = (prob > threshold).astype(np.uint8) * 255

    mask = Image.fromarray(mask).resize((original_w, original_h), Image.NEAREST)
    return np.array(mask)


def main():
    program_start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="推論閾值（預設 0.30）")
    parser.add_argument("--target-size", type=int, nargs=2, default=[1024, 1024],
                        metavar=("WIDTH", "HEIGHT"),
                        help="whole-image 推論前統一 resize 尺寸，例如 --target-size 1024 1024")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="運算裝置：auto / cpu / cuda")
    args = parser.parse_args()

    threshold = args.threshold
    target_size = tuple(args.target_size)
    device = select_device(args.device)

    # 建立輸出目錄
    log_dir    = OUT_BASE / "logs"
    before_dir = OUT_BASE / "before" / MODEL_NAME
    after_dir  = OUT_BASE / "after"  / MODEL_NAME
    for d in [log_dir, before_dir, after_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 載入設定
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_info = get_device_info(device)

    print(f"Device     : {device}")
    print(f"Threshold  : {threshold}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Checkpoint : {CKPT_PATH}")

    # 載入模型
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print("Model loaded.\n")

    transform = get_test_transforms()
    ds_cfg = cfg["dataset"]
    rgb_dir = Path(ds_cfg["root"]) / "rgb"
    split_file = Path(ds_cfg["splits_dir"]) / "test.txt"

    with open(split_file) as f:
        stems = [l.strip() for l in f if l.strip()]

    rgb_index = {
        p.stem.lower(): p for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    }

    log_path = log_dir / f"{MODEL_NAME}_log.txt"
    log_lines = [
        "DDRNet FTL Whole-image 推論記錄",
        f"時間        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint  : {CKPT_PATH}",
        f"閾值        : {threshold}",
        f"輸入尺寸    : {target_size[0]}x{target_size[1]}",
        f"影像數      : {len(stems)}",
        f"輸出目錄    : {OUT_BASE}",
        f"運算裝置    : {device_info['device']}",
        f"CPU         : {device_info['cpu_name']}",
        f"CUDA 可用   : {device_info['cuda_available']}",
        f"GPU         : {device_info['gpu_name']}",
        f"GPU 記憶體  : {device_info['gpu_total_memory_gb']}",
        f"PyTorch     : {device_info['torch_version']}",
        "-" * 78,
        f'{"影像ID":<12}  {"裂縫像素數":>12}  {"覆蓋率":>8}  {"判定":>6}  {"耗時秒":>8}',
        "-" * 78,
    ]

    success_count = 0
    missing_count = 0
    crack_image_count = 0
    no_crack_image_count = 0
    total_crack_px = 0
    coverage_list = []
    elapsed_list = []

    print(f"Processing {len(stems)} images...")
    inference_start = time.perf_counter()

    for stem in tqdm(stems, desc="DDRNet FTL Whole"):
        rgb_p = rgb_index.get(stem.lower())
        if rgb_p is None:
            missing_count += 1
            log_lines.append(f"{stem:<12}  RGB not found")
            continue

        item_start = time.perf_counter()

        rgb = np.array(Image.open(rgb_p).convert("RGB"))
        mask = infer_whole_image(
            rgb, model, device, transform,
            threshold=threshold,
            target_size=target_size
        )

        shutil.copy(str(rgb_p), before_dir / f"{stem}.png")

        overlay = overlay_mask(rgb, mask)
        Image.fromarray(overlay).save(after_dir / f"{stem}.png")

        crack_px = int((mask >= 128).sum())
        coverage = crack_px / (mask.shape[0] * mask.shape[1]) * 100
        judged = "有裂縫" if crack_px > 0 else "無裂縫"

        if crack_px > 0:
            crack_image_count += 1
        else:
            no_crack_image_count += 1

        elapsed = time.perf_counter() - item_start

        success_count += 1
        total_crack_px += crack_px
        coverage_list.append(coverage)
        elapsed_list.append(elapsed)

        log_lines.append(
            f"{stem:<12}  {crack_px:>12}px  {coverage:>7.2f}%  {judged:>6}  {elapsed:>7.3f}s"
        )

    inference_elapsed = time.perf_counter() - inference_start
    program_elapsed = time.perf_counter() - program_start

    judge_rate = (crack_image_count / success_count * 100) if success_count > 0 else 0.0
    avg_coverage = float(np.mean(coverage_list)) if coverage_list else 0.0
    avg_time = float(np.mean(elapsed_list)) if elapsed_list else 0.0
    min_time = float(np.min(elapsed_list)) if elapsed_list else 0.0
    max_time = float(np.max(elapsed_list)) if elapsed_list else 0.0

    log_lines += [
        "-" * 78,
        "整體統計",
        f"總影像數          : {len(stems)} 張",
        f"成功處理影像數    : {success_count} 張",
        f"缺少 RGB 影像數   : {missing_count} 張",
        f"判定有裂縫影像數  : {crack_image_count} 張",
        f"判定無裂縫影像數  : {no_crack_image_count} 張",
        f"判斷率            : {judge_rate:.2f}%",
        f"總裂縫像素數      : {total_crack_px}px",
        f"平均覆蓋率        : {avg_coverage:.4f}%",
        f"推論處理總耗時    : {inference_elapsed:.3f} 秒",
        f"程式總耗時        : {program_elapsed:.3f} 秒",
        f"平均每張耗時      : {avg_time:.3f} 秒",
        f"最快單張耗時      : {min_time:.3f} 秒",
        f"最慢單張耗時      : {max_time:.3f} 秒",
        "-" * 78,
        "硬體資訊",
        f"運算裝置          : {device_info['device']}",
        f"CPU               : {device_info['cpu_name']}",
        f"CUDA 可用         : {device_info['cuda_available']}",
        f"GPU               : {device_info['gpu_name']}",
        f"GPU 記憶體        : {device_info['gpu_total_memory_gb']}",
        f"PyTorch           : {device_info['torch_version']}",
        "-" * 78,
        f"完成：{success_count} / {len(stems)} 張",
        f"before 目錄: {before_dir}",
        f"after  目錄: {after_dir}",
    ]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("\n✅ 完成")
    print(f"   logs  → {log_path}")
    print(f"   before→ {before_dir}")
    print(f"   after → {after_dir}")


if __name__ == "__main__":
    main()
